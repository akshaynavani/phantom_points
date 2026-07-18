#include "phantom/generator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "phantom/io.hpp"
#include "phantom/params.hpp"

namespace fs = std::filesystem;

namespace phantom {

namespace {

Config load_or_empty(const std::string& path) {
    return path.empty() ? Config{} : Config::load(path);
}

// Stream ids derived from the per-scene seed so each stage is independently reproducible.
constexpr uint64_t kStreamScan = 1;
constexpr uint64_t kStreamRef = 2;
constexpr uint64_t kStreamRefine = 3;

}  // namespace

BatchGenerator::BatchGenerator(GeneratorOptions opts)
    : opts_(std::move(opts)),
      config_(load_or_empty(opts_.config_path)),
      scene_gen_(scene_params_from_config(config_)),
      sim_(lidar_params_from_config(config_)) {
    if (!opts_.pair_config_path.empty()) {
        // The reference sensor inherits everything from the main config and overrides
        // only what the pair config specifies (typically noise / intensity response).
        Config merged = config_;
        for (const auto& kv : Config::load(opts_.pair_config_path).values()) merged.set(kv.first, kv.second);
        ref_sim_ = std::make_unique<LidarSimulator>(lidar_params_from_config(merged));
        const LidarParams& a = sim_.params();
        const LidarParams& b = ref_sim_->params();
        if (a.beams != b.beams || a.columns != b.columns)
            throw std::invalid_argument("pair config must keep the same beam count and columns");
    }
    if (!opts_.model_path.empty()) {
        refiner_ = PointNetRefiner::load(opts_.model_path);
        opts_.refine.intensity_saturation = sim_.params().intensity.saturation;
    }
    if (opts_.threads == 0) opts_.threads = std::max(1u, std::thread::hardware_concurrency());
}

BatchGenerator::SceneResult BatchGenerator::generate_one(size_t index) const {
    SceneResult res;
    res.index = index;
    const std::string name = scan_name(index);
    const fs::path root(opts_.out_dir);

    if (opts_.skip_existing && fs::exists(root / "meta" / (name + ".json"))) {
        res.skipped = true;
        return res;
    }

    const uint64_t scene_seed = Rng::derive(opts_.seed, index);
    Scene scene = scene_gen_.generate(scene_seed);
    Scan scan = sim_.simulate(scene, Rng::derive(scene_seed, kStreamScan));

    if (ref_sim_) {
        Scan ref = ref_sim_->simulate(scene, Rng::derive(scene_seed, kStreamRef));
        write_scan_files((root / "ref").string(), name, ref);
    }

    size_t dropped = 0;
    if (refiner_.loaded()) {
        Rng rng(Rng::derive(scene_seed, kStreamRefine));
        dropped = refiner_.refine(scan, rng, opts_.refine).dropped_points;
    }

    write_scan_files(root.string(), name, scan);
    write_boxes((root / "boxes" / (name + ".txt")).string(), scan);
    write_scene_meta((root / "meta" / (name + ".json")).string(), index, scene, scan, dropped);

    res.layout = scene.layout;
    res.points = scan.points.size();
    res.objects = scan.objects.size();
    res.seed = scene_seed;
    return res;
}

GenerationSummary BatchGenerator::run(const Progress& progress) {
    prepare_dataset_dirs(opts_.out_dir, true);
    if (ref_sim_) prepare_dataset_dirs((fs::path(opts_.out_dir) / "ref").string(), false);
    write_dataset_json();

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<SceneResult> results(opts_.count);
    std::atomic<size_t> next{0}, done{0};
    std::mutex err_mu, progress_mu;
    std::string first_error;

    auto worker = [&]() {
        for (;;) {
            size_t k = next.fetch_add(1);
            if (k >= opts_.count) return;
            {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!first_error.empty()) return;
            }
            try {
                results[k] = generate_one(opts_.start_index + k);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (first_error.empty())
                    first_error = "scene " + std::to_string(opts_.start_index + k) + ": " + e.what();
                return;
            }
            size_t d = done.fetch_add(1) + 1;
            if (progress) {
                std::lock_guard<std::mutex> lk(progress_mu);
                progress(d, opts_.count);
            }
        }
    };

    std::vector<std::thread> pool;
    unsigned n_threads = static_cast<unsigned>(std::min<size_t>(opts_.threads, std::max<size_t>(opts_.count, 1)));
    for (unsigned t = 0; t < n_threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    if (!first_error.empty()) throw std::runtime_error(first_error);

    write_index(results);

    GenerationSummary s;
    for (const auto& r : results) {
        if (r.skipped) { ++s.skipped; continue; }
        ++s.scenes;
        s.total_points += r.points;
        s.total_objects += r.objects;
    }
    s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return s;
}

void BatchGenerator::write_dataset_json() const {
    const LidarParams& P = sim_.params();
    std::ofstream out(fs::path(opts_.out_dir) / "dataset.json");
    if (!out) throw std::runtime_error("cannot write dataset.json");
    out << "{\n";
    out << "  \"generator\": \"phantom_points\",\n";
    out << "  \"seed\": " << opts_.seed << ",\n";
    out << "  \"config\": \"" << json_escape(opts_.config_path) << "\",\n";
    out << "  \"pair_config\": \"" << json_escape(opts_.pair_config_path) << "\",\n";
    out << "  \"refiner\": \"" << json_escape(opts_.model_path) << "\",\n";
    out << "  \"sensor\": {\n";
    out << "    \"beams\": " << P.beams << ",\n";
    out << "    \"columns\": " << P.columns << ",\n";
    out << "    \"min_range\": " << P.min_range << ",\n";
    out << "    \"max_range\": " << P.max_range << ",\n";
    out << "    \"sensor_height\": " << P.sensor_height << ",\n";
    out << "    \"noise_enabled\": " << (P.enable_noise ? "true" : "false") << ",\n";
    out << "    \"elevations_deg\": [";
    for (size_t i = 0; i < P.elevations_deg.size(); ++i) out << (i ? ", " : "") << P.elevations_deg[i];
    out << "]\n  },\n";
    out << "  \"classes\": [";
    for (int c = 0; c < static_cast<int>(SemanticClass::Count); ++c)
        out << (c ? ", " : "") << '"' << class_name(static_cast<SemanticClass>(c)) << '"';
    out << "]\n}\n";
}

void BatchGenerator::write_index(const std::vector<SceneResult>& rows) const {
    const fs::path path = fs::path(opts_.out_dir) / "index.csv";
    // Merge with an existing index so sharded runs (different start_index) accumulate.
    std::map<size_t, std::string> lines;
    if (fs::exists(path)) {
        std::ifstream in(path);
        std::string line;
        std::getline(in, line);  // header
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            size_t comma = line.find(',');
            lines[std::stoull(line.substr(0, comma))] = line;
        }
    }
    for (const auto& r : rows) {
        if (r.skipped) continue;
        std::ostringstream ss;
        ss << r.index << ',' << r.layout << ',' << r.points << ',' << r.objects << ',' << r.seed;
        lines[r.index] = ss.str();
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write index.csv");
    out << "scene_id,layout,num_points,num_objects,seed\n";
    for (const auto& kv : lines) out << kv.second << '\n';
}

}  // namespace phantom
