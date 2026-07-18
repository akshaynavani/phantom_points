// phantom_gen — batch synthetic LiDAR scenario generator.
//
//   phantom_gen --out out/dataset --count 10000 --config configs/default.ini --threads 16
//   phantom_gen --out out/pairs --count 2000 --pair-config configs/reference_sensor.ini
//   phantom_gen --out out/refined --count 10000 --model runs/refiner/refiner.ppnt

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "phantom/generator.hpp"

namespace {

void usage() {
    std::cout <<
        "usage: phantom_gen [options]\n"
        "  --out DIR             output dataset directory (default out/dataset)\n"
        "  --count N             number of scenes to generate (default 100)\n"
        "  --start N             first scene index, for sharded generation (default 0)\n"
        "  --seed S              global seed (default 1)\n"
        "  --threads T           worker threads (default: all cores)\n"
        "  --config FILE         INI file with [scene] [lidar] [noise] [intensity]\n"
        "  --pair-config FILE    also render each scene with this reference sensor into DIR/ref\n"
        "  --model FILE          PointNet refiner weights (.ppnt) applied to every scan\n"
        "  --no-model-dropout    refiner adjusts range/intensity only, never drops points\n"
        "  --deterministic-drop  threshold refiner dropout at p=0.5 instead of sampling\n"
        "  --skip-existing       skip scenes whose meta/<id>.json already exists\n"
        "  -q, --quiet           no progress output\n"
        "  -h, --help            show this message\n";
}

std::string need(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    phantom::GeneratorOptions opts;
    bool quiet = false;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--out") opts.out_dir = need(i, argc, argv);
            else if (a == "--count") opts.count = std::stoull(need(i, argc, argv));
            else if (a == "--start") opts.start_index = std::stoull(need(i, argc, argv));
            else if (a == "--seed") opts.seed = std::stoull(need(i, argc, argv));
            else if (a == "--threads") opts.threads = static_cast<unsigned>(std::stoul(need(i, argc, argv)));
            else if (a == "--config") opts.config_path = need(i, argc, argv);
            else if (a == "--pair-config") opts.pair_config_path = need(i, argc, argv);
            else if (a == "--model") opts.model_path = need(i, argc, argv);
            else if (a == "--no-model-dropout") opts.refine.allow_dropout = false;
            else if (a == "--deterministic-drop") opts.refine.sample_dropout = false;
            else if (a == "--skip-existing") opts.skip_existing = true;
            else if (a == "-q" || a == "--quiet") quiet = true;
            else if (a == "-h" || a == "--help") { usage(); return 0; }
            else throw std::invalid_argument("unknown argument: " + a);
        }

        phantom::BatchGenerator gen(opts);
        const auto& lp = gen.lidar_params();
        if (!quiet) {
            std::cout << "phantom_gen: " << opts.count << " scenes -> " << opts.out_dir << " ("
                      << lp.beams << " beams x " << lp.columns << " columns"
                      << (opts.model_path.empty() ? "" : ", refiner on")
                      << (opts.pair_config_path.empty() ? "" : ", paired reference") << ")\n";
        }

        size_t step = std::max<size_t>(1, opts.count / 100);
        auto progress = [&](size_t done, size_t total) {
            if (quiet) return;
            if (done % step == 0 || done == total)
                std::cout << "\r  " << done << " / " << total << std::flush;
        };
        phantom::GenerationSummary s = gen.run(progress);

        if (!quiet) {
            std::cout << "\n  generated " << s.scenes << " scenes";
            if (s.skipped) std::cout << " (skipped " << s.skipped << " existing)";
            std::cout << " in " << s.seconds << " s";
            if (s.scenes) {
                std::cout << "\n  avg points/scan " << s.total_points / s.scenes << ", avg objects/scene "
                          << static_cast<double>(s.total_objects) / s.scenes;
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "phantom_gen: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
