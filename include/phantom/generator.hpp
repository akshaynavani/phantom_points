#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "phantom/config.hpp"
#include "phantom/lidar.hpp"
#include "phantom/pointnet.hpp"
#include "phantom/scene.hpp"

namespace phantom {

struct GeneratorOptions {
    std::string out_dir = "out/dataset";
    std::string config_path;       // main sensor + scene config (INI)
    std::string pair_config_path;  // optional reference sensor config -> <out>/ref
    std::string model_path;        // optional .ppnt refiner weights
    size_t count = 100;
    size_t start_index = 0;
    uint64_t seed = 1;
    unsigned threads = 0;          // 0 = hardware concurrency
    bool skip_existing = false;
    RefineOptions refine;
};

struct GenerationSummary {
    size_t scenes = 0;
    size_t skipped = 0;
    size_t total_points = 0;
    size_t total_objects = 0;
    double seconds = 0.0;
};

/// Generates labeled scans in parallel. Scene `i` is fully determined by (seed, i), so
/// datasets can be produced in shards (`start_index`) and any scene can be regenerated.
class BatchGenerator {
public:
    explicit BatchGenerator(GeneratorOptions opts);

    using Progress = std::function<void(size_t done, size_t total)>;
    GenerationSummary run(const Progress& progress = nullptr);

    const LidarParams& lidar_params() const { return sim_.params(); }

private:
    struct SceneResult {
        size_t index = 0;
        std::string layout;
        size_t points = 0;
        size_t objects = 0;
        uint64_t seed = 0;
        bool skipped = false;
    };

    SceneResult generate_one(size_t index) const;
    void write_dataset_json() const;
    void write_index(const std::vector<SceneResult>& rows) const;

    GeneratorOptions opts_;
    Config config_;
    SceneGenerator scene_gen_;
    LidarSimulator sim_;
    std::unique_ptr<LidarSimulator> ref_sim_;
    PointNetRefiner refiner_;
};

}  // namespace phantom
