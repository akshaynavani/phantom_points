#pragma once

#include <cstdint>
#include <vector>

#include "phantom/noise.hpp"
#include "phantom/scene.hpp"

namespace phantom {

enum PointFlags : uint8_t {
    kFlagNone = 0,
    kFlagMixedPixel = 1 << 0,  // range blended across a depth discontinuity
    kFlagSpurious = 1 << 1,    // phantom return not backed by geometry
    kFlagRefined = 1 << 2,     // modified by the PointNet refiner
};

/// One LiDAR return, expressed in the sensor frame (x forward, y left, z up).
struct LidarPoint {
    float x = 0, y = 0, z = 0;
    float intensity = 0;
    float range = 0;
    uint16_t ring = 0;     // beam index, 0 = lowest elevation
    uint16_t column = 0;   // azimuth bin
    uint16_t label = 0;    // SemanticClass
    uint16_t instance = 0; // 0 for stuff classes
    uint8_t flags = kFlagNone;
};

/// Per-object visibility statistics used for occlusion labels.
struct ObjectVisibility {
    uint16_t instance_id = 0;
    int expected_hits = 0;  // rays that would hit the object if nothing occluded it
    int visible_hits = 0;   // rays whose first hit is the object (before dropout)
    int final_points = 0;   // points surviving dropout
    float occlusion = 0.f;  // 1 - visible / expected
    int occlusion_level = 0;  // 0 visible, 1 partly, 2 largely, 3 unknown / fully occluded
};

struct LidarParams {
    int columns = 2048;              // azimuth steps per revolution
    std::vector<float> elevations_deg;  // ascending; filled from beams/min/max if empty
    int beams = 64;
    float elevation_min_deg = -24.8f;
    float elevation_max_deg = 2.0f;
    float min_range = 0.9f;
    float max_range = 120.f;
    float sensor_height = 1.73f;
    float beam_divergence_mrad = 3.0f;  // full-angle divergence; 0 disables edge sampling
    int edge_samples = 4;               // sub-rays on the divergence cone
    bool enable_noise = true;
    NoiseParams noise;
    IntensityParams intensity;

    /// Populate `elevations_deg` from (beams, min, max) when no explicit table was given.
    void resolve_elevations();
};

struct Scan {
    std::vector<LidarPoint> points;
    std::vector<ObjectBox> objects;  // ground truth, in the sensor frame
    std::vector<ObjectVisibility> visibility;  // parallel to `objects`
    int beams = 0;
    int columns = 0;
};

class LidarSimulator {
public:
    explicit LidarSimulator(LidarParams params);

    Scan simulate(const Scene& scene, uint64_t seed) const;

    const LidarParams& params() const { return params_; }
    float column_azimuth(int column) const;

private:
    Vec3 sensor_origin() const { return {0.f, 0.f, params_.sensor_height}; }
    void compute_visibility(const Scene& scene, Scan& scan,
                            const std::vector<int>& visible_hits) const;
    void add_spurious(Scan& scan, Rng& rng) const;

    LidarParams params_;
    std::vector<float> elev_rad_;
};

}  // namespace phantom
