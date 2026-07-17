#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "phantom/lidar.hpp"
#include "phantom/rng.hpp"

namespace phantom {

/// Dense layer with BatchNorm already folded in: y = W x + b, optional ReLU.
struct DenseLayer {
    int in_dim = 0;
    int out_dim = 0;
    bool relu = false;
    std::vector<float> weight;  // out_dim x in_dim, row-major
    std::vector<float> bias;    // out_dim

    /// Apply to `n` row vectors stored contiguously in `x` (n x in_dim) -> `y` (n x out_dim).
    void forward(const float* x, int n, float* y) const;
};

struct RefineOptions {
    bool sample_dropout = true;   // Bernoulli on sigmoid(logit); false = threshold at 0.5
    bool refine_intensity = true;
    bool refine_range = true;
    bool allow_dropout = true;
    float intensity_saturation = 1.0f;
};

struct RefineStats {
    size_t input_points = 0;
    size_t dropped_points = 0;
    double mean_abs_delta_range = 0.0;
    double mean_abs_delta_intensity = 0.0;
};

/// C++ inference for the sector-wise PointNet refiner trained in python/phantom_points.
/// Weights come from the `.ppnt` file written by `phantom_points.export`.
class PointNetRefiner {
public:
    PointNetRefiner() = default;

    static PointNetRefiner load(const std::string& path);

    bool loaded() const { return !local_.empty(); }
    int in_features() const { return in_features_; }
    int num_classes() const { return num_classes_; }
    int sectors() const { return sectors_; }

    /// Per-point features; must match python/phantom_points/features.py exactly.
    void build_features(const LidarPoint& p, int beams, float* out) const;

    /// Forward pass on one sector: `feats` is n x in_features, returns n x 3 raw outputs
    /// [delta_range_raw, delta_intensity_raw, drop_logit].
    std::vector<float> forward(const std::vector<float>& feats, int n) const;

    /// Refine a scan in place: bounded range/intensity residuals and learned dropout.
    RefineStats refine(Scan& scan, Rng& rng, const RefineOptions& opts = {}) const;

    /// Construct from in-memory layers (used by tests).
    static PointNetRefiner from_layers(std::vector<DenseLayer> local, std::vector<DenseLayer> global,
                                       std::vector<DenseLayer> head, int num_classes, int sectors,
                                       float feature_scale, float max_dr, float max_di);

private:
    void validate() const;

    std::vector<DenseLayer> local_, global_, head_;
    int in_features_ = 0;
    int num_classes_ = 0;
    int sectors_ = 32;
    float feature_scale_ = 50.f;
    float max_delta_range_ = 0.5f;
    float max_delta_intensity_ = 0.3f;
};

}  // namespace phantom
