#pragma once

#include "phantom/rng.hpp"

namespace phantom {

/// Return-level corruption parameters. Every term can be disabled by setting it to 0.
struct NoiseParams {
    // Range noise: sigma(r) = range_sigma_base + range_sigma_rel * r
    float range_sigma_base = 0.015f;  // m
    float range_sigma_rel = 0.0008f;  // fraction of range
    // Beam pointing jitter (encoder + mirror wobble).
    float azimuth_jitter_deg = 0.01f;
    float elevation_jitter_deg = 0.005f;
    // Random dropout independent of signal (packet loss, dust).
    float dropout_base = 0.01f;
    // Range-dependent dropout: p = dropout_range_coeff * (r / max_range)^2
    float dropout_range_coeff = 0.08f;
    // Soft detection threshold on received intensity (signal-limited dropout).
    float detect_threshold = 0.004f;
    float detect_softness = 0.002f;
    // Mixed-pixel / edge effect at depth discontinuities.
    float mixed_pixel_prob = 0.3f;
    float edge_depth_jump = 0.6f;  // m; sub-beam depth spread that counts as an edge
    // Multiplicative intensity noise and 8-bit quantisation.
    float intensity_noise_rel = 0.05f;
    bool quantize_intensity = true;
    // Spurious returns (rain / dust / multipath), expected count per scan.
    float spurious_per_scan = 30.f;
};

/// Physical intensity model: Lambertian + retro-reflective lobe, atmospheric attenuation,
/// inverse-power range falloff and receiver saturation.
struct IntensityParams {
    float gain = 1.0f;
    float reference_range = 10.0f;    // range at which a rho=1 normal target reads `gain`
    float falloff_exponent = 2.0f;    // 2 = ideal inverse-square
    float near_field_range = 2.0f;    // falloff is clamped below this range (optics overlap)
    float attenuation = 0.0015f;      // atmospheric extinction alpha (1/m); two-way: exp(-2 a r)
    float retro_exponent = 12.0f;     // sharpness of the retro-reflective lobe
    float saturation = 1.0f;          // receiver clipping level
};

/// Received intensity for a hit at range `r` with incidence cosine `cos_theta`.
float compute_intensity(const IntensityParams& p, float reflectivity, float retro, float r,
                        float cos_theta);

/// Standard deviation of range noise at range `r`.
inline float range_sigma(const NoiseParams& n, float r) {
    return n.range_sigma_base + n.range_sigma_rel * r;
}

/// Probability that a return with intensity `I` at range `r` is detected (not dropped).
float detection_probability(const NoiseParams& n, float intensity, float r, float max_range);

/// Apply multiplicative noise, saturation and optional 8-bit quantisation to an intensity.
float corrupt_intensity(const NoiseParams& n, float intensity, float saturation, Rng& rng);

}  // namespace phantom
