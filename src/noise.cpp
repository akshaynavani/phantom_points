#include "phantom/noise.hpp"

#include <cmath>

#include "phantom/math.hpp"

namespace phantom {

float compute_intensity(const IntensityParams& p, float reflectivity, float retro, float r,
                        float cos_theta) {
    cos_theta = clamp(cos_theta, 0.f, 1.f);
    float lambert = reflectivity * cos_theta;
    float retro_term = retro * std::pow(cos_theta, p.retro_exponent);
    float re = std::max(r, p.near_field_range);
    float falloff = std::pow(p.reference_range / re, p.falloff_exponent);
    float atmos = std::exp(-2.f * p.attenuation * r);
    float I = p.gain * (lambert + retro_term) * falloff * atmos;
    return clamp(I, 0.f, p.saturation);
}

float detection_probability(const NoiseParams& n, float intensity, float r, float max_range) {
    float p = 1.f - n.dropout_base;
    if (max_range > 0.f) {
        float q = r / max_range;
        p *= 1.f - clamp(n.dropout_range_coeff * q * q, 0.f, 1.f);
    }
    if (n.detect_softness > 0.f) {
        float z = (intensity - n.detect_threshold) / n.detect_softness;
        p *= 1.f / (1.f + std::exp(-z));
    } else if (intensity < n.detect_threshold) {
        p = 0.f;
    }
    return clamp(p, 0.f, 1.f);
}

float corrupt_intensity(const NoiseParams& n, float intensity, float saturation, Rng& rng) {
    float I = intensity * (1.f + rng.normal(0.f, n.intensity_noise_rel));
    I = clamp(I, 0.f, saturation);
    if (n.quantize_intensity && saturation > 0.f) {
        I = std::round(I / saturation * 255.f) / 255.f * saturation;
    }
    return I;
}

}  // namespace phantom
