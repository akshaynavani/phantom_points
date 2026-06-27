#include "phantom/params.hpp"

namespace phantom {

SceneParams scene_params_from_config(const Config& c) {
    SceneParams p;
    p.road_half_width = c.get_float("scene.road_half_width", p.road_half_width);
    p.sidewalk_width = c.get_float("scene.sidewalk_width", p.sidewalk_width);
    p.scene_extent = c.get_float("scene.extent", p.scene_extent);
    p.cars_min = c.get_int("scene.cars_min", p.cars_min);
    p.cars_max = c.get_int("scene.cars_max", p.cars_max);
    p.trucks_min = c.get_int("scene.trucks_min", p.trucks_min);
    p.trucks_max = c.get_int("scene.trucks_max", p.trucks_max);
    p.pedestrians_min = c.get_int("scene.pedestrians_min", p.pedestrians_min);
    p.pedestrians_max = c.get_int("scene.pedestrians_max", p.pedestrians_max);
    p.cyclists_min = c.get_int("scene.cyclists_min", p.cyclists_min);
    p.cyclists_max = c.get_int("scene.cyclists_max", p.cyclists_max);
    p.poles_min = c.get_int("scene.poles_min", p.poles_min);
    p.poles_max = c.get_int("scene.poles_max", p.poles_max);
    p.trees_min = c.get_int("scene.trees_min", p.trees_min);
    p.trees_max = c.get_int("scene.trees_max", p.trees_max);
    p.buildings_min = c.get_int("scene.buildings_min", p.buildings_min);
    p.buildings_max = c.get_int("scene.buildings_max", p.buildings_max);
    p.p_intersection = c.get_float("scene.p_intersection", p.p_intersection);
    p.p_parking = c.get_float("scene.p_parking", p.p_parking);
    p.sensor_clearance = c.get_float("scene.sensor_clearance", p.sensor_clearance);
    return p;
}

LidarParams lidar_params_from_config(const Config& c) {
    LidarParams p;
    p.columns = c.get_int("lidar.columns", p.columns);
    p.beams = c.get_int("lidar.beams", p.beams);
    p.elevation_min_deg = c.get_float("lidar.elevation_min_deg", p.elevation_min_deg);
    p.elevation_max_deg = c.get_float("lidar.elevation_max_deg", p.elevation_max_deg);
    p.elevations_deg = c.get_floats("lidar.elevations_deg");
    p.min_range = c.get_float("lidar.min_range", p.min_range);
    p.max_range = c.get_float("lidar.max_range", p.max_range);
    p.sensor_height = c.get_float("lidar.sensor_height", p.sensor_height);
    p.beam_divergence_mrad = c.get_float("lidar.beam_divergence_mrad", p.beam_divergence_mrad);
    p.edge_samples = c.get_int("lidar.edge_samples", p.edge_samples);
    p.enable_noise = c.get_bool("noise.enabled", p.enable_noise);

    NoiseParams& n = p.noise;
    n.range_sigma_base = c.get_float("noise.range_sigma_base", n.range_sigma_base);
    n.range_sigma_rel = c.get_float("noise.range_sigma_rel", n.range_sigma_rel);
    n.azimuth_jitter_deg = c.get_float("noise.azimuth_jitter_deg", n.azimuth_jitter_deg);
    n.elevation_jitter_deg = c.get_float("noise.elevation_jitter_deg", n.elevation_jitter_deg);
    n.dropout_base = c.get_float("noise.dropout_base", n.dropout_base);
    n.dropout_range_coeff = c.get_float("noise.dropout_range_coeff", n.dropout_range_coeff);
    n.detect_threshold = c.get_float("noise.detect_threshold", n.detect_threshold);
    n.detect_softness = c.get_float("noise.detect_softness", n.detect_softness);
    n.mixed_pixel_prob = c.get_float("noise.mixed_pixel_prob", n.mixed_pixel_prob);
    n.edge_depth_jump = c.get_float("noise.edge_depth_jump", n.edge_depth_jump);
    n.intensity_noise_rel = c.get_float("noise.intensity_noise_rel", n.intensity_noise_rel);
    n.quantize_intensity = c.get_bool("noise.quantize_intensity", n.quantize_intensity);
    n.spurious_per_scan = c.get_float("noise.spurious_per_scan", n.spurious_per_scan);

    IntensityParams& i = p.intensity;
    i.gain = c.get_float("intensity.gain", i.gain);
    i.reference_range = c.get_float("intensity.reference_range", i.reference_range);
    i.falloff_exponent = c.get_float("intensity.falloff_exponent", i.falloff_exponent);
    i.near_field_range = c.get_float("intensity.near_field_range", i.near_field_range);
    i.attenuation = c.get_float("intensity.attenuation", i.attenuation);
    i.retro_exponent = c.get_float("intensity.retro_exponent", i.retro_exponent);
    i.saturation = c.get_float("intensity.saturation", i.saturation);

    p.resolve_elevations();
    return p;
}

}  // namespace phantom
