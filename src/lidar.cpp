#include "phantom/lidar.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace phantom {

void LidarParams::resolve_elevations() {
    if (!elevations_deg.empty()) {
        std::sort(elevations_deg.begin(), elevations_deg.end());
        beams = static_cast<int>(elevations_deg.size());
        return;
    }
    if (beams <= 0) throw std::invalid_argument("lidar needs at least one beam");
    elevations_deg.resize(beams);
    for (int i = 0; i < beams; ++i) {
        float t = beams == 1 ? 0.5f : static_cast<float>(i) / (beams - 1);
        elevations_deg[i] = elevation_min_deg + t * (elevation_max_deg - elevation_min_deg);
    }
}

LidarSimulator::LidarSimulator(LidarParams params) : params_(std::move(params)) {
    params_.resolve_elevations();
    if (params_.columns <= 0) throw std::invalid_argument("lidar needs at least one column");
    elev_rad_.reserve(params_.elevations_deg.size());
    for (float e : params_.elevations_deg) elev_rad_.push_back(deg2rad(e));
}

float LidarSimulator::column_azimuth(int column) const {
    return -kPi + 2.f * kPi * (static_cast<float>(column) + 0.5f) / params_.columns;
}

Scan LidarSimulator::simulate(const Scene& scene, uint64_t seed) const {
    Rng rng(seed);
    const LidarParams& P = params_;
    const NoiseParams& N = P.noise;
    const bool noisy = P.enable_noise;
    const Vec3 origin = sensor_origin();

    Scan scan;
    scan.beams = P.beams;
    scan.columns = P.columns;
    scan.points.reserve(static_cast<size_t>(P.beams) * P.columns / 2);

    uint16_t max_id = 0;
    for (const auto& o : scene.objects) max_id = std::max(max_id, o.instance_id);
    std::vector<int> visible_hits(static_cast<size_t>(max_id) + 1, 0);

    // Sub-ray offsets on the divergence cone (in azimuth / elevation radians).
    const float half_div = 0.5f * P.beam_divergence_mrad * 1e-3f;
    const int n_sub = (half_div > 0.f) ? P.edge_samples : 0;

    for (int c = 0; c < P.columns; ++c) {
        const float az0 = column_azimuth(c);
        for (int b = 0; b < P.beams; ++b) {
            float az = az0, el = elev_rad_[b];
            if (noisy) {
                az += rng.normal(0.f, deg2rad(N.azimuth_jitter_deg));
                el += rng.normal(0.f, deg2rad(N.elevation_jitter_deg));
            }
            Ray ray{origin, spherical_dir(az, el)};
            Hit hit;
            if (!scene.raycast(ray, P.min_range, P.max_range, hit)) continue;

            float r = hit.t;
            uint8_t flags = kFlagNone;

            // Edge / mixed-pixel effect: the beam footprint straddles a depth discontinuity.
            if (n_sub > 0) {
                float r_sum = r, w_sum = 1.f, spread = 0.f;
                bool any_miss = false;
                for (int k = 0; k < n_sub; ++k) {
                    float phi = 2.f * kPi * k / n_sub;
                    Ray sub{origin, spherical_dir(az + half_div * std::cos(phi),
                                                  el + half_div * std::sin(phi))};
                    Hit sh;
                    if (scene.raycast(sub, P.min_range, P.max_range, sh)) {
                        spread = std::max(spread, std::fabs(sh.t - r));
                        r_sum += sh.t;
                        w_sum += 1.f;
                    } else {
                        any_miss = true;
                    }
                }
                bool edge = spread > N.edge_depth_jump || any_miss;
                if (edge && noisy && rng.bernoulli(N.mixed_pixel_prob)) {
                    r = r_sum / w_sum;
                    flags |= kFlagMixedPixel;
                }
            }

            const Primitive& prim = *hit.prim;
            if (prim.instance_id > 0 && prim.instance_id <= max_id) ++visible_hits[prim.instance_id];

            float cos_theta = -dot(ray.dir, hit.normal);
            float I = compute_intensity(P.intensity, prim.reflectivity, prim.retro, r, cos_theta);

            if (noisy) {
                if (!rng.bernoulli(detection_probability(N, I, r, P.max_range))) continue;
                r += rng.normal(0.f, range_sigma(N, r));
                if (r < P.min_range) continue;
                I = corrupt_intensity(N, I, P.intensity.saturation, rng);
            }

            // Points are reported along the nominal (encoder) direction, not the jittered one:
            // this is what makes pointing jitter visible as geometric noise.
            Vec3 d = noisy ? spherical_dir(az0, elev_rad_[b]) : ray.dir;
            LidarPoint pt;
            pt.x = d.x * r;
            pt.y = d.y * r;
            pt.z = d.z * r;
            pt.range = r;
            pt.intensity = I;
            pt.ring = static_cast<uint16_t>(b);
            pt.column = static_cast<uint16_t>(c);
            pt.label = static_cast<uint16_t>(prim.label);
            pt.instance = prim.instance_id;
            pt.flags = flags;
            scan.points.push_back(pt);
        }
    }

    if (noisy) add_spurious(scan, rng);

    // Ground-truth boxes in the sensor frame.
    scan.objects = scene.objects;
    for (auto& o : scan.objects) o.center.z -= P.sensor_height;
    compute_visibility(scene, scan, visible_hits);
    return scan;
}

void LidarSimulator::add_spurious(Scan& scan, Rng& rng) const {
    const LidarParams& P = params_;
    int n = rng.poisson(P.noise.spurious_per_scan);
    for (int i = 0; i < n; ++i) {
        int c = rng.uniform_int(0, P.columns - 1);
        int b = rng.uniform_int(0, P.beams - 1);
        // Phantom returns cluster at short range (backscatter falls off quickly).
        float r = P.min_range + std::min(-std::log(std::max(rng.uniform(), 1e-6f)) * 6.f,
                                         P.max_range * 0.3f);
        Vec3 d = spherical_dir(column_azimuth(c), elev_rad_[b]);
        LidarPoint pt;
        pt.x = d.x * r;
        pt.y = d.y * r;
        pt.z = d.z * r;
        pt.range = r;
        pt.intensity = rng.uniform(0.f, 0.05f);
        pt.ring = static_cast<uint16_t>(b);
        pt.column = static_cast<uint16_t>(c);
        pt.label = static_cast<uint16_t>(SemanticClass::Unlabeled);
        pt.flags = kFlagSpurious;
        scan.points.push_back(pt);
    }
}

void LidarSimulator::compute_visibility(const Scene& scene, Scan& scan,
                                        const std::vector<int>& visible_hits) const {
    const LidarParams& P = params_;
    const Vec3 origin = sensor_origin();
    const float az_step = 2.f * kPi / P.columns;

    std::vector<int> final_points(visible_hits.size(), 0);
    for (const auto& pt : scan.points)
        if (pt.instance > 0 && pt.instance < final_points.size()) ++final_points[pt.instance];

    scan.visibility.clear();
    scan.visibility.reserve(scene.objects.size());
    for (const ObjectBox& obj : scene.objects) {
        ObjectVisibility vis;
        vis.instance_id = obj.instance_id;
        vis.visible_hits = obj.instance_id < visible_hits.size() ? visible_hits[obj.instance_id] : 0;
        vis.final_points = obj.instance_id < final_points.size() ? final_points[obj.instance_id] : 0;

        // Isolate the object's primitives into a mini-scene.
        Scene solo;
        for (const Primitive& p : scene.primitives)
            if (p.instance_id == obj.instance_id) solo.primitives.push_back(p);
        solo.finalize();

        // Angular window spanned by the box corners, relative to the box-centre azimuth.
        Vec3 rel = obj.center - origin;
        float az_c = std::atan2(rel.y, rel.x);
        float da_min = 0.f, da_max = 0.f, el_min = kPi, el_max = -kPi;
        for (int i = 0; i < 8; ++i) {
            Vec3 corner{(i & 1 ? 0.5f : -0.5f) * obj.size.x, (i & 2 ? 0.5f : -0.5f) * obj.size.y,
                        (i & 4 ? 0.5f : -0.5f) * obj.size.z};
            Vec3 w = obj.center + rotate_z(corner, obj.yaw) - origin;
            float da = std::atan2(w.y, w.x) - az_c;
            while (da > kPi) da -= 2.f * kPi;
            while (da < -kPi) da += 2.f * kPi;
            da_min = std::min(da_min, da);
            da_max = std::max(da_max, da);
            float el = std::atan2(w.z, std::hypot(w.x, w.y));
            el_min = std::min(el_min, el);
            el_max = std::max(el_max, el);
        }

        int expected = 0;
        int c0 = static_cast<int>(std::floor((az_c + da_min + kPi) / az_step)) - 1;
        int c1 = static_cast<int>(std::ceil((az_c + da_max + kPi) / az_step)) + 1;
        for (int ci = c0; ci <= c1; ++ci) {
            int c = ((ci % P.columns) + P.columns) % P.columns;
            float az = column_azimuth(c);
            for (int b = 0; b < P.beams; ++b) {
                float el = elev_rad_[b];
                if (el < el_min - 0.01f || el > el_max + 0.01f) continue;
                Hit h;
                if (solo.raycast(Ray{origin, spherical_dir(az, el)}, P.min_range, P.max_range, h))
                    ++expected;
            }
        }
        vis.expected_hits = expected;
        if (expected == 0) {
            vis.occlusion = 1.f;
            vis.occlusion_level = 3;
        } else {
            vis.occlusion = clamp(1.f - static_cast<float>(vis.visible_hits) / expected, 0.f, 1.f);
            vis.occlusion_level = vis.occlusion < 0.1f ? 0 : vis.occlusion < 0.5f ? 1
                                : vis.occlusion < 0.9f ? 2 : 3;
        }
        scan.visibility.push_back(vis);
    }
}

}  // namespace phantom
