// Minimal self-contained unit tests (no external framework). Run via `ctest` or directly.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "phantom/config.hpp"
#include "phantom/geometry.hpp"
#include "phantom/io.hpp"
#include "phantom/lidar.hpp"
#include "phantom/noise.hpp"
#include "phantom/pointnet.hpp"
#include "phantom/scene.hpp"

using namespace phantom;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        ++g_checks;                                                                      \
        if (!(cond)) {                                                                   \
            ++g_failures;                                                                \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond "\n";     \
        }                                                                                \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

struct TestCase {
    const char* name;
    std::function<void()> fn;
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Register {
    Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

#define TEST(name)                                  \
    void name();                                    \
    Register reg_##name(#name, name);               \
    void name()

Primitive box_at(Vec3 c, Vec3 half, float yaw = 0.f, uint16_t id = 1) {
    Primitive p;
    p.type = PrimitiveType::Box;
    p.center = c;
    p.half = half;
    p.yaw = yaw;
    p.instance_id = id;
    p.label = SemanticClass::Car;
    p.reflectivity = 0.5f;
    return p;
}

Primitive ground_plane() {
    Primitive g;
    g.type = PrimitiveType::Plane;
    g.normal = {0, 0, 1};
    g.label = SemanticClass::Road;
    g.reflectivity = 0.2f;
    return g;
}

LidarParams clean_lidar() {
    LidarParams p;
    p.beams = 16;
    p.elevation_min_deg = -15.f;
    p.elevation_max_deg = 5.f;
    p.columns = 360;
    p.enable_noise = false;
    p.beam_divergence_mrad = 0.f;
    return p;
}

// ---------------------------------------------------------------------------

TEST(box_intersection_front_face) {
    Primitive b = box_at({10, 0, 1}, {1, 1, 1});
    Hit h;
    CHECK(intersect(Ray{{0, 0, 1}, {1, 0, 0}}, b, 0.f, 100.f, h));
    CHECK_NEAR(h.t, 9.f, 1e-4f);
    CHECK_NEAR(h.normal.x, -1.f, 1e-5f);
    CHECK(!intersect(Ray{{0, 0, 1}, {-1, 0, 0}}, b, 0.f, 100.f, h));
}

TEST(rotated_box_intersection) {
    // 45 deg yaw: the near corner points at the sensor, so the hit is closer than 9 m.
    Primitive b = box_at({10, 0, 1}, {1, 1, 1}, kPi / 4.f);
    Hit h;
    CHECK(intersect(Ray{{0, 0, 1}, {1, 0, 0}}, b, 0.f, 100.f, h));
    CHECK_NEAR(h.t, 10.f - std::sqrt(2.f), 1e-3f);
}

TEST(cylinder_and_sphere_intersection) {
    Primitive c;
    c.type = PrimitiveType::Cylinder;
    c.center = {5, 0, 0};
    c.radius = 0.5f;
    c.height = 2.f;
    Hit h;
    CHECK(intersect(Ray{{0, 0, 1}, {1, 0, 0}}, c, 0.f, 100.f, h));
    CHECK_NEAR(h.t, 4.5f, 1e-4f);
    CHECK(!intersect(Ray{{0, 0, 3}, {1, 0, 0}}, c, 0.f, 100.f, h));  // above the top

    Primitive s;
    s.type = PrimitiveType::Sphere;
    s.center = {0, 8, 0};
    s.radius = 2.f;
    CHECK(intersect(Ray{{0, 0, 0}, {0, 1, 0}}, s, 0.f, 100.f, h));
    CHECK_NEAR(h.t, 6.f, 1e-4f);
    CHECK_NEAR(h.normal.y, -1.f, 1e-4f);
}

TEST(plane_patch_bounds) {
    Primitive p = ground_plane();
    p.half = {2, 2, 0};
    Hit h;
    Vec3 d = normalize({1, 0, -1});
    CHECK(intersect(Ray{{0, 0, 1}, d}, p, 0.f, 100.f, h));   // lands at x = 1
    CHECK(!intersect(Ray{{0, 0, 5}, d}, p, 0.f, 100.f, h));  // lands at x = 5, outside
}

TEST(config_parsing) {
    Config c = Config::parse("# comment\n[lidar]\nbeams = 32 ; inline\nelevations_deg = -1, 0.5, 2\n"
                             "[noise]\nenabled = false\n");
    CHECK(c.get_int("lidar.beams", 0) == 32);
    CHECK(c.get_floats("lidar.elevations_deg").size() == 3);
    CHECK(!c.get_bool("noise.enabled", true));
    CHECK_NEAR(c.get_float("lidar.max_range", 77.f), 77.f, 0.f);
}

TEST(intensity_falloff_and_incidence) {
    IntensityParams p;
    p.attenuation = 0.f;
    float near = compute_intensity(p, 0.5f, 0.f, 10.f, 1.f);
    float far = compute_intensity(p, 0.5f, 0.f, 20.f, 1.f);
    CHECK_NEAR(near / far, 4.f, 1e-3f);  // inverse square
    float grazing = compute_intensity(p, 0.5f, 0.f, 10.f, 0.5f);
    CHECK_NEAR(grazing, 0.5f * near, 1e-5f);
    CHECK(compute_intensity(p, 1.f, 5.f, 1.f, 1.f) <= p.saturation);
}

TEST(detection_probability_monotonic) {
    NoiseParams n;
    float lo = detection_probability(n, 0.001f, 50.f, 120.f);
    float hi = detection_probability(n, 0.1f, 50.f, 120.f);
    float far = detection_probability(n, 0.1f, 110.f, 120.f);
    CHECK(lo < hi);
    CHECK(far < hi);
    CHECK(hi <= 1.f && lo >= 0.f);
}

TEST(scene_generation_is_deterministic) {
    SceneGenerator gen;
    Scene a = gen.generate(1234), b = gen.generate(1234), c = gen.generate(999);
    CHECK(a.primitives.size() == b.primitives.size());
    CHECK(a.objects.size() == b.objects.size());
    if (!a.objects.empty() && a.objects.size() == b.objects.size()) {
        CHECK_NEAR(a.objects[0].center.x, b.objects[0].center.x, 0.f);
    }
    CHECK(a.primitives.size() != c.primitives.size() || a.layout != c.layout ||
          (a.objects.empty() || c.objects.empty() || a.objects[0].center.x != c.objects[0].center.x));
}

TEST(clean_scan_of_flat_ground) {
    Scene s;
    s.primitives.push_back(ground_plane());
    s.finalize();
    LidarSimulator sim(clean_lidar());
    Scan scan = sim.simulate(s, 1);
    CHECK(!scan.points.empty());
    bool all_on_ground = true;
    for (const auto& p : scan.points)
        if (std::fabs(p.z + sim.params().sensor_height) > 1e-3f) all_on_ground = false;
    CHECK(all_on_ground);
    // Only downward-looking beams can hit the ground.
    for (const auto& p : scan.points) CHECK(sim.params().elevations_deg[p.ring] < 0.f);
}

TEST(occlusion_is_detected) {
    Scene s;
    s.primitives.push_back(ground_plane());
    s.primitives.push_back(box_at({8, 0, 1}, {0.5f, 2.f, 1.f}, 0.f, 1));   // occluder
    s.primitives.push_back(box_at({20, 0, 0.75f}, {2.f, 1.f, 0.75f}, 0.f, 2));  // hidden car
    ObjectBox a; a.instance_id = 1; a.label = SemanticClass::Car; a.center = {8, 0, 1}; a.size = {1, 4, 2};
    ObjectBox b; b.instance_id = 2; b.label = SemanticClass::Car; b.center = {20, 0, 0.75f}; b.size = {4, 2, 1.5f};
    s.objects = {a, b};
    s.finalize();

    LidarSimulator sim(clean_lidar());
    Scan scan = sim.simulate(s, 1);
    CHECK(scan.visibility.size() == 2);
    CHECK(scan.visibility[0].occlusion_level == 0);
    CHECK(scan.visibility[0].expected_hits > 0);
    CHECK(scan.visibility[1].occlusion > scan.visibility[0].occlusion);
    CHECK(scan.visibility[1].occlusion_level >= 2);
}

TEST(noise_changes_scan) {
    SceneGenerator gen;
    Scene s = gen.generate(7);
    LidarParams p = clean_lidar();
    Scan clean = LidarSimulator(p).simulate(s, 1);
    p.enable_noise = true;
    Scan noisy = LidarSimulator(p).simulate(s, 1);
    CHECK(!clean.points.empty());
    CHECK(clean.points.size() != noisy.points.size());
}

PointNetRefiner tiny_refiner(float drop_bias) {
    const int C = static_cast<int>(SemanticClass::Count);
    const int F = 6 + C;
    auto layer = [](int in, int out, bool relu, float w) {
        DenseLayer L;
        L.in_dim = in; L.out_dim = out; L.relu = relu;
        L.weight.assign(static_cast<size_t>(in) * out, w);
        L.bias.assign(out, 0.f);
        return L;
    };
    DenseLayer outl = layer(8 + 8, 3, false, 0.f);
    outl.bias[2] = drop_bias;
    return PointNetRefiner::from_layers({layer(F, 8, true, 0.1f)}, {layer(8, 8, true, 0.1f)}, {outl},
                                        C, 4, 50.f, 0.5f, 0.3f);
}

TEST(identity_refiner_keeps_points) {
    Scene s;
    s.primitives.push_back(ground_plane());
    s.finalize();
    Scan scan = LidarSimulator(clean_lidar()).simulate(s, 1);
    Scan before = scan;
    Rng rng(3);
    RefineOptions opts;
    opts.sample_dropout = false;
    RefineStats st = tiny_refiner(-20.f).refine(scan, rng, opts);
    CHECK(st.dropped_points == 0);
    CHECK(scan.points.size() == before.points.size());
    for (size_t i = 0; i < scan.points.size() && i < before.points.size(); ++i) {
        CHECK_NEAR(scan.points[i].range, before.points[i].range, 1e-4f);
        CHECK((scan.points[i].flags & kFlagRefined) != 0);
    }
}

TEST(refiner_dropout_removes_points) {
    Scene s;
    s.primitives.push_back(ground_plane());
    s.finalize();
    Scan scan = LidarSimulator(clean_lidar()).simulate(s, 1);
    Rng rng(3);
    RefineOptions opts;
    opts.sample_dropout = false;
    tiny_refiner(20.f).refine(scan, rng, opts);
    CHECK(scan.points.empty());
}

TEST(points_bin_round_trip) {
    Scene s;
    s.primitives.push_back(ground_plane());
    s.finalize();
    Scan scan = LidarSimulator(clean_lidar()).simulate(s, 1);
    auto path = (std::filesystem::temp_directory_path() / "phantom_points_test.bin").string();
    write_points_bin(path, scan);
    Scan back = read_points_bin(path);
    std::filesystem::remove(path);
    CHECK(back.points.size() == scan.points.size());
    if (!back.points.empty()) {
        CHECK_NEAR(back.points[0].x, scan.points[0].x, 0.f);
        CHECK_NEAR(back.points[0].intensity, scan.points[0].intensity, 0.f);
    }
}

}  // namespace

int main() {
    for (const auto& t : registry()) {
        int before = g_failures;
        t.fn();
        std::cout << (g_failures == before ? "[ OK ] " : "[FAIL] ") << t.name << "\n";
    }
    std::cout << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures == 0 ? 0 : 1;
}
