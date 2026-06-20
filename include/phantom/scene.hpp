#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "phantom/geometry.hpp"
#include "phantom/rng.hpp"

namespace phantom {

/// Ground-truth 3D bounding box for an object instance (yaw-only, KITTI-like).
struct ObjectBox {
    uint16_t instance_id = 0;
    SemanticClass label = SemanticClass::Unlabeled;
    Vec3 center;  // geometric centre
    Vec3 size;    // length (x), width (y), height (z)
    float yaw = 0.f;
    float speed = 0.f;  // m/s along heading, 0 for parked / static
};

struct Scene {
    std::vector<Primitive> primitives;
    std::vector<ObjectBox> objects;
    std::string layout;  // "straight", "intersection", "parking"
    uint64_t seed = 0;

    /// Closest intersection along the ray over all primitives.
    bool raycast(const Ray& ray, float t_min, float t_max, Hit& out) const;

    /// Bounding radius of each primitive from the sensor origin, used for early-outs.
    void finalize();

private:
    std::vector<Vec3> bound_center_;
    std::vector<float> bound_radius_;
};

/// Knobs for the procedural scene sampler. Counts are sampled uniformly in [min, max].
struct SceneParams {
    float road_half_width = 7.0f;
    float sidewalk_width = 3.0f;
    float scene_extent = 80.0f;  // half-size of the square world patch (m)

    int cars_min = 4, cars_max = 18;
    int trucks_min = 0, trucks_max = 3;
    int pedestrians_min = 0, pedestrians_max = 12;
    int cyclists_min = 0, cyclists_max = 4;
    int poles_min = 6, poles_max = 20;
    int trees_min = 4, trees_max = 20;
    int buildings_min = 4, buildings_max = 12;

    float p_intersection = 0.3f;  // probability of a 4-way intersection layout
    float p_parking = 0.15f;      // probability of a parking-lot layout
    float sensor_clearance = 3.0f;  // keep objects this far from the ego vehicle
};

class SceneGenerator {
public:
    explicit SceneGenerator(SceneParams params = {}) : params_(params) {}

    Scene generate(uint64_t seed) const;

    const SceneParams& params() const { return params_; }

private:
    struct Placement {
        float x, y, r;
    };

    void add_ground(Scene& s, bool intersection, bool parking) const;
    void add_vehicles(Scene& s, Rng& rng, std::vector<Placement>& occ, bool intersection,
                      bool parking, uint16_t& next_id) const;
    void add_vulnerable(Scene& s, Rng& rng, std::vector<Placement>& occ, uint16_t& next_id) const;
    void add_static(Scene& s, Rng& rng, std::vector<Placement>& occ, bool intersection) const;

    bool try_place(Rng& rng, std::vector<Placement>& occ, float r, float xmin, float xmax,
                   float ymin, float ymax, float& x, float& y) const;
    bool on_road(float x, float y, bool intersection) const;

    SceneParams params_;
};

}  // namespace phantom
