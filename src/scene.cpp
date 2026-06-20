#include "phantom/scene.hpp"

#include <cmath>

namespace phantom {

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

void Scene::finalize() {
    bound_center_.resize(primitives.size());
    bound_radius_.resize(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i) {
        const Primitive& p = primitives[i];
        switch (p.type) {
            case PrimitiveType::Plane:
                bound_center_[i] = p.center;
                bound_radius_[i] = -1.f;  // planes are always tested
                break;
            case PrimitiveType::Box:
                bound_center_[i] = p.center;
                bound_radius_[i] = length(p.half);
                break;
            case PrimitiveType::Cylinder: {
                float hh = 0.5f * p.height;
                bound_center_[i] = p.center + Vec3{0, 0, hh};
                bound_radius_[i] = std::sqrt(p.radius * p.radius + hh * hh);
                break;
            }
            case PrimitiveType::Sphere:
                bound_center_[i] = p.center;
                bound_radius_[i] = p.radius;
                break;
        }
    }
}

bool Scene::raycast(const Ray& ray, float t_min, float t_max, Hit& out) const {
    const bool have_bounds = bound_radius_.size() == primitives.size();
    float best = t_max;
    bool found = false;
    for (size_t i = 0; i < primitives.size(); ++i) {
        if (have_bounds && bound_radius_[i] > 0.f) {
            // Bounding sphere rejection: skip if the ray misses or the sphere is behind `best`.
            Vec3 oc = bound_center_[i] - ray.origin;
            float tc = dot(oc, ray.dir);
            float r = bound_radius_[i];
            if (tc + r < t_min || tc - r > best) continue;
            float d2 = dot(oc, oc) - tc * tc;
            if (d2 > r * r) continue;
        }
        Hit h;
        if (intersect(ray, primitives[i], t_min, best, h)) {
            best = h.t;
            out = h;
            found = true;
        }
    }
    return found;
}

// ---------------------------------------------------------------------------
// SceneGenerator
// ---------------------------------------------------------------------------

namespace {

Primitive make_box(Vec3 center, Vec3 half, float yaw, SemanticClass label, uint16_t id,
                   float refl) {
    Primitive p;
    p.type = PrimitiveType::Box;
    p.center = center;
    p.half = half;
    p.yaw = yaw;
    p.label = label;
    p.instance_id = id;
    p.reflectivity = refl;
    return p;
}

Primitive make_cylinder(Vec3 base, float radius, float height, SemanticClass label, uint16_t id,
                        float refl) {
    Primitive p;
    p.type = PrimitiveType::Cylinder;
    p.center = base;
    p.radius = radius;
    p.height = height;
    p.label = label;
    p.instance_id = id;
    p.reflectivity = refl;
    return p;
}

Primitive make_sphere(Vec3 center, float radius, SemanticClass label, uint16_t id, float refl) {
    Primitive p;
    p.type = PrimitiveType::Sphere;
    p.center = center;
    p.radius = radius;
    p.label = label;
    p.instance_id = id;
    p.reflectivity = refl;
    return p;
}

Primitive make_patch(Vec3 center, float hx, float hy, float yaw, SemanticClass label, float refl) {
    Primitive p;
    p.type = PrimitiveType::Plane;
    p.center = center;
    p.normal = {0, 0, 1};
    p.half = {hx, hy, 0};
    p.yaw = yaw;
    p.label = label;
    p.reflectivity = refl;
    return p;
}

}  // namespace

bool SceneGenerator::on_road(float x, float y, bool intersection) const {
    float edge = params_.road_half_width + params_.sidewalk_width;
    if (std::fabs(y) < edge) return true;
    if (intersection && std::fabs(x) < edge) return true;
    return false;
}

bool SceneGenerator::try_place(Rng& rng, std::vector<Placement>& occ, float r, float xmin,
                               float xmax, float ymin, float ymax, float& x, float& y) const {
    for (int attempt = 0; attempt < 40; ++attempt) {
        float cx = rng.uniform(xmin, xmax);
        float cy = rng.uniform(ymin, ymax);
        if (std::hypot(cx, cy) < params_.sensor_clearance + r) continue;
        bool ok = true;
        for (const auto& o : occ) {
            float dx = cx - o.x, dy = cy - o.y;
            float rr = r + o.r + 0.3f;
            if (dx * dx + dy * dy < rr * rr) { ok = false; break; }
        }
        if (!ok) continue;
        occ.push_back({cx, cy, r});
        x = cx;
        y = cy;
        return true;
    }
    return false;
}

void SceneGenerator::add_ground(Scene& s, bool intersection, bool parking) const {
    const float E = params_.scene_extent;
    const float rw = params_.road_half_width;
    const float sw = params_.sidewalk_width;

    // Terrain: infinite plane at z = 0 (unlabeled / off-road ground).
    Primitive ground;
    ground.type = PrimitiveType::Plane;
    ground.center = {0, 0, 0};
    ground.normal = {0, 0, 1};
    ground.label = SemanticClass::Unlabeled;
    ground.reflectivity = 0.25f;
    s.primitives.push_back(ground);

    if (parking) {
        s.primitives.push_back(make_patch({0, 0, 0.01f}, E, E * 0.6f, 0.f, SemanticClass::Road, 0.12f));
        return;
    }

    // Asphalt is darker than terrain; patches sit 1 cm above to win the depth test.
    s.primitives.push_back(make_patch({0, 0, 0.01f}, E, rw, 0.f, SemanticClass::Road, 0.10f));
    if (intersection)
        s.primitives.push_back(make_patch({0, 0, 0.011f}, rw, E, 0.f, SemanticClass::Road, 0.10f));

    // Curbs + sidewalks as thin raised slabs on both sides.
    const float curb_h = 0.15f;
    for (int side : {-1, 1}) {
        float yc = side * (rw + 0.5f * sw);
        if (!intersection) {
            s.primitives.push_back(make_box({0, yc, 0.5f * curb_h}, {E, 0.5f * sw, 0.5f * curb_h}, 0.f,
                                            SemanticClass::Sidewalk, 0, 0.30f));
        } else {
            // Split into two halves so the cross street stays clear.
            float seg = 0.5f * (E - rw - sw);
            float xc = rw + sw + seg;
            for (int xs : {-1, 1}) {
                s.primitives.push_back(make_box({xs * xc, yc, 0.5f * curb_h},
                                                {seg, 0.5f * sw, 0.5f * curb_h}, 0.f,
                                                SemanticClass::Sidewalk, 0, 0.30f));
                s.primitives.push_back(make_box({yc, xs * xc, 0.5f * curb_h},
                                                {0.5f * sw, seg, 0.5f * curb_h}, 0.f,
                                                SemanticClass::Sidewalk, 0, 0.30f));
            }
        }
    }
}

void SceneGenerator::add_vehicles(Scene& s, Rng& rng, std::vector<Placement>& occ,
                                  bool intersection, bool parking, uint16_t& next_id) const {
    const float E = params_.scene_extent * 0.9f;
    const float rw = params_.road_half_width;

    auto spawn = [&](bool truck) {
        float L = truck ? rng.uniform(6.0f, 10.5f) : rng.uniform(3.8f, 4.9f);
        float W = truck ? rng.uniform(2.3f, 2.55f) : rng.uniform(1.65f, 1.95f);
        float H = truck ? rng.uniform(2.8f, 3.8f) : rng.uniform(1.4f, 1.65f);
        float r = 0.5f * std::hypot(L, W);

        float x = 0, y = 0, yaw = 0, speed = 0;
        bool placed = false;
        if (parking) {
            // Rows of perpendicular parking spots.
            int row = rng.uniform_int(-3, 3);
            float ry = row * 6.0f;
            placed = try_place(rng, occ, r, -E, E, ry - 0.3f, ry + 0.3f, x, y);
            yaw = (rng.bernoulli(0.5f) ? 0.5f : -0.5f) * kPi + rng.normal(0.f, 0.05f);
        } else {
            bool cross = intersection && rng.bernoulli(0.35f);
            bool parked = !truck && rng.bernoulli(0.35f);
            int dir = rng.bernoulli(0.5f) ? 1 : -1;
            float lane = parked ? dir * (rw - 0.5f * W - 0.2f) : dir * rng.uniform(0.25f, 0.75f) * rw;
            if (!cross) {
                placed = try_place(rng, occ, r, -E, E, lane - 0.2f, lane + 0.2f, x, y);
                yaw = (dir > 0 ? 0.f : kPi) + rng.normal(0.f, parked ? 0.03f : 0.05f);
            } else {
                placed = try_place(rng, occ, r, lane - 0.2f, lane + 0.2f, -E, E, x, y);
                yaw = (dir > 0 ? 0.5f * kPi : -0.5f * kPi) + rng.normal(0.f, 0.05f);
            }
            speed = parked ? 0.f : rng.uniform(0.f, truck ? 20.f : 25.f);
        }
        if (!placed) return;

        uint16_t id = next_id++;
        SemanticClass cls = truck ? SemanticClass::Truck : SemanticClass::Car;
        float paint = rng.uniform(0.04f, 0.65f);  // black .. white paint
        float clearance = truck ? 0.45f : 0.22f;
        Vec3 fwd = rotate_z({1, 0, 0}, yaw);

        if (truck) {
            float cab_L = 2.2f;
            float cargo_L = L - cab_L - 0.2f;
            float cab_H = H * 0.8f;
            Vec3 cab_c = Vec3{x, y, 0} + fwd * (0.5f * L - 0.5f * cab_L);
            cab_c.z = clearance + 0.5f * (cab_H - clearance);
            s.primitives.push_back(make_box(cab_c, {0.5f * cab_L, 0.5f * W, 0.5f * (cab_H - clearance)},
                                            yaw, cls, id, paint));
            Vec3 cargo_c = Vec3{x, y, 0} - fwd * (0.5f * L - 0.5f * cargo_L);
            cargo_c.z = clearance + 0.5f * (H - clearance);
            s.primitives.push_back(make_box(cargo_c, {0.5f * cargo_L, 0.5f * W, 0.5f * (H - clearance)},
                                            yaw, cls, id, rng.uniform(0.3f, 0.7f)));
        } else {
            float body_H = H * 0.6f;
            Vec3 body_c{x, y, clearance + 0.5f * (body_H - clearance)};
            s.primitives.push_back(make_box(body_c, {0.5f * L, 0.5f * W, 0.5f * (body_H - clearance)},
                                            yaw, cls, id, paint));
            // Cabin: shorter, narrower, slightly rearward; glass is a weak reflector.
            float cabin_L = L * rng.uniform(0.45f, 0.6f);
            Vec3 cabin_c = Vec3{x, y, 0} - fwd * (L * 0.08f);
            cabin_c.z = body_H + 0.5f * (H - body_H);
            s.primitives.push_back(make_box(cabin_c, {0.5f * cabin_L, 0.45f * W, 0.5f * (H - body_H)},
                                            yaw, cls, id, 0.08f));
            // Retro-reflective licence plate at the rear.
            Primitive plate = make_box(Vec3{x, y, 0} - fwd * (0.5f * L + 0.01f) + Vec3{0, 0, 0.55f},
                                       {0.01f, 0.26f, 0.06f}, yaw, cls, id, 0.5f);
            plate.retro = 0.8f;
            s.primitives.push_back(plate);
        }

        ObjectBox box;
        box.instance_id = id;
        box.label = cls;
        box.center = {x, y, 0.5f * H};
        box.size = {L, W, H};
        box.yaw = yaw;
        box.speed = speed;
        s.objects.push_back(box);
    };

    int n_cars = rng.uniform_int(params_.cars_min, params_.cars_max);
    int n_trucks = rng.uniform_int(params_.trucks_min, params_.trucks_max);
    if (parking) { n_cars *= 2; n_trucks = 0; }
    for (int i = 0; i < n_trucks; ++i) spawn(true);
    for (int i = 0; i < n_cars; ++i) spawn(false);
}

void SceneGenerator::add_vulnerable(Scene& s, Rng& rng, std::vector<Placement>& occ,
                                    uint16_t& next_id) const {
    const float E = params_.scene_extent * 0.8f;
    const float rw = params_.road_half_width;
    const float sw = params_.sidewalk_width;

    int n_ped = rng.uniform_int(params_.pedestrians_min, params_.pedestrians_max);
    for (int i = 0; i < n_ped; ++i) {
        float x, y;
        int side = rng.bernoulli(0.5f) ? 1 : -1;
        bool crossing = rng.bernoulli(0.2f);
        float ymin = crossing ? -rw : side * (rw + 0.3f);
        float ymax = crossing ? rw : side * (rw + sw - 0.3f);
        if (ymin > ymax) std::swap(ymin, ymax);
        if (!try_place(rng, occ, 0.35f, -E, E, ymin, ymax, x, y)) continue;

        uint16_t id = next_id++;
        float h = rng.uniform(1.45f, 1.85f);
        float base_z = crossing ? 0.f : 0.15f;
        float torso_h = h - 0.25f;
        float cloth = rng.uniform(0.05f, 0.5f);
        s.primitives.push_back(make_cylinder({x, y, base_z}, rng.uniform(0.18f, 0.26f), torso_h,
                                             SemanticClass::Pedestrian, id, cloth));
        s.primitives.push_back(make_sphere({x, y, base_z + torso_h + 0.12f}, 0.11f,
                                           SemanticClass::Pedestrian, id, 0.3f));
        ObjectBox box;
        box.instance_id = id;
        box.label = SemanticClass::Pedestrian;
        box.center = {x, y, base_z + 0.5f * h};
        box.size = {0.6f, 0.6f, h};
        box.yaw = rng.uniform(-kPi, kPi);
        box.speed = rng.uniform(0.f, 1.8f);
        s.objects.push_back(box);
    }

    int n_cyc = rng.uniform_int(params_.cyclists_min, params_.cyclists_max);
    for (int i = 0; i < n_cyc; ++i) {
        float x, y;
        int dir = rng.bernoulli(0.5f) ? 1 : -1;
        float lane = dir * (rw - 0.8f);
        if (!try_place(rng, occ, 1.0f, -E, E, lane - 0.2f, lane + 0.2f, x, y)) continue;

        uint16_t id = next_id++;
        float yaw = (dir > 0 ? 0.f : kPi) + rng.normal(0.f, 0.05f);
        s.primitives.push_back(make_box({x, y, 0.5f}, {0.85f, 0.08f, 0.45f}, yaw,
                                        SemanticClass::Cyclist, id, 0.35f));
        s.primitives.push_back(make_cylinder({x, y, 0.9f}, 0.22f, 0.75f, SemanticClass::Cyclist, id,
                                             rng.uniform(0.05f, 0.5f)));
        s.primitives.push_back(make_sphere({x, y, 1.77f}, 0.12f, SemanticClass::Cyclist, id, 0.4f));
        ObjectBox box;
        box.instance_id = id;
        box.label = SemanticClass::Cyclist;
        box.center = {x, y, 0.95f};
        box.size = {1.8f, 0.6f, 1.9f};
        box.yaw = yaw;
        box.speed = rng.uniform(2.f, 8.f);
        s.objects.push_back(box);
    }
}

void SceneGenerator::add_static(Scene& s, Rng& rng, std::vector<Placement>& occ,
                                bool intersection) const {
    const float E = params_.scene_extent;
    const float rw = params_.road_half_width;
    const float sw = params_.sidewalk_width;

    // Poles along the curb line.
    int n_poles = rng.uniform_int(params_.poles_min, params_.poles_max);
    for (int i = 0; i < n_poles; ++i) {
        float x, y;
        int side = rng.bernoulli(0.5f) ? 1 : -1;
        float yc = side * (rw + 0.4f);
        if (!try_place(rng, occ, 0.3f, -E * 0.9f, E * 0.9f, yc - 0.1f, yc + 0.1f, x, y)) continue;
        s.primitives.push_back(make_cylinder({x, y, 0.15f}, rng.uniform(0.07f, 0.2f),
                                             rng.uniform(3.5f, 9.f), SemanticClass::Pole, 0,
                                             rng.uniform(0.2f, 0.5f)));
    }

    // Trees behind the sidewalk.
    int n_trees = rng.uniform_int(params_.trees_min, params_.trees_max);
    for (int i = 0; i < n_trees; ++i) {
        float canopy = rng.uniform(1.2f, 3.0f);
        float x = 0, y = 0;
        bool placed = false;
        for (int k = 0; k < 5 && !placed; ++k) {
            int side = rng.bernoulli(0.5f) ? 1 : -1;
            float y0 = side * (rw + sw + canopy * 0.6f);
            float y1 = side * (rw + sw + canopy + 6.f);
            if (y0 > y1) std::swap(y0, y1);
            placed = try_place(rng, occ, canopy * 0.6f, -E * 0.9f, E * 0.9f, y0, y1, x, y);
            if (placed && on_road(x, y, intersection)) { occ.pop_back(); placed = false; }
        }
        if (!placed) continue;
        float trunk_h = rng.uniform(1.8f, 3.5f);
        s.primitives.push_back(make_cylinder({x, y, 0}, rng.uniform(0.12f, 0.3f), trunk_h,
                                             SemanticClass::Vegetation, 0, 0.35f));
        s.primitives.push_back(make_sphere({x, y, trunk_h + canopy * 0.7f}, canopy,
                                           SemanticClass::Vegetation, 0, rng.uniform(0.4f, 0.7f)));
    }

    // Buildings set back from the street.
    int n_bld = rng.uniform_int(params_.buildings_min, params_.buildings_max);
    for (int i = 0; i < n_bld; ++i) {
        float L = rng.uniform(8.f, 28.f), W = rng.uniform(6.f, 16.f), H = rng.uniform(4.f, 25.f);
        float r = 0.5f * std::hypot(L, W);
        float x = 0, y = 0;
        bool placed = false;
        for (int k = 0; k < 5 && !placed; ++k) {
            int side = rng.bernoulli(0.5f) ? 1 : -1;
            float y0 = side * (rw + sw + 2.f + 0.5f * W);
            float y1 = side * (E * 0.95f);
            if (y0 > y1) std::swap(y0, y1);
            placed = try_place(rng, occ, r, -E, E, y0, y1, x, y);
            if (placed) {
                // Reject if any footprint corner would intrude on a road corridor.
                bool bad = on_road(x, y, intersection) ||
                           (intersection && std::fabs(x) < rw + sw + r);
                if (bad) { occ.pop_back(); placed = false; }
            }
        }
        if (!placed) continue;
        s.primitives.push_back(make_box({x, y, 0.5f * H}, {0.5f * L, 0.5f * W, 0.5f * H},
                                        rng.normal(0.f, 0.05f), SemanticClass::Building, 0,
                                        rng.uniform(0.15f, 0.6f)));
    }
}

Scene SceneGenerator::generate(uint64_t seed) const {
    Rng rng(seed);
    Scene s;
    s.seed = seed;

    float u = rng.uniform();
    bool parking = u < params_.p_parking;
    bool intersection = !parking && u < params_.p_parking + params_.p_intersection;
    s.layout = parking ? "parking" : (intersection ? "intersection" : "straight");

    std::vector<Placement> occ;
    uint16_t next_id = 1;

    add_ground(s, intersection, parking);
    add_vehicles(s, rng, occ, intersection, parking, next_id);
    if (!parking) add_vulnerable(s, rng, occ, next_id);
    add_static(s, rng, occ, intersection);

    s.finalize();
    return s;
}

}  // namespace phantom
