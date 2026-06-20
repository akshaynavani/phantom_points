#include "phantom/geometry.hpp"

#include <cmath>

namespace phantom {

const char* class_name(SemanticClass c) {
    switch (c) {
        case SemanticClass::Unlabeled: return "unlabeled";
        case SemanticClass::Road: return "road";
        case SemanticClass::Sidewalk: return "sidewalk";
        case SemanticClass::Car: return "car";
        case SemanticClass::Truck: return "truck";
        case SemanticClass::Pedestrian: return "pedestrian";
        case SemanticClass::Cyclist: return "cyclist";
        case SemanticClass::Pole: return "pole";
        case SemanticClass::Building: return "building";
        case SemanticClass::Vegetation: return "vegetation";
        default: return "unknown";
    }
}

bool intersect_plane(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out) {
    float denom = dot(p.normal, ray.dir);
    if (std::fabs(denom) < kEps) return false;
    float t = dot(p.center - ray.origin, p.normal) / denom;
    if (t <= t_min || t >= t_max) return false;

    if (p.half.x > 0.f) {
        // Bounded patch: check extents in the plane's local yaw frame (planes are horizontal).
        Vec3 local = rotate_z(ray.at(t) - p.center, -p.yaw);
        if (std::fabs(local.x) > p.half.x || std::fabs(local.y) > p.half.y) return false;
    }
    out.t = t;
    out.normal = denom < 0.f ? p.normal : -p.normal;
    out.prim = &p;
    return true;
}

bool intersect_box(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out) {
    // Transform the ray into the box's local frame (yaw-only rotation) and run slab test.
    Vec3 o = rotate_z(ray.origin - p.center, -p.yaw);
    Vec3 d = rotate_z(ray.dir, -p.yaw);

    float tn = t_min, tf = t_max;
    int axis_n = -1;
    float sign_n = 1.f;
    const float o_arr[3] = {o.x, o.y, o.z};
    const float d_arr[3] = {d.x, d.y, d.z};
    const float h_arr[3] = {p.half.x, p.half.y, p.half.z};

    for (int i = 0; i < 3; ++i) {
        if (std::fabs(d_arr[i]) < kEps) {
            if (o_arr[i] < -h_arr[i] || o_arr[i] > h_arr[i]) return false;
            continue;
        }
        float inv = 1.f / d_arr[i];
        float t0 = (-h_arr[i] - o_arr[i]) * inv;
        float t1 = (h_arr[i] - o_arr[i]) * inv;
        float s = -1.f;  // entering through the negative face
        if (t0 > t1) { std::swap(t0, t1); s = 1.f; }
        if (t0 > tn) { tn = t0; axis_n = i; sign_n = s; }
        if (t1 < tf) tf = t1;
        if (tn > tf) return false;
    }
    if (axis_n < 0) return false;  // origin inside the box: ignore (sensor never inside objects)

    Vec3 n_local{0, 0, 0};
    if (axis_n == 0) n_local.x = sign_n;
    if (axis_n == 1) n_local.y = sign_n;
    if (axis_n == 2) n_local.z = sign_n;

    out.t = tn;
    out.normal = rotate_z(n_local, p.yaw);
    out.prim = &p;
    return true;
}

bool intersect_cylinder(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out) {
    Vec3 o = ray.origin - p.center;
    const Vec3& d = ray.dir;
    float best = t_max;
    Vec3 best_n;
    bool found = false;

    // Side surface: (ox + t dx)^2 + (oy + t dy)^2 = r^2
    float a = d.x * d.x + d.y * d.y;
    if (a > kEps) {
        float b = 2.f * (o.x * d.x + o.y * d.y);
        float c = o.x * o.x + o.y * o.y - p.radius * p.radius;
        float disc = b * b - 4.f * a * c;
        if (disc >= 0.f) {
            float sq = std::sqrt(disc);
            float roots[2] = {(-b - sq) / (2.f * a), (-b + sq) / (2.f * a)};
            for (float t : roots) {
                if (t <= t_min || t >= best) continue;
                float z = o.z + t * d.z;
                if (z < 0.f || z > p.height) continue;
                best = t;
                best_n = normalize(Vec3{o.x + t * d.x, o.y + t * d.y, 0.f});
                found = true;
                break;
            }
        }
    }
    // Top cap (bottom cap sits on the ground and is never visible from above ground).
    if (std::fabs(d.z) > kEps) {
        float t = (p.height - o.z) / d.z;
        if (t > t_min && t < best) {
            float x = o.x + t * d.x, y = o.y + t * d.y;
            if (x * x + y * y <= p.radius * p.radius) {
                best = t;
                best_n = Vec3{0, 0, 1};
                found = true;
            }
        }
    }
    if (!found) return false;
    out.t = best;
    out.normal = best_n;
    out.prim = &p;
    return true;
}

bool intersect_sphere(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out) {
    Vec3 oc = ray.origin - p.center;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - p.radius * p.radius;
    float disc = b * b - c;
    if (disc < 0.f) return false;
    float sq = std::sqrt(disc);
    float t = -b - sq;
    if (t <= t_min) t = -b + sq;
    if (t <= t_min || t >= t_max) return false;
    out.t = t;
    out.normal = normalize(ray.at(t) - p.center);
    out.prim = &p;
    return true;
}

bool intersect(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out) {
    switch (p.type) {
        case PrimitiveType::Plane: return intersect_plane(ray, p, t_min, t_max, out);
        case PrimitiveType::Box: return intersect_box(ray, p, t_min, t_max, out);
        case PrimitiveType::Cylinder: return intersect_cylinder(ray, p, t_min, t_max, out);
        case PrimitiveType::Sphere: return intersect_sphere(ray, p, t_min, t_max, out);
    }
    return false;
}

float footprint_radius(const Primitive& p) {
    switch (p.type) {
        case PrimitiveType::Box: return std::sqrt(p.half.x * p.half.x + p.half.y * p.half.y);
        case PrimitiveType::Cylinder:
        case PrimitiveType::Sphere: return p.radius;
        case PrimitiveType::Plane: return 0.f;
    }
    return 0.f;
}

}  // namespace phantom
