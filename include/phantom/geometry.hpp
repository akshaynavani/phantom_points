#pragma once

#include <cstdint>
#include <limits>

#include "phantom/math.hpp"

namespace phantom {

/// Semantic classes written to the per-point label files (lower 16 bits).
enum class SemanticClass : uint16_t {
    Unlabeled = 0,
    Road = 1,
    Sidewalk = 2,
    Car = 3,
    Truck = 4,
    Pedestrian = 5,
    Cyclist = 6,
    Pole = 7,
    Building = 8,
    Vegetation = 9,
    Count
};

const char* class_name(SemanticClass c);

enum class PrimitiveType : uint8_t { Plane, Box, Cylinder, Sphere };

/// A single analytic surface in the scene.
///
/// - Plane:    infinite plane through `center` with normal `normal`, optionally
///             restricted to |local x| <= half.x, |local y| <= half.y (if half.x > 0).
/// - Box:      oriented box (yaw only), `center` is the geometric centre, `half` = half extents.
/// - Cylinder: vertical cylinder, `center` is the base centre, `radius`, `height`.
/// - Sphere:   `center`, `radius` (used for tree canopies / bushes).
struct Primitive {
    PrimitiveType type = PrimitiveType::Box;
    Vec3 center;
    Vec3 half;          // box half extents / plane patch half extents
    Vec3 normal{0, 0, 1};
    float yaw = 0.f;
    float radius = 0.f;
    float height = 0.f;

    SemanticClass label = SemanticClass::Unlabeled;
    uint16_t instance_id = 0;  // 0 = stuff (road, buildings, ...)
    float reflectivity = 0.2f; // Lambertian albedo at the sensor wavelength [0, 1]
    float retro = 0.f;         // retro-reflective component (licence plates, signs)
};

struct Hit {
    float t = std::numeric_limits<float>::infinity();
    Vec3 normal;
    const Primitive* prim = nullptr;
    bool valid() const { return prim != nullptr; }
};

/// Intersect `ray` with `p` and return the nearest positive hit (t in (t_min, t_max)).
bool intersect(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out);

bool intersect_plane(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out);
bool intersect_box(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out);
bool intersect_cylinder(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out);
bool intersect_sphere(const Ray& ray, const Primitive& p, float t_min, float t_max, Hit& out);

/// 2D footprint (XY) of a primitive as a conservative circle, used for placement checks.
float footprint_radius(const Primitive& p);

}  // namespace phantom
