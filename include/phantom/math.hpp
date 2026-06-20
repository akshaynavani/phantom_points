#pragma once

#include <algorithm>
#include <cmath>

namespace phantom {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1e-6f;

inline float deg2rad(float d) { return d * kPi / 180.0f; }
inline float rad2deg(float r) { return r * 180.0f / kPi; }

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) {
    float l = length(v);
    return l > kEps ? v / l : Vec3{0.f, 0.f, 0.f};
}

/// Rotation about the +Z (up) axis. Scenes are mostly yaw-only.
inline Vec3 rotate_z(const Vec3& v, float yaw) {
    float c = std::cos(yaw), s = std::sin(yaw);
    return {c * v.x - s * v.y, s * v.x + c * v.y, v.z};
}

struct Ray {
    Vec3 origin;
    Vec3 dir;  // unit length
    Vec3 at(float t) const { return origin + dir * t; }
};

/// Direction for a spinning LiDAR beam: azimuth in XY plane, elevation from horizon.
inline Vec3 spherical_dir(float azimuth, float elevation) {
    float ce = std::cos(elevation);
    return {ce * std::cos(azimuth), ce * std::sin(azimuth), std::sin(elevation)};
}

template <typename T>
inline T clamp(T v, T lo, T hi) { return std::max(lo, std::min(hi, v)); }

}  // namespace phantom
