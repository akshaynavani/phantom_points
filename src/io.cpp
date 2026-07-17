#include "phantom/io.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace phantom {

namespace {

std::ofstream open_out(const std::string& path, bool binary) {
    std::ofstream out(path, binary ? std::ios::binary | std::ios::trunc : std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + path);
    return out;
}

template <typename T>
void write_vector(const std::string& path, const std::vector<T>& v) {
    auto out = open_out(path, true);
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    if (!out) throw std::runtime_error("short write to " + path);
}

}  // namespace

std::string scan_name(size_t index) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%06zu", index);
    return buf;
}

void prepare_dataset_dirs(const std::string& root, bool with_annotations) {
    fs::create_directories(fs::path(root) / "velodyne");
    fs::create_directories(fs::path(root) / "labels");
    fs::create_directories(fs::path(root) / "aux");
    if (with_annotations) {
        fs::create_directories(fs::path(root) / "boxes");
        fs::create_directories(fs::path(root) / "meta");
    }
}

void write_points_bin(const std::string& path, const Scan& scan) {
    std::vector<float> buf;
    buf.reserve(scan.points.size() * 4);
    for (const auto& p : scan.points) {
        buf.push_back(p.x);
        buf.push_back(p.y);
        buf.push_back(p.z);
        buf.push_back(p.intensity);
    }
    write_vector(path, buf);
}

void write_labels(const std::string& path, const Scan& scan) {
    std::vector<uint32_t> buf;
    buf.reserve(scan.points.size());
    for (const auto& p : scan.points)
        buf.push_back(static_cast<uint32_t>(p.label) | (static_cast<uint32_t>(p.instance) << 16));
    write_vector(path, buf);
}

void write_aux(const std::string& path, const Scan& scan) {
    std::vector<uint16_t> buf;
    buf.reserve(scan.points.size() * 3);
    for (const auto& p : scan.points) {
        buf.push_back(p.ring);
        buf.push_back(p.column);
        buf.push_back(static_cast<uint16_t>(p.flags));
    }
    write_vector(path, buf);
}

void write_boxes(const std::string& path, const Scan& scan) {
    auto out = open_out(path, false);
    out << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < scan.objects.size(); ++i) {
        const ObjectBox& b = scan.objects[i];
        ObjectVisibility v;
        if (i < scan.visibility.size()) v = scan.visibility[i];
        out << class_name(b.label) << ' ' << b.center.x << ' ' << b.center.y << ' ' << b.center.z << ' '
            << b.size.x << ' ' << b.size.y << ' ' << b.size.z << ' ' << b.yaw << ' '
            << v.occlusion_level << ' ' << v.occlusion << ' ' << v.final_points << ' ' << b.speed << ' '
            << b.instance_id << '\n';
    }
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

void write_scene_meta(const std::string& path, size_t index, const Scene& scene, const Scan& scan,
                      size_t dropped_by_refiner) {
    size_t class_counts[static_cast<size_t>(SemanticClass::Count)] = {};
    size_t mixed = 0, spurious = 0;
    for (const auto& p : scan.points) {
        if (p.label < static_cast<uint16_t>(SemanticClass::Count)) ++class_counts[p.label];
        if (p.flags & kFlagMixedPixel) ++mixed;
        if (p.flags & kFlagSpurious) ++spurious;
    }
    auto out = open_out(path, false);
    out << "{\n";
    out << "  \"scene_id\": " << index << ",\n";
    out << "  \"seed\": " << scene.seed << ",\n";
    out << "  \"layout\": \"" << json_escape(scene.layout) << "\",\n";
    out << "  \"num_points\": " << scan.points.size() << ",\n";
    out << "  \"num_objects\": " << scan.objects.size() << ",\n";
    out << "  \"num_primitives\": " << scene.primitives.size() << ",\n";
    out << "  \"mixed_pixel_points\": " << mixed << ",\n";
    out << "  \"spurious_points\": " << spurious << ",\n";
    out << "  \"refiner_dropped\": " << dropped_by_refiner << ",\n";
    out << "  \"class_points\": {";
    for (size_t c = 0; c < static_cast<size_t>(SemanticClass::Count); ++c) {
        out << (c ? ", " : "") << '"' << class_name(static_cast<SemanticClass>(c)) << "\": " << class_counts[c];
    }
    out << "}\n}\n";
}

void write_scan_files(const std::string& root, const std::string& name, const Scan& scan) {
    fs::path r(root);
    write_points_bin((r / "velodyne" / (name + ".bin")).string(), scan);
    write_labels((r / "labels" / (name + ".label")).string(), scan);
    write_aux((r / "aux" / (name + ".aux")).string(), scan);
}

Scan read_points_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::streamsize bytes = in.tellg();
    in.seekg(0);
    if (bytes % (4 * sizeof(float)) != 0) throw std::runtime_error("malformed point file " + path);
    std::vector<float> buf(static_cast<size_t>(bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(buf.data()), bytes);
    Scan scan;
    scan.points.resize(buf.size() / 4);
    for (size_t i = 0; i < scan.points.size(); ++i) {
        LidarPoint& p = scan.points[i];
        p.x = buf[4 * i];
        p.y = buf[4 * i + 1];
        p.z = buf[4 * i + 2];
        p.intensity = buf[4 * i + 3];
        p.range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }
    return scan;
}

}  // namespace phantom
