#pragma once

#include <string>

#include "phantom/lidar.hpp"
#include "phantom/scene.hpp"

namespace phantom {

/// Six-digit zero-padded scan name, e.g. 42 -> "000042".
std::string scan_name(size_t index);

/// Create the standard sub-directories (velodyne, labels, aux, boxes, meta) under `root`.
void prepare_dataset_dirs(const std::string& root, bool with_annotations = true);

/// velodyne/<name>.bin  — float32 [x, y, z, intensity] per point (KITTI layout).
void write_points_bin(const std::string& path, const Scan& scan);

/// labels/<name>.label  — uint32 per point: semantic | (instance << 16) (SemanticKITTI layout).
void write_labels(const std::string& path, const Scan& scan);

/// aux/<name>.aux       — uint16 [ring, column, flags] per point.
void write_aux(const std::string& path, const Scan& scan);

/// boxes/<name>.txt     — class x y z length width height yaw occ_level occlusion points speed instance
void write_boxes(const std::string& path, const Scan& scan);

/// meta/<name>.json     — layout, seed and summary statistics for one scene.
void write_scene_meta(const std::string& path, size_t index, const Scene& scene, const Scan& scan,
                      size_t dropped_by_refiner);

/// Write points + labels + aux for one scan under `root` (used for both main and ref scans).
void write_scan_files(const std::string& root, const std::string& name, const Scan& scan);

/// Minimal JSON string escaping.
std::string json_escape(const std::string& s);

/// Read a KITTI-style .bin back (x, y, z, intensity only). Used by tests and tools.
Scan read_points_bin(const std::string& path);

}  // namespace phantom
