"""Python tooling for phantom_points: PointNet refiner training/export and dataset analysis."""

__version__ = "0.1.0"

CLASS_NAMES = [
    "unlabeled",
    "road",
    "sidewalk",
    "car",
    "truck",
    "pedestrian",
    "cyclist",
    "pole",
    "building",
    "vegetation",
]
NUM_CLASSES = len(CLASS_NAMES)

# Point flags, mirrored from include/phantom/lidar.hpp
FLAG_MIXED_PIXEL = 1 << 0
FLAG_SPURIOUS = 1 << 1
FLAG_REFINED = 1 << 2
