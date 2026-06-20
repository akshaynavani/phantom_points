# phantom_points — Implementation Plan

Synthetic LiDAR scene generator. A C++ pipeline procedurally builds driving scenes,
ray-casts them with a configurable spinning-LiDAR sensor model (noise, occlusion,
intensity falloff), and optionally passes each scan through a PointNet-based
"realism refiner" that learns per-point range residuals, intensity corrections and
dropout probabilities from real data. Output is labeled scans (KITTI-style `.bin` +
per-point semantic labels + 3D boxes) at the 10k+ scenario scale. A Python toolkit
trains the PointNet model, exports its weights for the dependency-free C++
inference engine, and runs statistical distribution analysis comparing synthetic
scans to a reference set.

## Key design decisions

- **C++17, CMake, no heavy deps.** Core math, ray casting, config parsing and
  PointNet inference are implemented in-tree so the generator builds anywhere.
- **Analytic ray casting** against primitives (ground plane, oriented boxes,
  vertical cylinders, spheres) instead of a mesh BVH — enough for vehicles,
  pedestrians, poles, buildings, vegetation, and fast for 10k scenes.
- **Sensor model** parameterised by beam elevation table (default: 64-beam,
  HDL-64-like), horizontal resolution, max range, range noise sigma
  (range-proportional), beam divergence, dropout, and intensity model
  `I = rho * cos(theta) / r^2` with atmospheric attenuation and saturation.
- **PointNet refiner** (shared per-point MLP -> global max-pool -> concat ->
  segmentation-style head) predicts `(delta_range, delta_intensity, drop_logit)`.
  Trained in PyTorch; weights exported to a flat little-endian binary format that
  a hand-written C++ forward pass reads. Model is optional at generation time.
- **Determinism:** every scenario is seeded by `(global_seed, scene_index)` so any
  scene can be regenerated individually.
- **Config:** simple INI files (`configs/*.ini`) parsed in-tree.

## Phases

### Phase 1 — C++ core: math, geometry and procedural scenes
- `CMakeLists.txt`, `include/phantom/{math.hpp,geometry.hpp,scene.hpp,rng.hpp}`,
  `src/{geometry.cpp,scene.cpp}`
- Vec3 / ray types, ray–primitive intersection (plane, OBB, cylinder, sphere)
  with surface normals.
- Semantic classes + material reflectivity table.
- Procedural `SceneGenerator`: road layout, parked/moving cars, trucks,
  pedestrians, cyclists, poles, buildings, vegetation, non-overlap placement.

### Phase 2 — LiDAR sensor model
- `include/phantom/{config.hpp,lidar.hpp,noise.hpp}`, `src/{config.cpp,lidar.cpp,noise.cpp}`,
  `configs/default.ini`
- INI config parser; beam pattern (uniform or explicit table).
- Ray caster producing points with range, intensity, ring, azimuth, label, instance id.
- Noise (range-proportional Gaussian, angular jitter, random + range-dependent
  dropout), occlusion (closest hit, partial-beam mixed returns), intensity falloff.

### Phase 3 — PointNet realism refiner
- `python/phantom_points/{model.py,dataset.py,train.py,export.py}`
- `include/phantom/pointnet.hpp`, `src/pointnet.cpp`
- PyTorch PointNet refiner + training loop on paired (synthetic, reference) scans.
- Binary weight export; C++ inference (linear + folded batch-norm + ReLU, max-pool)
  and application of predicted residuals / dropout to scans.

### Phase 4 — Batch scenario generation + I/O
- `include/phantom/{io.hpp,generator.hpp}`, `src/{io.cpp,generator.cpp}`,
  `apps/phantom_gen.cpp`, `tests/test_core.cpp`
- Writers: KITTI `.bin` (x,y,z,intensity), `.label` (uint32 semantic|instance),
  boxes `.txt`, per-scene `meta.json`, dataset `index.csv`.
- Multithreaded batch generator CLI (`--count 10000 --threads N --model ...`).
- C++ unit tests registered with CTest.

### Phase 5 — Statistical distribution analysis + Python tests
- `python/phantom_points/{io.py,stats.py}`, `python/analyze.py`, `python/tests/`
- Readers for generated datasets; per-feature distributions (range, intensity,
  points per ring, per-class point counts, elevation).
- KS test, Wasserstein distance, Jensen–Shannon divergence, summary report
  (JSON + optional matplotlib plots).
- pytest suite for model shapes, export round-trip and stats functions.

Afterwards: `.gitignore`, `README.md` with architecture diagram and run/test docs.
