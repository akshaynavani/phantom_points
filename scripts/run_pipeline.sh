#!/usr/bin/env bash
# End-to-end: build -> paired data -> train refiner -> 10k refined scenarios -> report.
# Override sizes with env vars, e.g. PAIRS=200 SCENES=1000 EPOCHS=5 ./scripts/run_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PAIRS="${PAIRS:-2000}"
SCENES="${SCENES:-10000}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-1}"
OUT="${OUT:-out}"
PY="${PYTHON:-python3}"

echo "==> building"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "==> generating ${PAIRS} paired scenes (source + reference sensor)"
./build/phantom_gen --out "$OUT/pairs" --count "$PAIRS" --seed 1000 \
  --config configs/default.ini --pair-config configs/reference_sensor.ini

echo "==> training PointNet refiner"
PYTHONPATH=python "$PY" -m phantom_points.train --data "$OUT/pairs" --epochs "$EPOCHS" --out runs/refiner

echo "==> generating ${SCENES} refined scenarios"
./build/phantom_gen --out "$OUT/dataset" --count "$SCENES" --seed "$SEED" \
  --config configs/default.ini --model runs/refiner/refiner.ppnt

echo "==> realism report: base simulator vs reference sensor"
"$PY" python/analyze.py --synthetic "$OUT/pairs" --reference "$OUT/pairs" --reference-ref \
  --out reports/base_vs_reference --plots

echo "==> realism report: refined scenarios vs reference sensor"
"$PY" python/analyze.py --synthetic "$OUT/dataset" --reference "$OUT/pairs" --reference-ref \
  --out reports/refined_vs_reference --plots
