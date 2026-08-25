#!/usr/bin/env bash
#
# Edge TPU compile-stage instrumentation sweep.
#
# Measures DFIT's compile-stage cost across models of varying operator-table
# size, so the relationship between graph size and instrumentation cost can be
# established rather than asserted.
#
# Method, and why each part matters:
#   - Each configuration runs REPEATS times inside a timed loop, so 1 ms clock
#     granularity becomes ~1/REPEATS ms of resolution.
#   - Every model gets WARMUP untimed runs first. Without this the first
#     configuration measured absorbs cold page-cache cost and looks slower than
#     it is -- which produced a spurious "instrumentation makes it faster"
#     result before the warmup was added.
#   - Modes are measured PASSES times each so run-to-run spread is visible.
#     A single pass cannot distinguish a real effect from noise.
set -uo pipefail

REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-5}"
PASSES="${PASSES:-3}"
MODEL_DIR="${MODEL_DIR:-models}"
OUT_DIR="results/edgetpu_sweep"
CSV="$OUT_DIR/sweep.csv"

if [[ ! -x ./build/dfabitctl ]]; then
  echo "error: ./build/dfabitctl not found. Build first: cmake --build build -j" >&2
  exit 1
fi
if ! command -v edgetpu_compiler > /dev/null 2>&1; then
  echo "error: edgetpu_compiler not on PATH" >&2
  exit 1
fi

MODELS=(
  mobilenet_v1_1.0_224_quant
  mobilenet_v2_1.0_224_quant
  inception_v1_224_quant
  inception_v2_224_quant
  ssd_mobilenet_v2_coco_quant_postprocess
  deeplabv3_mnv2_pascal_quant
  inception_v3_299_quant
  inception_v4_299_quant
)

MODES=(baseline sampled selective full)
BASE_URL="https://raw.githubusercontent.com/google-coral/test_data/master"

mkdir -p "$MODEL_DIR" "$OUT_DIR"

echo "=== fetching and compiling models ==="
AVAILABLE=()
for model in "${MODELS[@]}"; do
  tflite="$MODEL_DIR/${model}.tflite"
  compiled="$MODEL_DIR/${model}_edgetpu.tflite"
  summary="$MODEL_DIR/${model}_summary.log"

  if [[ ! -f "$tflite" ]]; then
    echo "  downloading $model"
    if ! curl -sfL "$BASE_URL/${model}.tflite" -o "$tflite"; then
      echo "  SKIP $model (download failed)" >&2
      rm -f "$tflite"
      continue
    fi
  fi

  if [[ ! -f "$summary" ]]; then
    echo "  compiling $model"
    # -s writes the memory summary and operator table to stdout. The
    # auto-generated *_edgetpu.log contains only the operator table, so the
    # summary must be captured separately or the memory metrics come back zero.
    if ! (cd "$MODEL_DIR" && edgetpu_compiler -s "${model}.tflite" > "${model}_summary.log" 2>&1); then
      echo "  SKIP $model (compile failed; see $summary)" >&2
      continue
    fi
  fi

  if [[ ! -f "$compiled" ]]; then
    echo "  SKIP $model (no compiled model produced)" >&2
    continue
  fi

  AVAILABLE+=("$model")
done

if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
  echo "error: no models available to sweep" >&2
  exit 1
fi

echo
echo "=== sweeping ${#AVAILABLE[@]} models x ${#MODES[@]} modes x $PASSES passes x $REPEATS reps ==="

echo "model,total_operations,ops_mapped,ops_fallback,tpu_mapping_ratio,on_chip_fraction,pass,mode,total_ms,per_run_ms" > "$CSV"

for model in "${AVAILABLE[@]}"; do
  compiled="$MODEL_DIR/${model}_edgetpu.tflite"
  summary="$MODEL_DIR/${model}_summary.log"

  probe_dir="$OUT_DIR/probe_${model}"
  ./build/dfabitctl --backend edgetpu --model "$compiled" \
    --compile-report "$summary" --out "$probe_dir" --mode full > /dev/null 2>&1

  metrics="$probe_dir/reports/compile_metrics.csv"
  field() { grep "^$1," "$metrics" 2>/dev/null | head -1 | cut -d, -f2; }

  total_ops=$(field total_operations)
  mapped=$(field ops_mapped_to_tpu)
  fallback=$(field ops_fallback_to_cpu)
  ratio=$(field tpu_mapping_ratio)
  onchip=$(field on_chip_param_fraction)

  echo "  $model: ${total_ops:-?} ops, ratio ${ratio:-?}, on-chip ${onchip:-?}"

  for _ in $(seq "$WARMUP"); do
    ./build/dfabitctl --backend edgetpu --model "$compiled" \
      --compile-report "$summary" --out "$OUT_DIR/warm" --mode full > /dev/null 2>&1
  done

  for pass in $(seq "$PASSES"); do
    for mode in "${MODES[@]}"; do
      start=$(date +%s%N)
      for _ in $(seq "$REPEATS"); do
        ./build/dfabitctl --backend edgetpu --model "$compiled" \
          --compile-report "$summary" \
          --out "$OUT_DIR/run_${model}_${mode}" --mode "$mode" > /dev/null 2>&1
      done
      end=$(date +%s%N)

      total_ms=$(( (end - start) / 1000000 ))
      per_run=$(awk -v t="$total_ms" -v r="$REPEATS" 'BEGIN { printf "%.3f", t / r }')

      echo "$model,${total_ops:-},${mapped:-},${fallback:-},${ratio:-},${onchip:-},$pass,$mode,$total_ms,$per_run" >> "$CSV"
      printf "    pass%s %-10s %6s ms total  %8s ms/run\n" "$pass" "$mode" "$total_ms" "$per_run"
    done
  done
done

echo
echo "=== summary (median per-run ms across passes) ==="
awk -F, 'NR > 1 {
  key = $1 "," $2 "," $8
  vals[key] = vals[key] " " $10
}
END {
  print "model,total_operations,mode,median_ms"
  for (k in vals) {
    n = split(vals[k], a, " ")
    m = 0
    for (i = 1; i <= n; i++) if (a[i] != "") { v[++m] = a[i] + 0 }
    for (i = 1; i < m; i++) for (j = i + 1; j <= m; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
    med = (m % 2) ? v[(m + 1) / 2] : (v[m / 2] + v[m / 2 + 1]) / 2
    printf "%s,%.3f\n", k, med
    delete v
  }
}' "$CSV" | sort -t, -k2 -n

echo
echo "wrote $CSV"
echo
echo "NOTE: overhead here is DFIT's own session cost, not a fraction of"
echo "compilation. Compare against the compile_time_ms metric before quoting a"
echo "percentage -- the two denominators differ by orders of magnitude."
