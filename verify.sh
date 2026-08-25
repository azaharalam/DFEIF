#!/usr/bin/env bash
#
# Checks the hardware-free claims in REPRODUCE.md and reports pass or fail.
#
# Structural values -- operator counts, MAC totals, executor counts -- are
# deterministic and are compared exactly. Timing values are not checked here:
# they depend on the host, and a threshold that passes on one machine and fails
# on another is worse than no check at all. Run sweep_cerebras.sh for those.
#
# Usage:  bash verify.sh
# Exit:   0 if every check passed, 1 otherwise.
set -uo pipefail

PASS=0
FAIL=0
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

green() { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
red()   { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }

check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    green "$label = $actual"
  else
    red "$label: expected $expected, got ${actual:-<empty>}"
  fi
}

# Compare two floats to a relative tolerance, since MAC totals are printed in
# scientific notation and exact string equality would be brittle.
check_approx() {
  local label="$1" expected="$2" actual="$3" tol="${4:-0.001}"
  if [ -z "$actual" ]; then red "$label: no value"; return; fi
  local ok
  ok=$(awk -v e="$expected" -v a="$actual" -v t="$tol" \
       'BEGIN { d = (a-e)/e; if (d<0) d=-d; print (d<=t) ? "1" : "0" }')
  if [ "$ok" = "1" ]; then
    green "$label = $actual"
  else
    red "$label: expected ~$expected, got $actual"
  fi
}

metric() { grep -h "^$2," "$1" 2>/dev/null | head -1 | cut -d, -f2; }

# Extracts a metric row matching an attribute, e.g. phase=forward.
metric_attr() {
  grep -h "^$2," "$1" 2>/dev/null | grep "$3" | head -1 | cut -d, -f2
}

stage_graph() {  # <gz> <dir>  -> lays out a model_dir the adapter can discover
  local gz="$1" dir="$2"
  mkdir -p "$dir/cerebras_logs/x/executors/000001"
  gunzip -c "$gz" > "$dir/cerebras_logs/x/executors/000001/cirh.mlir"
}

echo "DFIT verification"
echo

# ---------------------------------------------------------------- build check
echo "build"
if [ ! -x ./build/dfabitctl ]; then
  red "./build/dfabitctl not found -- run: cmake -S . -B build -DDFABIT_BUILD_PYTHON=OFF && cmake --build build -j"
  echo; echo "cannot continue"; exit 1
fi
green "dfabitctl present"

if ./build/dfabitctl --help 2>&1 | grep -q -- "--model-dir"; then
  green "--model-dir supported"
else
  red "--model-dir missing: build predates Cerebras IR discovery"
fi

if [ -x ./build/smoketest ]; then
  if ./build/smoketest >/dev/null 2>&1; then green "smoketest"; else red "smoketest"; fi
fi
echo

# ------------------------------------------------- T1.1 ViT operator recovery
echo "T1.1  Cerebras operator recovery and MAC computation (ViT)"
GZ=dfit_ship/vit/cirh_10288ops.mlir.gz
if [ ! -f "$GZ" ]; then
  red "$GZ not found -- shipped artifacts missing"
else
  stage_graph "$GZ" "$TMP/vit"
  ./build/dfabitctl --backend cerebras --model-dir "$TMP/vit" \
    --out "$TMP/vit_out" --mode full --detail full >/dev/null 2>&1
  M="$TMP/vit_out/reports/compile_metrics.csv"

  check        "cirh_operators"        10262      "$(metric "$M" cirh_operators)"
  check_approx "forward MACs"          5.00569e13 "$(metric_attr "$M" cirh_phase_macs 'phase=forward')"
  check_approx "backward MACs"         9.97843e13 "$(metric_attr "$M" cirh_phase_macs 'phase=backward')"
  check        "optimizer MACs"        0          "$(metric_attr "$M" cirh_phase_macs 'phase=optimizer')"
  check        "operators with lineage" 10247     "$(metric "$M" cirh_operators_with_lineage)"
fi
echo

# --------------------------------------------------- T1.2 all shipped graphs
echo "T1.2  All shipped graphs parse to their expected operator counts"
while IFS=' ' read -r model tag expected; do
  [ -z "$model" ] && continue
  GZ="dfit_ship/$model/$tag.mlir.gz"
  if [ ! -f "$GZ" ]; then red "$model/$tag missing"; continue; fi
  D="$TMP/${model}_${tag}"
  stage_graph "$GZ" "$D"
  ./build/dfabitctl --backend cerebras --model-dir "$D" \
    --out "${D}_out" --mode full >/dev/null 2>&1
  check "$model/$tag" "$expected" "$(metric "${D}_out/reports/compile_metrics.csv" cirh_operators)"
done << 'EOF'
vit cirh_811ops 810
vit cirh_10288ops 10262
bert cirh_1604ops 1602
bert cirh_19956ops 19904
dit cirh_2800ops 2800
dit cirh_27586ops 27537
esm2 cirh_26910ops 26841
mixtral cirh_2420ops 2329
mixtral cirh_16401ops 16169
dino cirh_32670ops 32569
llama3_8b cirh_19401ops 19400
EOF
echo

# ------------------------------------------------------- T1.4 multi-executor
echo "T1.4  Per-executor attribution"
if [ -f dfit_ship/vit/cirh_10288ops.mlir.gz ] && [ -f dfit_ship/vit/cirh_811ops.mlir.gz ]; then
  mkdir -p "$TMP/multi/cerebras_logs/x/executors/000001" \
           "$TMP/multi/cerebras_logs/x/executors/000002"
  gunzip -c dfit_ship/vit/cirh_10288ops.mlir.gz > "$TMP/multi/cerebras_logs/x/executors/000001/cirh.mlir"
  gunzip -c dfit_ship/vit/cirh_811ops.mlir.gz  > "$TMP/multi/cerebras_logs/x/executors/000002/cirh.mlir"
  ./build/dfabitctl --backend cerebras --model-dir "$TMP/multi" \
    --out "$TMP/multi_out" --mode full --detail full >/dev/null 2>&1
  M="$TMP/multi_out/reports/compile_metrics.csv"
  check "cirh_executors" 2 "$(metric "$M" cirh_executors)"
  check "executor 000001 operators" 10262 "$(metric_attr "$M" cirh_operators 'executor=000001')"
  check "executor 000002 operators" 810   "$(metric_attr "$M" cirh_operators 'executor=000002')"
else
  red "ViT graphs missing; cannot check multi-executor attribution"
fi
echo

# ------------------------------------------ T1.5 trace volume by detail level
echo "T1.5  Trace volume scales with instrumentation depth"
if [ -f dfit_ship/vit/cirh_10288ops.mlir.gz ]; then
  stage_graph dfit_ship/vit/cirh_10288ops.mlir.gz "$TMP/tv"
  prev=0
  ordered=1
  for lvl in ids lite full; do
    ./build/dfabitctl --backend cerebras --model-dir "$TMP/tv" \
      --out "$TMP/tv_$lvl" --mode full --detail "$lvl" >/dev/null 2>&1
    b=$(grep trace_bytes_written "$TMP/tv_$lvl/reports/run_summary.txt" 2>/dev/null | cut -d= -f2)
    [ -z "$b" ] && b=0
    printf "       %-5s %10s bytes\n" "$lvl" "$b"
    if [ "$b" -le "$prev" ]; then ordered=0; fi
    prev="$b"
  done
  if [ "$ordered" = "1" ]; then
    green "trace volume increases ids < lite < full"
  else
    red "trace volume not monotonic across detail levels"
  fi
fi
echo

# -------------------------------------- T1.6 no unmeasured metric is reported
echo "T1.6  No overhead figure is produced without a measurement"
if [ -f examples/cerebras/graph.txt ]; then
  ./build/dfabitctl --backend cerebras --graph examples/cerebras/graph.txt \
    --out "$TMP/nm" --mode full >/dev/null 2>&1
  if [ -f "$TMP/nm/reports/overhead_not_measured.txt" ]; then
    green "overhead_not_measured.txt written"
  else
    red "expected overhead_not_measured.txt"
  fi
  if [ -d "$TMP/nm/tools/overhead_profiler" ] && \
     [ -n "$(ls -A "$TMP/nm/tools/overhead_profiler" 2>/dev/null)" ]; then
    red "overhead artifacts produced without a paired measurement"
  else
    green "no overhead artifacts produced"
  fi
fi
echo

# ------------------------------- SambaNova honest degradation without hardware
echo "SambaNova  capability discovery reports absent hardware honestly"
if [ -f examples/sambanova/graph.txt ]; then
  ./build/dfabitctl --backend sambanova --graph examples/sambanova/graph.txt \
    --out "$TMP/sn" --mode full >/dev/null 2>&1
  C="$TMP/sn/tool_portability_capabilities.csv"
  if [ -f "$C" ]; then
    check "profiler_metrics_available" 0 "$(grep '^profiler_metrics_available,' "$C" | cut -d, -f2)"
    check "custom_env_controls"        0 "$(grep '^custom_env_controls,' "$C" | cut -d, -f2)"
  else
    red "no capability report produced"
  fi
fi

# ------------------------------------------------ T2 Edge TPU compile stage
echo "T2  Edge TPU compiler-report analysis"
if [ ! -d dfit_ship_tpu ]; then
  echo "       dfit_ship_tpu/ not present -- SKIP"
else
  # The adapter requires --model, but reads only the compiler report when one
  # is supplied. A placeholder therefore suffices, which is why the models
  # themselves (441 MB) are not shipped.
  STUB="$TMP/stub.tflite"
  : > "$STUB"

  while IFS=' ' read -r name ops sub ratio onchip; do
    [ -z "$name" ] && continue
    LOG="dfit_ship_tpu/${name}_summary.log"
    if [ ! -f "$LOG" ]; then red "$name: summary log missing"; continue; fi

    ./build/dfabitctl --backend edgetpu --model "$STUB" \
      --compile-report "$LOG" --out "$TMP/t2_$name" --mode full >/dev/null 2>&1
    M="$TMP/t2_$name/reports/compile_metrics.csv"

    check        "$name total_operations"    "$ops"   "$(metric "$M" total_operations)"
    check        "$name edgetpu_subgraphs"   "$sub"   "$(metric "$M" edgetpu_subgraphs)"
    check_approx "$name tpu_mapping_ratio"   "$ratio" "$(metric "$M" tpu_mapping_ratio)" 0.001
    check_approx "$name on_chip_fraction"    "$onchip" "$(metric "$M" on_chip_param_fraction)" 0.01
  done << 'T2EOF'
GEMM_1024_quant 1 1 1.0 1.0
mobilenet_v1_1.0_224_quant 31 1 1.0 1.0
mobilenet_v2_1.0_224_quant 66 1 1.0 1.0
deeplabv3_mnv2_pascal_quant 72 1 0.888889 1.0
tfhub_tf2_resnet_50_imagenet_ptq 78 1 1.0 0.270
inception_v2_224_quant 98 1 1.0 0.558
ssd_mobilenet_v2_coco_quant_postprocess 111 1 0.972973 1.0
inception_v3_299_quant 132 1 1.0 0.237
resnet101_224_quant 143 1 1.0 0.167
inception_v4_299_quant 205 1 1.0 0.140
yolov5n_320_quant 277 1 0.916968 1.0
SpMM_1024_quant 1 1 1.0 1.0
efficientnet-edgetpu-S_quant 66 1 1.0 1.0
tf2_mobilenet_v3_edgetpu_1.0_224_ptq 79 1 1.0 1.0
keras_post_training_unet_mv2_128_quant 81 1 1.0 0.998
inception_v1_224_quant 83 1 1.0 0.945
T2EOF
fi
echo

# --------------------------------- T2b fresh compile agrees with shipped log
echo "T2b Recompiling reproduces the shipped report (needs edgetpu_compiler)"
if ! command -v edgetpu_compiler >/dev/null 2>&1; then
  echo "       edgetpu_compiler not on PATH -- SKIP"
elif [ ! -f models/mobilenet_v2_1.0_224_quant.tflite ]; then
  echo "       models/mobilenet_v2_1.0_224_quant.tflite not present -- SKIP"
else
  ( cd "$TMP" && edgetpu_compiler -s \
      "$OLDPWD/models/mobilenet_v2_1.0_224_quant.tflite" \
      > fresh_summary.log 2>&1 )
  : > "$TMP/stub2.tflite"
  ./build/dfabitctl --backend edgetpu --model "$TMP/stub2.tflite" \
    --compile-report "$TMP/fresh_summary.log" \
    --out "$TMP/t2b" --mode full >/dev/null 2>&1
  check "fresh compile total_operations" 66 \
    "$(metric "$TMP/t2b/reports/compile_metrics.csv" total_operations)"
  check_approx "fresh compile tpu_mapping_ratio" 1.0 \
    "$(metric "$TMP/t2b/reports/compile_metrics.csv" tpu_mapping_ratio)"
fi
echo

# ------------------------------------------------ T2c latency (needs device)
echo "T2c Edge TPU latency (needs a Coral device)"
if ! python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; import sys; sys.exit(0 if list_edge_tpus() else 1)" >/dev/null 2>&1; then
  echo "       no Coral device enumerated -- SKIP"
elif [ ! -f models/mobilenet_v2_1.0_224_quant_edgetpu.tflite ]; then
  echo "       compiled model not present -- SKIP"
else
  PYTHONPATH=python python3 -m dfabit.edgetpu.bench \
    --model models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
    --iters 100 --warmup 20 --instrument full \
    --runtime-log "$TMP/rt.csv" --quiet >/dev/null 2>&1
  lat=$(grep '^run,latency_ms' "$TMP/rt.csv" 2>/dev/null | cut -d, -f3)
  if [ -z "$lat" ]; then
    red "no latency recorded"
  else
    ok=$(awk -v l="$lat" 'BEGIN { print (l > 1 && l < 30) ? 1 : 0 }')
    if [ "$ok" = "1" ]; then
      green "MobileNetV2 latency ${lat} ms (expected 3-15 ms on USB)"
    else
      red "MobileNetV2 latency ${lat} ms outside the plausible 1-30 ms range"
    fi
  fi
fi

echo
echo "--------------------------------------------"
printf "passed %d, failed %d\n" "$PASS" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo "All hardware-free checks passed."
  echo "Timing claims are not checked here; run sweep_cerebras.sh for those."
  exit 0
fi
echo "Some checks failed. See REPRODUCE.md for what each one asserts."
exit 1
