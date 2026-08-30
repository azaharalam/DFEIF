#!/usr/bin/env bash
# Runs both instrumentation tools across both backends in one pass and collects
# their summaries into per-backend tables.
#
# A single dfabitctl invocation drives every registered tool, so nothing here
# is tool-specific: the loop prepares a backend's inputs and the framework does
# the rest. That is the portability claim in operational form -- the same
# command, the same output schema, two backends with nothing architecturally in
# common.
#
# The two backends do not report the same things, and that is expected. Edge TPU
# executes a sequential operator list against one 8 MB SRAM, so peak activation
# working set is a physically meaningful figure there and can be read against
# the compiler's own parameter residency report. Cerebras streams weights from
# external memory and partitions activations across the wafer, so the same
# figure describes the graph rather than the hardware; the forward-to-backward
# stash is the comparable quantity. Source-line attribution likewise reaches
# model code on Cerebras, where the compiler records callsites, and stops at
# operator type on Edge TPU, where the report carries none.
set -uo pipefail

OUT=results/tools
mkdir -p "$OUT"

if [ ! -x ./build/dfabitctl ]; then
  echo "error: ./build/dfabitctl not found. Build first." >&2
  exit 1
fi

metric() { grep -h "^$2," "$1" 2>/dev/null | head -1 | cut -d, -f2; }

MEM_ETPU="$OUT/edgetpu_memory.csv"
ATT_ETPU="$OUT/edgetpu_attribution.csv"
MEM_CB="$OUT/cerebras_memory.csv"
ATT_CB="$OUT/cerebras_attribution.csv"

echo "model,operators,parameter_bytes,peak_activation_bytes,total_footprint_bytes,on_chip_reported,off_chip_reported" > "$MEM_ETPU"
echo "model,operators,with_module,with_source_site,source_sites,module_only,unattributed" > "$ATT_ETPU"
echo "model,graph,operators,total_footprint_bytes,peak_live_bytes,retained_bytes,retained_tensors,mean_reuse_distance,dominant_phase,dominant_module" > "$MEM_CB"
echo "model,graph,operators,with_module,with_source_site,source_sites,module_only,unattributed" > "$ATT_CB"

# ------------------------------------------------------------------ Edge TPU
echo "=== Edge TPU"
printf "%-42s %6s %11s %10s %8s %8s\n" "model" "ops" "params_MiB" "peak_MiB" "sites" "src_ops"

for compiled in models/*_edgetpu.tflite; do
  [ -f "$compiled" ] || continue
  base=$(basename "$compiled" _edgetpu.tflite)
  log="models/${base}_summary.log"
  src="models/${base}.tflite"
  [ -f "$log" ] && [ -f "$src" ] || continue

  d="$OUT/etpu_$base"
  rm -rf "$d"
  ./build/dfabitctl --backend edgetpu --model "$compiled" \
    --compile-report "$log" --out "$d" --mode full --detail full >/dev/null 2>&1

  M="$d/tools/dataflow_memory_proxy/dataflow_memory_proxy_summary.csv"
  A="$d/tools/semantic_attribution/semantic_attribution_summary.csv"
  C="$d/reports/compile_metrics.csv"
  [ -f "$M" ] || continue

  ops=$(metric "$M" operators)
  peak=$(metric "$M" peak_live_bytes)
  foot=$(metric "$M" total_footprint_bytes)
  par=$(metric "$C" model_parameter_bytes)
  on=$(grep "On-chip memory used" "$log" | cut -d: -f2 | tr -d ' ')
  off=$(grep "Off-chip memory used" "$log" | cut -d: -f2 | tr -d ' ')
  echo "$base,${ops:-},${par:-},${peak:-},${foot:-},${on:-},${off:-}" >> "$MEM_ETPU"

  wm=$(metric "$A" operators_with_module)
  ws=$(metric "$A" operators_with_source_site)
  ss=$(metric "$A" source_sites)
  mo=$(metric "$A" operators_attributed_by_module_only)
  un=$(metric "$A" operators_unattributed)
  echo "$base,${ops:-},${wm:-0},${ws:-0},${ss:-0},${mo:-0},${un:-0}" >> "$ATT_ETPU"

  printf "%-42s %6s %11.2f %10.2f %8s %8s\n" "${base:0:42}" "${ops:-?}" \
    "$(awk -v v="${par:-0}" 'BEGIN{print v/1048576}')" \
    "$(awk -v v="${peak:-0}" 'BEGIN{print v/1048576}')" \
    "${ss:-0}" "${ws:-0}"
done

# ------------------------------------------------------------------ Cerebras
echo
echo "=== Cerebras"
printf "%-26s %7s %11s %12s %7s %8s\n" "model/graph" "ops" "total_GiB" "retained_GiB" "sites" "src_ops"

for model_dir in dfit_ship/*/; do
  [ -d "$model_dir" ] || continue
  m=$(basename "$model_dir")
  for gz in "$model_dir"cirh_*.mlir.gz; do
    [ -f "$gz" ] || continue
    graph=$(basename "$gz" .mlir.gz)

    work=/tmp/toolsweep/${m}_${graph}/cerebras_logs/x/executors/000001
    mkdir -p "$work"
    gunzip -c "$gz" > "$work/cirh.mlir"

    d="$OUT/cb_${m}_${graph}"
    rm -rf "$d"
    ./build/dfabitctl --backend cerebras \
      --model-dir "/tmp/toolsweep/${m}_${graph}" \
      --out "$d" --mode full --detail full >/dev/null 2>&1

    M="$d/tools/dataflow_memory_proxy/dataflow_memory_proxy_summary.csv"
    A="$d/tools/semantic_attribution/semantic_attribution_summary.csv"
    [ -f "$M" ] || continue

    ops=$(metric "$M" operators)
    foot=$(metric "$M" total_footprint_bytes)
    peak=$(metric "$M" peak_live_bytes)
    ret=$(metric "$M" retained_to_backward_bytes)
    rett=$(metric "$M" retained_to_backward_tensors)
    reuse=$(metric "$M" mean_reuse_distance)
    dph=$(metric "$M" dominant_phase)
    dmod=$(metric "$M" dominant_module)
    echo "$m,$graph,${ops:-},${foot:-},${peak:-},${ret:-},${rett:-},${reuse:-},${dph:-},${dmod:-}" >> "$MEM_CB"

    wm=$(metric "$A" operators_with_module)
    ws=$(metric "$A" operators_with_source_site)
    ss=$(metric "$A" source_sites)
    mo=$(metric "$A" operators_attributed_by_module_only)
    un=$(metric "$A" operators_unattributed)
    echo "$m,$graph,${ops:-},${wm:-0},${ws:-0},${ss:-0},${mo:-0},${un:-0}" >> "$ATT_CB"

    printf "%-26s %7s %11.2f %12.2f %7s %8s\n" "${m}/${graph:5:12}" "${ops:-?}" \
      "$(awk -v v="${foot:-0}" 'BEGIN{print v/1073741824}')" \
      "$(awk -v v="${ret:-0}" 'BEGIN{print v/1073741824}')" \
      "${ss:-0}" "${ws:-0}"
  done
done

rm -rf /tmp/toolsweep

echo
echo "wrote $MEM_ETPU"
echo "wrote $ATT_ETPU"
echo "wrote $MEM_CB"
echo "wrote $ATT_CB"
echo
echo "Per-site attribution tables are in each run directory under"
echo "tools/semantic_attribution/semantic_attribution_summary.csv."
