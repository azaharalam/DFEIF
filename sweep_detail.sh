#!/usr/bin/env bash
# Four-level instrumentation-depth sweep across the compiled Coral models.
#
# The axis varied is --detail, which gates how much each event carries and which
# events are emitted at all. baseline is measured separately as the
# uninstrumented reference (tracing off entirely).
#
# Timing note, and it matters: process launch costs roughly 4 ms on this
# machine, which is larger than the instrumentation being measured. Repeating
# the session INSIDE one process (--repeat-session) amortizes that away, so the
# reported per-session figure reflects framework work rather than fork/exec.
# Numbers here are therefore NOT comparable to wall-clock timing of a single
# dfabitctl invocation from the shell -- that would include ~4 ms of launch.
set -uo pipefail

REPEATS="${REPEATS:-200}"   # sessions per process
WARMUP="${WARMUP:-1}"       # untimed warmup invocations per model
PASSES="${PASSES:-3}"       # repeated measurements, to expose spread
MODEL_DIR="${MODEL_DIR:-models}"
OUT="results/detail_sweep"
CSV="$OUT/detail_sweep.csv"

if [ ! -x ./build/dfabitctl ]; then
  echo "error: ./build/dfabitctl not found. Build first: cmake --build build -j" >&2
  exit 1
fi

if ! ./build/dfabitctl --help 2>&1 | grep -q -- "--detail"; then
  echo "error: this build lacks --detail. Apply the detail-level patch first." >&2
  exit 1
fi

mkdir -p "$OUT"
echo "model,total_operations,tpu_mapping_ratio,on_chip_fraction,pass,level,sessions,total_ms,per_session_ms,trace_events,trace_bytes" > "$CSV"

for compiled in "$MODEL_DIR"/*_edgetpu.tflite; do
  [ -f "$compiled" ] || continue
  base=$(basename "$compiled" _edgetpu.tflite)
  summary="$MODEL_DIR/${base}_summary.log"
  if [ ! -f "$summary" ]; then
    echo "SKIP $base (no ${base}_summary.log)"
    continue
  fi

  probe="$OUT/probe_$base"
  ./build/dfabitctl --backend edgetpu --model "$compiled" \
    --compile-report "$summary" --out "$probe" --mode full >/dev/null 2>&1
  m="$probe/reports/compile_metrics.csv"
  field() { grep "^$1," "$m" 2>/dev/null | head -1 | cut -d, -f2; }
  ops=$(field total_operations)
  ratio=$(field tpu_mapping_ratio)
  onchip=$(field on_chip_param_fraction)
  echo "  $base: ${ops:-?} ops, ratio ${ratio:-?}"

  for _ in $(seq "$WARMUP"); do
    ./build/dfabitctl --backend edgetpu --model "$compiled" \
      --compile-report "$summary" --out "$OUT/warm" \
      --mode full --detail full --repeat-session "$REPEATS" >/dev/null 2>&1
  done

  for pass in $(seq "$PASSES"); do
    for level in baseline ids lite full; do
      if [ "$level" = "baseline" ]; then
        mode=baseline; detail=full
      else
        mode=full; detail="$level"
      fi

      dir="$OUT/run_${base}_${level}"
      start=$(date +%s%N)
      ./build/dfabitctl --backend edgetpu --model "$compiled" \
        --compile-report "$summary" --out "$dir" \
        --mode "$mode" --detail "$detail" \
        --repeat-session "$REPEATS" >/dev/null 2>&1
      end=$(date +%s%N)

      total=$(( (end - start) / 1000000 ))
      per=$(awk -v t="$total" -v r="$REPEATS" 'BEGIN{printf "%.4f", t/r}')
      ev=$(grep trace_event_count "$dir/reports/run_summary.txt" 2>/dev/null | cut -d= -f2)
      by=$(grep trace_bytes_written "$dir/reports/run_summary.txt" 2>/dev/null | cut -d= -f2)

      echo "$base,${ops:-},${ratio:-},${onchip:-},$pass,$level,$REPEATS,$total,$per,${ev:-0},${by:-0}" >> "$CSV"
      printf "    pass%s %-9s %7s ms total  %9s ms/session  %4s events %7s B\n" \
        "$pass" "$level" "$total" "$per" "${ev:-0}" "${by:-0}"
    done
  done
done

echo
echo "=== median ms/session by model and level ==="
awk -F, 'NR>1 { k=$1","$2","$6; v[k]=v[k]" "$9 }
END { print "model,ops,level,median_ms"
  for (k in v) { n=split(v[k],a," "); m=0
    for(i=1;i<=n;i++) if(a[i]!="") x[++m]=a[i]+0
    for(i=1;i<m;i++) for(j=i+1;j<=m;j++) if(x[j]<x[i]){t=x[i];x[i]=x[j];x[j]=t}
    printf "%s,%.4f\n", k, (m%2)?x[(m+1)/2]:(x[m/2]+x[m/2+1])/2
    delete x } }' "$CSV" | sort -t, -k2 -n -k3

echo
echo "wrote $CSV"
echo
echo "NOTE: per-session figures exclude process launch (~4 ms). State this in"
echo "the methodology; a reviewer timing a single invocation will see more."
