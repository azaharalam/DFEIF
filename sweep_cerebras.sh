#!/usr/bin/env bash
# Four-level instrumentation-depth sweep over the shipped Cerebras artifacts.
#
# Method, and why each part is here:
#   - WARMUP passes are timed but discarded. Without them the first measured
#     pass runs 20-30% high on page-cache and CPU-frequency effects, which is
#     what produced the spurious 50% overhead figures in an earlier run.
#   - Each configuration is repeated REPEATS times inside ONE process
#     (--repeat-session), so process launch (~4 ms) does not dominate.
#   - PASSES independent measurements per configuration expose run-to-run
#     spread; the median is reported. A single pass cannot distinguish a real
#     effect from noise.
#
# DENOMINATOR, which must be stated wherever these numbers appear:
# --mode baseline still parses the entire cirh.mlir. Parsing happens in
# CompileEnd regardless of whether tracing is on, so this measures the cost of
# EVENT EMISSION against a baseline that already includes the full parse. It is
# not instrumentation-versus-nothing.
set -uo pipefail

REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-1}"
PASSES="${PASSES:-3}"
SHIP_DIR="${SHIP_DIR:-dfit_ship}"
OUT="results/cerebras_sweep"
CSV="$OUT/cerebras_sweep.csv"

if [ ! -x ./build/dfabitctl ]; then
  echo "error: ./build/dfabitctl not found" >&2
  exit 1
fi
if ! ./build/dfabitctl --help 2>&1 | grep -q -- "--model-dir"; then
  echo "error: this build lacks --model-dir" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "model,graph,operators,pass,level,sessions,total_ms,per_session_ms,trace_events,trace_bytes" > "$CSV"

for model_dir in "$SHIP_DIR"/*/; do
  m=$(basename "$model_dir")
  for gz in "$model_dir"cirh_*.mlir.gz; do
    [ -f "$gz" ] || continue
    graph=$(basename "$gz" .mlir.gz)

    work=/tmp/cbsweep/${m}_${graph}/cerebras_logs/x/executors/000001
    mkdir -p "$work"
    gunzip -c "$gz" > "$work/cirh.mlir"
    root=/tmp/cbsweep/${m}_${graph}

    probe="$OUT/probe_${m}_${graph}"
    ./build/dfabitctl --backend cerebras --model-dir "$root" \
      --out "$probe" --mode full --detail full >/dev/null 2>&1
    ops=$(grep -h '^cirh_operators,' "$probe/reports/compile_metrics.csv" 2>/dev/null | head -1 | cut -d, -f2)
    echo "  $m/$graph: ${ops:-?} operators"

    # Warmup: timed but discarded.
    for _ in $(seq "$WARMUP"); do
      ./build/dfabitctl --backend cerebras --model-dir "$root" \
        --out "$OUT/warm" --mode full --detail full \
        --repeat-session "$REPEATS" >/dev/null 2>&1
    done

    for pass in $(seq "$PASSES"); do
      for level in baseline ids lite full; do
        if [ "$level" = "baseline" ]; then mode=baseline; detail=full
        else mode=full; detail="$level"; fi

        dir="$OUT/run_${m}_${graph}_${level}"
        start=$(date +%s%N)
        ./build/dfabitctl --backend cerebras --model-dir "$root" \
          --out "$dir" --mode "$mode" --detail "$detail" \
          --repeat-session "$REPEATS" >/dev/null 2>&1
        end=$(date +%s%N)

        total=$(( (end - start) / 1000000 ))
        per=$(awk -v t="$total" -v r="$REPEATS" 'BEGIN{printf "%.3f", t/r}')
        ev=$(grep trace_event_count "$dir/reports/run_summary.txt" 2>/dev/null | cut -d= -f2)
        by=$(grep trace_bytes_written "$dir/reports/run_summary.txt" 2>/dev/null | cut -d= -f2)

        echo "$m,$graph,${ops:-},$pass,$level,$REPEATS,$total,$per,${ev:-0},${by:-0}" >> "$CSV"
        printf "    pass%s %-9s %8s ms/session  %6s events %9s B\n" \
          "$pass" "$level" "$per" "${ev:-0}" "${by:-0}"
      done
    done
  done
done

echo
echo "=== median ms/session, and overhead against baseline ==="
awk -F, 'NR>1 { k=$1"/"$2","$3","$5; v[k]=v[k]" "$8 }
END {
  for (k in v) {
    n=split(v[k],a," "); m=0
    for(i=1;i<=n;i++) if(a[i]!="") x[++m]=a[i]+0
    for(i=1;i<m;i++) for(j=i+1;j<=m;j++) if(x[j]<x[i]){t=x[i];x[i]=x[j];x[j]=t}
    med[k]=(m%2)?x[(m+1)/2]:(x[m/2]+x[m/2+1])/2
    delete x
    split(k,p,","); key=p[1]","p[2]
    if (p[3]=="baseline") base[key]=med[k]
    keys[key]=1
  }
  printf "%-28s %8s %10s %10s %10s %10s\n","model/graph","ops","baseline","ids","lite","full"
  for (key in keys) {
    split(key,p,",")
    b=base[key]
    printf "%-28s %8s %10.0f", p[1], p[2], b
    split("ids lite full", L, " ")
    for (i=1;i<=3;i++) {
      v2=med[key","L[i]]
      if (b>0 && v2>0) printf " %6.0f(%+.1f%%)", v2, (v2-b)/b*100
      else printf " %10s", "-"
    }
    printf "\n"
  }
}' "$CSV" | sort

echo
echo "wrote $CSV"
echo
echo "NOTE: baseline includes the full cirh.mlir parse; these percentages are"
echo "the cost of event emission on top of parsing, not instrumentation versus"
echo "an uninstrumented run. Pass 1 is discarded as warmup."
