# Reproducing the reported numbers

Each claim below maps to a command and the value it should return. Tier 1 needs
no hardware and no vendor toolchain. Tier 2 needs the Edge TPU compiler. Tier 3
needs an attached Coral device.

`verify.sh` runs every Tier 1 check and reports pass or fail:

```bash
bash verify.sh
```

From a clean clone this is 91 checks. Tier 2 and Tier 3 checks report SKIP when
their prerequisite is absent.

---

## Setup

```bash
cmake -S . -B build -DDFABIT_BUILD_PYTHON=OFF
cmake --build build -j
./build/smoketest
```

Several checks below read a shipped Cerebras dump. Stage it once:

```bash
mkdir -p /tmp/v/cerebras_logs/x/executors/000001
gunzip -c dfit_ship/vit/cirh_10288ops.mlir.gz \
  > /tmp/v/cerebras_logs/x/executors/000001/cirh.mlir

./build/dfabitctl --backend cerebras --model-dir /tmp/v --out /tmp/vo --mode full
```

---

## Tier 1: no hardware required

### T1.1 Operator recovery and MAC computation (ViT)

```bash
grep -E "cirh_operators|cirh_phase_macs|cirh_operators_with_lineage" \
  /tmp/vo/reports/compile_metrics.csv
```

| metric | expected |
|---|---|
| `cirh_operators` | 10262 |
| `cirh_phase_macs` forward | 5.00569e+13 |
| `cirh_phase_macs` backward | 9.97843e+13 |
| `cirh_phase_macs` optimizer | 0 |
| `cirh_operators_with_lineage` | 10247 |

The backward-to-forward MAC ratio is 1.993, against a theoretical 2.

### T1.2 All shipped graphs parse to their expected operator counts

```bash
for gz in dfit_ship/*/cirh_*.mlir.gz; do
  d=/tmp/p/$(basename $(dirname $gz))_$(basename $gz .mlir.gz)
  mkdir -p $d/cerebras_logs/x/executors/000001
  gunzip -c $gz > $d/cerebras_logs/x/executors/000001/cirh.mlir
  ./build/dfabitctl --backend cerebras --model-dir $d --out $d/out --mode full >/dev/null 2>&1
  printf "%-32s %s\n" "$(basename $gz)" \
    "$(grep '^cirh_operators,' $d/out/reports/compile_metrics.csv | cut -d, -f2)"
done
```

| graph | operators |
|---|---|
| vit/cirh_811ops | 810 |
| vit/cirh_10288ops | 10262 |
| bert/cirh_1604ops | 1602 |
| bert/cirh_19956ops | 19904 |
| dit/cirh_2800ops | 2800 |
| dit/cirh_27586ops | 27537 |
| esm2/cirh_26910ops | 26841 |
| mixtral/cirh_2420ops | 2329 |
| mixtral/cirh_16401ops | 16169 |
| dino/cirh_32670ops | 32569 |
| llama3_8b/cirh_19401ops | 19400 |

### T1.3 Per-executor attribution

A single compile emits a training graph and an evaluation graph. Both are
parsed and reported separately.

```bash
grep -E "cirh_executors|executor=" /tmp/vo/reports/compile_metrics.csv | head
```

| metric | expected |
|---|---|
| `cirh_executors` | 2 |
| executor 000001 operators | 10262 |
| executor 000002 operators | 810 |

### T1.4 Trace volume scales with instrumentation depth

```bash
for d in ids lite full; do
  ./build/dfabitctl --backend cerebras --model-dir /tmp/v \
    --out /tmp/t_$d --mode full --detail $d >/dev/null 2>&1
  printf "%-6s %s bytes\n" $d "$(stat -c%s /tmp/t_$d/trace/events.jsonl 2>/dev/null || echo 0)"
done
```

Expect a strict increase across `ids`, `lite`, `full`. Volumes are deterministic
for a given graph.

### T1.5 Overhead is reported only from a paired measurement

The overhead figures in Section 4.2 come from paired runs: an uninstrumented arm
and an instrumented arm, executed alternately and compared on medians. This
check confirms the other half of that contract, that no figure appears without
the pair.

Run without `--baseline-run-cmd`:

```bash
ls /tmp/vo/tools/overhead_profiler/ 2>/dev/null
cat /tmp/vo/reports/overhead_not_measured.txt
```

Expect no overhead artifacts and a note naming the flag that supplies the
baseline arm. With `--baseline-run-cmd` present the same tool reports the
measured slowdown; `sweep_detail.sh` in Tier 3 exercises that path.

### T1.6 Semantic attribution (Section 5.1)

```bash
./build/dfabitctl --backend cerebras --model-dir /tmp/v \
  --tool semantic_attribution --out /tmp/sa --mode full >/dev/null 2>&1
head -8 /tmp/sa/tools/semantic_attribution/semantic_attribution_summary.csv
```

| metric | expected (ViT) |
|---|---|
| `operators` | 10262 |
| `operators_with_module` | 10247 |
| `operators_with_source_site` | 998 |
| `source_sites` | 41 |
| `operators_unattributed` | 15 |

The largest source site produces 173 operators carrying 63.6 TFLOP. The
unattributed remainder is compiler-materialized constants.

### T1.7 Dataflow memory proxy (Section 5.2)

Cerebras:

```bash
./build/dfabitctl --backend cerebras --model-dir /tmp/v \
  --tool dataflow_memory_proxy --out /tmp/mp --mode full >/dev/null 2>&1
grep -E "retained_to_backward|dominant_phase" \
  /tmp/mp/tools/dataflow_memory_proxy/dataflow_memory_proxy_summary.csv
```

| metric | expected (ViT) |
|---|---|
| `retained_to_backward_bytes` | 294912501044 |
| `retained_to_backward_tensors` | 442 |
| `dominant_phase` | backward |

Edge TPU, using a shipped model:

```bash
./build/dfabitctl --backend edgetpu \
  --model models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
  --compile-report models/deeplabv3_mnv2_pascal_quant_summary.log \
  --tool dataflow_memory_proxy --out /tmp/mpe --mode full >/dev/null 2>&1
grep -E "^operators,|peak_live_bytes" \
  /tmp/mpe/tools/dataflow_memory_proxy/dataflow_memory_proxy_summary.csv
```

| metric | expected (DeepLab-v3) |
|---|---|
| `operators` | 72 |
| `peak_live_bytes` | 7938240 |

That is 7.57 MiB of activation working set against an 8 MiB budget, while the
compiler reports 2.26 MiB of cached parameters and nothing streaming.

### T1.8 Program analyzer (Section 4.3)

The same tool on both backends, with peak compute and bandwidth supplied by the
adapter rather than the tool.

```bash
./build/dfabitctl --backend cerebras --model-dir /tmp/v \
  --tool program_analyzer --out /tmp/pa --mode full >/dev/null 2>&1
grep -E "arithmetic_intensity|ridge_point|boundness" \
  /tmp/pa/tools/program_analyzer/program_analysis_summary.csv

./build/dfabitctl --backend edgetpu \
  --model models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --compile-report models/mobilenet_v2_1.0_224_quant_summary.log \
  --tool program_analyzer --out /tmp/pae --mode full >/dev/null 2>&1
grep -E "total_macs|arithmetic_intensity|ridge_point|boundness" \
  /tmp/pae/tools/program_analyzer/program_analysis_summary.csv
```

| | Cerebras ViT | Edge TPU MobileNetV2 |
|---|---|---|
| `total_macs` | 1.50169e+14 | 300775552 |
| `arithmetic_intensity` | 42.8467 | 16.9865 |
| `ridge_point` | 3.125 | 250 |
| `boundness` | compute | memory |

The Cerebras value agrees with `cirh_arithmetic_intensity`, computed
independently by the adapter. The Edge TPU MAC count matches the published
MobileNetV2 figure.

### T1.9 Tool selection

```bash
./build/dfabitctl --list-tools
```

Expect five names: `dataflow_memory_proxy`, `overhead_profiler`,
`portability_report`, `program_analyzer`, `semantic_attribution`.

Naming one runs only that tool:

```bash
./build/dfabitctl --backend cerebras --model-dir /tmp/v \
  --tool semantic_attribution --out /tmp/one --mode full >/dev/null 2>&1
ls /tmp/one/tools/
```

Expect a single directory. An unregistered name fails with
`failed to create tool: <name>` and a non-zero exit.

---

## Tier 2: Edge TPU compiler required

### T2.1 Recompiling reproduces the shipped report

```bash
edgetpu_compiler -s -o /tmp/rc models/mobilenet_v2_1.0_224_quant.tflite
diff <(grep "Total number of operations" /tmp/rc/mobilenet_v2_1.0_224_quant_edgetpu.log) \
     <(grep "Total number of operations" models/mobilenet_v2_1.0_224_quant_summary.log)
```

Expect no difference. The shipped summaries are not stale.

### T2.2 DFIT drives the compiler

```bash
mkdir -p /tmp/live && cp models/mobilenet_v2_1.0_224_quant.tflite /tmp/live/
./build/dfabitctl --backend edgetpu \
  --model /tmp/live/mobilenet_v2_1.0_224_quant.tflite --work-dir /tmp/live \
  --compile-cmd 'edgetpu_compiler -s -o /tmp/live /tmp/live/mobilenet_v2_1.0_224_quant.tflite' \
  --tool program_analyzer --out /tmp/liveout --mode full
grep "^total_macs" /tmp/liveout/tools/program_analyzer/program_analysis_summary.csv
```

Expect `300775552`, the same value as T1.8 from the pre-compiled artifacts.

---

## Tier 3: Coral device required

### T3.1 Measured latency

```bash
PYTHONPATH=python python -m dfabit.edgetpu.bench \
  --model models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --iters 200 --warmup 20 --instrument full --runtime-log /tmp/rt.csv
grep "^run,latency_ms" /tmp/rt.csv
```

Expect 3-15 ms on a USB accelerator. Absolute values depend on the host and the
USB link, so the check asserts a plausible range rather than a fixed number.

### T3.2 Instrumentation overhead

```bash
bash sweep_detail.sh
```

Reports slowdown across `ids`, `lite` and `full` against an uninstrumented
baseline. Expect a monotonic increase. The sweep uses warmup passes and repeated
measurement; single-pass timings are not reliable.