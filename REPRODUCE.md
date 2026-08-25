# Reproducing DFIT's Reported Numbers

This document maps each quantitative claim to the command that produces it and
the value that command should return. Everything listed under Tier 1 runs on an
ordinary Linux machine with no accelerator attached.

Tolerances are stated per claim. Structural values (operator counts, MAC totals,
mapping ratios) are deterministic and must match exactly. Timing values vary
with the host and are given as ranges.

---

## Requirements

Tier 1 needs only a C++17 compiler and CMake 3.16+. No accelerator, no vendor
SDK, no network access.

```bash
cmake -S . -B build -DDFABIT_BUILD_PYTHON=OFF
cmake --build build -j
./build/smoketest        # expect: all smoke tests passed
```

Tier 2 additionally needs `edgetpu_compiler` (a free download; it runs without
the device) and, for latency only, a Coral USB Accelerator.

Tier 3 needs an ALCF Cerebras allocation and is not expected to be exercised by
reviewers. It is documented for completeness.

---

## Tier 1 — Cerebras compiler-artifact analysis (no hardware)

The repository ships `cirh.mlir` dumps from seven model families compiled on a
Cerebras CS-2 at ALCF, under `dfit_ship/`. These are the compiler's own output,
gzipped; nothing about them is synthesised.

### T1.1 Operator recovery and MAC computation

```bash
mkdir -p /tmp/v/cerebras_logs/x/executors/000001
gunzip -c dfit_ship/vit/cirh_10288ops.mlir.gz \
  > /tmp/v/cerebras_logs/x/executors/000001/cirh.mlir
./build/dfabitctl --backend cerebras --model-dir /tmp/v \
  --out /tmp/vo --mode full --detail full
grep -E "cirh_operators,|cirh_phase_macs" /tmp/vo/reports/compile_metrics.csv
```

Expected, exactly:

| metric | value |
|---|---|
| `cirh_operators` | 10262 |
| `cirh_phase_macs` phase=forward | 5.00569e+13 |
| `cirh_phase_macs` phase=backward | 9.97843e+13 |
| `cirh_phase_macs` phase=optimizer | 0 |

The forward figure is independently checkable: ViT-Base/16 is ~17.6 GMAC per
image and this graph has batch 2850, giving ~5.02e13. The backward/forward ratio
of 1.993 matches the theoretical 2x for backpropagation. Optimizer MACs are zero
because Adam performs elementwise updates and no multiply-accumulates.

### T1.2 All eleven shipped graphs parse

```bash
for m in vit bert dit esm2 mixtral dino llama3_8b; do
  for gz in dfit_ship/$m/cirh_*.mlir.gz; do
    [ -f "$gz" ] || continue
    tag=$(basename "$gz" .mlir.gz)
    d=/tmp/va/${m}_${tag}/cerebras_logs/x/executors/000001
    mkdir -p "$d" && gunzip -c "$gz" > "$d/cirh.mlir"
    ./build/dfabitctl --backend cerebras --model-dir /tmp/va/${m}_${tag} \
      --out /tmp/vao/${m}_${tag} --mode full >/dev/null 2>&1
    echo "$m $tag: $(grep -h '^cirh_operators,' \
      /tmp/vao/${m}_${tag}/reports/compile_metrics.csv | cut -d, -f2)"
  done
done
```

Expected operator counts, exactly:

| model | graph | operators |
|---|---|---|
| vit | cirh_811ops | 810 |
| vit | cirh_10288ops | 10262 |
| bert | cirh_1604ops | 1602 |
| bert | cirh_19956ops | 19904 |
| dit | cirh_2800ops | 2800 |
| dit | cirh_27586ops | 27537 |
| esm2 | cirh_26910ops | 26841 |
| mixtral | cirh_2420ops | 2329 |
| mixtral | cirh_16401ops | 16169 |
| dino | cirh_32670ops | 32569 |
| llama3_8b | cirh_19401ops | 19400 |

### T1.3 Semantic lineage recovery

Every operator carries the module path it came from, the originating ATen
operator, and the Python source line, all read from the compiler's own `loc()`
records rather than reconstructed.

```bash
grep -c "" /tmp/vo/reports/metadata_ops.csv
head -3 /tmp/vo/reports/metadata_ops.csv
```

Expected: 10263 lines (10262 operators plus header). Operator rows carry
`module_path`, `aten_op` and `src` attributes; for the ViT graph over 99% of
operators have a non-empty module path.

### T1.4 Multiple executors per compile

One `cszoo fit --compile_only` invocation emits several compiled graphs. DFIT
attributes metrics per executor rather than picking one.

```bash
mkdir -p /tmp/vm/cerebras_logs/x/executors/00000{1,2}
gunzip -c dfit_ship/vit/cirh_10288ops.mlir.gz \
  > /tmp/vm/cerebras_logs/x/executors/000001/cirh.mlir
gunzip -c dfit_ship/vit/cirh_811ops.mlir.gz \
  > /tmp/vm/cerebras_logs/x/executors/000002/cirh.mlir
./build/dfabitctl --backend cerebras --model-dir /tmp/vm \
  --out /tmp/vmo --mode full --detail full
grep -E "cirh_executors|cirh_operators," /tmp/vmo/reports/compile_metrics.csv
```

Expected: `cirh_executors` = 2, with separate `cirh_operators` rows of 10262
(executor=000001) and 810 (executor=000002).

### T1.5 Instrumentation overhead by depth

```bash
bash sweep_cerebras.sh
```

Runs 11 graphs at four instrumentation depths, one warmup pass and three timed
passes of 20 in-process repetitions each. Takes roughly 40 minutes.

Expected medians across the eleven graphs:

| level | median overhead | observed range |
|---|---|---|
| ids | +3.8% | +0.2% to +5.5% |
| lite | +5.4% | +2.0% to +7.4% |
| full | +12.4% | +5.3% to +13.6% |

The ordering baseline < ids < lite < full holds for every graph. Absolute
per-session times are host-dependent and will differ; the percentages should
reproduce within a few points.

Two properties of this measurement must be carried into any use of the numbers:

1. `--mode baseline` still parses the entire `cirh.mlir`. Parsing happens
   regardless of whether tracing is enabled, so these percentages measure the
   cost of event emission on top of parsing, not instrumentation against an
   uninstrumented run.
2. Timings are taken with `--repeat-session`, which amortizes process launch
   (~4 ms) across repetitions. Timing a single invocation from the shell
   includes that launch cost and will not match.

Trace volume is deterministic and should match exactly:

| graph | ids bytes | lite bytes | full bytes |
|---|---|---|---|
| vit/cirh_10288ops | 1938765 | 2626647 | 5508424 |
| dino/cirh_32670ops | 6139653 | 8330959 | 17974171 |

### T1.6 No metric is reported that was not measured

Running without a paired baseline command produces no overhead figures at all,
and says so.

```bash
./build/dfabitctl --backend edgetpu \
  --model models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --out /tmp/nm --mode full
ls /tmp/nm/tools/overhead_profiler/ 2>/dev/null
cat /tmp/nm/reports/overhead_not_measured.txt
```

Expected: no `overhead_profiler` directory, and a note explaining that no paired
measurement was run. This is a deliberate property: earlier revisions of this
code emitted a fixed 8% slowdown from hardcoded constants when no measurement
had taken place.

---

## Tier 2 — Edge TPU (compiler only, or with a Coral device)

`edgetpu_compiler` runs without the accelerator attached, so the compile-stage
claims reproduce on any machine.

### T2.1 Operator mapping and memory residency

```bash
edgetpu_compiler -s models/mobilenet_v2_1.0_224_quant.tflite \
  > /tmp/mnv2_summary.log 2>&1
./build/dfabitctl --backend edgetpu \
  --model models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --compile-report /tmp/mnv2_summary.log \
  --out /tmp/etpu --mode full
grep -E "tpu_mapping_ratio|on_chip_param_fraction|edgetpu_subgraphs|total_operations" \
  /tmp/etpu/reports/compile_metrics.csv
```

Expected for MobileNetV2: `tpu_mapping_ratio` 1, `on_chip_param_fraction` 1,
`edgetpu_subgraphs` 1, `total_operations` 66.

### T2.2 Partial mapping and off-chip streaming

The interesting cases are models that do not map cleanly:

| model | tpu_mapping_ratio | on_chip_param_fraction | note |
|---|---|---|---|
| ssd_mobilenet_v2_coco_quant_postprocess | 0.973 | 1.000 | 3 operators fall back to CPU |
| deeplabv3_mnv2_pascal_quant | 0.889 | 1.000 | 8 operators fall back |
| inception_v4_299_quant | 1.000 | 0.140 | 36.3 MiB streams off-chip per inference |
| yolov5n_320_quant | 0.917 | 1.000 | 23 operators fall back |

Fallback reasons are preserved verbatim from the compiler in the `status`
attribute of each `operator_count` row.

### T2.3 Latency (requires the device)

```bash
PYTHONPATH=python python -m dfabit.edgetpu.bench \
  --model models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
  --iters 200 --warmup 20 --instrument full \
  --runtime-log /tmp/rt.csv
```

Expected on a USB 3.0 Coral: median latency 4-5 ms for MobileNetV2. USB 2.0 is
roughly three times slower.

Runtime instrumentation overhead on this device is below the measurement noise
floor: across 500-iteration runs at four instrumentation depths, median latency
varied by less than 0.3 ms against a within-run standard deviation of 0.20-0.61
ms, with no consistent ordering. We report an upper bound rather than a point
estimate.

---

## Tier 3 — Live Cerebras instrumentation (requires an ALCF allocation)

Documented for completeness; not expected to be reproduced.

```bash
export MODELDIR="$HOME/R_2.10.0/modelzoo/src/cerebras/modelzoo/models/vision/vision_transformer"
export VENV="$HOME/R_2.10.0/venv_cerebras_pt/bin/activate"
OUT="$HOME/dfit_live_$(date -u +%Y%m%d_%H%M%S)"

./dfabitctl --backend cerebras \
  --model-dir "$OUT/cmp" \
  --work-dir "$MODELDIR" \
  --compile-cmd "source $VENV && cszoo fit configs/params_vit_base_modified.yaml --compile_only --num_csx 1 --disable_version_check --model_dir $OUT/cmp" \
  --out "$OUT/dfit" --mode full --detail full
```

DFIT invokes the compile, waits for it, discovers every `cirh.mlir` the compile
emitted under the model directory, parses them, and attributes metrics per
executor. A representative run recorded `compile_command_elapsed_ms` of 146624
against a cached compile; a cold compile takes substantially longer.

---

## Not reproducible

**SambaNova.** The adapter is implemented against the documented SambaFlow
compile and run interface. It is not evaluated here: the SN30 training cluster
we had access to was decommissioned during this work and replaced with
inference-only SN40L endpoints, which expose no compiler artifacts. No
SambaNova compiler traces are included in this artifact.

The adapter still exercises a property the framework claims. Run without a
device, capability discovery reports what is absent rather than assuming:

```bash
./build/dfabitctl --backend sambanova --graph examples/sambanova/graph.txt \
  --out /tmp/sn --mode full
cat /tmp/sn/tool_portability_capabilities.csv
```

Expected on a machine with no SambaNova stack: `profiler_metrics_available` 0
and `custom_env_controls` 0, with no overhead figures produced.
