# DFIT — A Framework for Building Instrumentation Tools Across Heterogeneous Dataflow Accelerators

DFIT builds instrumentation tools that work across accelerators whose compilers
and runtimes expose very different amounts of information. A tool written once
against DFIT's normalized event schema runs on any backend the framework
supports; each backend adapter reports what it can actually observe, and the
framework produces no metric that was not measured.

## Building

C++17 and CMake 3.16 or newer. No other dependencies.

```bash
cmake -S . -B build -DDFABIT_BUILD_PYTHON=OFF
cmake --build build -j
./build/smoketest
```

`all smoke tests passed` means the build is good.

## Backends

| backend | what it reads | hardware needed to reproduce |
|---|---|---|
| `cerebras` | `cirh.mlir` compiler IR | none — dumps are shipped in `dfit_ship/` |
| `edgetpu` | `edgetpu_compiler -s` report | none for compile stage; Coral USB stick for latency |
| `sambanova` | SambaFlow compile and run artifacts | SN30, which is no longer available (see below) |
| `gpu_mlir` | MLIR files | none |

## Quick start

Analyze a shipped Cerebras compiler dump:

```bash
mkdir -p /tmp/v/cerebras_logs/x/executors/000001
gunzip -c dfit_ship/vit/cirh_10288ops.mlir.gz \
  > /tmp/v/cerebras_logs/x/executors/000001/cirh.mlir

./build/dfabitctl --backend cerebras --model-dir /tmp/v \
  --out /tmp/vo --mode full --detail full

grep cirh_ /tmp/vo/reports/compile_metrics.csv
```

That recovers 10,262 operators with their shapes, dtypes, MAC counts, execution
phase, originating PyTorch module path, ATen operator and source line — all read
from the compiler's own output.

## Command line

```
dfabitctl --backend cerebras  --model-dir <dir>            # discover cirh.mlir emitted by a compile
dfabitctl --backend cerebras  --graph <file> [--sidecar <file>]
dfabitctl --backend edgetpu   --model <file.tflite> [--compile-report <log>]
dfabitctl --backend gpu_mlir  --mlir <file>
dfabitctl --backend sambanova --graph <file> | --app-dir <dir>

  --out <dir>                 output directory (required)
  --mode baseline|sampled|selective|full     tracing policy
  --detail ids|lite|full      instrumentation depth (default full)
  --repeat-session <n>        run the session n times in one process
  --compile-cmd <cmd>         command to invoke for the compile stage
  --run-cmd <cmd>             command to invoke for the run stage
  --baseline-run-cmd <cmd>    uninstrumented arm for overhead measurement
```

### Two independent axes

`--mode` selects the tracing policy: whether tracing is on, and which events
pass the filters. `--detail` selects how much each event carries. They are
independent, and `--mode baseline` disables tracing entirely.

### Measuring overhead

DFIT will not report a slowdown it did not measure. Overhead metrics appear only
when `--baseline-run-cmd` supplies an uninstrumented arm to compare against;
without one, the run writes `reports/overhead_not_measured.txt` and emits no
latency or throughput figures.

```bash
./build/dfabitctl --backend edgetpu --model model_edgetpu.tflite \
  --work-dir . \
  --baseline-run-cmd 'python bench.py --instrument none' \
  --run-cmd          'python bench.py --instrument full' \
  --iters 10 --warmup 2 --out results/
```

The two arms run alternately so drift over the measurement window affects both
equally, failed iterations are excluded and reported, and the comparison uses
medians.

## Layout

```
include/dfabit/         public headers
src/adapters/           one directory per backend
src/analysis/           overhead, scalability, lightweight-fit engines
src/tools/builtin/      instrumentation tools built on the framework
python/dfabit/          Edge TPU benchmark harness, Cerebras MLIR parser
dfit_ship/              Cerebras cirh.mlir dumps, 7 model families
examples/               small fixtures used by the smoke tests
sweep_cerebras.sh       instrumentation-depth sweep over the shipped dumps
sweep_edgetpu.sh        the same for Edge TPU compile-stage timings
parse_cirh.py           standalone Cerebras MLIR parser (mirrors the C++ one)
extract_macs.py         per-operator MACs and bytes from TFLite models
```

## Reproducing the reported numbers

`REPRODUCE.md` maps each claim to a command and the value it should return, and
`verify.sh` runs the hardware-free subset and reports pass or fail.

```bash
bash verify.sh
```

## Python environment

The Edge TPU harness needs `numpy<2`. PyCoral is compiled against the NumPy 1.x
ABI and fails to import under NumPy 2 with `_ARRAY_API not found`. Installing
other packages into the same environment can silently upgrade NumPy; pin it.

```bash
conda create -n coral python=3.9 -y && conda activate coral
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
pip install "numpy<2"
```

The Cerebras MLIR parser needs only the standard library.

## SambaNova

The SambaNova adapter is implemented against the documented SambaFlow compile
and run interface but is not evaluated. The SN30 training cluster we had access
to was decommissioned during this work and replaced with inference-only SN40L
endpoints, which expose no compiler artifacts, so no SambaNova compiler traces
are included here.

The adapter is retained because its behaviour without hardware exercises a
design property the framework claims: capability discovery reports what a
backend actually exposes rather than assuming. Run on a machine with no
SambaNova stack, it reports `profiler_metrics_available` and
`custom_env_controls` as false and produces no overhead figures.

## A note on measurement

Several results in this work came out differently once they were measured
properly rather than measured once. Process launch cost (~4 ms) exceeded the
effect being measured until timings moved in-process. Page-cache effects made
the first pass of a sweep run 20-30% high until warmup passes were added. Both
produced plausible-looking numbers that did not survive replication. The sweep
scripts include warmup and repeated passes for this reason, and single-pass
timings from them should not be trusted.
