# DFIT - A Framework for Building Instrumentation Tools Across Heterogeneous Dataflow Accelerators

DFIT (DataFlow Accelerator Instrumentation Tool) is an instrumentation framework
for dataflow architectures. Unlike CPUs and GPUs, where instrumentation
frameworks enable fine-grained introspection, dataflow hardware exposes little
information. DFIT instruments operations, regions, and compiler transformations
using semantic program representation, and reconstructs execution traces through
metadata and artifact correlation. This preserves semantic continuity across
compilation and execution boundaries, so the same tool logic runs unmodified on
Cerebras, Edge TPU, and SambaNova. Collected data is exported as normalized
tables for analysis.

## 1. Install

C++17 and CMake 3.16+. No other dependencies.

```bash
cmake -S . -B build -DDFABIT_BUILD_PYTHON=OFF
cmake --build build -j
./build/smoketest
```

For Edge TPU latency runs only, PyCoral needs NumPy 1.x:

```bash
conda create -n coral python=3.9 -y && conda activate coral
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
pip install "numpy<2"
```

## 2. Run on hardware

Cerebras.

```bash
./build/dfabitctl --backend cerebras \
  --model-dir  $HOME/run/model_dir \
  --compile-cmd 'cszoo fit configs/params_vit_base.yaml --model_dir $HOME/run/model_dir' \
  --tool semantic_attribution \
  --out /tmp/out --mode full --detail full
```

Edge TPU.

```bash
./build/dfabitctl --backend edgetpu \
  --model /path/model.tflite --work-dir /path \
  --compile-cmd 'edgetpu_compiler -s -o /path /path/model.tflite' \
  --run-cmd     'python -m dfabit.edgetpu.bench --model /path/model_edgetpu.tflite --iters 200 --warmup 20' \
  --tool dataflow_memory_proxy \
  --out /tmp/out --mode full --detail full
```

DFIT invokes the compiler and the runtime, and collects at both stages.

| flag | purpose |
|---|---|
| `--backend <name>` | which adapter to use |
| `--model <file>` | model to instrument, before compilation |
| `--model-dir <dir>` | Cerebras: directory the compile writes into |
| `--work-dir <dir>` | where `--compile-cmd` and `--run-cmd` execute |
| `--compile-cmd <cmd>` | command DFIT runs for the compile stage |
| `--run-cmd <cmd>` | command DFIT runs for the execution stage |
| `--baseline-run-cmd <cmd>` | uninstrumented arm; without it no overhead is reported |
| `--tool <name>` | run a named tool, repeatable; defaults to all |
| `--list-tools` | print registered tool names and exit |
| `--out <dir>` | output directory, required |
| `--mode baseline\|sampled\|selective\|full` | tracing policy |
| `--detail ids\|lite\|full` | how much each event carries, default full |

## 3. Run without hardware

Compiler artifacts for both backends are shipped, so every structural claim in
the paper checks without an accelerator.

```bash
bash verify.sh
```

91 checks covering operator recovery, MAC computation, per-executor
attribution, trace volume, and capability reporting. Checks that need the
Edge TPU compiler or a Coral device report SKIP.

```bash
bash sweep_tools.sh
```

Runs the tools across all shipped artifacts, 11 Cerebras graphs and 15 Edge TPU
models, and writes per-backend summaries into `results/tools/`. These are the
numbers behind the analysis and use-case sections.

`REPRODUCE.md` maps each claim to its command and expected value.

## 4. Backends

| backend | reads | hardware to reproduce |
|---|---|---|
| `cerebras` | `cirh.mlir` compiler IR | none, dumps in `dfit_ship/` |
| `edgetpu` | compiler report and `.tflite` | none for compile stage, Coral USB for latency |
| `sambanova` | SambaFlow artifacts | SN30, no longer available |
| `gpu_mlir` | MLIR files | none |

## 5. Invocation forms

```
dfabitctl --backend cerebras  --model-dir <dir>       discover cirh.mlir from a compile
dfabitctl --backend cerebras  --graph <file> [--sidecar <file>]
dfabitctl --backend edgetpu   --model <file.tflite> [--compile-report <log>]
dfabitctl --backend gpu_mlir  --mlir <file>
dfabitctl --backend sambanova --graph <file> | --app-dir <dir>
```

`--mode` and `--detail` are independent. The first selects the tracing policy
and which events pass the filters, the second how much each event carries.
`--mode baseline` disables tracing. `--no-<name>-tool` drops one tool from the
default set. Full flag list in Section 2.

## 6. Adding a tool

A tool subclasses `Tool`, implements the lifecycle hooks it needs, and registers
a factory. Nothing else in the framework changes.

```cpp
// src/tools/builtin/my_tool.cc
class MyTool final : public Tool {
 public:
  std::string name() const override { return "my_tool"; }

  core::Status OnCompileEnd(api::Context* ctx,
                            const adapters::CompileArtifactSet& ca) override {
    ops_ = ToolServices::FilterOps(*ctx, ctx->metadata_ops());
    return core::Status::Ok();
  }

  core::Status OnShutdown(api::Context* ctx) override {
    // write whatever the analysis produced
    return core::Status::Ok();
  }

  // remaining hooks return core::Status::Ok()

 private:
  std::vector<metadata::OpDesc> ops_;
};

std::unique_ptr<Tool> CreateMyTool() { return std::make_unique<MyTool>(); }
```

Register the factory in `src/tools/register_builtin_tools.cc`:

```cpp
ToolRegistry::Instance().Register("my_tool", &CreateMyTool);
```

Add the source to the `dfabit_core` target in `CMakeLists.txt`, beside the other
entries under `src/tools/builtin/`:

```cmake
add_library(dfabit_core
  ...
  src/tools/builtin/semantic_attribution_tool.cc
  src/tools/builtin/my_tool.cc          # <- add here
  ...
)
```

Rebuild and run it by name:

```bash
cmake --build build -j
./build/dfabitctl --list-tools
./build/dfabitctl --backend cerebras --model-dir /tmp/v --tool my_tool --out /tmp/out
```

The tool receives operator descriptors and metric samples the adapter has
already normalized, so the same source runs on every backend. What differs is
how much each descriptor carries, which the tool reads from the capability
descriptor.

## 7. Adding a backend

A backend adapter implements `BackendAdapter`, declares what the toolchain
exposes, and parses the format it emits.

```cpp
// src/adapters/mybackend/mybackend_adapter.cc
AdapterCapabilities MyAdapter::DiscoverCapabilities(const api::Context& ctx) const {
  AdapterCapabilities caps;
  caps.visible_graph_ir = FileExists(ctx.GetProperty("mybackend_graph"));
  caps.op_level_events  = true;
  return caps;
}

core::Status MyAdapter::CompileEnd(api::Context* ctx,
                                   CompileArtifactSet* artifacts) {
  // parse the compiler's output into metadata::OpDesc entries
  ctx->SetMetadataOps(model_.ops);
  return core::Status::Ok();
}
```

Register it in `src/adapters/register_builtin_adapters.cc` and add the sources
to `CMakeLists.txt` alongside the other adapters:

```cpp
BackendRegistry::Instance().Register("mybackend", &CreateMyAdapter);
```

```cmake
add_library(dfabit_core
  ...
  src/adapters/mybackend/mybackend_adapter.cc
  src/adapters/mybackend/mybackend_parser.cc
  ...
)
```

The lifecycle structure carries over from an existing adapter. The work is the
reader for the backend's own format, and two are included as references:
`src/adapters/cerebras/cirh_parser.cc` reads MLIR, and
`src/adapters/edgetpu/edgetpu_flatbuffer.cc` reads a TFLite model. Every
registered tool then runs against the new backend without modification.

## 8. Layout

```
include/dfabit/         public headers
src/adapters/           one directory per backend
src/analysis/           overhead, scalability, lightweight-fit engines
src/tools/builtin/      instrumentation tools built on the framework
python/dfabit/          Edge TPU benchmark harness, Cerebras MLIR parser
dfit_ship/              Cerebras cirh.mlir dumps, 7 model families
dfit_ship_tpu/          Edge TPU compiler summaries, 16 models
examples/               fixtures for the smoke tests
sweep_tools.sh          both analysis tools across both backends
sweep_cerebras.sh       instrumentation-depth sweep over the shipped dumps
sweep_edgetpu.sh        the same for Edge TPU compile-stage timings
parse_cirh.py           standalone Cerebras MLIR parser
extract_macs.py         per-operator MACs and bytes from TFLite models
```

## 9. SambaNova

The SambaNova results in the paper were measured on an SN30 cluster while the
work was in progress. Those runs produced the overhead figures across the four
instrumentation modes, the workload, trace-density and fragmentation
scalability data, the lightweight-fit slopes, and the roofline and boundness
analyses, over seven benchmarks: BERT, GPT, DLRM, LayerNorm, NLPStack, RecSys
and Transformer.

That cluster was decommissioned and replaced with inference-only SN40L
endpoints, which expose no compiler artifacts. We can no longer regenerate the
compile-stage traces the checks would read, so SambaNova is not included in
`REPRODUCE.md` and `verify.sh`.