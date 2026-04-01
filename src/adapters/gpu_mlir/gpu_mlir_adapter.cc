#include "dfabit/adapters/gpu_mlir/gpu_mlir_adapter.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#include "dfabit/adapters/backend_registry.h"

namespace dfabit::adapters::gpu_mlir {

namespace {

std::string ReadFileText(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {};
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

std::string DetectMlirPath(const dfabit::api::Context& ctx) {
  const auto explicit_path = ctx.GetProperty("mlir_module_path");
  if (!explicit_path.empty()) {
    return explicit_path;
  }
  const auto from_run = ctx.run_context().GetAttribute("mlir_module_path");
  if (!from_run.empty()) {
    return from_run;
  }
  return {};
}

std::vector<dfabit::runtime::RuntimeRecord> ParseRuntimeMetricsText(const std::string& text) {
  std::vector<dfabit::runtime::RuntimeRecord> out;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ls(line);
    std::string symbol;
    std::string stage;
    std::string metric_name;
    std::string metric_value;
    std::string unit;

    if (!std::getline(ls, symbol, ',')) {
      continue;
    }
    if (!std::getline(ls, stage, ',')) {
      continue;
    }
    if (!std::getline(ls, metric_name, ',')) {
      continue;
    }
    if (!std::getline(ls, metric_value, ',')) {
      continue;
    }
    if (!std::getline(ls, unit, ',')) {
      unit.clear();
    }

    dfabit::runtime::RuntimeRecord rec;
    rec.symbol = symbol;
    rec.stage = stage;
    rec.metric_name = metric_name;
    try {
      rec.metric_value = std::stod(metric_value);
    } catch (...) {
      rec.metric_value = 0.0;
    }
    rec.unit = unit;
    out.push_back(std::move(rec));
  }
  return out;
}

class GpuMlirAdapterRegistrar {
 public:
  GpuMlirAdapterRegistrar() {
    (void)dfabit::adapters::BackendRegistry::Instance().Register(
        "gpu_mlir",
        &CreateGpuMlirAdapter);
  }
};

GpuMlirAdapterRegistrar g_registrar;

}  // namespace

GpuMlirAdapter::GpuMlirAdapter() = default;

std::string GpuMlirAdapter::name() const {
  return "gpu_mlir";
}

std::string GpuMlirAdapter::provider() const {
  return "generic_gpu_mlir";
}

dfabit::core::Status GpuMlirAdapter::InitializeSession(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  ctx->SetProperty("active_adapter", name());
  return dfabit::core::Status::Ok();
}

AdapterCapabilities GpuMlirAdapter::DiscoverCapabilities(const dfabit::api::Context& ctx) const {
  (void)ctx;
  AdapterCapabilities caps;
  caps.visible_mlir = true;
  caps.visible_graph_ir = true;
  caps.compile_report_available = true;
  caps.runtime_log_available = true;
  caps.profiler_metrics_available = true;
  caps.op_level_events = true;
  caps.subgraph_level_events = true;
  caps.partition_level_events = false;
  caps.custom_env_controls = true;
  caps.supported_stages = {"pre_fusion", "post_fusion", "pre_lowering", "post_lowering"};
  caps.supported_artifact_types = {"mlir_module", "manifest", "compile_report", "runtime_log"};
  caps.supported_metric_names = {"latency_ms", "throughput", "occupancy", "bytes"};
  return caps;
}

dfabit::core::Status GpuMlirAdapter::PrepareArtifacts(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !compile_artifacts || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid PrepareArtifacts arguments"};
  }

  const auto mlir_path = DetectMlirPath(*ctx);
  if (mlir_path.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "mlir_module_path is not set"};
  }

  compile_artifacts->inputs.clear();
  compile_artifacts->outputs.clear();
  runtime_artifacts->inputs.clear();
  runtime_artifacts->outputs.clear();
  runtime_artifacts->metrics.clear();

  ArtifactRef mlir_artifact;
  mlir_artifact.kind = ArtifactKind::kMlirModule;
  mlir_artifact.name = "input_mlir";
  mlir_artifact.path = mlir_path;
  mlir_artifact.stage = "pre_lowering";
  compile_artifacts->inputs.push_back(std::move(mlir_artifact));

  ArtifactRef manifest_artifact;
  manifest_artifact.kind = ArtifactKind::kManifest;
  manifest_artifact.name = "semantic_manifest";
  manifest_artifact.path = OutputPath(*ctx, "mlir_manifest.json");
  manifest_artifact.stage = "pre_lowering";
  compile_artifacts->outputs.push_back(manifest_artifact);

  const auto runtime_log = ctx->GetProperty("runtime_metrics_path");
  if (!runtime_log.empty()) {
    ArtifactRef runtime_artifact;
    runtime_artifact.kind = ArtifactKind::kRuntimeLog;
    runtime_artifact.name = "runtime_metrics";
    runtime_artifact.path = runtime_log;
    runtime_artifact.stage = "post_lowering";
    runtime_artifacts->inputs.push_back(std::move(runtime_artifact));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::LoadManifest(
    dfabit::api::Context* ctx,
    const ArtifactRef& manifest_artifact) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  if (manifest_artifact.path.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "manifest path is empty"};
  }

  if (!std::filesystem::exists(manifest_artifact.path)) {
    return {dfabit::core::StatusCode::kNotFound, "manifest not found: " + manifest_artifact.path};
  }

  ctx->SetProperty("manifest_path", manifest_artifact.path);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::CompileBegin(
    dfabit::api::Context* ctx,
    const CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  if (compile_artifacts.inputs.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "compile input artifacts are empty"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::CompileEnd(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts) {
  if (!ctx || !compile_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid CompileEnd arguments"};
  }
  if (compile_artifacts->inputs.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "compile input artifacts are empty"};
  }

  const auto& mlir_artifact = compile_artifacts->inputs.front();
  auto st = BuildModelFromMlir(ctx, mlir_artifact.path, mlir_artifact.stage, &model_);
  if (!st.ok()) {
    return st;
  }

  for (auto& op : model_.ops) {
    cost_model_.Populate(&op);
  }

  const auto manifest_path = OutputPath(*ctx, "mlir_manifest.json");
  st = manifest_exporter_.WriteJson(model_, manifest_path);
  if (!st.ok()) {
    return st;
  }

  bool found_manifest = false;
  for (auto& artifact : compile_artifacts->outputs) {
    if (artifact.kind == ArtifactKind::kManifest) {
      artifact.path = manifest_path;
      artifact.attributes["graph_name"] = model_.graph_name;
      artifact.attributes["op_count"] = std::to_string(model_.ops.size());
      found_manifest = true;
    }
  }
  if (!found_manifest) {
    ArtifactRef manifest_artifact;
    manifest_artifact.kind = ArtifactKind::kManifest;
    manifest_artifact.name = "semantic_manifest";
    manifest_artifact.path = manifest_path;
    manifest_artifact.stage = mlir_artifact.stage;
    manifest_artifact.attributes["graph_name"] = model_.graph_name;
    manifest_artifact.attributes["op_count"] = std::to_string(model_.ops.size());
    compile_artifacts->outputs.push_back(std::move(manifest_artifact));
  }

  IndexModel(model_);
  ctx->SetMetadataOps(model_.ops);
  ctx->mutable_run_context().SetAttribute("graph_name", model_.graph_name);
  ctx->SetProperty("manifest_path", manifest_path);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::LoadBegin(
    dfabit::api::Context* ctx,
    const RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  if (model_.ops.empty()) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "model is not initialized"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::LoadEnd(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid LoadEnd arguments"};
  }

  MetricSample sample;
  sample.name = "op_count";
  sample.value = static_cast<double>(model_.ops.size());
  sample.unit = "count";
  sample.stage = "load";
  runtime_artifacts->metrics.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::RunBegin(
    dfabit::api::Context* ctx,
    const RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::RunEnd(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid RunEnd arguments"};
  }
  return CollectRuntimeMetrics(ctx, runtime_artifacts);
}

dfabit::core::Status GpuMlirAdapter::SubgraphBegin(
    dfabit::api::Context* ctx,
    std::string subgraph_name,
    std::uint64_t stable_id) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  ctx->SetProperty("active_subgraph", std::move(subgraph_name));
  ctx->SetProperty("active_subgraph_id", std::to_string(stable_id));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::SubgraphEnd(
    dfabit::api::Context* ctx,
    std::string subgraph_name,
    std::uint64_t stable_id) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  ctx->SetProperty("last_subgraph", std::move(subgraph_name));
  ctx->SetProperty("last_subgraph_id", std::to_string(stable_id));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::CollectRuntimeMetrics(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid CollectRuntimeMetrics arguments"};
  }

  for (const auto& artifact : runtime_artifacts->inputs) {
    if (artifact.kind != ArtifactKind::kRuntimeLog) {
      continue;
    }

    const auto text = ReadFileText(artifact.path);
    if (text.empty()) {
      continue;
    }

    const auto runtime_records = ParseRuntimeMetricsText(text);
    std::vector<dfabit::runtime::CorrelatedRuntimeRecord> correlated;
    auto st = runtime_correlator_.Correlate(runtime_records, &correlated);
    if (!st.ok()) {
      return st;
    }

    auto metrics = runtime_correlator_.ToMetricSamples(correlated);
    runtime_artifacts->metrics.insert(
        runtime_artifacts->metrics.end(),
        std::make_move_iterator(metrics.begin()),
        std::make_move_iterator(metrics.end()));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::Shutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  model_.ops.clear();
  runtime_correlator_.Reset();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status GpuMlirAdapter::BuildModelFromMlir(
    dfabit::api::Context* ctx,
    const std::string& mlir_path,
    const std::string& stage,
    dfabit::metadata::ModelDesc* model) const {
  if (!ctx || !model) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid BuildModelFromMlir arguments"};
  }

  dfabit::mlir::MlirModuleSnapshot snapshot;
  auto st = loader_.LoadFromFile(mlir_path, stage, &snapshot);
  if (!st.ok()) {
    return st;
  }

  st = tagger_.BuildModelDescription(snapshot, name(), model);
  if (!st.ok()) {
    return st;
  }

  st = tagger_.TagStableIds(model);
  if (!st.ok()) {
    return st;
  }

  return dfabit::core::Status::Ok();
}

void GpuMlirAdapter::IndexModel(const dfabit::metadata::ModelDesc& model) {
  runtime_correlator_.IndexModel(model);
  for (const auto& op : model.ops) {
    artifact_correlator_.RegisterSymbol(op.stable_id, op.op_name);
    const auto symbol_it = op.attributes.find("symbol");
    if (symbol_it != op.attributes.end()) {
      artifact_correlator_.RegisterAlias(symbol_it->second, op.stable_id);
    }
  }
}

std::string GpuMlirAdapter::OutputPath(
    const dfabit::api::Context& ctx,
    const std::string& filename) const {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  if (base.empty()) {
    return filename;
  }
  return (std::filesystem::path(base) / filename).string();
}

std::unique_ptr<BackendAdapter> CreateGpuMlirAdapter() {
  return std::make_unique<GpuMlirAdapter>();
}

}  // namespace dfabit::adapters::gpu_mlir