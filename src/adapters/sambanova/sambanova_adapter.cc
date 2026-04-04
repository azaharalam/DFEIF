#include "dfabit/adapters/sambanova/sambanova_adapter.h"

#include <chrono>
#include <filesystem>
#include <utility>
#include <vector>

namespace dfabit::adapters::sambanova {

namespace {

std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::string DetectPath(const dfabit::api::Context& ctx, const std::string& key) {
  const auto from_property = ctx.GetProperty(key);
  if (!from_property.empty()) {
    return from_property;
  }
  return ctx.run_context().GetAttribute(key);
}

}  // namespace

SambaNovaAdapter::SambaNovaAdapter() = default;

std::string SambaNovaAdapter::name() const {
  return "sambanova";
}

std::string SambaNovaAdapter::provider() const {
  return "sambanova_hidden_ir";
}

dfabit::core::Status SambaNovaAdapter::InitializeSession(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  partition_tracker_.Reset();
  ctx->SetProperty("active_adapter", name());
  return dfabit::core::Status::Ok();
}

AdapterCapabilities SambaNovaAdapter::DiscoverCapabilities(const dfabit::api::Context& ctx) const {
  (void)ctx;
  AdapterCapabilities caps;
  caps.visible_mlir = false;
  caps.visible_llvm = false;
  caps.visible_graph_ir = true;
  caps.compile_report_available = true;
  caps.runtime_log_available = true;
  caps.profiler_metrics_available = true;
  caps.op_level_events = false;
  caps.subgraph_level_events = true;
  caps.partition_level_events = true;
  caps.custom_env_controls = true;
  caps.supported_stages = {"compile", "load", "run", "partition"};
  caps.supported_artifact_types = {"graph_ir", "manifest", "compile_report", "runtime_log", "profile"};
  caps.supported_metric_names = {
      "latency_ms",
      "throughput",
      "utilization",
      "partition_ms",
      "compile_command_elapsed_ms",
      "run_command_elapsed_ms"};
  return caps;
}

dfabit::core::Status SambaNovaAdapter::PrepareArtifacts(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !compile_artifacts || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid PrepareArtifacts arguments"};
  }

  compile_artifacts->inputs.clear();
  compile_artifacts->outputs.clear();
  compile_artifacts->metrics.clear();
  runtime_artifacts->inputs.clear();
  runtime_artifacts->outputs.clear();
  runtime_artifacts->metrics.clear();

  const auto graph_path = DetectPath(*ctx, "sambanova_graph_path");
  if (graph_path.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "sambanova_graph_path is not set"};
  }

  ArtifactRef graph;
  graph.kind = ArtifactKind::kGraphIr;
  graph.name = "sambanova_graph";
  graph.path = graph_path;
  graph.stage = "compile";
  compile_artifacts->inputs.push_back(std::move(graph));

  const auto sidecar_path = DetectPath(*ctx, "sambanova_sidecar_path");
  if (!sidecar_path.empty()) {
    ArtifactRef sidecar;
    sidecar.kind = ArtifactKind::kManifest;
    sidecar.name = "sambanova_sidecar";
    sidecar.path = sidecar_path;
    sidecar.stage = "compile";
    compile_artifacts->inputs.push_back(std::move(sidecar));
  }

  const auto compile_report_path = DetectPath(*ctx, "sambanova_compile_report_path");
  if (!compile_report_path.empty()) {
    ArtifactRef report;
    report.kind = ArtifactKind::kCompileReport;
    report.name = "sambanova_compile_report";
    report.path = compile_report_path;
    report.stage = "compile";
    compile_artifacts->outputs.push_back(std::move(report));
  }

  const auto runtime_log_path = DetectPath(*ctx, "sambanova_runtime_log_path");
  if (!runtime_log_path.empty()) {
    ArtifactRef log;
    log.kind = ArtifactKind::kRuntimeLog;
    log.name = "sambanova_runtime_log";
    log.path = runtime_log_path;
    log.stage = "run";
    runtime_artifacts->inputs.push_back(std::move(log));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::LoadManifest(
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

dfabit::core::Status SambaNovaAdapter::CompileBegin(
    dfabit::api::Context* ctx,
    const CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  if (compile_artifacts.inputs.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "compile inputs are empty"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::CompileEnd(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts) {
  if (!ctx || !compile_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid CompileEnd arguments"};
  }
  if (compile_artifacts->inputs.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "compile inputs are empty"};
  }

  auto st = workflow_.MaybeRunCompile(ctx, compile_artifacts);
  if (!st.ok()) {
    return st;
  }

  const auto& graph_artifact = compile_artifacts->inputs.front();
  st = BuildModel(ctx, graph_artifact.path, graph_artifact.stage);
  if (!st.ok()) {
    return st;
  }

  for (const auto& artifact : compile_artifacts->outputs) {
    if (artifact.kind != ArtifactKind::kCompileReport) {
      continue;
    }
    std::vector<dfabit::adapters::shared::CompileReportRecord> records;
    st = compile_report_parser_.ParseFile(artifact.path, &records);
    if (st.ok()) {
      auto metrics = compile_report_parser_.ToMetricSamples(records);
      compile_artifacts->metrics.insert(
          compile_artifacts->metrics.end(),
          std::make_move_iterator(metrics.begin()),
          std::make_move_iterator(metrics.end()));
    }
  }

  ctx->SetMetadataOps(model_.ops);
  ctx->mutable_run_context().SetAttribute("graph_name", model_.graph_name);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::LoadBegin(
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

dfabit::core::Status SambaNovaAdapter::LoadEnd(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid LoadEnd arguments"};
  }

  MetricSample sample;
  sample.name = "graph_nodes";
  sample.value = static_cast<double>(model_.ops.size());
  sample.unit = "count";
  sample.stage = "load";
  runtime_artifacts->metrics.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::RunBegin(
    dfabit::api::Context* ctx,
    const RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::RunEnd(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid RunEnd arguments"};
  }

  auto st = workflow_.MaybeRunExecution(ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = CollectRuntimeMetrics(ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  for (const auto& rec : partition_tracker_.Snapshot()) {
    MetricSample sample;
    sample.name = "partition_ms";
    sample.value = static_cast<double>(rec.end_ts_ns - rec.begin_ts_ns) / 1.0e6;
    sample.unit = "ms";
    sample.stage = rec.stage;
    sample.stable_id = rec.stable_id;
    sample.attributes["partition_name"] = rec.partition_name;
    runtime_artifacts->metrics.push_back(std::move(sample));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::SubgraphBegin(
    dfabit::api::Context* ctx,
    std::string subgraph_name,
    std::uint64_t stable_id) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return partition_tracker_.Begin(subgraph_name, "partition", stable_id, NowNs());
}

dfabit::core::Status SambaNovaAdapter::SubgraphEnd(
    dfabit::api::Context* ctx,
    std::string subgraph_name,
    std::uint64_t stable_id) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)stable_id;
  return partition_tracker_.End(subgraph_name, NowNs());
}

dfabit::core::Status SambaNovaAdapter::CollectRuntimeMetrics(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid CollectRuntimeMetrics arguments"};
  }

  for (const auto& artifact : runtime_artifacts->inputs) {
    if (artifact.kind != ArtifactKind::kRuntimeLog) {
      continue;
    }

    std::vector<dfabit::adapters::shared::RuntimeLogRecord> records;
    auto st = runtime_log_parser_.ParseFile(artifact.path, &records);
    if (!st.ok()) {
      continue;
    }

    auto metrics = runtime_log_parser_.ToMetricSamples(records);
    runtime_artifacts->metrics.insert(
        runtime_artifacts->metrics.end(),
        std::make_move_iterator(metrics.begin()),
        std::make_move_iterator(metrics.end()));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::Shutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  partition_tracker_.Reset();
  runtime_correlator_.Reset();
  model_.ops.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SambaNovaAdapter::BuildModel(
    dfabit::api::Context* ctx,
    const std::string& graph_path,
    const std::string& stage) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  auto st = graph_importer_.ImportFromFile(graph_path, name(), &model_);
  if (!st.ok()) {
    return st;
  }

  const auto sidecar_path = DetectPath(*ctx, "sambanova_sidecar_path");
  if (!sidecar_path.empty()) {
    std::vector<dfabit::hidden_ir::SidecarEntry> entries;
    st = sidecar_loader_.LoadFile(sidecar_path, &entries);
    if (st.ok()) {
      for (const auto& entry : entries) {
        artifact_correlator_.RegisterSymbol(entry.stable_id, entry.symbol);
        artifact_correlator_.RegisterAlias(entry.symbol, entry.stable_id);
        for (auto& op : model_.ops) {
          const auto sym_it = op.attributes.find("symbol");
          if (sym_it != op.attributes.end() && sym_it->second == entry.symbol) {
            if (entry.stable_id != 0) {
              op.stable_id = entry.stable_id;
            }
            op.stage_tag = entry.stage.empty() ? stage : entry.stage;
            for (const auto& kv : entry.attributes) {
              op.attributes[kv.first] = kv.second;
            }
          }
        }
      }
    }
  }

  IndexModel();
  return dfabit::core::Status::Ok();
}

void SambaNovaAdapter::IndexModel() {
  runtime_correlator_.IndexModel(model_);
  for (const auto& op : model_.ops) {
    artifact_correlator_.RegisterSymbol(op.stable_id, op.op_name);
    const auto it = op.attributes.find("symbol");
    if (it != op.attributes.end()) {
      artifact_correlator_.RegisterAlias(it->second, op.stable_id);
    }
  }
}

std::string SambaNovaAdapter::OutputPath(
    const dfabit::api::Context& ctx,
    const std::string& file) const {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  if (base.empty()) {
    return file;
  }
  return (std::filesystem::path(base) / file).string();
}

std::unique_ptr<BackendAdapter> CreateSambaNovaAdapter() {
  return std::make_unique<SambaNovaAdapter>();
}

}  // namespace dfabit::adapters::sambanova