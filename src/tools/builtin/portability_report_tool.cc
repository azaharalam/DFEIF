#include "dfabit/tools/builtin/portability_report_tool.h"

#include <filesystem>
#include <fstream>
#include <utility>

#include "dfabit/adapters/backend_registry.h"
#include "dfabit/tools/tool_context.h"
#include "dfabit/tools/tool_registry.h"

namespace dfabit::tools::builtin {

namespace {

class PortabilityReportToolRegistrar {
 public:
  PortabilityReportToolRegistrar() {
    (void)dfabit::tools::ToolRegistry::Instance().Register(
        "portability_report",
        &CreatePortabilityReportTool);
  }
};

PortabilityReportToolRegistrar g_registrar;

}  // namespace

std::string PortabilityReportTool::name() const {
  return "portability_report";
}

dfabit::core::Status PortabilityReportTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return WriteCapabilities(ctx);
}

dfabit::core::Status PortabilityReportTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  collected_metrics_.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  const auto filtered_ops = ToolServices::FilterOps(*ctx, ctx->metadata_ops());
  auto st = ToolServices::WriteOpTableCsv(OutputPath(*ctx, "tool_portability_ops.csv"), filtered_ops);
  if (!st.ok()) {
    return st;
  }

  const auto filtered_metrics = ToolServices::FilterMetrics(*ctx, collected_metrics_);
  return ToolServices::WriteMetricsCsv(
      OutputPath(*ctx, "tool_portability_metrics.csv"),
      filtered_metrics);
}

dfabit::core::Status PortabilityReportTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  dfabit::adapters::MetricSample sample;
  sample.name = "compile_input_artifacts";
  sample.value = static_cast<double>(compile_artifacts.inputs.size());
  sample.unit = "count";
  sample.stage = "compile_begin";
  collected_metrics_.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  for (const auto& metric : compile_artifacts.metrics) {
    collected_metrics_.push_back(metric);
  }

  dfabit::adapters::MetricSample sample;
  sample.name = "compile_output_artifacts";
  sample.value = static_cast<double>(compile_artifacts.outputs.size());
  sample.unit = "count";
  sample.stage = "compile_end";
  collected_metrics_.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  dfabit::adapters::MetricSample sample;
  sample.name = "runtime_input_artifacts";
  sample.value = static_cast<double>(runtime_artifacts.inputs.size());
  sample.unit = "count";
  sample.stage = "load_begin";
  collected_metrics_.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  for (const auto& metric : runtime_artifacts.metrics) {
    collected_metrics_.push_back(metric);
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  for (const auto& metric : runtime_artifacts.metrics) {
    collected_metrics_.push_back(metric);
  }

  dfabit::adapters::MetricSample sample;
  sample.name = "observed_runtime_metrics";
  sample.value = static_cast<double>(runtime_artifacts.metrics.size());
  sample.unit = "count";
  sample.stage = "run_end";
  collected_metrics_.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PortabilityReportTool::WriteCapabilities(dfabit::api::Context* ctx) const {
  const auto adapter_name = ctx->GetProperty("active_adapter");
  if (adapter_name.empty()) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "active_adapter is not set"};
  }

  auto adapter = dfabit::adapters::BackendRegistry::Instance().Create(adapter_name);
  if (!adapter) {
    return {dfabit::core::StatusCode::kNotFound, "adapter not found: " + adapter_name};
  }

  const auto caps = adapter->DiscoverCapabilities(*ctx);
  std::ofstream ofs(OutputPath(*ctx, "tool_portability_capabilities.csv"));
  if (!ofs.is_open()) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to open portability capabilities csv"};
  }

  ofs << "key,value\n";
  ofs << "adapter_name," << adapter->name() << "\n";
  ofs << "provider," << adapter->provider() << "\n";
  ofs << "visible_mlir," << caps.visible_mlir << "\n";
  ofs << "visible_llvm," << caps.visible_llvm << "\n";
  ofs << "visible_graph_ir," << caps.visible_graph_ir << "\n";
  ofs << "compile_report_available," << caps.compile_report_available << "\n";
  ofs << "runtime_log_available," << caps.runtime_log_available << "\n";
  ofs << "profiler_metrics_available," << caps.profiler_metrics_available << "\n";
  ofs << "op_level_events," << caps.op_level_events << "\n";
  ofs << "subgraph_level_events," << caps.subgraph_level_events << "\n";
  ofs << "partition_level_events," << caps.partition_level_events << "\n";
  ofs << "custom_env_controls," << caps.custom_env_controls << "\n";

  return dfabit::core::Status::Ok();
}

std::string PortabilityReportTool::OutputPath(
    const dfabit::api::Context& ctx,
    const std::string& filename) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  if (base.empty()) {
    return filename;
  }
  std::filesystem::create_directories(base);
  return (std::filesystem::path(base) / filename).string();
}

std::unique_ptr<Tool> CreatePortabilityReportTool() {
  return std::make_unique<PortabilityReportTool>();
}

}  // namespace dfabit::tools::builtin