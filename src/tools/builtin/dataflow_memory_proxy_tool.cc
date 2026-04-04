#include "dfabit/tools/builtin/dataflow_memory_proxy_tool.h"

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <utility>

#include "dfabit/tools/tool_context.h"

namespace dfabit::tools::builtin {

namespace {

std::size_t CountMetricsForId(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    std::uint64_t stable_id) {
  std::size_t count = 0;
  for (const auto& metric : metrics) {
    if (metric.stable_id == stable_id) {
      ++count;
    }
  }
  return count;
}

double SumMetricByNameForId(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    std::uint64_t stable_id,
    const std::string& name) {
  double total = 0.0;
  for (const auto& metric : metrics) {
    if (metric.stable_id == stable_id && metric.name == name) {
      total += metric.value;
    }
  }
  return total;
}

}  // namespace

std::string DataflowMemoryProxyTool::name() const {
  return "dataflow_memory_proxy";
}

dfabit::core::Status DataflowMemoryProxyTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  filtered_ops_.clear();
  runtime_metrics_.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  const auto out_dir = OutputDir(*ctx);
  std::filesystem::create_directories(out_dir);

  auto st = WritePerOpTable(
      (std::filesystem::path(out_dir) / "dataflow_memory_proxy_table.csv").string(),
      filtered_ops_,
      runtime_metrics_);
  if (!st.ok()) {
    return st;
  }

  st = WriteSummary(
      (std::filesystem::path(out_dir) / "dataflow_memory_proxy_summary.csv").string(),
      filtered_ops_,
      runtime_metrics_);
  if (!st.ok()) {
    return st;
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)compile_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)compile_artifacts;
  filtered_ops_ = ToolServices::FilterOps(*ctx, ctx->metadata_ops());
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  runtime_metrics_ = ToolServices::FilterMetrics(*ctx, runtime_artifacts.metrics);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::WritePerOpTable(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to open dataflow proxy table: " + path};
  }

  ofs << "stable_id,op_name,dialect,stage_tag,input_count,output_count,estimated_bytes,estimated_flops,"
         "runtime_metric_count,observed_bytes,observed_latency_ms,pressure_score\n";

  for (const auto& op : ops) {
    const auto runtime_metric_count = CountMetricsForId(runtime_metrics, op.stable_id);
    const double observed_bytes =
        SumMetricByNameForId(runtime_metrics, op.stable_id, "bytes");
    const double observed_latency_ms =
        SumMetricByNameForId(runtime_metrics, op.stable_id, "latency_ms") +
        SumMetricByNameForId(runtime_metrics, op.stable_id, "launch_duration_ms");

    const double pressure_score =
        static_cast<double>(op.estimated_bytes) +
        observed_bytes +
        0.001 * static_cast<double>(op.estimated_flops);

    ofs << op.stable_id << ","
        << op.op_name << ","
        << op.dialect << ","
        << op.stage_tag << ","
        << op.inputs.size() << ","
        << op.outputs.size() << ","
        << op.estimated_bytes << ","
        << op.estimated_flops << ","
        << runtime_metric_count << ","
        << observed_bytes << ","
        << observed_latency_ms << ","
        << pressure_score << "\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::WriteSummary(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::uint64_t total_estimated_bytes = 0;
  std::uint64_t total_estimated_flops = 0;
  double total_observed_bytes = 0.0;
  double total_observed_latency_ms = 0.0;
  std::size_t attributed_runtime_metric_count = 0;

  std::unordered_map<std::string, std::uint64_t> stage_bytes;
  for (const auto& op : ops) {
    total_estimated_bytes += op.estimated_bytes;
    total_estimated_flops += op.estimated_flops;
    stage_bytes[op.stage_tag] += op.estimated_bytes;
  }

  for (const auto& metric : runtime_metrics) {
    if (metric.stable_id != 0) {
      ++attributed_runtime_metric_count;
    }
    if (metric.name == "bytes") {
      total_observed_bytes += metric.value;
    }
    if (metric.name == "latency_ms" || metric.name == "launch_duration_ms") {
      total_observed_latency_ms += metric.value;
    }
  }

  std::string dominant_stage = "unknown";
  std::uint64_t dominant_stage_bytes = 0;
  for (const auto& kv : stage_bytes) {
    if (kv.second > dominant_stage_bytes) {
      dominant_stage = kv.first;
      dominant_stage_bytes = kv.second;
    }
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to open dataflow proxy summary: " + path};
  }

  ofs << "op_count,total_estimated_bytes,total_estimated_flops,total_observed_bytes,"
         "total_observed_latency_ms,attributed_runtime_metric_count,dominant_stage,dominant_stage_bytes\n";
  ofs << ops.size() << ","
      << total_estimated_bytes << ","
      << total_estimated_flops << ","
      << total_observed_bytes << ","
      << total_observed_latency_ms << ","
      << attributed_runtime_metric_count << ","
      << dominant_stage << ","
      << dominant_stage_bytes << "\n";

  return dfabit::core::Status::Ok();
}

std::string DataflowMemoryProxyTool::OutputDir(const dfabit::api::Context& ctx) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  return (std::filesystem::path(base) / "tools" / "dataflow_memory_proxy").string();
}

std::unique_ptr<Tool> CreateDataflowMemoryProxyTool() {
  return std::make_unique<DataflowMemoryProxyTool>();
}

}  // namespace dfabit::tools::builtin