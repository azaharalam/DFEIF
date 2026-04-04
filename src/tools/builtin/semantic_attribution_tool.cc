#include "dfabit/tools/builtin/semantic_attribution_tool.h"

#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <utility>

#include "dfabit/tools/tool_context.h"

namespace dfabit::tools::builtin {

std::string SemanticAttributionTool::name() const {
  return "semantic_attribution";
}

dfabit::core::Status SemanticAttributionTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  filtered_ops_.clear();
  runtime_metrics_.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  const auto out_dir = OutputDir(*ctx);
  std::filesystem::create_directories(out_dir);

  auto st = WriteAttributionTable(
      (std::filesystem::path(out_dir) / "semantic_attribution_table.csv").string(),
      filtered_ops_,
      runtime_metrics_);
  if (!st.ok()) {
    return st;
  }

  st = WriteSummary(
      (std::filesystem::path(out_dir) / "semantic_attribution_summary.csv").string(),
      filtered_ops_,
      runtime_metrics_);
  if (!st.ok()) {
    return st;
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)compile_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)compile_artifacts;
  filtered_ops_ = ToolServices::FilterOps(*ctx, ctx->metadata_ops());
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  runtime_metrics_ = ToolServices::FilterMetrics(*ctx, runtime_artifacts.metrics);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::WriteAttributionTable(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::unordered_set<std::uint64_t> observed_ids;
  for (const auto& metric : runtime_metrics) {
    if (metric.stable_id != 0) {
      observed_ids.insert(metric.stable_id);
    }
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to open semantic attribution table: " + path};
  }

  ofs << "stable_id,op_name,dialect,stage_tag,estimated_flops,estimated_bytes,observed_runtime_metrics,attributed\n";
  for (const auto& op : ops) {
    std::size_t metric_count = 0;
    for (const auto& metric : runtime_metrics) {
      if (metric.stable_id == op.stable_id) {
        ++metric_count;
      }
    }

    ofs << op.stable_id << ","
        << op.op_name << ","
        << op.dialect << ","
        << op.stage_tag << ","
        << op.estimated_flops << ","
        << op.estimated_bytes << ","
        << metric_count << ","
        << (observed_ids.find(op.stable_id) != observed_ids.end() ? 1 : 0) << "\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::WriteSummary(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::unordered_set<std::uint64_t> observed_ids;
  for (const auto& metric : runtime_metrics) {
    if (metric.stable_id != 0) {
      observed_ids.insert(metric.stable_id);
    }
  }

  std::size_t attributed_count = 0;
  for (const auto& op : ops) {
    if (observed_ids.find(op.stable_id) != observed_ids.end()) {
      ++attributed_count;
    }
  }

  const double coverage =
      ops.empty() ? 0.0 : (100.0 * static_cast<double>(attributed_count) / static_cast<double>(ops.size()));

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to open semantic attribution summary: " + path};
  }

  ofs << "op_count,attributed_op_count,attribution_coverage_pct,runtime_metric_count\n";
  ofs << ops.size() << ","
      << attributed_count << ","
      << coverage << ","
      << runtime_metrics.size() << "\n";

  return dfabit::core::Status::Ok();
}

std::string SemanticAttributionTool::OutputDir(const dfabit::api::Context& ctx) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  return (std::filesystem::path(base) / "tools" / "semantic_attribution").string();
}

std::unique_ptr<Tool> CreateSemanticAttributionTool() {
  return std::make_unique<SemanticAttributionTool>();
}

}  // namespace dfabit::tools::builtin