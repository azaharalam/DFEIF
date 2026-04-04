#include "dfabit/tools/builtin/overhead_profiler_tool.h"

#include <filesystem>
#include <utility>

#include "dfabit/analysis/reporting.h"
#include "dfabit/tools/register_builtin_tools.h"
#include "dfabit/tools/tool_context.h"

namespace dfabit::tools::builtin {

std::string OverheadProfilerTool::name() const {
  return "overhead_profiler";
}

dfabit::core::Status OverheadProfilerTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  overhead_engine_.Reset();
  scalability_runner_.Reset();
  compile_metrics_.clear();
  load_metrics_.clear();
  run_metrics_.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  const auto out_dir = OutputDir(*ctx);
  std::filesystem::create_directories(out_dir);

  if (!overhead_engine_.samples().empty()) {
    dfabit::analysis::Reporting reporting;
    auto st = reporting.WriteOverheadBundle(out_dir, overhead_engine_);
    if (!st.ok()) {
      return st;
    }

    dfabit::analysis::LightweightFitEngine fit_engine;
    dfabit::analysis::LightweightFitResult fit_result;
    st = fit_engine.Fit(overhead_engine_.samples(), &fit_result);
    if (!st.ok()) {
      return st;
    }

    st = reporting.WriteLightweightBundle(out_dir, fit_result);
    if (!st.ok()) {
      return st;
    }

    st = reporting.WriteScalabilityBundle(out_dir, scalability_runner_);
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)compile_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  compile_metrics_ = compile_artifacts.metrics;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  load_metrics_ = runtime_artifacts.metrics;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadProfilerTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  run_metrics_ = runtime_artifacts.metrics;

  const auto& cfg = ctx->run_context().config();
  const std::string backend_name = cfg.backend.backend_name.empty()
                                       ? ctx->GetProperty("active_adapter")
                                       : cfg.backend.backend_name;
  const std::string mode_name = ctx->GetProperty("run_mode").empty()
                                    ? "full"
                                    : ctx->GetProperty("run_mode");
  const std::string workload_name = cfg.model_name.empty()
                                        ? "unnamed_model"
                                        : cfg.model_name;

  overhead_engine_.AddMetricSamples(
      backend_name,
      mode_name,
      workload_name,
      runtime_artifacts.metrics);

  scalability_runner_.AddFromOverheadSamples(
      "event_count",
      overhead_engine_.samples());

  return dfabit::core::Status::Ok();
}

std::string OverheadProfilerTool::OutputDir(const dfabit::api::Context& ctx) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  return (std::filesystem::path(base) / "tools" / "overhead_profiler").string();
}

std::unique_ptr<Tool> CreateOverheadProfilerTool() {
  return std::make_unique<OverheadProfilerTool>();
}

}  // namespace dfabit::tools::builtin