#include "dfabit/adapters/cerebras/cerebras_workflow.h"

#include <filesystem>
#include <utility>

namespace dfabit::adapters::cerebras {

std::string CerebrasWorkflow::OutputDir(const dfabit::api::Context& ctx) {
  return (std::filesystem::path(ctx.run_context().config().output.base_output_dir) /
          "platform" / "cerebras")
      .string();
}

std::string CerebrasWorkflow::DetectPath(
    const dfabit::api::Context& ctx,
    const std::string& key) {
  const auto p = ctx.GetProperty(key);
  if (!p.empty()) {
    return p;
  }
  return ctx.run_context().GetAttribute(key);
}

void CerebrasWorkflow::AppendProcessMetrics(
    const std::string& stage,
    const dfabit::platform::ProcessResult& result,
    std::vector<dfabit::adapters::MetricSample>* metrics) {
  dfabit::adapters::MetricSample exit_metric;
  exit_metric.name = stage + "_command_exit_code";
  exit_metric.value = static_cast<double>(result.exit_code);
  exit_metric.unit = "code";
  exit_metric.stage = stage;
  metrics->push_back(std::move(exit_metric));

  dfabit::adapters::MetricSample elapsed_metric;
  elapsed_metric.name = stage + "_command_elapsed_ms";
  elapsed_metric.value = result.elapsed_ms;
  elapsed_metric.unit = "ms";
  elapsed_metric.stage = stage;
  metrics->push_back(std::move(elapsed_metric));
}

dfabit::core::Status CerebrasWorkflow::MaybeRunCompile(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts) {
  if (!ctx || !compile_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid compile workflow arguments"};
  }

  const auto cmd = DetectPath(*ctx, "cerebras_compile_cmd");
  if (cmd.empty()) {
    return dfabit::core::Status::Ok();
  }

  const auto work_dir = DetectPath(*ctx, "cerebras_work_dir");
  const auto out_dir = OutputDir(*ctx);
  std::filesystem::create_directories(out_dir);

  dfabit::platform::ProcessSpec spec;
  spec.name = "cerebras_compile";
  spec.command = cmd;
  spec.working_directory = work_dir.empty() ? "." : work_dir;
  spec.stdout_path = (std::filesystem::path(out_dir) / "compile.stdout").string();
  spec.stderr_path = (std::filesystem::path(out_dir) / "compile.stderr").string();

  dfabit::platform::ProcessResult result;
  const auto st = runner_.Run(spec, &result);
  AppendProcessMetrics("compile", result, &compile_artifacts->metrics);
  return st;
}

dfabit::core::Status CerebrasWorkflow::MaybeRunExecution(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid run workflow arguments"};
  }

  const auto cmd = DetectPath(*ctx, "cerebras_run_cmd");
  if (cmd.empty()) {
    return dfabit::core::Status::Ok();
  }

  const auto work_dir = DetectPath(*ctx, "cerebras_work_dir");
  const auto out_dir = OutputDir(*ctx);
  std::filesystem::create_directories(out_dir);

  dfabit::platform::ProcessSpec spec;
  spec.name = "cerebras_run";
  spec.command = cmd;
  spec.working_directory = work_dir.empty() ? "." : work_dir;
  spec.stdout_path = (std::filesystem::path(out_dir) / "run.stdout").string();
  spec.stderr_path = (std::filesystem::path(out_dir) / "run.stderr").string();

  dfabit::platform::ProcessResult result;
  const auto st = runner_.Run(spec, &result);
  AppendProcessMetrics("run", result, &runtime_artifacts->metrics);
  return st;
}

}  // namespace dfabit::adapters::cerebras