#include <filesystem>
#include <iostream>
#include <memory>

#include "dfabit/adapters/backend_registry.h"
#include "dfabit/api/context.h"
#include "dfabit/core/framework_config.h"
#include "dfabit/tools/builtin/portability_report_tool.h"
#include "dfabit/tools/tool_manager.h"
#include "dfabit/tools/tool_registry.h"

int main() {
  const auto adapter_names = dfabit::adapters::BackendRegistry::Instance().List();
  const auto tool_names = dfabit::tools::ToolRegistry::Instance().List();

  std::cerr << "dfabitctl: tool layer build OK\n";
  for (const auto& name : adapter_names) {
    std::cerr << "adapter=" << name << "\n";
  }
  for (const auto& name : tool_names) {
    std::cerr << "tool=" << name << "\n";
  }

  dfabit::core::RunConfig run_cfg;
  run_cfg.run_id = "demo_run";
  run_cfg.output.base_output_dir = "build/tool_reports";
  run_cfg.policy.mode = dfabit::core::InstrumentationMode::kFull;
  run_cfg.policy.detail_level = dfabit::core::DetailLevel::kFull;

  dfabit::api::Context ctx(run_cfg);
  ctx.SetProperty("active_adapter", "gpu_mlir");

  dfabit::tools::ToolManager manager;
  auto tool = dfabit::tools::ToolRegistry::Instance().Create("portability_report");
  if (!tool) {
    std::cerr << "failed to create portability_report tool\n";
    return 1;
  }

  auto st = manager.AddTool(std::move(tool));
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnRegister(&ctx);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnInit(&ctx);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  dfabit::adapters::CompileArtifactSet compile_artifacts;
  dfabit::adapters::RuntimeArtifactSet runtime_artifacts;

  dfabit::adapters::MetricSample metric;
  metric.name = "instrumented_latency_ms";
  metric.value = 12.4;
  metric.unit = "ms";
  metric.stage = "run";
  runtime_artifacts.metrics.push_back(metric);

  st = manager.OnCompileBegin(&ctx, compile_artifacts);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnCompileEnd(&ctx, compile_artifacts);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnLoadBegin(&ctx, runtime_artifacts);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnLoadEnd(&ctx, runtime_artifacts);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnRunBegin(&ctx, runtime_artifacts);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnRunEnd(&ctx, runtime_artifacts);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  st = manager.OnShutdown(&ctx);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  return 0;
}