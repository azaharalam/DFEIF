#include <filesystem>
#include <iostream>
#include <string>

#include "dfabit/adapters/backend_registry.h"
#include "dfabit/adapters/register_builtin_adapters.h"
#include "dfabit/cli/runner.h"
#include "dfabit/core/file_utils.h"
#include "dfabit/tools/register_builtin_tools.h"
#include "dfabit/tools/tool_registry.h"

namespace {

bool RequirePath(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    std::cerr << "missing expected path: " << path << "\n";
    return false;
  }
  return true;
}

bool RunGpuSmoke() {
  dfabit::cli::CliOptions options;
  options.backend = "gpu_mlir";
  options.mlir_path = "examples/gpu_mlir/model.mlir";
  options.runtime_log_path = "examples/gpu_mlir/runtime_metrics.csv";
  options.output_dir = "test_out/gpu";
  options.mode = "full";

  const auto st = dfabit::cli::Run(options);
  if (!st.ok()) {
    std::cerr << "gpu smoke failed: " << st.message() << "\n";
    return false;
  }

  return RequirePath("test_out/gpu/reports/runtime_metrics.csv") &&
         RequirePath("test_out/gpu/tools/overhead_profiler/overhead_summary.csv") &&
         RequirePath("test_out/gpu/tools/semantic_attribution/semantic_attribution_summary.csv") &&
         RequirePath("test_out/gpu/tools/dataflow_memory_proxy/dataflow_memory_proxy_summary.csv");
}

bool RunCerebrasSmoke() {
  dfabit::cli::CliOptions options;
  options.backend = "cerebras";
  options.graph_path = "examples/cerebras/graph.txt";
  options.sidecar_path = "examples/cerebras/sidecar.csv";
  options.compile_report_path = "examples/cerebras/compile_report.csv";
  options.runtime_log_path = "examples/cerebras/runtime_log.csv";
  options.output_dir = "test_out/cerebras";
  options.mode = "full";
  options.work_dir = ".";
  options.compile_cmd = "echo cerebras compile";
  options.run_cmd = "echo cerebras run";

  const auto st = dfabit::cli::Run(options);
  if (!st.ok()) {
    std::cerr << "cerebras smoke failed: " << st.message() << "\n";
    return false;
  }

  return RequirePath("test_out/cerebras/platform/cerebras/compile.stdout") &&
         RequirePath("test_out/cerebras/platform/cerebras/run.stdout") &&
         RequirePath("test_out/cerebras/reports/runtime_metrics.csv");
}

bool RunSambaNovaSmoke() {
  dfabit::cli::CliOptions options;
  options.backend = "sambanova";
  options.graph_path = "examples/sambanova/graph.txt";
  options.sidecar_path = "examples/sambanova/sidecar.csv";
  options.compile_report_path = "examples/sambanova/compile_report.csv";
  options.runtime_log_path = "examples/sambanova/runtime_log.csv";
  options.output_dir = "test_out/sambanova";
  options.mode = "full";
  options.work_dir = ".";
  options.compile_cmd = "echo sambanova compile";
  options.run_cmd = "echo sambanova run";

  const auto st = dfabit::cli::Run(options);
  if (!st.ok()) {
    std::cerr << "sambanova smoke failed: " << st.message() << "\n";
    return false;
  }

  return RequirePath("test_out/sambanova/platform/sambanova/compile.stdout") &&
         RequirePath("test_out/sambanova/platform/sambanova/run.stdout") &&
         RequirePath("test_out/sambanova/reports/runtime_metrics.csv");
}

bool RunRegistrySmoke() {
  auto st = dfabit::adapters::RegisterBuiltinAdapters();
  if (!st.ok()) {
    std::cerr << "adapter registration failed\n";
    return false;
  }

  st = dfabit::tools::RegisterBuiltinTools();
  if (!st.ok()) {
    std::cerr << "tool registration failed\n";
    return false;
  }

  if (!dfabit::adapters::BackendRegistry::Instance().HasAdapter("gpu_mlir")) {
    std::cerr << "missing gpu_mlir adapter\n";
    return false;
  }
  if (!dfabit::adapters::BackendRegistry::Instance().HasAdapter("cerebras")) {
    std::cerr << "missing cerebras adapter\n";
    return false;
  }
  if (!dfabit::adapters::BackendRegistry::Instance().HasAdapter("sambanova")) {
    std::cerr << "missing sambanova adapter\n";
    return false;
  }

  if (!dfabit::tools::ToolRegistry::Instance().HasTool("portability_report")) {
    std::cerr << "missing portability_report tool\n";
    return false;
  }
  if (!dfabit::tools::ToolRegistry::Instance().HasTool("overhead_profiler")) {
    std::cerr << "missing overhead_profiler tool\n";
    return false;
  }
  if (!dfabit::tools::ToolRegistry::Instance().HasTool("semantic_attribution")) {
    std::cerr << "missing semantic_attribution tool\n";
    return false;
  }
  if (!dfabit::tools::ToolRegistry::Instance().HasTool("dataflow_memory_proxy")) {
    std::cerr << "missing dataflow_memory_proxy tool\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  std::filesystem::remove_all("test_out");

  if (!RunRegistrySmoke()) {
    return 1;
  }
  if (!RunGpuSmoke()) {
    return 2;
  }
  if (!RunCerebrasSmoke()) {
    return 3;
  }
  if (!RunSambaNovaSmoke()) {
    return 4;
  }

  std::cout << "all smoke tests passed\n";
  return 0;
}