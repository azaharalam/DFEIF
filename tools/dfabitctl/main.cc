#include <iostream>
#include <string>
#include <vector>

#include "dfabit/cli/experiment_config.h"
#include "dfabit/cli/experiment_runner.h"
#include "dfabit/cli/runner.h"
#include "dfabit/tools/register_builtin_tools.h"
#include "dfabit/tools/tool_registry.h"

namespace {

void PrintUsage() {
  std::cerr
      << "single run:\n"
      << "  dfabitctl --backend gpu_mlir --mlir <file> --out <dir> [--mode full|baseline|selective|sampled]\n"
      << "  dfabitctl --backend cerebras --graph <file> [--sidecar <file>] [--compile-report <file>] "
         "[--runtime-log <file>] [--work-dir <dir>] [--compile-cmd <cmd>] [--run-cmd <cmd>] "
         "--out <dir> [--mode full|baseline|selective|sampled]\n"
      << "  dfabitctl --backend sambanova --graph <file> [--sidecar <file>] [--compile-report <file>] "
         "[--runtime-log <file>] [--work-dir <dir>] [--compile-cmd <cmd>] [--run-cmd <cmd>] "
         "--out <dir> [--mode full|baseline|selective|sampled]\n"
      << "  dfabitctl --backend edgetpu --model <file.tflite> [--compile-report <log>] "
         "[--runtime-log <csv>] --out <dir> [--mode full|baseline|selective|sampled]\n"
      << "\n"
      << "  --model-dir <dir>        cerebras: discover cirh.mlir from the compile\n"
      << "  --detail ids|lite|full   instrumentation depth (default full)\n"
      << "\n"
      << "tools:\n"
      << "  --tool <name>            run a named tool; repeatable. Naming none\n"
      << "                           runs every registered tool.\n"
      << "  --list-tools             print registered tool names and exit\n"
      << "  --no-<name>-tool         drop one tool from the default set\n"
      << "\n"
      << "batch run:\n"
      << "  dfabitctl --config <experiment.cfg>\n";
}

bool RequireValue(int argc, char** argv, int i) {
  return i + 1 < argc && argv[i + 1] != nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    PrintUsage();
    return 1;
  }

  dfabit::cli::CliOptions options;
  std::string config_path;

  // Number of times to execute the session inside this process. Process launch
  // costs ~4 ms, which swamps the instrumentation being measured; repeating
  // in-process amortizes that away so the timing reflects the framework.
  int repeat_session = 1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--backend") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --backend\n";
        return 1;
      }
      options.backend = argv[++i];
    } else if (arg == "--out") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --out\n";
        return 1;
      }
      options.output_dir = argv[++i];
    } else if (arg == "--mlir") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --mlir\n";
        return 1;
      }
      options.mlir_path = argv[++i];
    } else if (arg == "--graph") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --graph\n";
        return 1;
      }
      options.graph_path = argv[++i];
    } else if (arg == "--sidecar") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --sidecar\n";
        return 1;
      }
      options.sidecar_path = argv[++i];
    } else if (arg == "--compile-report") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --compile-report\n";
        return 1;
      }
      options.compile_report_path = argv[++i];
    } else if (arg == "--runtime-log") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --runtime-log\n";
        return 1;
      }
      options.runtime_log_path = argv[++i];
    } else if (arg == "--work-dir") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --work-dir\n";
        return 1;
      }
      options.work_dir = argv[++i];
    } else if (arg == "--compile-cmd") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --compile-cmd\n";
        return 1;
      }
      options.compile_cmd = argv[++i];
    } else if (arg == "--run-cmd") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --run-cmd\n";
        return 1;
      }
      options.run_cmd = argv[++i];
    } else if (arg == "--model-dir") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --model-dir\n";
        return 1;
      }
      options.model_dir = argv[++i];
    } else if (arg == "--model") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --model\n";
        return 1;
      }
      options.model_path = argv[++i];
    } else if (arg == "--repeat-session") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --repeat-session\n";
        return 1;
      }
      try {
        repeat_session = std::stoi(argv[++i]);
      } catch (...) {
        std::cerr << "invalid value for --repeat-session\n";
        return 1;
      }
    } else if (arg == "--detail") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --detail\n";
        return 1;
      }
      options.detail = argv[++i];
    } else if (arg == "--mode") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --mode\n";
        return 1;
      }
      options.mode = argv[++i];
    } else if (arg == "--repeat") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --repeat\n";
        return 1;
      }
      try {
        options.repeat = std::stoi(argv[++i]);
      } catch (...) {
        std::cerr << "invalid value for --repeat\n";
        return 1;
      }
    } else if (arg == "--sampling-ratio") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --sampling-ratio\n";
        return 1;
      }
      try {
        options.sampling_ratio = std::stod(argv[++i]);
      } catch (...) {
        std::cerr << "invalid value for --sampling-ratio\n";
        return 1;
      }
    } else if (arg == "--include-op") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --include-op\n";
        return 1;
      }
      options.include_ops.push_back(argv[++i]);
    } else if (arg == "--config") {
      if (!RequireValue(argc, argv, i)) {
        std::cerr << "missing value for --config\n";
        return 1;
      }
      config_path = argv[++i];
    } else if (arg == "--tool") {
      if (i + 1 >= argc) {
        std::cerr << "--tool requires a tool name\n";
        return 1;
      }
      options.requested_tools.emplace_back(argv[++i]);
    } else if (arg == "--list-tools") {
      dfabit::tools::RegisterBuiltinTools();
      for (const auto& name : dfabit::tools::ToolRegistry::Instance().List()) {
        std::cout << name << "\n";
      }
      return 0;
    } else if (arg == "--no-portability-tool") {
      options.enable_portability_tool = false;
    } else if (arg == "--no-overhead-profiler-tool") {
      options.enable_overhead_profiler_tool = false;
    } else if (arg == "--no-semantic-attribution-tool") {
      options.enable_semantic_attribution_tool = false;
    } else if (arg == "--no-dataflow-memory-proxy-tool") {
      options.enable_dataflow_memory_proxy_tool = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      PrintUsage();
      return 1;
    }
  }

  if (!config_path.empty()) {
    dfabit::cli::ExperimentConfigLoader loader;
    std::vector<dfabit::cli::ExperimentSpec> specs;
    auto st = loader.LoadFile(config_path, &specs);
    if (!st.ok()) {
      std::cerr << st.message() << "\n";
      return 1;
    }

    dfabit::cli::ExperimentRunner runner;
    std::vector<dfabit::cli::ExperimentRunRecord> records;
    st = runner.RunSpecs(specs, &records);
    if (!st.ok()) {
      std::cerr << st.message() << "\n";
      return 1;
    }

    const auto index_path = "experiment_index.csv";
    st = runner.WriteIndexCsv(index_path, records);
    if (!st.ok()) {
      std::cerr << st.message() << "\n";
      return 1;
    }

    std::cerr << "batch run completed\n";
    return 0;
  }

  if (options.repeat > 1) {
    std::vector<dfabit::cli::ExperimentSpec> specs(1);
    specs[0].name = "cli_repeat";
    specs[0].options = options;
    specs[0].modes = {options.mode};

    dfabit::cli::ExperimentRunner runner;
    std::vector<dfabit::cli::ExperimentRunRecord> records;
    const auto st = runner.RunSpecs(specs, &records);
    if (!st.ok()) {
      std::cerr << st.message() << "\n";
      return 1;
    }

    const auto index_path = "repeat_index.csv";
    auto write_st = runner.WriteIndexCsv(index_path, records);
    if (!write_st.ok()) {
      std::cerr << write_st.message() << "\n";
      return 1;
    }

    std::cerr << "repeat run completed\n";
    return 0;
  }

  if (repeat_session < 1) {
    std::cerr << "--repeat-session must be >= 1\n";
    return 1;
  }

  for (int rep = 0; rep < repeat_session; ++rep) {
    const auto st = dfabit::cli::Run(options);
    if (!st.ok()) {
      std::cerr << st.message() << "\n";
      return 1;
    }
  }

  std::cerr << "run completed successfully\n";
  return 0;
}
