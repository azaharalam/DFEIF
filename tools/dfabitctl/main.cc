#include <iostream>
#include <string>
#include <vector>

#include "dfabit/cli/experiment_config.h"
#include "dfabit/cli/experiment_runner.h"
#include "dfabit/cli/runner.h"

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

  const auto st = dfabit::cli::Run(options);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  std::cerr << "run completed successfully\n";
  return 0;
}