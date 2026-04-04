#include <iostream>
#include <string>

#include "dfabit/cli/runner.h"

namespace {

void PrintUsage() {
  std::cerr
      << "usage:\n"
      << "  dfabitctl --backend gpu_mlir --mlir <file> --out <dir>\n"
      << "  dfabitctl --backend cerebras --graph <file> [--sidecar <file>] "
         "[--compile-report <file>] [--runtime-log <file>] --out <dir>\n"
      << "  dfabitctl --backend sambanova --graph <file> [--sidecar <file>] "
         "[--compile-report <file>] [--runtime-log <file>] --out <dir>\n";
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
    } else if (arg == "--no-portability-tool") {
      options.enable_portability_tool = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      PrintUsage();
      return 1;
    }
  }

  const auto st = dfabit::cli::Run(options);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  std::cerr << "run completed successfully\n";
  return 0;
}