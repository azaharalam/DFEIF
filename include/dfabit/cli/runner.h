#pragma once

#include <string>
#include <vector>

#include "dfabit/core/status.h"

namespace dfabit::cli {

struct CliOptions {
  std::string backend;
  std::string output_dir;

  std::string mlir_path;

  std::string graph_path;
  std::string sidecar_path;
  std::string compile_report_path;
  std::string runtime_log_path;

  std::string mode = "full";
  int repeat = 1;
  double sampling_ratio = 1.0;
  std::vector<std::string> include_ops;

  bool enable_portability_tool = true;
  bool enable_overhead_profiler_tool = true;
};

dfabit::core::Status Run(const CliOptions& options);

}  // namespace dfabit::cli