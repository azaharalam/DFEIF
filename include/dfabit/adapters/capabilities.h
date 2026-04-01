#pragma once

#include <string>
#include <vector>

namespace dfabit::adapters {

struct AdapterCapabilities {
  bool visible_mlir = false;
  bool visible_llvm = false;
  bool visible_graph_ir = false;
  bool compile_report_available = false;
  bool runtime_log_available = false;
  bool profiler_metrics_available = false;
  bool op_level_events = false;
  bool subgraph_level_events = false;
  bool partition_level_events = false;
  bool custom_env_controls = false;
  std::vector<std::string> supported_stages;
  std::vector<std::string> supported_artifact_types;
  std::vector<std::string> supported_metric_names;
};

}  // namespace dfabit::adapters