#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "dfabit/core/status.h"

namespace dfabit::analysis {

struct OpRecord {
  std::string op_id;      // "0:12"
  std::string op_name;    // e.g., "CONV_2D"
  std::string layer;      // pretty string (Python can fill)
  int64_t mac_ops = 0;
  int64_t total_bytes = 0;
  double measured_latency_ms = 0.0;  // v1 default 0; later fill by segmentation

  // Stable kernel id used in your datasheet
  std::string kernel_id;  // e.g. "coral.mobilenetv2.conv2d12"
};

dfabit::core::Status WriteProgramAnalysisCsv(
    const std::string& out_path,
    const std::vector<OpRecord>& ops);

}  // namespace dfabit::analysis
