#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"

namespace dfabit::adapters::edgetpu {

// One row of the Edge TPU compiler's operator table:
//   Operator                       Count      Status
//   CONV_2D                        36         Mapped to Edge TPU
//   PAD                            2          Operation is otherwise supported...
struct OperatorMappingRecord {
  std::string op_name;       // CONV_2D
  int count = 0;             // 36
  std::string status;        // full status text
  bool mapped_to_tpu = false;
  std::uint64_t stable_id = 0;
};

// The scalar fields the compiler reports above the operator table. Sizes are
// normalized to bytes; the compiler prints them as "3.93MiB", "0.00B", etc.
struct EdgeTpuCompileSummary {
  std::string compiler_version;
  std::string input_model;
  std::string output_model;

  double input_size_bytes = 0.0;
  double output_size_bytes = 0.0;
  double on_chip_used_bytes = 0.0;
  double on_chip_remaining_bytes = 0.0;
  double off_chip_streamed_bytes = 0.0;

  int subgraph_count = 0;
  int total_operations = 0;
  double compile_time_ms = 0.0;

  bool has_summary = false;
};

// Parses `edgetpu_compiler -s` output (stdout or the *_edgetpu.log file).
//
// This is the richest compiler-visible information the Edge TPU exposes: which
// operators the compiler could place on the accelerator, which fell back to the
// CPU, how the model partitioned into subgraphs, and how parameter memory split
// between on-chip cache and off-chip streaming. Everything here is read from the
// compiler's own report -- nothing is estimated.
class EdgeTpuCompilerLogParser {
 public:
  dfabit::core::Status ParseFile(
      const std::string& path,
      EdgeTpuCompileSummary* summary,
      std::vector<OperatorMappingRecord>* operators) const;

  dfabit::core::Status ParseText(
      const std::string& text,
      EdgeTpuCompileSummary* summary,
      std::vector<OperatorMappingRecord>* operators) const;

  std::vector<MetricSample> ToMetricSamples(
      const EdgeTpuCompileSummary& summary,
      const std::vector<OperatorMappingRecord>& operators) const;

  // "3.93MiB" -> 4121097.0 ; "0.00B" -> 0.0 ; returns false if unparseable.
  static bool ParseByteSize(const std::string& text, double* out_bytes);
};

}  // namespace dfabit::adapters::edgetpu