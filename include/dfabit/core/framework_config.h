#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dfabit::core {

enum class InstrumentationMode {
  kFull = 0,
  kSelective = 1,
  kSampled = 2
};

enum class DetailLevel {
  kIds = 0,
  kLite = 1,
  kFull = 2
};

struct OutputConfig {
  std::string base_output_dir;
  std::string trace_jsonl_path;
  std::string report_dir;
  std::string program_analysis_csv_path;
};

struct TraceConfig {
  bool enabled = true;
  std::size_t buffer_capacity = 65536;
  bool flush_per_write = false;
};

struct PolicyConfig {
  InstrumentationMode mode = InstrumentationMode::kFull;
  DetailLevel detail_level = DetailLevel::kFull;
  std::vector<std::string> include_ops;
  std::vector<std::string> include_stages;
  std::vector<std::string> include_dialects;
  double sampling_ratio = 1.0;
  std::uint64_t sampling_stride = 1;
  std::uint64_t sampling_salt = 0;
};

struct BackendConfig {
  std::string backend_name;
  std::string provider_name;
  std::string adapter_name;
};

struct RunConfig {
  std::string run_id;
  std::string model_name;
  OutputConfig output;
  TraceConfig trace;
  PolicyConfig policy;
  BackendConfig backend;
  std::unordered_map<std::string, std::string> attributes;
};

struct FrameworkConfig {
  std::string session_id;
  OutputConfig default_output;
  TraceConfig default_trace;
  PolicyConfig default_policy;
  BackendConfig default_backend;
  std::unordered_map<std::string, std::string> attributes;
};

}  // namespace dfabit::core