#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"

namespace dfabit::analysis {

struct OverheadSample {
  std::string backend_name;
  std::string mode;
  std::string workload_name;
  std::size_t event_count = 0;
  std::size_t trace_bytes = 0;
  double baseline_latency_ms = 0.0;
  double instrumented_latency_ms = 0.0;
  double flush_latency_ms = 0.0;
  double dropped_rate = 0.0;
  double throughput_baseline = 0.0;
  double throughput_instrumented = 0.0;
};

struct OverheadSummary {
  std::string backend_name;
  std::string mode;
  std::size_t sample_count = 0;
  std::size_t total_event_count = 0;
  std::size_t total_trace_bytes = 0;
  double mean_baseline_latency_ms = 0.0;
  double mean_instrumented_latency_ms = 0.0;
  double mean_flush_latency_ms = 0.0;
  double mean_dropped_rate = 0.0;
  double mean_slowdown_pct = 0.0;
  double p95_slowdown_pct = 0.0;
  double max_slowdown_pct = 0.0;
};

class OverheadEngine {
 public:
  void Reset();

  void AddSample(OverheadSample sample);
  void AddMetricSamples(
      const std::string& backend_name,
      const std::string& mode,
      const std::string& workload_name,
      const std::vector<dfabit::adapters::MetricSample>& metrics);

  const std::vector<OverheadSample>& samples() const { return samples_; }

  dfabit::core::Status Summarize(OverheadSummary* summary) const;
  dfabit::core::Status WriteSamplesCsv(const std::string& path) const;
  dfabit::core::Status WriteSummaryCsv(const std::string& path) const;

 private:
  static double SlowdownPct(const OverheadSample& sample);

  std::vector<OverheadSample> samples_;
};

}  // namespace dfabit::analysis