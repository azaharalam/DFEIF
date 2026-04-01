#include "dfabit/analysis/overhead_engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <utility>

namespace dfabit::analysis {

namespace {

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double Percentile95(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t idx =
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
  return values[std::min(idx, values.size() - 1)];
}

double ParseMetricValue(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    const std::string& name,
    double default_value) {
  for (const auto& metric : metrics) {
    if (metric.name == name) {
      return metric.value;
    }
  }
  return default_value;
}

std::size_t ParseMetricSizeValue(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    const std::string& name,
    std::size_t default_value) {
  for (const auto& metric : metrics) {
    if (metric.name == name) {
      if (metric.value < 0.0) {
        return 0;
      }
      return static_cast<std::size_t>(metric.value);
    }
  }
  return default_value;
}

}  // namespace

void OverheadEngine::Reset() {
  samples_.clear();
}

void OverheadEngine::AddSample(OverheadSample sample) {
  samples_.push_back(std::move(sample));
}

void OverheadEngine::AddMetricSamples(
    const std::string& backend_name,
    const std::string& mode,
    const std::string& workload_name,
    const std::vector<dfabit::adapters::MetricSample>& metrics) {
  OverheadSample sample;
  sample.backend_name = backend_name;
  sample.mode = mode;
  sample.workload_name = workload_name;
  sample.event_count = ParseMetricSizeValue(metrics, "event_count", 0);
  sample.trace_bytes = ParseMetricSizeValue(metrics, "trace_bytes", 0);
  sample.baseline_latency_ms = ParseMetricValue(metrics, "baseline_latency_ms", 0.0);
  sample.instrumented_latency_ms = ParseMetricValue(metrics, "instrumented_latency_ms", 0.0);
  sample.flush_latency_ms = ParseMetricValue(metrics, "flush_latency_ms", 0.0);
  sample.dropped_rate = ParseMetricValue(metrics, "dropped_rate", 0.0);
  sample.throughput_baseline = ParseMetricValue(metrics, "throughput_baseline", 0.0);
  sample.throughput_instrumented = ParseMetricValue(metrics, "throughput_instrumented", 0.0);
  samples_.push_back(std::move(sample));
}

dfabit::core::Status OverheadEngine::Summarize(OverheadSummary* summary) const {
  if (!summary) {
    return {dfabit::core::StatusCode::kInvalidArgument, "summary is null"};
  }
  if (samples_.empty()) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "no overhead samples"};
  }

  std::vector<double> baselines;
  std::vector<double> instrumented;
  std::vector<double> flushes;
  std::vector<double> dropped;
  std::vector<double> slowdowns;

  baselines.reserve(samples_.size());
  instrumented.reserve(samples_.size());
  flushes.reserve(samples_.size());
  dropped.reserve(samples_.size());
  slowdowns.reserve(samples_.size());

  summary->backend_name = samples_.front().backend_name;
  summary->mode = samples_.front().mode;
  summary->sample_count = samples_.size();

  for (const auto& sample : samples_) {
    summary->total_event_count += sample.event_count;
    summary->total_trace_bytes += sample.trace_bytes;
    baselines.push_back(sample.baseline_latency_ms);
    instrumented.push_back(sample.instrumented_latency_ms);
    flushes.push_back(sample.flush_latency_ms);
    dropped.push_back(sample.dropped_rate);
    slowdowns.push_back(SlowdownPct(sample));
  }

  summary->mean_baseline_latency_ms = Mean(baselines);
  summary->mean_instrumented_latency_ms = Mean(instrumented);
  summary->mean_flush_latency_ms = Mean(flushes);
  summary->mean_dropped_rate = Mean(dropped);
  summary->mean_slowdown_pct = Mean(slowdowns);
  summary->p95_slowdown_pct = Percentile95(slowdowns);
  summary->max_slowdown_pct =
      *std::max_element(slowdowns.begin(), slowdowns.end());

  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadEngine::WriteSamplesCsv(const std::string& path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open overhead samples csv: " + path};
  }

  ofs << "backend_name,mode,workload_name,event_count,trace_bytes,baseline_latency_ms,"
         "instrumented_latency_ms,flush_latency_ms,dropped_rate,throughput_baseline,"
         "throughput_instrumented,slowdown_pct\n";

  for (const auto& sample : samples_) {
    ofs << sample.backend_name << ","
        << sample.mode << ","
        << sample.workload_name << ","
        << sample.event_count << ","
        << sample.trace_bytes << ","
        << sample.baseline_latency_ms << ","
        << sample.instrumented_latency_ms << ","
        << sample.flush_latency_ms << ","
        << sample.dropped_rate << ","
        << sample.throughput_baseline << ","
        << sample.throughput_instrumented << ","
        << SlowdownPct(sample) << "\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status OverheadEngine::WriteSummaryCsv(const std::string& path) const {
  OverheadSummary summary;
  const auto st = Summarize(&summary);
  if (!st.ok()) {
    return st;
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open overhead summary csv: " + path};
  }

  ofs << "backend_name,mode,sample_count,total_event_count,total_trace_bytes,"
         "mean_baseline_latency_ms,mean_instrumented_latency_ms,mean_flush_latency_ms,"
         "mean_dropped_rate,mean_slowdown_pct,p95_slowdown_pct,max_slowdown_pct\n";

  ofs << summary.backend_name << ","
      << summary.mode << ","
      << summary.sample_count << ","
      << summary.total_event_count << ","
      << summary.total_trace_bytes << ","
      << summary.mean_baseline_latency_ms << ","
      << summary.mean_instrumented_latency_ms << ","
      << summary.mean_flush_latency_ms << ","
      << summary.mean_dropped_rate << ","
      << summary.mean_slowdown_pct << ","
      << summary.p95_slowdown_pct << ","
      << summary.max_slowdown_pct << "\n";

  return dfabit::core::Status::Ok();
}

double OverheadEngine::SlowdownPct(const OverheadSample& sample) {
  if (sample.baseline_latency_ms <= 0.0) {
    return 0.0;
  }
  return ((sample.instrumented_latency_ms - sample.baseline_latency_ms) /
          sample.baseline_latency_ms) *
         100.0;
}

}  // namespace dfabit::analysis