#include "dfabit/analysis/scalability_runner.h"

#include <algorithm>
#include <fstream>
#include <numeric>
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

double SlowdownPct(const OverheadSample& sample) {
  if (sample.baseline_latency_ms <= 0.0) {
    return 0.0;
  }
  return ((sample.instrumented_latency_ms - sample.baseline_latency_ms) /
          sample.baseline_latency_ms) *
         100.0;
}

}  // namespace

void ScalabilityRunner::Reset() {
  points_.clear();
}

void ScalabilityRunner::AddPoint(ScalabilityPoint point) {
  points_.push_back(std::move(point));
}

void ScalabilityRunner::AddFromOverheadSamples(
    const std::string& axis_name,
    const std::vector<OverheadSample>& samples) {
  for (const auto& sample : samples) {
    ScalabilityPoint point;
    point.backend_name = sample.backend_name;
    point.mode = sample.mode;
    point.axis_name = axis_name;
    if (axis_name == "event_count") {
      point.axis_value = static_cast<double>(sample.event_count);
    } else if (axis_name == "trace_bytes") {
      point.axis_value = static_cast<double>(sample.trace_bytes);
    } else {
      point.axis_value = static_cast<double>(sample.event_count);
    }
    point.slowdown_pct = SlowdownPct(sample);
    point.trace_mb = static_cast<double>(sample.trace_bytes) / (1024.0 * 1024.0);
    point.events_per_run = static_cast<double>(sample.event_count);
    points_.push_back(std::move(point));
  }
}

dfabit::core::Status ScalabilityRunner::Summarize(ScalabilitySummary* summary) const {
  if (!summary) {
    return {dfabit::core::StatusCode::kInvalidArgument, "summary is null"};
  }
  if (points_.empty()) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "no scalability points"};
  }

  summary->backend_name = points_.front().backend_name;
  summary->mode = points_.front().mode;
  summary->axis_name = points_.front().axis_name;
  summary->point_count = points_.size();

  std::vector<double> slowdowns;
  std::vector<double> traces;
  slowdowns.reserve(points_.size());
  traces.reserve(points_.size());

  for (const auto& point : points_) {
    slowdowns.push_back(point.slowdown_pct);
    traces.push_back(point.trace_mb);
  }

  summary->min_slowdown_pct = *std::min_element(slowdowns.begin(), slowdowns.end());
  summary->max_slowdown_pct = *std::max_element(slowdowns.begin(), slowdowns.end());
  summary->mean_slowdown_pct = Mean(slowdowns);
  summary->min_trace_mb = *std::min_element(traces.begin(), traces.end());
  summary->max_trace_mb = *std::max_element(traces.begin(), traces.end());
  summary->mean_trace_mb = Mean(traces);

  return dfabit::core::Status::Ok();
}

dfabit::core::Status ScalabilityRunner::WritePointsCsv(const std::string& path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open scalability points csv: " + path};
  }

  ofs << "backend_name,mode,axis_name,axis_value,slowdown_pct,trace_mb,events_per_run\n";
  for (const auto& point : points_) {
    ofs << point.backend_name << ","
        << point.mode << ","
        << point.axis_name << ","
        << point.axis_value << ","
        << point.slowdown_pct << ","
        << point.trace_mb << ","
        << point.events_per_run << "\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status ScalabilityRunner::WriteSummaryCsv(const std::string& path) const {
  ScalabilitySummary summary;
  const auto st = Summarize(&summary);
  if (!st.ok()) {
    return st;
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open scalability summary csv: " + path};
  }

  ofs << "backend_name,mode,axis_name,point_count,min_slowdown_pct,max_slowdown_pct,"
         "mean_slowdown_pct,min_trace_mb,max_trace_mb,mean_trace_mb\n";
  ofs << summary.backend_name << ","
      << summary.mode << ","
      << summary.axis_name << ","
      << summary.point_count << ","
      << summary.min_slowdown_pct << ","
      << summary.max_slowdown_pct << ","
      << summary.mean_slowdown_pct << ","
      << summary.min_trace_mb << ","
      << summary.max_trace_mb << ","
      << summary.mean_trace_mb << "\n";

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::analysis