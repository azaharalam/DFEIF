#pragma once

#include <string>
#include <vector>

#include "dfabit/analysis/lightweight_fit.h"
#include "dfabit/analysis/overhead_engine.h"
#include "dfabit/core/status.h"

namespace dfabit::analysis {

struct ScalabilityPoint {
  std::string backend_name;
  std::string mode;
  std::string axis_name;
  double axis_value = 0.0;
  double slowdown_pct = 0.0;
  double trace_mb = 0.0;
  double events_per_run = 0.0;
};

struct ScalabilitySummary {
  std::string backend_name;
  std::string mode;
  std::string axis_name;
  std::size_t point_count = 0;
  double min_slowdown_pct = 0.0;
  double max_slowdown_pct = 0.0;
  double mean_slowdown_pct = 0.0;
  double min_trace_mb = 0.0;
  double max_trace_mb = 0.0;
  double mean_trace_mb = 0.0;
};

class ScalabilityRunner {
 public:
  void Reset();

  void AddPoint(ScalabilityPoint point);
  void AddFromOverheadSamples(
      const std::string& axis_name,
      const std::vector<OverheadSample>& samples);

  const std::vector<ScalabilityPoint>& points() const { return points_; }

  dfabit::core::Status Summarize(ScalabilitySummary* summary) const;
  dfabit::core::Status WritePointsCsv(const std::string& path) const;
  dfabit::core::Status WriteSummaryCsv(const std::string& path) const;

 private:
  std::vector<ScalabilityPoint> points_;
};

}  // namespace dfabit::analysis