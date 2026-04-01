#pragma once

#include <string>
#include <vector>

#include "dfabit/analysis/overhead_engine.h"
#include "dfabit/core/status.h"

namespace dfabit::analysis {

struct LightweightFitResult {
  std::string backend_name;
  std::string mode;
  std::size_t count = 0;
  double slope_us_per_event = 0.0;
  double intercept_ms_per_event = 0.0;
  double r2_event = 0.0;
  double slope_us_per_kb = 0.0;
  double intercept_ms_per_kb = 0.0;
  double r2_kb = 0.0;
  double slope_pct_per_mb = 0.0;
};

class LightweightFitEngine {
 public:
  dfabit::core::Status Fit(
      const std::vector<OverheadSample>& samples,
      LightweightFitResult* result) const;

  dfabit::core::Status WriteCsv(
      const std::string& path,
      const LightweightFitResult& result) const;

 private:
  struct LinearFit {
    double slope = 0.0;
    double intercept = 0.0;
    double r2 = 0.0;
  };

  static double SlowdownMs(const OverheadSample& sample);
  static double SlowdownPct(const OverheadSample& sample);
  static dfabit::core::Status LinearRegression(
      const std::vector<double>& xs,
      const std::vector<double>& ys,
      LinearFit* fit);
};

}  // namespace dfabit::analysis