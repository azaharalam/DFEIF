#include "dfabit/analysis/lightweight_fit.h"

#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

namespace dfabit::analysis {

double LightweightFitEngine::SlowdownMs(const OverheadSample& sample) {
  return sample.instrumented_latency_ms - sample.baseline_latency_ms;
}

double LightweightFitEngine::SlowdownPct(const OverheadSample& sample) {
  if (sample.baseline_latency_ms <= 0.0) {
    return 0.0;
  }
  return (SlowdownMs(sample) / sample.baseline_latency_ms) * 100.0;
}

dfabit::core::Status LightweightFitEngine::LinearRegression(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    LinearFit* fit) {
  if (!fit) {
    return {dfabit::core::StatusCode::kInvalidArgument, "fit is null"};
  }
  if (xs.size() != ys.size() || xs.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid regression inputs"};
  }

  const double n = static_cast<double>(xs.size());
  const double sum_x = std::accumulate(xs.begin(), xs.end(), 0.0);
  const double sum_y = std::accumulate(ys.begin(), ys.end(), 0.0);
  const double mean_x = sum_x / n;
  const double mean_y = sum_y / n;

  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const double dx = xs[i] - mean_x;
    const double dy = ys[i] - mean_y;
    num += dx * dy;
    den += dx * dx;
  }

  fit->slope = den == 0.0 ? 0.0 : num / den;
  fit->intercept = mean_y - fit->slope * mean_x;

  double ss_tot = 0.0;
  double ss_res = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const double y_hat = fit->intercept + fit->slope * xs[i];
    const double dy = ys[i] - mean_y;
    const double residual = ys[i] - y_hat;
    ss_tot += dy * dy;
    ss_res += residual * residual;
  }

  fit->r2 = ss_tot == 0.0 ? 1.0 : 1.0 - (ss_res / ss_tot);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status LightweightFitEngine::Fit(
    const std::vector<OverheadSample>& samples,
    LightweightFitResult* result) const {
  if (!result) {
    return {dfabit::core::StatusCode::kInvalidArgument, "result is null"};
  }
  if (samples.empty()) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "no overhead samples"};
  }

  std::vector<double> event_xs;
  std::vector<double> kb_xs;
  std::vector<double> slowdown_ms;
  std::vector<double> slowdown_pct;

  event_xs.reserve(samples.size());
  kb_xs.reserve(samples.size());
  slowdown_ms.reserve(samples.size());
  slowdown_pct.reserve(samples.size());

  result->backend_name = samples.front().backend_name;
  result->mode = samples.front().mode;
  result->count = samples.size();

  for (const auto& sample : samples) {
    event_xs.push_back(static_cast<double>(sample.event_count));
    kb_xs.push_back(static_cast<double>(sample.trace_bytes) / 1024.0);
    slowdown_ms.push_back(SlowdownMs(sample));
    slowdown_pct.push_back(SlowdownPct(sample));
  }

  LinearFit event_fit;
  auto st = LinearRegression(event_xs, slowdown_ms, &event_fit);
  if (!st.ok()) {
    return st;
  }

  LinearFit kb_fit;
  st = LinearRegression(kb_xs, slowdown_ms, &kb_fit);
  if (!st.ok()) {
    return st;
  }

  LinearFit mb_pct_fit;
  std::vector<double> mb_xs;
  mb_xs.reserve(samples.size());
  for (const auto& sample : samples) {
    mb_xs.push_back(static_cast<double>(sample.trace_bytes) / (1024.0 * 1024.0));
  }
  st = LinearRegression(mb_xs, slowdown_pct, &mb_pct_fit);
  if (!st.ok()) {
    return st;
  }

  result->slope_us_per_event = event_fit.slope * 1000.0;
  result->intercept_ms_per_event = event_fit.intercept;
  result->r2_event = event_fit.r2;
  result->slope_us_per_kb = kb_fit.slope * 1000.0;
  result->intercept_ms_per_kb = kb_fit.intercept;
  result->r2_kb = kb_fit.r2;
  result->slope_pct_per_mb = mb_pct_fit.slope;

  return dfabit::core::Status::Ok();
}

dfabit::core::Status LightweightFitEngine::WriteCsv(
    const std::string& path,
    const LightweightFitResult& result) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open lightweight fit csv: " + path};
  }

  ofs << "backend_name,mode,count,slope_us_per_event,intercept_ms_per_event,r2_event,"
         "slope_us_per_kb,intercept_ms_per_kb,r2_kb,slope_pct_per_mb\n";
  ofs << result.backend_name << ","
      << result.mode << ","
      << result.count << ","
      << result.slope_us_per_event << ","
      << result.intercept_ms_per_event << ","
      << result.r2_event << ","
      << result.slope_us_per_kb << ","
      << result.intercept_ms_per_kb << ","
      << result.r2_kb << ","
      << result.slope_pct_per_mb << "\n";

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::analysis