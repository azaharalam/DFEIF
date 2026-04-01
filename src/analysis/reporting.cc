#include "dfabit/analysis/reporting.h"

#include <filesystem>

namespace dfabit::analysis {

dfabit::core::Status Reporting::WriteOverheadBundle(
    const std::string& output_dir,
    const OverheadEngine& engine) const {
  std::filesystem::create_directories(output_dir);

  auto st = engine.WriteSamplesCsv(
      (std::filesystem::path(output_dir) / "overhead_samples.csv").string());
  if (!st.ok()) {
    return st;
  }

  st = engine.WriteSummaryCsv(
      (std::filesystem::path(output_dir) / "overhead_summary.csv").string());
  if (!st.ok()) {
    return st;
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status Reporting::WriteLightweightBundle(
    const std::string& output_dir,
    const LightweightFitResult& result) const {
  std::filesystem::create_directories(output_dir);
  LightweightFitEngine engine;
  return engine.WriteCsv(
      (std::filesystem::path(output_dir) / "lightweight_fit.csv").string(),
      result);
}

dfabit::core::Status Reporting::WriteScalabilityBundle(
    const std::string& output_dir,
    const ScalabilityRunner& runner) const {
  std::filesystem::create_directories(output_dir);

  auto st = runner.WritePointsCsv(
      (std::filesystem::path(output_dir) / "scalability_points.csv").string());
  if (!st.ok()) {
    return st;
  }

  st = runner.WriteSummaryCsv(
      (std::filesystem::path(output_dir) / "scalability_summary.csv").string());
  if (!st.ok()) {
    return st;
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::analysis