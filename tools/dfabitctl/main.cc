#include <filesystem>
#include <iostream>

#include "dfabit/analysis/lightweight_fit.h"
#include "dfabit/analysis/overhead_engine.h"
#include "dfabit/analysis/reporting.h"
#include "dfabit/analysis/scalability_runner.h"
#include "dfabit/adapters/backend_registry.h"

int main() {
  const auto names = dfabit::adapters::BackendRegistry::Instance().List();
  std::cerr << "dfabitctl: overhead and scalability layer build OK\n";
  for (const auto& name : names) {
    std::cerr << "adapter=" << name << "\n";
  }

  dfabit::analysis::OverheadEngine overhead;
  overhead.AddSample({
      "gpu_mlir",
      "full",
      "demo",
      1024,
      65536,
      10.0,
      10.8,
      0.2,
      0.0,
      100.0,
      92.0});

  dfabit::analysis::LightweightFitEngine fit_engine;
  dfabit::analysis::LightweightFitResult fit_result;
  const auto fit_st = fit_engine.Fit(overhead.samples(), &fit_result);
  if (!fit_st.ok()) {
    std::cerr << fit_st.message() << "\n";
    return 1;
  }

  dfabit::analysis::ScalabilityRunner scalability;
  scalability.AddFromOverheadSamples("event_count", overhead.samples());

  const std::string out_dir = "build/phase_reports";
  dfabit::analysis::Reporting reporting;
  auto st = reporting.WriteOverheadBundle(out_dir, overhead);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }
  st = reporting.WriteLightweightBundle(out_dir, fit_result);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }
  st = reporting.WriteScalabilityBundle(out_dir, scalability);
  if (!st.ok()) {
    std::cerr << st.message() << "\n";
    return 1;
  }

  return 0;
}