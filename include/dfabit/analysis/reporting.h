#pragma once

#include <string>

#include "dfabit/analysis/lightweight_fit.h"
#include "dfabit/analysis/overhead_engine.h"
#include "dfabit/analysis/scalability_runner.h"
#include "dfabit/core/status.h"

namespace dfabit::analysis {

class Reporting {
 public:
  dfabit::core::Status WriteOverheadBundle(
      const std::string& output_dir,
      const OverheadEngine& engine) const;

  dfabit::core::Status WriteLightweightBundle(
      const std::string& output_dir,
      const LightweightFitResult& result) const;

  dfabit::core::Status WriteScalabilityBundle(
      const std::string& output_dir,
      const ScalabilityRunner& runner) const;
};

}  // namespace dfabit::analysis