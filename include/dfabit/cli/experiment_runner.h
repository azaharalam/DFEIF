#pragma once

#include <string>
#include <vector>

#include "dfabit/cli/experiment_config.h"
#include "dfabit/core/status.h"

namespace dfabit::cli {

struct ExperimentRunRecord {
  std::string experiment_name;
  std::string mode;
  int trial = 0;
  std::string output_dir;
  bool success = false;
  std::string message;
};

class ExperimentRunner {
 public:
  dfabit::core::Status RunSpecs(
      const std::vector<ExperimentSpec>& specs,
      std::vector<ExperimentRunRecord>* records) const;

  dfabit::core::Status WriteIndexCsv(
      const std::string& path,
      const std::vector<ExperimentRunRecord>& records) const;
};

}  // namespace dfabit::cli