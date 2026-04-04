#pragma once

#include <string>
#include <vector>

#include "dfabit/cli/runner.h"
#include "dfabit/core/status.h"

namespace dfabit::cli {

struct ExperimentSpec {
  std::string name;
  CliOptions options;
  std::vector<std::string> modes;
};

class ExperimentConfigLoader {
 public:
  dfabit::core::Status LoadFile(
      const std::string& path,
      std::vector<ExperimentSpec>* specs) const;
};

}  // namespace dfabit::cli