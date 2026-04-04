#pragma once

#include <string>

#include "dfabit/core/status.h"

namespace dfabit::platform {

struct ProcessSpec {
  std::string name;
  std::string command;
  std::string working_directory;
  std::string stdout_path;
  std::string stderr_path;
};

struct ProcessResult {
  int exit_code = 0;
  double elapsed_ms = 0.0;
  std::string stdout_path;
  std::string stderr_path;
};

class ProcessRunner {
 public:
  dfabit::core::Status Run(
      const ProcessSpec& spec,
      ProcessResult* result) const;
};

}  // namespace dfabit::platform