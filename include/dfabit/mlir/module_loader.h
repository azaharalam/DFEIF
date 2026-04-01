#pragma once

#include <string>
#include <vector>

#include "dfabit/core/status.h"

namespace dfabit::mlir {

struct MlirModuleSnapshot {
  std::string module_name;
  std::string stage;
  std::string source_path;
  std::string text;
  std::vector<std::string> lines;
};

class MlirModuleLoader {
 public:
  dfabit::core::Status LoadFromFile(
      const std::string& path,
      const std::string& stage,
      MlirModuleSnapshot* snapshot) const;

  dfabit::core::Status LoadFromText(
      std::string text,
      const std::string& module_name,
      const std::string& stage,
      MlirModuleSnapshot* snapshot) const;
};

}  // namespace dfabit::mlir