#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/core/status.h"
#include "dfabit/tools/tool.h"

namespace dfabit::tools {

class ToolRegistry {
 public:
  static ToolRegistry& Instance();

  dfabit::core::Status Register(std::string tool_name, ToolFactory factory);
  bool HasTool(const std::string& tool_name) const;
  std::unique_ptr<Tool> Create(const std::string& tool_name) const;
  std::vector<std::string> List() const;

 private:
  ToolRegistry() = default;

  std::unordered_map<std::string, ToolFactory> factories_;
};

}  // namespace dfabit::tools