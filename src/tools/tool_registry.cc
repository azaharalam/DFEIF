#include "dfabit/tools/tool_registry.h"

#include <algorithm>
#include <utility>

namespace dfabit::tools {

ToolRegistry& ToolRegistry::Instance() {
  static ToolRegistry registry;
  return registry;
}

dfabit::core::Status ToolRegistry::Register(std::string tool_name, ToolFactory factory) {
  if (tool_name.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "tool_name is empty"};
  }
  if (!factory) {
    return {dfabit::core::StatusCode::kInvalidArgument, "factory is null"};
  }

  const auto it = factories_.find(tool_name);
  if (it != factories_.end()) {
    return {
        dfabit::core::StatusCode::kAlreadyExists,
        "tool already registered: " + tool_name};
  }

  factories_.emplace(std::move(tool_name), factory);
  return dfabit::core::Status::Ok();
}

bool ToolRegistry::HasTool(const std::string& tool_name) const {
  return factories_.find(tool_name) != factories_.end();
}

std::unique_ptr<Tool> ToolRegistry::Create(const std::string& tool_name) const {
  const auto it = factories_.find(tool_name);
  if (it == factories_.end()) {
    return nullptr;
  }
  return it->second();
}

std::vector<std::string> ToolRegistry::List() const {
  std::vector<std::string> out;
  out.reserve(factories_.size());
  for (const auto& kv : factories_) {
    out.push_back(kv.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace dfabit::tools