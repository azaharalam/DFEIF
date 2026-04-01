#include "dfabit/adapters/backend_registry.h"

#include <algorithm>
#include <utility>

namespace dfabit::adapters {

BackendRegistry& BackendRegistry::Instance() {
  static BackendRegistry registry;
  return registry;
}

dfabit::core::Status BackendRegistry::Register(
    std::string adapter_name,
    BackendAdapterFactory factory) {
  if (adapter_name.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "adapter_name is empty"};
  }
  if (!factory) {
    return {dfabit::core::StatusCode::kInvalidArgument, "factory is null"};
  }
  const auto it = factories_.find(adapter_name);
  if (it != factories_.end()) {
    return {
        dfabit::core::StatusCode::kAlreadyExists,
        "adapter already registered: " + adapter_name};
  }
  factories_.emplace(std::move(adapter_name), factory);
  return dfabit::core::Status::Ok();
}

bool BackendRegistry::HasAdapter(const std::string& adapter_name) const {
  return factories_.find(adapter_name) != factories_.end();
}

std::unique_ptr<BackendAdapter> BackendRegistry::Create(
    const std::string& adapter_name) const {
  const auto it = factories_.find(adapter_name);
  if (it == factories_.end()) {
    return nullptr;
  }
  return it->second();
}

std::vector<std::string> BackendRegistry::List() const {
  std::vector<std::string> out;
  out.reserve(factories_.size());
  for (const auto& kv : factories_) {
    out.push_back(kv.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace dfabit::adapters