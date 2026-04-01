#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/adapters/backend_adapter.h"
#include "dfabit/core/status.h"

namespace dfabit::adapters {

class BackendRegistry {
 public:
  static BackendRegistry& Instance();

  dfabit::core::Status Register(std::string adapter_name, BackendAdapterFactory factory);
  bool HasAdapter(const std::string& adapter_name) const;
  std::unique_ptr<BackendAdapter> Create(const std::string& adapter_name) const;
  std::vector<std::string> List() const;

 private:
  BackendRegistry() = default;

  std::unordered_map<std::string, BackendAdapterFactory> factories_;
};

}  // namespace dfabit::adapters