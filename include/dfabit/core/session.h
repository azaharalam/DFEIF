#pragma once

#include <string>
#include <utility>

#include "dfabit/core/framework_config.h"

namespace dfabit::core {

class Session {
 public:
  Session() = default;
  explicit Session(FrameworkConfig cfg) : config_(std::move(cfg)) {}

  const FrameworkConfig& config() const { return config_; }
  FrameworkConfig& mutable_config() { return config_; }

  const std::string& id() const { return config_.session_id; }
  void set_id(std::string id) { config_.session_id = std::move(id); }

 private:
  FrameworkConfig config_;
};

}  // namespace dfabit::core