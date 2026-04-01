#pragma once

#include <utility>

#include "dfabit/core/framework_config.h"

namespace dfabit::api {
class Context;
}

namespace dfabit::core {

class Framework {
 public:
  Framework() = default;
  explicit Framework(FrameworkConfig cfg) : config_(std::move(cfg)) {}

  const FrameworkConfig& config() const { return config_; }
  void SetConfig(FrameworkConfig cfg) { config_ = std::move(cfg); }

  dfabit::api::Context CreateContext() const;
  dfabit::api::Context CreateContext(RunConfig run_cfg) const;

 private:
  FrameworkConfig config_;
};

}  // namespace dfabit::core