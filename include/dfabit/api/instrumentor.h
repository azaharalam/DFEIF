#pragma once
#include <string>
#include "dfabit/core/status.h"

namespace dfabit::api {

class Context;

class Instrumentor {
 public:
  virtual ~Instrumentor() = default;
  virtual std::string name() const = 0;

  // Execute the tool on the current Context (adapter + policy + sinks).
  virtual dfabit::core::Status Run(Context* ctx) = 0;
};

}  // namespace dfabit::api
