#pragma once
#include <stdexcept>
#include <string>
#include "dfabit/core/status.h"

namespace dfabit::core {

class DFABITError final : public std::runtime_error {
 public:
  explicit DFABITError(const std::string& what) : std::runtime_error(what) {}
  explicit DFABITError(const Status& s) : std::runtime_error(s.message()) {}
};

}  // namespace dfabit::core
