#pragma once

#include <cstdint>

namespace dfabit::policy {

struct SamplingSpec {
  double ratio = 1.0;
  std::uint64_t stride = 1;
  std::uint64_t salt = 0;

  bool enabled() const {
    return ratio < 1.0 || stride > 1;
  }
};

}  // namespace dfabit::policy