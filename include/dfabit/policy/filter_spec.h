#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>

namespace dfabit::policy {

struct FilterSpec {
  std::unordered_set<std::string> op_names;
  std::unordered_set<std::string> stages;
  std::unordered_set<std::string> dialects;
  std::uint64_t min_stable_id = 0;
  std::uint64_t max_stable_id = std::numeric_limits<std::uint64_t>::max();

  bool empty() const {
    return op_names.empty() &&
           stages.empty() &&
           dialects.empty() &&
           min_stable_id == 0 &&
           max_stable_id == std::numeric_limits<std::uint64_t>::max();
  }
};

}  // namespace dfabit::policy