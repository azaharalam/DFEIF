#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "dfabit/core/status.h"

namespace dfabit::hidden_ir {

struct SidecarEntry {
  std::string symbol;
  std::uint64_t stable_id = 0;
  std::string stage;
  std::unordered_map<std::string, std::string> attributes;
};

class SidecarLoader {
 public:
  dfabit::core::Status LoadFile(
      const std::string& path,
      std::vector<SidecarEntry>* entries) const;

  dfabit::core::Status LoadText(
      const std::string& text,
      std::vector<SidecarEntry>* entries) const;
};

}  // namespace dfabit::hidden_ir