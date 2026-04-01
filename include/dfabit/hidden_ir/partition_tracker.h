#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/core/status.h"

namespace dfabit::hidden_ir {

struct PartitionRecord {
  std::string partition_name;
  std::string stage;
  std::uint64_t stable_id = 0;
  std::uint64_t begin_ts_ns = 0;
  std::uint64_t end_ts_ns = 0;
};

class PartitionTracker {
 public:
  void Reset();

  dfabit::core::Status Begin(
      const std::string& partition_name,
      const std::string& stage,
      std::uint64_t stable_id,
      std::uint64_t ts_ns);

  dfabit::core::Status End(
      const std::string& partition_name,
      std::uint64_t ts_ns);

  std::vector<PartitionRecord> Snapshot() const;

 private:
  std::unordered_map<std::string, PartitionRecord> active_;
  std::vector<PartitionRecord> completed_;
};

}  // namespace dfabit::hidden_ir