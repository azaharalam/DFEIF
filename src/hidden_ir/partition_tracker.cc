#include "dfabit/hidden_ir/partition_tracker.h"

namespace dfabit::hidden_ir {

void PartitionTracker::Reset() {
  active_.clear();
  completed_.clear();
}

dfabit::core::Status PartitionTracker::Begin(
    const std::string& partition_name,
    const std::string& stage,
    std::uint64_t stable_id,
    std::uint64_t ts_ns) {
  if (partition_name.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "partition_name is empty"};
  }
  if (active_.find(partition_name) != active_.end()) {
    return {dfabit::core::StatusCode::kAlreadyExists, "partition already active: " + partition_name};
  }

  PartitionRecord rec;
  rec.partition_name = partition_name;
  rec.stage = stage;
  rec.stable_id = stable_id;
  rec.begin_ts_ns = ts_ns;
  active_.emplace(partition_name, rec);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status PartitionTracker::End(
    const std::string& partition_name,
    std::uint64_t ts_ns) {
  const auto it = active_.find(partition_name);
  if (it == active_.end()) {
    return {dfabit::core::StatusCode::kNotFound, "partition not active: " + partition_name};
  }

  it->second.end_ts_ns = ts_ns;
  completed_.push_back(it->second);
  active_.erase(it);
  return dfabit::core::Status::Ok();
}

std::vector<PartitionRecord> PartitionTracker::Snapshot() const {
  return completed_;
}

}  // namespace dfabit::hidden_ir