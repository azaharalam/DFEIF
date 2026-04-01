#include "dfabit/policy/policy_engine.h"

#include <algorithm>
#include <cmath>

namespace dfabit::policy {

namespace {

bool Contains(const std::unordered_set<std::string>& s, const std::string& v) {
  return s.find(v) != s.end();
}

std::unordered_set<std::string> ToSet(const std::vector<std::string>& v) {
  return std::unordered_set<std::string>(v.begin(), v.end());
}

}  // namespace

PolicyEngine::PolicyEngine(dfabit::core::PolicyConfig cfg) {
  Reset(std::move(cfg));
}

void PolicyEngine::Reset(dfabit::core::PolicyConfig cfg) {
  config_ = std::move(cfg);
  filter_.op_names = ToSet(config_.include_ops);
  filter_.stages = ToSet(config_.include_stages);
  filter_.dialects = ToSet(config_.include_dialects);
  sampling_.ratio = config_.sampling_ratio;
  sampling_.stride = config_.sampling_stride == 0 ? 1 : config_.sampling_stride;
  sampling_.salt = config_.sampling_salt;
}

bool PolicyEngine::ShouldInstrument(const dfabit::metadata::OpDesc& op) const {
  if (!filter_.op_names.empty() && !Contains(filter_.op_names, op.op_name)) {
    return false;
  }
  if (!filter_.stages.empty() && !Contains(filter_.stages, op.stage_tag)) {
    return false;
  }
  if (!filter_.dialects.empty() && !Contains(filter_.dialects, op.dialect)) {
    return false;
  }
  if (op.stable_id < filter_.min_stable_id || op.stable_id > filter_.max_stable_id) {
    return false;
  }
  if (config_.mode == dfabit::core::InstrumentationMode::kSampled) {
    return ShouldEmit(op.stable_id);
  }
  return true;
}

bool PolicyEngine::ShouldEmit(std::uint64_t stable_id) const {
  const std::uint64_t mixed = Mix(stable_id ^ sampling_.salt);

  if (sampling_.stride > 1 && (mixed % sampling_.stride) != 0) {
    return false;
  }

  if (sampling_.ratio <= 0.0) {
    return false;
  }
  if (sampling_.ratio >= 1.0) {
    return true;
  }

  constexpr double kScale = 1000000.0;
  const double bucket = static_cast<double>(mixed % static_cast<std::uint64_t>(kScale));
  return bucket < std::floor(sampling_.ratio * kScale);
}

std::uint64_t PolicyEngine::Mix(std::uint64_t v) {
  v ^= v >> 33;
  v *= 0xff51afd7ed558ccdULL;
  v ^= v >> 33;
  v *= 0xc4ceb9fe1a85ec53ULL;
  v ^= v >> 33;
  return v;
}

}  // namespace dfabit::policy