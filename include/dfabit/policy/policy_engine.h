#pragma once

#include <cstdint>

#include "dfabit/core/framework_config.h"
#include "dfabit/metadata/op_desc.h"
#include "dfabit/policy/filter_spec.h"
#include "dfabit/policy/sampling_spec.h"

namespace dfabit::policy {

class PolicyEngine {
 public:
  PolicyEngine() = default;
  explicit PolicyEngine(dfabit::core::PolicyConfig cfg);

  void Reset(dfabit::core::PolicyConfig cfg);

  const dfabit::core::PolicyConfig& config() const { return config_; }
  const FilterSpec& filter() const { return filter_; }
  const SamplingSpec& sampling() const { return sampling_; }

  bool ShouldInstrument(const dfabit::metadata::OpDesc& op) const;
  bool ShouldEmit(std::uint64_t stable_id) const;

 private:
  static std::uint64_t Mix(std::uint64_t v);

  dfabit::core::PolicyConfig config_;
  FilterSpec filter_;
  SamplingSpec sampling_;
};

}  // namespace dfabit::policy