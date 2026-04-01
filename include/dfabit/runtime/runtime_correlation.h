#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"

namespace dfabit::runtime {

struct RuntimeRecord {
  std::string symbol;
  std::string stage;
  std::string metric_name;
  double metric_value = 0.0;
  std::string unit;
};

struct CorrelatedRuntimeRecord {
  std::uint64_t stable_id = 0;
  std::string op_name;
  std::string stage;
  std::string metric_name;
  double metric_value = 0.0;
  std::string unit;
};

class RuntimeCorrelator {
 public:
  void Reset();
  void IndexModel(const dfabit::metadata::ModelDesc& model);

  dfabit::core::Status Correlate(
      const std::vector<RuntimeRecord>& runtime_records,
      std::vector<CorrelatedRuntimeRecord>* correlated) const;

  std::vector<dfabit::adapters::MetricSample> ToMetricSamples(
      const std::vector<CorrelatedRuntimeRecord>& correlated) const;

 private:
  std::unordered_map<std::string, std::uint64_t> symbol_to_id_;
  std::unordered_map<std::uint64_t, std::string> id_to_name_;
};

}  // namespace dfabit::runtime