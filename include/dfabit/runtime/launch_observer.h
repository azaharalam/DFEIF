#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"

namespace dfabit::runtime {

struct LaunchEventRecord {
  std::string type;
  std::string symbol;
  std::string stage;
  std::string name;
  double value = 0.0;
  std::string unit;
};

class LaunchObserver {
 public:
  dfabit::core::Status ParseFile(
      const std::string& path,
      std::vector<LaunchEventRecord>* records) const;

  dfabit::core::Status ParseText(
      const std::string& text,
      std::vector<LaunchEventRecord>* records) const;

  std::vector<dfabit::adapters::MetricSample> Correlate(
      const dfabit::metadata::ModelDesc& model,
      const std::vector<LaunchEventRecord>& records) const;
};

}  // namespace dfabit::runtime