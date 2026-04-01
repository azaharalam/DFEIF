#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"

namespace dfabit::adapters::shared {

struct RuntimeLogRecord {
  std::string phase;
  std::string name;
  std::string value;
  std::string unit;
  std::uint64_t stable_id = 0;
};

class RuntimeLogParser {
 public:
  dfabit::core::Status ParseFile(
      const std::string& path,
      std::vector<RuntimeLogRecord>* records) const;

  dfabit::core::Status ParseText(
      const std::string& text,
      std::vector<RuntimeLogRecord>* records) const;

  std::vector<MetricSample> ToMetricSamples(
      const std::vector<RuntimeLogRecord>& records) const;
};

}  // namespace dfabit::adapters::shared