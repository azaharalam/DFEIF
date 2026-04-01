#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"

namespace dfabit::adapters::shared {

struct CompileReportRecord {
  std::string stage;
  std::string name;
  std::string value;
  std::string unit;
};

class CompileReportParser {
 public:
  dfabit::core::Status ParseFile(
      const std::string& path,
      std::vector<CompileReportRecord>* records) const;

  dfabit::core::Status ParseText(
      const std::string& text,
      std::vector<CompileReportRecord>* records) const;

  std::vector<MetricSample> ToMetricSamples(
      const std::vector<CompileReportRecord>& records) const;
};

}  // namespace dfabit::adapters::shared