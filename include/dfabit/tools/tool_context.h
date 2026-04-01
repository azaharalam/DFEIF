#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/api/context.h"
#include "dfabit/core/framework_config.h"
#include "dfabit/core/status.h"
#include "dfabit/metadata/op_desc.h"
#include "dfabit/policy/policy_engine.h"

namespace dfabit::tools {

class ToolContextView {
 public:
  explicit ToolContextView(dfabit::api::Context* ctx) : ctx_(ctx) {}

  bool valid() const { return ctx_ != nullptr; }

  const dfabit::core::RunConfig& run_config() const {
    return ctx_->run_context().config();
  }

  const dfabit::core::FrameworkConfig& framework_config() const {
    return ctx_->framework_config();
  }

  const std::vector<dfabit::metadata::OpDesc>& metadata_ops() const {
    return ctx_->metadata_ops();
  }

  std::string GetProperty(const std::string& key) const {
    return ctx_->GetProperty(key);
  }

  void SetProperty(const std::string& key, std::string value) {
    ctx_->SetProperty(key, std::move(value));
  }

 private:
  dfabit::api::Context* ctx_ = nullptr;
};

class ToolServices {
 public:
  static dfabit::policy::PolicyEngine BuildPolicyEngine(const dfabit::api::Context& ctx);

  static std::vector<dfabit::metadata::OpDesc> FilterOps(
      const dfabit::api::Context& ctx,
      const std::vector<dfabit::metadata::OpDesc>& ops);

  static std::vector<dfabit::adapters::MetricSample> FilterMetrics(
      const dfabit::api::Context& ctx,
      const std::vector<dfabit::adapters::MetricSample>& metrics);

  static dfabit::core::Status WriteMetricsCsv(
      const std::string& path,
      const std::vector<dfabit::adapters::MetricSample>& metrics);

  static dfabit::core::Status WriteOpTableCsv(
      const std::string& path,
      const std::vector<dfabit::metadata::OpDesc>& ops);
};

}  // namespace dfabit::tools