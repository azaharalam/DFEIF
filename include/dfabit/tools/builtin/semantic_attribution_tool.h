#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"
#include "dfabit/metadata/op_desc.h"
#include "dfabit/tools/tool.h"

namespace dfabit::tools::builtin {

class SemanticAttributionTool final : public Tool {
 public:
  std::string name() const override;

  dfabit::core::Status OnRegister(dfabit::api::Context* ctx) override;
  dfabit::core::Status OnInit(dfabit::api::Context* ctx) override;
  dfabit::core::Status OnShutdown(dfabit::api::Context* ctx) override;

  dfabit::core::Status OnCompileBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::CompileArtifactSet& compile_artifacts) override;

  dfabit::core::Status OnCompileEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::CompileArtifactSet& compile_artifacts) override;

  dfabit::core::Status OnLoadBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status OnLoadEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status OnRunBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status OnRunEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) override;

 private:
  static std::string OutputDir(const dfabit::api::Context& ctx);

  dfabit::core::Status WriteAttributionTable(
      const std::string& path,
      const std::vector<dfabit::metadata::OpDesc>& ops,
      const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const;

  dfabit::core::Status WriteSummary(
      const std::string& path,
      const std::vector<dfabit::metadata::OpDesc>& ops,
      const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const;

  std::vector<dfabit::metadata::OpDesc> filtered_ops_;
  std::vector<dfabit::adapters::MetricSample> runtime_metrics_;
};

std::unique_ptr<Tool> CreateSemanticAttributionTool();

}  // namespace dfabit::tools::builtin