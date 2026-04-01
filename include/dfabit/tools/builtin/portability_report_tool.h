#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"
#include "dfabit/tools/tool.h"

namespace dfabit::tools::builtin {

class PortabilityReportTool final : public Tool {
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
  dfabit::core::Status WriteCapabilities(dfabit::api::Context* ctx) const;
  static std::string OutputPath(const dfabit::api::Context& ctx, const std::string& filename);

  std::vector<dfabit::adapters::MetricSample> collected_metrics_;
};

std::unique_ptr<Tool> CreatePortabilityReportTool();

}  // namespace dfabit::tools::builtin