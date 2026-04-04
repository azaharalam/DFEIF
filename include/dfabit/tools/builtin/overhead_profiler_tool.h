#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/analysis/lightweight_fit.h"
#include "dfabit/analysis/overhead_engine.h"
#include "dfabit/analysis/scalability_runner.h"
#include "dfabit/core/status.h"
#include "dfabit/tools/tool.h"

namespace dfabit::tools::builtin {

class OverheadProfilerTool final : public Tool {
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

  dfabit::analysis::OverheadEngine overhead_engine_;
  dfabit::analysis::ScalabilityRunner scalability_runner_;
  std::vector<dfabit::adapters::MetricSample> compile_metrics_;
  std::vector<dfabit::adapters::MetricSample> load_metrics_;
  std::vector<dfabit::adapters::MetricSample> run_metrics_;
};

std::unique_ptr<Tool> CreateOverheadProfilerTool();

}  // namespace dfabit::tools::builtin