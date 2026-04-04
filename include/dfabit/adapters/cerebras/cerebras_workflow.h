#pragma once

#include <string>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/api/context.h"
#include "dfabit/core/status.h"
#include "dfabit/platform/process_runner.h"

namespace dfabit::adapters::cerebras {

class CerebrasWorkflow {
 public:
  dfabit::core::Status MaybeRunCompile(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts);

  dfabit::core::Status MaybeRunExecution(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts);

 private:
  static std::string OutputDir(const dfabit::api::Context& ctx);
  static std::string DetectPath(const dfabit::api::Context& ctx, const std::string& key);
  static void AppendProcessMetrics(
      const std::string& stage,
      const dfabit::platform::ProcessResult& result,
      std::vector<dfabit::adapters::MetricSample>* metrics);

  dfabit::platform::ProcessRunner runner_;
};

}  // namespace dfabit::adapters::cerebras