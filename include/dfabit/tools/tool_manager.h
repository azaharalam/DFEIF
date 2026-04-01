#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/api/context.h"
#include "dfabit/core/status.h"
#include "dfabit/tools/tool.h"

namespace dfabit::tools {

class ToolManager {
 public:
  ToolManager() = default;

  dfabit::core::Status AddTool(std::unique_ptr<Tool> tool);
  std::size_t size() const { return tools_.size(); }
  bool empty() const { return tools_.empty(); }

  dfabit::core::Status OnRegister(dfabit::api::Context* ctx);
  dfabit::core::Status OnInit(dfabit::api::Context* ctx);
  dfabit::core::Status OnShutdown(dfabit::api::Context* ctx);

  dfabit::core::Status OnCompileBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::CompileArtifactSet& compile_artifacts);

  dfabit::core::Status OnCompileEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::CompileArtifactSet& compile_artifacts);

  dfabit::core::Status OnLoadBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts);

  dfabit::core::Status OnLoadEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts);

  dfabit::core::Status OnRunBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts);

  dfabit::core::Status OnRunEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts);

 private:
  template <typename Fn>
  dfabit::core::Status FanOut(const std::string& phase, Fn&& fn);

  std::vector<std::unique_ptr<Tool>> tools_;
};

}  // namespace dfabit::tools