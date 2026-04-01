#include "dfabit/tools/tool_manager.h"

#include <utility>

namespace dfabit::tools {

dfabit::core::Status ToolManager::AddTool(std::unique_ptr<Tool> tool) {
  if (!tool) {
    return {dfabit::core::StatusCode::kInvalidArgument, "tool is null"};
  }
  tools_.push_back(std::move(tool));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ToolManager::OnRegister(dfabit::api::Context* ctx) {
  return FanOut("register", [ctx](Tool* tool) { return tool->OnRegister(ctx); });
}

dfabit::core::Status ToolManager::OnInit(dfabit::api::Context* ctx) {
  return FanOut("init", [ctx](Tool* tool) { return tool->OnInit(ctx); });
}

dfabit::core::Status ToolManager::OnShutdown(dfabit::api::Context* ctx) {
  return FanOut("shutdown", [ctx](Tool* tool) { return tool->OnShutdown(ctx); });
}

dfabit::core::Status ToolManager::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  return FanOut("compile_begin", [ctx, &compile_artifacts](Tool* tool) {
    return tool->OnCompileBegin(ctx, compile_artifacts);
  });
}

dfabit::core::Status ToolManager::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  return FanOut("compile_end", [ctx, &compile_artifacts](Tool* tool) {
    return tool->OnCompileEnd(ctx, compile_artifacts);
  });
}

dfabit::core::Status ToolManager::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  return FanOut("load_begin", [ctx, &runtime_artifacts](Tool* tool) {
    return tool->OnLoadBegin(ctx, runtime_artifacts);
  });
}

dfabit::core::Status ToolManager::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  return FanOut("load_end", [ctx, &runtime_artifacts](Tool* tool) {
    return tool->OnLoadEnd(ctx, runtime_artifacts);
  });
}

dfabit::core::Status ToolManager::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  return FanOut("run_begin", [ctx, &runtime_artifacts](Tool* tool) {
    return tool->OnRunBegin(ctx, runtime_artifacts);
  });
}

dfabit::core::Status ToolManager::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  return FanOut("run_end", [ctx, &runtime_artifacts](Tool* tool) {
    return tool->OnRunEnd(ctx, runtime_artifacts);
  });
}

template <typename Fn>
dfabit::core::Status ToolManager::FanOut(const std::string& phase, Fn&& fn) {
  for (const auto& tool : tools_) {
    const auto st = fn(tool.get());
    if (!st.ok()) {
      return {
          st.code(),
          "tool phase failed [" + phase + "] for tool=" + tool->name() + ": " + st.message()};
    }
  }
  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::tools