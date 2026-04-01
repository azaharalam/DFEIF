#pragma once

#include <string>

#include "dfabit/adapters/backend_adapter.h"
#include "dfabit/api/context.h"
#include "dfabit/core/status.h"

namespace dfabit::tools {

class Tool {
 public:
  virtual ~Tool() = default;

  virtual std::string name() const = 0;

  virtual dfabit::core::Status OnRegister(dfabit::api::Context* ctx) = 0;
  virtual dfabit::core::Status OnInit(dfabit::api::Context* ctx) = 0;
  virtual dfabit::core::Status OnShutdown(dfabit::api::Context* ctx) = 0;

  virtual dfabit::core::Status OnCompileBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::CompileArtifactSet& compile_artifacts) = 0;

  virtual dfabit::core::Status OnCompileEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::CompileArtifactSet& compile_artifacts) = 0;

  virtual dfabit::core::Status OnLoadBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) = 0;

  virtual dfabit::core::Status OnLoadEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) = 0;

  virtual dfabit::core::Status OnRunBegin(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) = 0;

  virtual dfabit::core::Status OnRunEnd(
      dfabit::api::Context* ctx,
      const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) = 0;
};

using ToolFactory = std::unique_ptr<Tool>(*)();

}  // namespace dfabit::tools