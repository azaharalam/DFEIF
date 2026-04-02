#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <cstdint>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/adapters/capabilities.h"
#include "dfabit/api/context.h"
#include "dfabit/core/status.h"

namespace dfabit::adapters {

class BackendAdapter {
 public:
  virtual ~BackendAdapter() = default;

  virtual std::string name() const = 0;
  virtual std::string provider() const = 0;

  virtual dfabit::core::Status InitializeSession(dfabit::api::Context* ctx) = 0;
  virtual AdapterCapabilities DiscoverCapabilities(const dfabit::api::Context& ctx) const = 0;

  virtual dfabit::core::Status PrepareArtifacts(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts,
      RuntimeArtifactSet* runtime_artifacts) = 0;

  virtual dfabit::core::Status LoadManifest(
      dfabit::api::Context* ctx,
      const ArtifactRef& manifest_artifact) = 0;

  virtual dfabit::core::Status CompileBegin(
      dfabit::api::Context* ctx,
      const CompileArtifactSet& compile_artifacts) = 0;

  virtual dfabit::core::Status CompileEnd(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts) = 0;

  virtual dfabit::core::Status LoadBegin(
      dfabit::api::Context* ctx,
      const RuntimeArtifactSet& runtime_artifacts) = 0;

  virtual dfabit::core::Status LoadEnd(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) = 0;

  virtual dfabit::core::Status RunBegin(
      dfabit::api::Context* ctx,
      const RuntimeArtifactSet& runtime_artifacts) = 0;

  virtual dfabit::core::Status RunEnd(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) = 0;

  virtual dfabit::core::Status SubgraphBegin(
      dfabit::api::Context* ctx,
      std::string subgraph_name,
      std::uint64_t stable_id) = 0;

  virtual dfabit::core::Status SubgraphEnd(
      dfabit::api::Context* ctx,
      std::string subgraph_name,
      std::uint64_t stable_id) = 0;

  virtual dfabit::core::Status CollectRuntimeMetrics(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) = 0;

  virtual dfabit::core::Status Shutdown(dfabit::api::Context* ctx) = 0;
};

using BackendAdapterFactory = std::unique_ptr<BackendAdapter>(*)();

}  // namespace dfabit::adapters