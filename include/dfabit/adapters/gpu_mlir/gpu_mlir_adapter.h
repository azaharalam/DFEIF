#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dfabit/adapters/backend_adapter.h"
#include "dfabit/adapters/shared/artifact_correlator.h"
#include "dfabit/mlir/cost_model.h"
#include "dfabit/mlir/manifest_exporter.h"
#include "dfabit/mlir/module_instrumentor.h"
#include "dfabit/mlir/module_loader.h"
#include "dfabit/mlir/semantic_tagger.h"
#include "dfabit/runtime/launch_observer.h"
#include "dfabit/runtime/runtime_correlation.h"

namespace dfabit::adapters::gpu_mlir {

class GpuMlirAdapter final : public BackendAdapter {
 public:
  GpuMlirAdapter();

  std::string name() const override;
  std::string provider() const override;

  dfabit::core::Status InitializeSession(dfabit::api::Context* ctx) override;
  AdapterCapabilities DiscoverCapabilities(const dfabit::api::Context& ctx) const override;

  dfabit::core::Status PrepareArtifacts(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status LoadManifest(
      dfabit::api::Context* ctx,
      const ArtifactRef& manifest_artifact) override;

  dfabit::core::Status CompileBegin(
      dfabit::api::Context* ctx,
      const CompileArtifactSet& compile_artifacts) override;

  dfabit::core::Status CompileEnd(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts) override;

  dfabit::core::Status LoadBegin(
      dfabit::api::Context* ctx,
      const RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status LoadEnd(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status RunBegin(
      dfabit::api::Context* ctx,
      const RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status RunEnd(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status SubgraphBegin(
      dfabit::api::Context* ctx,
      std::string subgraph_name,
      std::uint64_t stable_id) override;

  dfabit::core::Status SubgraphEnd(
      dfabit::api::Context* ctx,
      std::string subgraph_name,
      std::uint64_t stable_id) override;

  dfabit::core::Status CollectRuntimeMetrics(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status Shutdown(dfabit::api::Context* ctx) override;

 private:
  dfabit::core::Status BuildModelFromMlir(
      dfabit::api::Context* ctx,
      const std::string& mlir_path,
      const std::string& stage,
      dfabit::metadata::ModelDesc* model,
      dfabit::mlir::MlirModuleSnapshot* snapshot) const;

  dfabit::core::Status WriteInstrumentedSnapshots(
      const dfabit::api::Context& ctx,
      const dfabit::mlir::MlirModuleSnapshot& snapshot,
      const dfabit::metadata::ModelDesc& model,
      CompileArtifactSet* compile_artifacts) const;

  void IndexModel(const dfabit::metadata::ModelDesc& model);
  std::string OutputPath(const dfabit::api::Context& ctx, const std::string& filename) const;

  dfabit::mlir::MlirModuleLoader loader_;
  dfabit::mlir::MlirSemanticTagger tagger_;
  dfabit::mlir::MlirManifestExporter manifest_exporter_;
  dfabit::mlir::MlirModuleInstrumentor module_instrumentor_;
  dfabit::mlir::MlirCostModel cost_model_;
  dfabit::runtime::RuntimeCorrelator runtime_correlator_;
  dfabit::runtime::LaunchObserver launch_observer_;
  dfabit::adapters::shared::ArtifactCorrelator artifact_correlator_;
  dfabit::metadata::ModelDesc model_;
};

std::unique_ptr<BackendAdapter> CreateGpuMlirAdapter();

}  // namespace dfabit::adapters::gpu_mlir