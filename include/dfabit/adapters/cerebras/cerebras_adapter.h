#pragma once

#include <string>
#include <vector>

#include "dfabit/adapters/backend_adapter.h"
#include "dfabit/adapters/shared/artifact_correlator.h"
#include "dfabit/adapters/shared/compile_report_parser.h"
#include "dfabit/adapters/shared/runtime_log_parser.h"
#include "dfabit/hidden_ir/graph_importer.h"
#include "dfabit/hidden_ir/partition_tracker.h"
#include "dfabit/hidden_ir/sidecar_loader.h"
#include "dfabit/runtime/runtime_correlation.h"

namespace dfabit::adapters::cerebras {

class CerebrasAdapter final : public BackendAdapter {
 public:
  CerebrasAdapter();

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
  dfabit::core::Status BuildModel(
      dfabit::api::Context* ctx,
      const std::string& graph_path,
      const std::string& stage);

  void IndexModel();

  std::string OutputPath(const dfabit::api::Context& ctx, const std::string& file) const;

  dfabit::hidden_ir::GraphImporter graph_importer_;
  dfabit::hidden_ir::SidecarLoader sidecar_loader_;
  dfabit::hidden_ir::PartitionTracker partition_tracker_;
  dfabit::adapters::shared::CompileReportParser compile_report_parser_;
  dfabit::adapters::shared::RuntimeLogParser runtime_log_parser_;
  dfabit::adapters::shared::ArtifactCorrelator artifact_correlator_;
  dfabit::runtime::RuntimeCorrelator runtime_correlator_;
  dfabit::metadata::ModelDesc model_;
};

std::unique_ptr<BackendAdapter> CreateCerebrasAdapter();

}  // namespace dfabit::adapters::cerebras