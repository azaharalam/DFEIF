#include "dfabit/adapters/shared/artifact_correlator.h"

#include <filesystem>
#include <utility>

namespace dfabit::adapters::shared {

void ArtifactCorrelator::RegisterSymbol(std::uint64_t stable_id, std::string symbol) {
  if (symbol.empty()) {
    return;
  }
  id_to_symbol_[stable_id] = symbol;
  alias_to_id_[symbol] = stable_id;
}

void ArtifactCorrelator::RegisterAlias(std::string alias, std::uint64_t stable_id) {
  if (alias.empty()) {
    return;
  }
  alias_to_id_[std::move(alias)] = stable_id;
}

dfabit::core::Status ArtifactCorrelator::CorrelateArtifacts(
    const std::vector<ArtifactRef>& artifacts,
    std::vector<CorrelatedArtifact>* correlated) const {
  if (!correlated) {
    return {dfabit::core::StatusCode::kInvalidArgument, "correlated is null"};
  }

  correlated->clear();
  correlated->reserve(artifacts.size());

  for (const auto& artifact : artifacts) {
    CorrelatedArtifact item;
    item.artifact = artifact;
    item.stable_id = Resolve(artifact);
    item.stage = artifact.stage;
    const auto it = id_to_symbol_.find(item.stable_id);
    if (it != id_to_symbol_.end()) {
      item.symbol = it->second;
    }
    correlated->push_back(std::move(item));
  }

  return dfabit::core::Status::Ok();
}

std::uint64_t ArtifactCorrelator::Resolve(const ArtifactRef& artifact) const {
  const auto attr_it = artifact.attributes.find("stable_id");
  if (attr_it != artifact.attributes.end()) {
    try {
      return static_cast<std::uint64_t>(std::stoull(attr_it->second));
    } catch (...) {
    }
  }

  const auto name_it = alias_to_id_.find(artifact.name);
  if (name_it != alias_to_id_.end()) {
    return name_it->second;
  }

  const auto path_name = std::filesystem::path(artifact.path).filename().string();
  const auto path_it = alias_to_id_.find(path_name);
  if (path_it != alias_to_id_.end()) {
    return path_it->second;
  }

  return 0;
}

}  // namespace dfabit::adapters::shared