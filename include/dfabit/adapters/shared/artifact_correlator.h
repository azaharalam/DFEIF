#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"

namespace dfabit::adapters::shared {

struct CorrelatedArtifact {
  ArtifactRef artifact;
  std::uint64_t stable_id = 0;
  std::string symbol;
  std::string stage;
};

class ArtifactCorrelator {
 public:
  void RegisterSymbol(std::uint64_t stable_id, std::string symbol);
  void RegisterAlias(std::string alias, std::uint64_t stable_id);

  dfabit::core::Status CorrelateArtifacts(
      const std::vector<ArtifactRef>& artifacts,
      std::vector<CorrelatedArtifact>* correlated) const;

  std::uint64_t Resolve(const ArtifactRef& artifact) const;

 private:
  std::unordered_map<std::uint64_t, std::string> id_to_symbol_;
  std::unordered_map<std::string, std::uint64_t> alias_to_id_;
};

}  // namespace dfabit::adapters::shared