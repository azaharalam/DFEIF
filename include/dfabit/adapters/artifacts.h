#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dfabit::adapters {

enum class ArtifactKind {
  kUnknown = 0,
  kMlirModule = 1,
  kGraphIr = 2,
  kManifest = 3,
  kCompileReport = 4,
  kRuntimeLog = 5,
  kProfile = 6,
  kBinary = 7,
  kText = 8
};

struct ArtifactRef {
  ArtifactKind kind = ArtifactKind::kUnknown;
  std::string name;
  std::string path;
  std::string stage;
  std::unordered_map<std::string, std::string> attributes;
};

struct MetricSample {
  std::string name;
  double value = 0.0;
  std::string unit;
  std::string stage;
  std::uint64_t stable_id = 0;
  std::unordered_map<std::string, std::string> attributes;
};

struct CompileArtifactSet {
  std::vector<ArtifactRef> inputs;
  std::vector<ArtifactRef> outputs;
  std::vector<MetricSample> metrics;
};

struct RuntimeArtifactSet {
  std::vector<ArtifactRef> inputs;
  std::vector<ArtifactRef> outputs;
  std::vector<MetricSample> metrics;
};

struct AdapterHooks {
  bool enable_compile_hooks = true;
  bool enable_load_hooks = true;
  bool enable_run_hooks = true;
  bool enable_subgraph_hooks = true;
  bool enable_metric_collection = true;
};

}  // namespace dfabit::adapters