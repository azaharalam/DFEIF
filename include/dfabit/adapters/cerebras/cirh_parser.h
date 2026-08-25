#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"

namespace dfabit::adapters::cerebras {

// One operator recovered from a Cerebras cirh.mlir dump.
//
// Cerebras is the only backend here that emits a real compiler IR, and its
// MLIR carries more than shapes: every operator references an MLIR `loc()`
// that resolves to the originating module path, the ATen operator it came
// from, and the Python callsite that produced it. That is transformation
// lineage stated by the compiler, not reconstructed by us.
struct CirhOp {
  std::string ssa;          // %681
  std::string op_name;      // MatMul
  std::string aten_op;      // aten::mm
  std::string module_path;  // model.vit_model...proj_q_dense_layer
  std::string phase;        // forward | backward | optimizer
  std::string shape;        // 561450x768
  std::string dtype;        // f16
  std::string src_file;     // torch/nn/modules/linear.py
  std::string src_line;     // 116
  double macs = 0.0;
  double total_bytes = 0.0;
  bool macs_determined = false;
  std::uint64_t stable_id = 0;
};

// Parses cirh.mlir. Operators whose arithmetic is not determined by their
// shapes are reported with macs=0 and macs_determined=false rather than
// guessed at, so an undercount is visible instead of silent.
class CirhParser {
 public:
  dfabit::core::Status ParseFile(
      const std::string& path,
      std::vector<CirhOp>* ops) const;

  dfabit::core::Status ParseText(
      const std::string& text,
      std::vector<CirhOp>* ops) const;

  // Builds the semantic model DFIT's tools consume.
  static dfabit::metadata::ModelDesc ToModel(
      const std::vector<CirhOp>& ops,
      const std::string& graph_name);

  // Aggregate metrics: op counts and MAC/byte totals, split by phase so a
  // training graph's forward cost stays separable from backward and optimizer.
  static std::vector<MetricSample> ToMetricSamples(
      const std::vector<CirhOp>& ops);

  // Locates the newest cirh.mlir beneath a model_dir written by cszoo, which
  // lays them out as cerebras_logs/<timestamp>/executors/<n>/cirh.mlir.
  static std::string FindNewestCirh(const std::string& model_dir);

  // Every cirh.mlir beneath a model_dir, sorted by path.
  //
  // One `cszoo fit` invocation emits several executors -- typically a training
  // graph and a separate eval graph -- each with its own IR. Picking one by
  // modification time is arbitrary and silently discards the rest, so callers
  // are given all of them and attribute metrics per executor. Summing across
  // executors would double-count the forward pass, since each contains its own
  // copy of it.
  static std::vector<std::string> FindAllCirh(const std::string& model_dir);
};

}  // namespace dfabit::adapters::cerebras
