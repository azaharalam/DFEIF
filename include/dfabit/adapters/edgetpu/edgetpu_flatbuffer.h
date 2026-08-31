#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dfabit/core/status.h"

namespace dfabit::adapters::edgetpu {

// Operator structure recovered from a TFLite model file.
//
// The Edge TPU compiler's report lists operator TYPES with counts and a mapping
// status, not individual operators, and the compiled model collapses to a
// single fused custom op that carries no structure at all. Neither is enough to
// reason about tensor lifetimes. The pre-compilation model still holds the
// graph as written, so that is what this reads.
//
// This parses the FlatBuffer directly rather than linking the TFLite runtime.
// Only the parts of the schema needed here are decoded: the operator list, the
// tensor table, and the buffer table. Going through the interpreter would also
// work, but it inserts DELEGATE pseudo-operators that are not in the model and
// reports tensor buffers for activations once tensors are allocated, both of
// which distort the counts.

struct TfLiteTensorInfo {
  std::string name;
  std::vector<std::int64_t> shape;
  std::string dtype;
  double bytes = 0.0;

  // True when the tensor has a non-empty backing buffer, which is what
  // distinguishes a weight from an activation. Asking an allocated interpreter
  // which tensors are readable does not work: it hands back zero-filled
  // buffers for activations too, which inflates the parameter total past the
  // size of the model file.
  bool is_constant = false;
};

struct TfLiteOpInfo {
  std::string op_name;
  std::vector<int> inputs;
  std::vector<int> outputs;

  // Multiply-accumulate count derived from tensor shapes. An operator whose
  // arithmetic cannot be inferred from shapes alone reports zero with
  // macs_determined false, rather than a guess: the Edge TPU's fused subgraph
  // operator is the main such case, and counting it as zero work would be as
  // wrong as inventing a figure for it.
  double macs = 0.0;
  bool macs_determined = false;
};

struct TfLiteGraph {
  std::vector<TfLiteOpInfo> ops;
  std::vector<TfLiteTensorInfo> tensors;
  double parameter_bytes = 0.0;
};

class TfLiteFlatBufferReader {
 public:
  // Reads the first subgraph of a .tflite model.
  dfabit::core::Status Read(const std::string& path, TfLiteGraph* graph) const;

  // Given a compiled model path, returns the path to the model it was compiled
  // from, or an empty string when that file is not present alongside it.
  static std::string DeriveSourceModelPath(const std::string& model_path);
};

}  // namespace dfabit::adapters::edgetpu
