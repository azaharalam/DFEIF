#include "dfabit/adapters/edgetpu/edgetpu_flatbuffer.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace dfabit::adapters::edgetpu {

namespace {

// A FlatBuffer table is preceded by a signed offset to its vtable. The vtable
// gives, for each field, the field's offset within the table, or zero when the
// field was not written. Reading is therefore always bounds-checked: a
// truncated or unexpected file yields zeros rather than reading past the end.
class ByteView {
 public:
  explicit ByteView(std::vector<std::uint8_t> data) : data_(std::move(data)) {}

  bool InBounds(std::size_t offset, std::size_t count) const {
    return offset + count <= data_.size();
  }
  std::uint8_t U8(std::size_t o) const {
    return InBounds(o, 1) ? data_[o] : 0;
  }
  std::uint16_t U16(std::size_t o) const {
    std::uint16_t v = 0;
    if (InBounds(o, 2)) std::memcpy(&v, &data_[o], 2);
    return v;
  }
  std::uint32_t U32(std::size_t o) const {
    std::uint32_t v = 0;
    if (InBounds(o, 4)) std::memcpy(&v, &data_[o], 4);
    return v;
  }
  std::int32_t I32(std::size_t o) const {
    std::int32_t v = 0;
    if (InBounds(o, 4)) std::memcpy(&v, &data_[o], 4);
    return v;
  }
  bool empty() const { return data_.empty(); }

 private:
  std::vector<std::uint8_t> data_;
};

std::size_t FieldOffset(const ByteView& b, std::size_t table, int field) {
  const std::int32_t soffset = b.I32(table);
  if (soffset == 0) return 0;
  const std::size_t vtable = table - static_cast<std::size_t>(soffset);
  const std::uint16_t vtable_bytes = b.U16(vtable);
  const std::size_t slot = 4 + static_cast<std::size_t>(field) * 2;
  if (slot >= vtable_bytes) return 0;
  const std::uint16_t relative = b.U16(vtable + slot);
  return relative ? table + relative : 0;
}

std::size_t Indirect(const ByteView& b, std::size_t pos) {
  return pos + b.U32(pos);
}

std::size_t VectorLength(const ByteView& b, std::size_t vec) {
  return vec ? b.U32(vec) : 0;
}

std::size_t VectorElement(std::size_t vec, std::size_t index, std::size_t stride) {
  return vec + 4 + index * stride;
}

std::string ReadString(const ByteView& b, std::size_t field) {
  if (!field) return std::string();
  const std::size_t s = Indirect(b, field);
  const std::uint32_t n = b.U32(s);
  std::string out;
  out.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out += static_cast<char>(b.U8(s + 4 + i));
  }
  return out;
}

struct DtypeInfo {
  const char* name;
  std::size_t bytes;
};

DtypeInfo Dtype(int code) {
  switch (code) {
    case 0:  return {"float32", 4};
    case 1:  return {"float16", 2};
    case 2:  return {"int32", 4};
    case 3:  return {"uint8", 1};
    case 4:  return {"int64", 8};
    case 5:  return {"string", 1};
    case 6:  return {"bool", 1};
    case 7:  return {"int16", 2};
    case 8:  return {"complex64", 8};
    case 9:  return {"int8", 1};
    case 10: return {"float64", 8};
    case 11: return {"complex128", 16};
    case 12: return {"uint64", 8};
    case 15: return {"uint32", 4};
    case 16: return {"uint16", 2};
    default: return {"unknown", 1};
  }
}

// Builtin operator codes, in schema order. A code outside this range is
// reported as UNKNOWN rather than guessed at; custom operators carry their own
// name string and are read from there instead.
const char* const kBuiltinOperators[] = {
    "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D",
    "DEPTH_TO_SPACE", "DEQUANTIZE", "EMBEDDING_LOOKUP", "FLOOR",
    "FULLY_CONNECTED", "HASHTABLE_LOOKUP", "L2_NORMALIZATION", "L2_POOL_2D",
    "LOCAL_RESPONSE_NORMALIZATION", "LOGISTIC", "LSH_PROJECTION", "LSTM",
    "MAX_POOL_2D", "MUL", "RELU", "RELU_N1_TO_1", "RELU6", "RESHAPE",
    "RESIZE_BILINEAR", "RNN", "SOFTMAX", "SPACE_TO_DEPTH", "SVDF", "TANH",
    "CONCAT_EMBEDDINGS", "SKIP_GRAM", "CALL", "CUSTOM", "EMBEDDING_LOOKUP_SPARSE",
    "PAD", "UNIDIRECTIONAL_SEQUENCE_RNN", "GATHER", "BATCH_TO_SPACE_ND",
    "SPACE_TO_BATCH_ND", "TRANSPOSE", "MEAN", "SUB", "DIV", "SQUEEZE",
    "UNIDIRECTIONAL_SEQUENCE_LSTM", "STRIDED_SLICE",
    "BIDIRECTIONAL_SEQUENCE_RNN", "EXP", "TOPK_V2", "SPLIT", "LOG_SOFTMAX",
    "DELEGATE", "BIDIRECTIONAL_SEQUENCE_LSTM", "CAST", "PRELU", "MAXIMUM",
    "ARG_MAX", "MINIMUM", "LESS", "NEG", "PADV2", "GREATER", "GREATER_EQUAL",
    "LESS_EQUAL", "SELECT", "SLICE", "SIN", "TRANSPOSE_CONV", "SPARSE_TO_DENSE",
    "TILE", "EXPAND_DIMS", "EQUAL", "NOT_EQUAL", "LOG", "SUM", "SQRT", "RSQRT",
    "SHAPE", "POW", "ARG_MIN", "FAKE_QUANT", "REDUCE_PROD", "REDUCE_MAX",
    "PACK", "LOGICAL_OR", "ONE_HOT", "LOGICAL_AND", "LOGICAL_NOT", "UNPACK",
    "REDUCE_MIN", "FLOOR_DIV", "REDUCE_ANY", "SQUARE", "ZEROS_LIKE", "FILL",
    "FLOOR_MOD", "RANGE", "RESIZE_NEAREST_NEIGHBOR", "LEAKY_RELU",
    "SQUARED_DIFFERENCE", "MIRROR_PAD", "ABS", "SPLIT_V", "UNIQUE", "CEIL",
    "REVERSE_V2", "ADD_N", "GATHER_ND", "COS", "WHERE", "RANK", "ELU",
    "REVERSE_SEQUENCE", "MATRIX_DIAG", "QUANTIZE", "MATRIX_SET_DIAG", "ROUND",
    "HARD_SWISH", "IF", "WHILE", "NON_MAX_SUPPRESSION_V4",
    "NON_MAX_SUPPRESSION_V5", "SCATTER_ND", "SELECT_V2", "DENSIFY",
    "SEGMENT_SUM", "BATCH_MATMUL"};

constexpr std::size_t kBuiltinCount =
    sizeof(kBuiltinOperators) / sizeof(kBuiltinOperators[0]);


double Numel(const std::vector<std::int64_t>& shape) {
  double n = 1.0;
  for (const auto d : shape) {
    if (d <= 0) return 0.0;
    n *= static_cast<double>(d);
  }
  return n;
}

// MACs for one operator from its tensor shapes. Operators that move or reshape
// data do no arithmetic and report zero as a determined result. Operators whose
// cost depends on values rather than shapes, and the fused Edge TPU subgraph
// operator, report zero undetermined.
double ComputeMacs(const std::string& op_name,
                   const std::vector<TfLiteTensorInfo>& tensors,
                   const std::vector<int>& inputs,
                   const std::vector<int>& outputs,
                   bool* determined) {
  *determined = false;

  const auto shape_of = [&](std::size_t which,
                            const std::vector<int>& idx) -> std::vector<std::int64_t> {
    if (which >= idx.size()) return {};
    const int t = idx[which];
    if (t < 0 || static_cast<std::size_t>(t) >= tensors.size()) return {};
    return tensors[t].shape;
  };

  static const char* kZeroMac[] = {
      "RESHAPE", "TRANSPOSE", "PAD", "PADV2", "CONCATENATION", "SPLIT",
      "SPLIT_V", "SLICE", "STRIDED_SLICE", "SQUEEZE", "EXPAND_DIMS", "PACK",
      "UNPACK", "GATHER", "CAST", "QUANTIZE", "DEQUANTIZE", "SHAPE",
      "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR", "MAX_POOL_2D",
      "AVERAGE_POOL_2D", "RELU", "RELU6", "LOGISTIC", "TANH", "SOFTMAX",
      "ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM", "MEAN", "SUM",
      "HARD_SWISH", "L2_NORMALIZATION"};
  for (const auto* z : kZeroMac) {
    if (op_name == z) {
      *determined = true;
      return 0.0;
    }
  }

  const auto out = shape_of(0, outputs);
  if (out.empty()) return 0.0;

  if (op_name == "FULLY_CONNECTED") {
    const auto w = shape_of(1, inputs);
    if (w.size() >= 2 && w[0] > 0) {
      const double units = static_cast<double>(w[0]);
      const double in_features = static_cast<double>(w[1]);
      const double batch = Numel(out) / units;
      *determined = true;
      return batch * units * in_features;
    }
    return 0.0;
  }

  if (op_name == "BATCH_MATMUL") {
    const auto a = shape_of(0, inputs);
    if (a.size() >= 2) {
      *determined = true;
      return Numel(out) * static_cast<double>(a.back());
    }
    return 0.0;
  }

  if (op_name == "CONV_2D" || op_name == "TRANSPOSE_CONV") {
    // weights [out_ch, kh, kw, in_ch]; one MAC per output element per tap.
    const auto w = shape_of(1, inputs);
    if (w.size() == 4) {
      *determined = true;
      return Numel(out) * static_cast<double>(w[1]) *
             static_cast<double>(w[2]) * static_cast<double>(w[3]);
    }
    return 0.0;
  }

  if (op_name == "DEPTHWISE_CONV_2D") {
    // weights [1, kh, kw, channels]; each output element sees kh*kw taps.
    const auto w = shape_of(1, inputs);
    if (w.size() == 4) {
      *determined = true;
      return Numel(out) * static_cast<double>(w[1]) *
             static_cast<double>(w[2]);
    }
    return 0.0;
  }

  return 0.0;
}
}  // namespace

dfabit::core::Status TfLiteFlatBufferReader::Read(
    const std::string& path,
    TfLiteGraph* graph) const {
  if (!graph) {
    return {dfabit::core::StatusCode::kInvalidArgument, "graph is null"};
  }
  *graph = TfLiteGraph{};

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    return {dfabit::core::StatusCode::kNotFound,
            "failed to open tflite model: " + path};
  }
  ByteView buf(std::vector<std::uint8_t>(
      (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()));
  if (buf.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "empty model: " + path};
  }

  // Model table: field 1 operator_codes, field 2 subgraphs, field 4 buffers.
  const std::size_t root = Indirect(buf, 0);
  const std::size_t opcodes_field = FieldOffset(buf, root, 1);
  const std::size_t subgraphs_field = FieldOffset(buf, root, 2);
  const std::size_t buffers_field = FieldOffset(buf, root, 4);
  if (!subgraphs_field) {
    return {dfabit::core::StatusCode::kInvalidArgument,
            "no subgraphs in model: " + path};
  }
  const std::size_t opcodes = opcodes_field ? Indirect(buf, opcodes_field) : 0;
  const std::size_t subgraphs = Indirect(buf, subgraphs_field);
  const std::size_t buffers = buffers_field ? Indirect(buf, buffers_field) : 0;

  std::vector<std::string> operator_names;
  for (std::size_t i = 0; i < VectorLength(buf, opcodes); ++i) {
    const std::size_t oc = Indirect(buf, VectorElement(opcodes, i, 4));
    int code = 0;
    if (const std::size_t f = FieldOffset(buf, oc, 3)) {
      code = buf.I32(f);
    } else if (const std::size_t legacy = FieldOffset(buf, oc, 0)) {
      code = static_cast<std::int8_t>(buf.U8(legacy));
    }
    std::string name =
        (code >= 0 && static_cast<std::size_t>(code) < kBuiltinCount)
            ? kBuiltinOperators[code]
            : "UNKNOWN";
    if (name == "CUSTOM") {
      const auto custom = ReadString(buf, FieldOffset(buf, oc, 1));
      if (!custom.empty()) name = custom;
    }
    operator_names.push_back(std::move(name));
  }

  std::vector<std::size_t> buffer_sizes;
  for (std::size_t i = 0; i < VectorLength(buf, buffers); ++i) {
    const std::size_t bt = Indirect(buf, VectorElement(buffers, i, 4));
    const std::size_t data = FieldOffset(buf, bt, 0);
    buffer_sizes.push_back(data ? buf.U32(Indirect(buf, data)) : 0);
  }

  // Only the first subgraph is read. TFLite models produced by the converters
  // used here have exactly one; control-flow operators can introduce more, and
  // those are out of scope.
  if (VectorLength(buf, subgraphs) == 0) {
    return {dfabit::core::StatusCode::kInvalidArgument,
            "model has no subgraph: " + path};
  }
  const std::size_t sg = Indirect(buf, VectorElement(subgraphs, 0, 4));
  const std::size_t tensors_field = FieldOffset(buf, sg, 0);
  const std::size_t ops_field = FieldOffset(buf, sg, 3);
  const std::size_t tensors = tensors_field ? Indirect(buf, tensors_field) : 0;
  const std::size_t ops = ops_field ? Indirect(buf, ops_field) : 0;

  for (std::size_t i = 0; i < VectorLength(buf, tensors); ++i) {
    const std::size_t t = Indirect(buf, VectorElement(tensors, i, 4));
    TfLiteTensorInfo info;

    if (const std::size_t shape_field = FieldOffset(buf, t, 0)) {
      const std::size_t shape = Indirect(buf, shape_field);
      for (std::size_t k = 0; k < VectorLength(buf, shape); ++k) {
        info.shape.push_back(buf.I32(VectorElement(shape, k, 4)));
      }
    }

    int type_code = 0;
    if (const std::size_t type_field = FieldOffset(buf, t, 1)) {
      type_code = buf.U8(type_field);
    }
    const auto dtype = Dtype(type_code);
    info.dtype = dtype.name;

    std::uint32_t buffer_index = 0;
    if (const std::size_t buffer_field = FieldOffset(buf, t, 2)) {
      buffer_index = buf.U32(buffer_field);
    }
    info.is_constant = buffer_index < buffer_sizes.size() &&
                       buffer_sizes[buffer_index] > 0;

    info.name = ReadString(buf, FieldOffset(buf, t, 3));
    if (info.name.empty()) {
      info.name = "tensor_" + std::to_string(i);
    }

    double elements = 1.0;
    for (const auto dim : info.shape) {
      if (dim <= 0) {
        elements = 0.0;
        break;
      }
      elements *= static_cast<double>(dim);
    }
    info.bytes = elements * static_cast<double>(dtype.bytes);

    if (info.is_constant) {
      graph->parameter_bytes += info.bytes;
    }
    graph->tensors.push_back(std::move(info));
  }

  for (std::size_t i = 0; i < VectorLength(buf, ops); ++i) {
    const std::size_t o = Indirect(buf, VectorElement(ops, i, 4));
    TfLiteOpInfo op;

    std::uint32_t opcode_index = 0;
    if (const std::size_t f = FieldOffset(buf, o, 0)) {
      opcode_index = buf.U32(f);
    }
    op.op_name = opcode_index < operator_names.size()
                     ? operator_names[opcode_index]
                     : "UNKNOWN";

    if (const std::size_t f = FieldOffset(buf, o, 1)) {
      const std::size_t v = Indirect(buf, f);
      for (std::size_t k = 0; k < VectorLength(buf, v); ++k) {
        op.inputs.push_back(buf.I32(VectorElement(v, k, 4)));
      }
    }
    if (const std::size_t f = FieldOffset(buf, o, 2)) {
      const std::size_t v = Indirect(buf, f);
      for (std::size_t k = 0; k < VectorLength(buf, v); ++k) {
        op.outputs.push_back(buf.I32(VectorElement(v, k, 4)));
      }
    }
    bool determined = false;
    op.macs = ComputeMacs(op.op_name, graph->tensors, op.inputs, op.outputs,
                          &determined);
    op.macs_determined = determined;

    graph->ops.push_back(std::move(op));
  }

  if (graph->ops.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument,
            "no operators recovered from: " + path};
  }
  return dfabit::core::Status::Ok();
}

std::string TfLiteFlatBufferReader::DeriveSourceModelPath(
    const std::string& model_path) {
  if (model_path.empty()) return std::string();

  const std::filesystem::path p(model_path);
  auto stem = p.stem().string();

  const std::string suffix = "_edgetpu";
  if (stem.size() > suffix.size() &&
      stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
    stem = stem.substr(0, stem.size() - suffix.size());
  } else {
    // Already the pre-compilation model.
    std::error_code ec;
    return std::filesystem::is_regular_file(model_path, ec) ? model_path
                                                            : std::string();
  }

  const auto candidate = (p.parent_path() / (stem + ".tflite")).string();
  std::error_code ec;
  return std::filesystem::is_regular_file(candidate, ec) ? candidate
                                                         : std::string();
}

}  // namespace dfabit::adapters::edgetpu
