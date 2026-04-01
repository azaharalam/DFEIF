#include "dfabit/mlir/manifest_exporter.h"

#include <fstream>
#include <sstream>

namespace dfabit::mlir {

namespace {

std::string EscapeJson(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '\"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\b': o << "\\b"; break;
      case '\f': o << "\\f"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          o << "\\u00";
          static const char* hex = "0123456789abcdef";
          o << hex[(c >> 4) & 0xF] << hex[c & 0xF];
        } else {
          o << c;
        }
    }
  }
  return o.str();
}

void WriteShape(std::ostringstream& ss, const std::vector<std::int64_t>& shape) {
  ss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      ss << ",";
    }
    ss << shape[i];
  }
  ss << "]";
}

void WriteTensor(std::ostringstream& ss, const dfabit::metadata::TensorDesc& tensor) {
  ss << "{";
  ss << "\"name\":\"" << EscapeJson(tensor.name) << "\",";
  ss << "\"shape\":";
  WriteShape(ss, tensor.shape);
  ss << ",";
  ss << "\"dtype\":\"" << EscapeJson(tensor.dtype) << "\",";
  ss << "\"layout\":\"" << EscapeJson(tensor.layout) << "\"";
  ss << "}";
}

void WriteOp(std::ostringstream& ss, const dfabit::metadata::OpDesc& op) {
  ss << "{";
  ss << "\"stable_id\":" << op.stable_id << ",";
  ss << "\"op_name\":\"" << EscapeJson(op.op_name) << "\",";
  ss << "\"dialect\":\"" << EscapeJson(op.dialect) << "\",";
  ss << "\"stage_tag\":\"" << EscapeJson(op.stage_tag) << "\",";
  ss << "\"estimated_flops\":" << op.estimated_flops << ",";
  ss << "\"estimated_bytes\":" << op.estimated_bytes << ",";
  ss << "\"inputs\":[";
  for (std::size_t i = 0; i < op.inputs.size(); ++i) {
    if (i != 0) {
      ss << ",";
    }
    WriteTensor(ss, op.inputs[i]);
  }
  ss << "],";
  ss << "\"outputs\":[";
  for (std::size_t i = 0; i < op.outputs.size(); ++i) {
    if (i != 0) {
      ss << ",";
    }
    WriteTensor(ss, op.outputs[i]);
  }
  ss << "],";
  ss << "\"attributes\":{";
  bool first = true;
  for (const auto& kv : op.attributes) {
    if (!first) {
      ss << ",";
    }
    first = false;
    ss << "\"" << EscapeJson(kv.first) << "\":\"" << EscapeJson(kv.second) << "\"";
  }
  ss << "}";
  ss << "}";
}

}  // namespace

dfabit::core::Status MlirManifestExporter::WriteJson(
    const dfabit::metadata::ModelDesc& model,
    const std::string& path) const {
  std::string text;
  const auto st = ToJson(model, &text);
  if (!st.ok()) {
    return st;
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to write manifest: " + path};
  }
  ofs << text;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status MlirManifestExporter::ToJson(
    const dfabit::metadata::ModelDesc& model,
    std::string* out) const {
  if (!out) {
    return {dfabit::core::StatusCode::kInvalidArgument, "out is null"};
  }

  std::ostringstream ss;
  ss << "{";
  ss << "\"model_name\":\"" << EscapeJson(model.model_name) << "\",";
  ss << "\"backend_name\":\"" << EscapeJson(model.backend_name) << "\",";
  ss << "\"graph_name\":\"" << EscapeJson(model.graph_name) << "\",";
  ss << "\"ops\":[";
  for (std::size_t i = 0; i < model.ops.size(); ++i) {
    if (i != 0) {
      ss << ",";
    }
    WriteOp(ss, model.ops[i]);
  }
  ss << "]";
  ss << "}";
  *out = ss.str();
  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::mlir