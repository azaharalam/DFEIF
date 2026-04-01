#include "dfabit/mlir/semantic_tagger.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string_view>
#include <utility>

namespace dfabit::mlir {

namespace {

std::string Trim(std::string s) {
  std::size_t begin = 0;
  while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
    ++begin;
  }
  std::size_t end = s.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(begin, end - begin);
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::vector<std::string> Split(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    out.push_back(item);
  }
  return out;
}

std::vector<std::int64_t> ParseShapeToken(const std::string& token) {
  const auto lt = token.find('<');
  const auto gt = token.rfind('>');
  if (lt == std::string::npos || gt == std::string::npos || gt <= lt + 1) {
    return {};
  }

  const auto inside = token.substr(lt + 1, gt - lt - 1);
  const auto parts = Split(inside, 'x');
  if (parts.size() < 2) {
    return {};
  }

  std::vector<std::int64_t> shape;
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    const auto piece = Trim(parts[i]);
    if (piece.empty()) {
      continue;
    }
    if (piece == "?") {
      shape.push_back(-1);
      continue;
    }

    bool numeric = std::all_of(piece.begin(), piece.end(), [](unsigned char c) {
      return std::isdigit(c);
    });
    if (!numeric) {
      continue;
    }
    shape.push_back(static_cast<std::int64_t>(std::stoll(piece)));
  }
  return shape;
}

std::string ParseDTypeToken(const std::string& token) {
  const auto lt = token.find('<');
  const auto gt = token.rfind('>');
  if (lt == std::string::npos || gt == std::string::npos || gt <= lt + 1) {
    return {};
  }

  const auto inside = token.substr(lt + 1, gt - lt - 1);
  const auto parts = Split(inside, 'x');
  if (parts.empty()) {
    return {};
  }
  return Trim(parts.back());
}

}  // namespace

dfabit::core::Status MlirSemanticTagger::BuildModelDescription(
    const MlirModuleSnapshot& snapshot,
    const std::string& backend_name,
    dfabit::metadata::ModelDesc* model) const {
  if (!model) {
    return {dfabit::core::StatusCode::kInvalidArgument, "model is null"};
  }

  model->backend_name = backend_name;
  model->graph_name = ExtractFunctionName(snapshot.text);
  model->model_name = snapshot.module_name;
  model->ops.clear();

  for (const auto& raw_line : snapshot.lines) {
    const auto line = Trim(raw_line);
    if (line.empty()) {
      continue;
    }
    if (StartsWith(line, "module") || StartsWith(line, "func.func") || StartsWith(line, "return")) {
      continue;
    }
    if (line.find('"') == std::string::npos) {
      continue;
    }

    dfabit::metadata::OpDesc op;
    op.op_name = ExtractOpName(line);
    if (op.op_name.empty()) {
      continue;
    }

    op.dialect = InferDialect(op.op_name);
    op.stage_tag = snapshot.stage;

    dfabit::metadata::TensorDesc input;
    input.name = "input0";
    input.shape = ExtractShape(line);
    input.dtype = ExtractDType(line);
    input.layout = "unknown";

    dfabit::metadata::TensorDesc output = input;
    output.name = "output0";

    if (!input.shape.empty() || !input.dtype.empty()) {
      op.inputs.push_back(input);
      op.outputs.push_back(output);
    }

    op.attributes["mlir_text"] = line;
    model->ops.push_back(std::move(op));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status MlirSemanticTagger::TagStableIds(
    dfabit::metadata::ModelDesc* model) const {
  if (!model) {
    return {dfabit::core::StatusCode::kInvalidArgument, "model is null"};
  }

  dfabit::metadata::StableIdAssigner assigner;
  for (auto& op : model->ops) {
    op.stable_id = assigner.Assign(op);
  }
  return dfabit::core::Status::Ok();
}

std::string MlirSemanticTagger::ExtractOpName(const std::string& line) {
  const auto first = line.find('"');
  if (first == std::string::npos) {
    return {};
  }
  const auto second = line.find('"', first + 1);
  if (second == std::string::npos || second <= first + 1) {
    return {};
  }
  return line.substr(first + 1, second - first - 1);
}

std::string MlirSemanticTagger::InferDialect(const std::string& op_name) {
  const auto pos = op_name.find('.');
  return pos == std::string::npos ? std::string("builtin") : op_name.substr(0, pos);
}

std::vector<std::int64_t> MlirSemanticTagger::ExtractShape(const std::string& line) {
  const std::string tensor_tag = "tensor<";
  const auto pos = line.find(tensor_tag);
  if (pos == std::string::npos) {
    return {};
  }
  const auto end = line.find('>', pos + tensor_tag.size());
  if (end == std::string::npos) {
    return {};
  }
  return ParseShapeToken(line.substr(pos, end - pos + 1));
}

std::string MlirSemanticTagger::ExtractDType(const std::string& line) {
  const std::string tensor_tag = "tensor<";
  const auto pos = line.find(tensor_tag);
  if (pos == std::string::npos) {
    return {};
  }
  const auto end = line.find('>', pos + tensor_tag.size());
  if (end == std::string::npos) {
    return {};
  }
  return ParseDTypeToken(line.substr(pos, end - pos + 1));
}

std::string MlirSemanticTagger::ExtractFunctionName(const std::string& text) {
  const auto pos = text.find("func.func");
  if (pos == std::string::npos) {
    return "main";
  }
  const auto at = text.find('@', pos);
  if (at == std::string::npos) {
    return "main";
  }
  const auto open = text.find('(', at);
  if (open == std::string::npos || open <= at + 1) {
    return "main";
  }
  return text.substr(at + 1, open - at - 1);
}

}  // namespace dfabit::mlir