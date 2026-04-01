#include "dfabit/hidden_ir/graph_importer.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>

namespace dfabit::hidden_ir {

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

std::vector<std::string> Split(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    out.push_back(Trim(item));
  }
  return out;
}

std::vector<std::int64_t> ParseShape(const std::string& s) {
  std::vector<std::int64_t> shape;
  if (s.empty()) {
    return shape;
  }
  for (const auto& item : Split(s, 'x')) {
    if (item.empty()) {
      continue;
    }
    if (item == "?") {
      shape.push_back(-1);
      continue;
    }
    bool numeric = true;
    for (unsigned char c : item) {
      if (!std::isdigit(c)) {
        numeric = false;
        break;
      }
    }
    if (!numeric) {
      continue;
    }
    shape.push_back(static_cast<std::int64_t>(std::stoll(item)));
  }
  return shape;
}

std::string BaseName(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

}  // namespace

dfabit::core::Status GraphImporter::ImportFromFile(
    const std::string& path,
    const std::string& backend_name,
    dfabit::metadata::ModelDesc* model) const {
  if (!model) {
    return {dfabit::core::StatusCode::kInvalidArgument, "model is null"};
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {dfabit::core::StatusCode::kNotFound, "failed to open graph file: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return ImportFromText(buffer.str(), BaseName(path), backend_name, model);
}

dfabit::core::Status GraphImporter::ImportFromText(
    const std::string& text,
    const std::string& graph_name,
    const std::string& backend_name,
    dfabit::metadata::ModelDesc* model) const {
  if (!model) {
    return {dfabit::core::StatusCode::kInvalidArgument, "model is null"};
  }

  model->model_name = graph_name;
  model->graph_name = graph_name;
  model->backend_name = backend_name;
  model->ops.clear();

  std::stringstream ss(text);
  std::string line;
  dfabit::metadata::StableIdAssigner assigner;

  while (std::getline(ss, line)) {
    line = Trim(line);
    if (line.empty()) {
      continue;
    }

    auto parts = Split(line, ',');
    if (parts.size() < 5) {
      parts = Split(line, '|');
    }
    if (parts.size() < 5) {
      continue;
    }

    dfabit::metadata::OpDesc op;
    const std::string symbol = parts[0];
    op.op_name = parts[1];
    op.stage_tag = parts[2];
    op.dialect = backend_name;

    dfabit::metadata::TensorDesc out;
    out.name = symbol;
    out.shape = ParseShape(parts[3]);
    out.dtype = parts[4];
    out.layout = "unknown";
    op.outputs.push_back(out);

    op.attributes["symbol"] = symbol;
    op.attributes["source_record"] = line;
    op.stable_id = assigner.Assign("hidden_ir", symbol + "|" + op.op_name + "|" + op.stage_tag);
    model->ops.push_back(std::move(op));
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::hidden_ir