#include "dfabit/mlir/module_instrumentor.h"

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace dfabit::mlir {

namespace {

std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    lines.push_back(line);
  }
  return lines;
}

std::string JoinLines(const std::vector<std::string>& lines) {
  std::ostringstream ss;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    ss << lines[i];
    if (i + 1 != lines.size()) {
      ss << '\n';
    }
  }
  return ss.str();
}

bool LooksLikeOpLine(const std::string& line) {
  return line.find('"') != std::string::npos && line.find('=') != std::string::npos;
}

std::string InjectAttributes(
    const std::string& line,
    const dfabit::metadata::OpDesc& op) {
  const auto brace = line.rfind('}');
  const std::string attr_text =
      " {dfabit.id = " + std::to_string(op.stable_id) +
      " : i64, dfabit.stage = \"" + op.stage_tag +
      "\", dfabit.op = \"" + op.op_name + "\"}";

  if (brace != std::string::npos) {
    return line.substr(0, brace) + attr_text + line.substr(brace);
  }

  const auto arrow = line.find(" : ");
  if (arrow != std::string::npos) {
    return line.substr(0, arrow) + attr_text + line.substr(arrow);
  }

  return line + attr_text;
}

}  // namespace

dfabit::core::Status MlirModuleInstrumentor::Instrument(
    const MlirModuleSnapshot& snapshot,
    const dfabit::metadata::ModelDesc& model,
    InstrumentedModuleSnapshot* out) const {
  if (!out) {
    return {dfabit::core::StatusCode::kInvalidArgument, "out is null"};
  }

  out->module_name = snapshot.module_name;
  out->stage = snapshot.stage;
  out->source_path = snapshot.source_path;
  out->original_text = snapshot.text;

  auto lines = SplitLines(snapshot.text);
  std::size_t op_index = 0;

  for (auto& line : lines) {
    if (!LooksLikeOpLine(line)) {
      continue;
    }
    if (op_index >= model.ops.size()) {
      break;
    }
    line = InjectAttributes(line, model.ops[op_index]);
    ++op_index;
  }

  out->instrumented_text = JoinLines(lines);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status MlirModuleInstrumentor::WriteSnapshot(
    const InstrumentedModuleSnapshot& snapshot,
    const std::string& original_path,
    const std::string& instrumented_path) const {
  {
    std::ofstream ofs(original_path);
    if (!ofs.is_open()) {
      return {
          dfabit::core::StatusCode::kInternal,
          "failed to write original MLIR snapshot: " + original_path};
    }
    ofs << snapshot.original_text;
  }

  {
    std::ofstream ofs(instrumented_path);
    if (!ofs.is_open()) {
      return {
          dfabit::core::StatusCode::kInternal,
          "failed to write instrumented MLIR snapshot: " + instrumented_path};
    }
    ofs << snapshot.instrumented_text;
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::mlir