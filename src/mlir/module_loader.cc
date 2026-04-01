#include "dfabit/mlir/module_loader.h"

#include <fstream>
#include <sstream>
#include <utility>

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

std::string Basename(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

}  // namespace

dfabit::core::Status MlirModuleLoader::LoadFromFile(
    const std::string& path,
    const std::string& stage,
    MlirModuleSnapshot* snapshot) const {
  if (!snapshot) {
    return {dfabit::core::StatusCode::kInvalidArgument, "snapshot is null"};
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {dfabit::core::StatusCode::kNotFound, "failed to open MLIR file: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return LoadFromText(buffer.str(), Basename(path), stage, snapshot);
}

dfabit::core::Status MlirModuleLoader::LoadFromText(
    std::string text,
    const std::string& module_name,
    const std::string& stage,
    MlirModuleSnapshot* snapshot) const {
  if (!snapshot) {
    return {dfabit::core::StatusCode::kInvalidArgument, "snapshot is null"};
  }

  snapshot->module_name = module_name;
  snapshot->stage = stage;
  snapshot->text = std::move(text);
  snapshot->lines = SplitLines(snapshot->text);
  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::mlir