#include "dfabit/cli/experiment_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace dfabit::cli {

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
    item = Trim(item);
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

void ApplyKeyValue(
    const std::string& key,
    const std::string& value,
    ExperimentSpec* spec) {
  if (key == "backend") {
    spec->options.backend = value;
  } else if (key == "out") {
    spec->options.output_dir = value;
  } else if (key == "mlir") {
    spec->options.mlir_path = value;
  } else if (key == "graph") {
    spec->options.graph_path = value;
  } else if (key == "sidecar") {
    spec->options.sidecar_path = value;
  } else if (key == "compile_report") {
    spec->options.compile_report_path = value;
  } else if (key == "runtime_log") {
    spec->options.runtime_log_path = value;
  } else if (key == "mode") {
    spec->options.mode = value;
  } else if (key == "modes") {
    spec->modes = Split(value, ',');
  } else if (key == "repeat") {
    try {
      spec->options.repeat = std::max(1, std::stoi(value));
    } catch (...) {
      spec->options.repeat = 1;
    }
  } else if (key == "sampling_ratio") {
    try {
      spec->options.sampling_ratio = std::stod(value);
    } catch (...) {
      spec->options.sampling_ratio = 1.0;
    }
  } else if (key == "include_ops") {
    spec->options.include_ops = Split(value, ',');
  } else if (key == "enable_portability_tool") {
    spec->options.enable_portability_tool =
        !(value == "0" || value == "false" || value == "False");
  } else if (key == "enable_overhead_profiler_tool") {
    spec->options.enable_overhead_profiler_tool =
        !(value == "0" || value == "false" || value == "False");
  } else if (key == "enable_semantic_attribution_tool") {
    spec->options.enable_semantic_attribution_tool =
        !(value == "0" || value == "false" || value == "False");
  }
}

}  // namespace

dfabit::core::Status ExperimentConfigLoader::LoadFile(
    const std::string& path,
    std::vector<ExperimentSpec>* specs) const {
  if (!specs) {
    return {dfabit::core::StatusCode::kInvalidArgument, "specs is null"};
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "failed to open experiment config: " + path};
  }

  specs->clear();

  ExperimentSpec current;
  bool has_current = false;

  std::string line;
  while (std::getline(ifs, line)) {
    line = Trim(line);
    if (line.empty() || StartsWith(line, "#")) {
      continue;
    }

    if (line.front() == '[' && line.back() == ']') {
      if (has_current) {
        if (current.modes.empty()) {
          current.modes.push_back(
              current.options.mode.empty() ? "full" : current.options.mode);
        }
        specs->push_back(current);
      }
      current = ExperimentSpec{};
      current.name = Trim(line.substr(1, line.size() - 2));
      has_current = true;
      continue;
    }

    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }

    if (!has_current) {
      current = ExperimentSpec{};
      current.name = "default";
      has_current = true;
    }

    const auto key = Trim(line.substr(0, eq));
    const auto value = Trim(line.substr(eq + 1));
    ApplyKeyValue(key, value, &current);
  }

  if (has_current) {
    if (current.modes.empty()) {
      current.modes.push_back(
          current.options.mode.empty() ? "full" : current.options.mode);
    }
    specs->push_back(current);
  }

  if (specs->empty()) {
    return {
        dfabit::core::StatusCode::kFailedPrecondition,
        "no experiment specs found in config"};
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::cli