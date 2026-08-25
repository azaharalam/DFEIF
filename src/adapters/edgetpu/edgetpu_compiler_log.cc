#include "dfabit/adapters/edgetpu/edgetpu_compiler_log.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

#include "dfabit/metadata/stable_id.h"

namespace dfabit::adapters::edgetpu {

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

// Splits "Key: Value" once. Returns false when the line has no colon.
bool SplitKeyValue(const std::string& line, std::string* key, std::string* value) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return false;
  }
  *key = Trim(line.substr(0, colon));
  *value = Trim(line.substr(colon + 1));
  return true;
}

int ParseInt(const std::string& s, int fallback) {
  try {
    return std::stoi(s);
  } catch (...) {
    return fallback;
  }
}

}  // namespace

bool EdgeTpuCompilerLogParser::ParseByteSize(const std::string& text, double* out_bytes) {
  if (!out_bytes) {
    return false;
  }

  const auto trimmed = Trim(text);
  if (trimmed.empty()) {
    return false;
  }

  // Leading numeric portion, then a unit suffix.
  std::size_t i = 0;
  while (i < trimmed.size() &&
         (std::isdigit(static_cast<unsigned char>(trimmed[i])) || trimmed[i] == '.' ||
          trimmed[i] == '-' || trimmed[i] == '+')) {
    ++i;
  }
  if (i == 0) {
    return false;
  }

  const double value = std::strtod(trimmed.substr(0, i).c_str(), nullptr);
  auto unit = Trim(trimmed.substr(i));
  for (auto& c : unit) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  double multiplier = 1.0;
  if (unit == "b" || unit.empty()) {
    multiplier = 1.0;
  } else if (unit == "kib") {
    multiplier = 1024.0;
  } else if (unit == "mib") {
    multiplier = 1024.0 * 1024.0;
  } else if (unit == "gib") {
    multiplier = 1024.0 * 1024.0 * 1024.0;
  } else if (unit == "kb") {
    multiplier = 1000.0;
  } else if (unit == "mb") {
    multiplier = 1000.0 * 1000.0;
  } else if (unit == "gb") {
    multiplier = 1000.0 * 1000.0 * 1000.0;
  } else {
    return false;
  }

  *out_bytes = value * multiplier;
  return true;
}

dfabit::core::Status EdgeTpuCompilerLogParser::ParseFile(
    const std::string& path,
    EdgeTpuCompileSummary* summary,
    std::vector<OperatorMappingRecord>* operators) const {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "failed to open edgetpu compiler log: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return ParseText(buffer.str(), summary, operators);
}

dfabit::core::Status EdgeTpuCompilerLogParser::ParseText(
    const std::string& text,
    EdgeTpuCompileSummary* summary,
    std::vector<OperatorMappingRecord>* operators) const {
  if (!summary || !operators) {
    return {dfabit::core::StatusCode::kInvalidArgument, "null output arguments"};
  }

  *summary = EdgeTpuCompileSummary{};
  operators->clear();

  const dfabit::metadata::StableIdAssigner assigner;

  std::stringstream ss(text);
  std::string line;
  bool in_operator_table = false;

  while (std::getline(ss, line)) {
    const auto trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    // The operator table starts at its header row and runs to the end of the
    // section. Rows are "NAME  COUNT  STATUS..." with variable whitespace.
    if (trimmed.rfind("Operator", 0) == 0 &&
        trimmed.find("Count") != std::string::npos &&
        trimmed.find("Status") != std::string::npos) {
      in_operator_table = true;
      continue;
    }

    if (in_operator_table) {
      std::stringstream ls(trimmed);
      std::string name;
      std::string count_token;
      if (!(ls >> name >> count_token)) {
        continue;
      }

      // A row must have a numeric count; anything else is prose and ends the
      // table (e.g. "Compilation succeeded!").
      bool numeric = !count_token.empty();
      for (const char c : count_token) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
          numeric = false;
          break;
        }
      }
      if (!numeric) {
        in_operator_table = false;
        continue;
      }

      std::string status;
      std::getline(ls, status);
      status = Trim(status);

      OperatorMappingRecord rec;
      rec.op_name = name;
      rec.count = ParseInt(count_token, 0);
      rec.status = status;
      // The compiler says "Mapped to Edge TPU" for placed ops; every other
      // status is a fallback reason.
      rec.mapped_to_tpu = status.find("Mapped to Edge TPU") != std::string::npos;
      rec.stable_id = assigner.Assign("edgetpu_op", name);
      operators->push_back(std::move(rec));
      continue;
    }

    // The version banner has no colon: "Edge TPU Compiler version 16.0.384591198"
    {
      const std::string vkey = "Edge TPU Compiler version ";
      if (trimmed.rfind(vkey, 0) == 0) {
        summary->compiler_version = Trim(trimmed.substr(vkey.size()));
        summary->has_summary = true;
        continue;
      }
    }

    std::string key;
    std::string value;
    if (!SplitKeyValue(trimmed, &key, &value)) {
      continue;
    }

    if (key == "Input model" || key == "Input") {
      summary->input_model = value;
      summary->has_summary = true;
    } else if (key == "Input size") {
      ParseByteSize(value, &summary->input_size_bytes);
      summary->has_summary = true;
    } else if (key == "Output model" || key == "Output") {
      summary->output_model = value;
    } else if (key == "Output size") {
      ParseByteSize(value, &summary->output_size_bytes);
    } else if (key == "On-chip memory used for caching model parameters") {
      ParseByteSize(value, &summary->on_chip_used_bytes);
      summary->has_summary = true;
    } else if (key == "On-chip memory remaining for caching model parameters") {
      ParseByteSize(value, &summary->on_chip_remaining_bytes);
    } else if (key == "Off-chip memory used for streaming uncached model parameters") {
      ParseByteSize(value, &summary->off_chip_streamed_bytes);
    } else if (key == "Number of Edge TPU subgraphs") {
      summary->subgraph_count = ParseInt(value, 0);
      summary->has_summary = true;
    } else if (key == "Total number of operations") {
      summary->total_operations = ParseInt(value, 0);
      summary->has_summary = true;
    }
  }

  // "Model compiled successfully in 941 ms." has no colon, so it never reaches
  // the key/value branch. Catch it with a direct scan.
  if (summary->compile_time_ms == 0.0) {
    const std::string key = "Model compiled successfully in ";
    const auto at = text.find(key);
    if (at != std::string::npos) {
      summary->compile_time_ms =
          std::strtod(text.substr(at + key.size()).c_str(), nullptr);
    }
  }

  return dfabit::core::Status::Ok();
}

std::vector<MetricSample> EdgeTpuCompilerLogParser::ToMetricSamples(
    const EdgeTpuCompileSummary& summary,
    const std::vector<OperatorMappingRecord>& operators) const {
  std::vector<MetricSample> out;

  const auto push = [&out](
                        const std::string& name,
                        double value,
                        const std::string& unit,
                        std::uint64_t stable_id) {
    MetricSample m;
    m.name = name;
    m.value = value;
    m.unit = unit;
    m.stage = "compile";
    m.stable_id = stable_id;
    m.attributes["source"] = "edgetpu_compiler_log";
    out.push_back(std::move(m));
  };

  if (summary.has_summary) {
    push("model_input_bytes", summary.input_size_bytes, "B", 0);
    push("model_output_bytes", summary.output_size_bytes, "B", 0);
    push("on_chip_param_bytes", summary.on_chip_used_bytes, "B", 0);
    push("on_chip_remaining_bytes", summary.on_chip_remaining_bytes, "B", 0);
    push("off_chip_streamed_bytes", summary.off_chip_streamed_bytes, "B", 0);
    push("edgetpu_subgraphs", static_cast<double>(summary.subgraph_count), "count", 0);
    push("total_operations", static_cast<double>(summary.total_operations), "count", 0);
    if (summary.compile_time_ms > 0.0) {
      push("compile_time_ms", summary.compile_time_ms, "ms", 0);
    }

    // Whether the whole parameter set fit on chip is the single most useful
    // derived fact here: off-chip streaming dominates latency when it happens.
    const double total_params =
        summary.on_chip_used_bytes + summary.off_chip_streamed_bytes;
    if (total_params > 0.0) {
      push(
          "on_chip_param_fraction",
          summary.on_chip_used_bytes / total_params,
          "ratio",
          0);
    }
  }

  int mapped_ops = 0;
  int fallback_ops = 0;

  for (const auto& op : operators) {
    MetricSample m;
    m.name = "operator_count";
    m.value = static_cast<double>(op.count);
    m.unit = "count";
    m.stage = "compile";
    m.stable_id = op.stable_id;
    m.attributes["operator"] = op.op_name;
    m.attributes["mapped_to_tpu"] = op.mapped_to_tpu ? "1" : "0";
    m.attributes["status"] = op.status;
    m.attributes["source"] = "edgetpu_compiler_log";
    out.push_back(std::move(m));

    if (op.mapped_to_tpu) {
      mapped_ops += op.count;
    } else {
      fallback_ops += op.count;
    }
  }

  if (!operators.empty()) {
    push("ops_mapped_to_tpu", static_cast<double>(mapped_ops), "count", 0);
    push("ops_fallback_to_cpu", static_cast<double>(fallback_ops), "count", 0);

    const int total = mapped_ops + fallback_ops;
    if (total > 0) {
      // The headline portability number for this backend: what fraction of the
      // model the accelerator could actually take.
      push(
          "tpu_mapping_ratio",
          static_cast<double>(mapped_ops) / static_cast<double>(total),
          "ratio",
          0);
    }
  }

  return out;
}

}  // namespace dfabit::adapters::edgetpu