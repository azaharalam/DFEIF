#include "dfabit/runtime/launch_observer.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace dfabit::runtime {

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

}  // namespace

dfabit::core::Status LaunchObserver::ParseFile(
    const std::string& path,
    std::vector<LaunchEventRecord>* records) const {
  if (!records) {
    return {dfabit::core::StatusCode::kInvalidArgument, "records is null"};
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "failed to open launch event log: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return ParseText(buffer.str(), records);
}

dfabit::core::Status LaunchObserver::ParseText(
    const std::string& text,
    std::vector<LaunchEventRecord>* records) const {
  if (!records) {
    return {dfabit::core::StatusCode::kInvalidArgument, "records is null"};
  }

  records->clear();
  std::stringstream ss(text);
  std::string line;

  while (std::getline(ss, line)) {
    line = Trim(line);
    if (line.empty()) {
      continue;
    }

    const auto parts = Split(line, ',');
    if (parts.size() < 5) {
      continue;
    }

    LaunchEventRecord rec;
    rec.type = parts[0];
    rec.symbol = parts[1];
    rec.stage = parts[2];
    rec.name = parts[3];
    try {
      rec.value = std::stod(parts[4]);
    } catch (...) {
      rec.value = 0.0;
    }
    rec.unit = parts.size() >= 6 ? parts[5] : "";
    records->push_back(std::move(rec));
  }

  return dfabit::core::Status::Ok();
}

std::vector<dfabit::adapters::MetricSample> LaunchObserver::Correlate(
    const dfabit::metadata::ModelDesc& model,
    const std::vector<LaunchEventRecord>& records) const {
  std::unordered_map<std::string, std::uint64_t> symbol_to_id;
  for (const auto& op : model.ops) {
    symbol_to_id[op.op_name] = op.stable_id;
    const auto it = op.attributes.find("symbol");
    if (it != op.attributes.end()) {
      symbol_to_id[it->second] = op.stable_id;
    }
  }

  std::unordered_map<std::string, double> begin_ns;
  std::vector<dfabit::adapters::MetricSample> out;

  for (const auto& rec : records) {
    const auto id_it = symbol_to_id.find(rec.symbol);
    const std::uint64_t stable_id = id_it == symbol_to_id.end() ? 0 : id_it->second;
    const std::string key = rec.symbol + "|" + rec.stage;

    if (rec.type == "begin") {
      begin_ns[key] = rec.value;
      continue;
    }

    if (rec.type == "end") {
      const auto it = begin_ns.find(key);
      if (it != begin_ns.end() && rec.unit == "ns") {
        dfabit::adapters::MetricSample sample;
        sample.name = "launch_duration_ms";
        sample.value = (rec.value - it->second) / 1.0e6;
        sample.unit = "ms";
        sample.stage = rec.stage;
        sample.stable_id = stable_id;
        sample.attributes["symbol"] = rec.symbol;
        out.push_back(std::move(sample));
      }
      continue;
    }

    if (rec.type == "metric") {
      dfabit::adapters::MetricSample sample;
      sample.name = rec.name;
      sample.value = rec.value;
      sample.unit = rec.unit;
      sample.stage = rec.stage;
      sample.stable_id = stable_id;
      sample.attributes["symbol"] = rec.symbol;
      out.push_back(std::move(sample));
    }
  }

  return out;
}

}  // namespace dfabit::runtime