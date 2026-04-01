#include "dfabit/adapters/shared/runtime_log_parser.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

namespace dfabit::adapters::shared {

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

std::vector<std::string> Split(const std::string& line, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, delim)) {
    parts.push_back(Trim(item));
  }
  return parts;
}

bool ParseStableId(const std::string& s, std::uint64_t* out) {
  if (!out || s.empty()) {
    return false;
  }
  char* end = nullptr;
  const auto v = std::strtoull(s.c_str(), &end, 10);
  if (end == nullptr || *end != '\0') {
    return false;
  }
  *out = static_cast<std::uint64_t>(v);
  return true;
}

bool IsNumeric(const std::string& s) {
  if (s.empty()) {
    return false;
  }
  char* end = nullptr;
  const auto value = std::strtod(s.c_str(), &end);
  (void)value;
  return end != nullptr && *end == '\0';
}

}  // namespace

dfabit::core::Status RuntimeLogParser::ParseFile(
    const std::string& path,
    std::vector<RuntimeLogRecord>* records) const {
  if (!records) {
    return {dfabit::core::StatusCode::kInvalidArgument, "records is null"};
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "failed to open runtime log: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return ParseText(buffer.str(), records);
}

dfabit::core::Status RuntimeLogParser::ParseText(
    const std::string& text,
    std::vector<RuntimeLogRecord>* records) const {
  if (!records) {
    return {dfabit::core::StatusCode::kInvalidArgument, "records is null"};
  }

  records->clear();

  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    const auto trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    auto parts = Split(trimmed, ',');
    if (parts.size() < 5) {
      parts = Split(trimmed, ':');
    }
    if (parts.size() < 4) {
      continue;
    }

    RuntimeLogRecord rec;
    rec.phase = parts[0];
    rec.name = parts[1];
    rec.value = parts[2];
    rec.unit = parts[3];
    if (parts.size() >= 5) {
      ParseStableId(parts[4], &rec.stable_id);
    }
    records->push_back(std::move(rec));
  }

  return dfabit::core::Status::Ok();
}

std::vector<MetricSample> RuntimeLogParser::ToMetricSamples(
    const std::vector<RuntimeLogRecord>& records) const {
  std::vector<MetricSample> out;
  out.reserve(records.size());

  for (const auto& rec : records) {
    MetricSample sample;
    sample.name = rec.name;
    sample.unit = rec.unit;
    sample.stage = rec.phase;
    sample.stable_id = rec.stable_id;
    if (IsNumeric(rec.value)) {
      sample.value = std::strtod(rec.value.c_str(), nullptr);
    } else {
      sample.attributes["raw_value"] = rec.value;
    }
    out.push_back(std::move(sample));
  }

  return out;
}

}  // namespace dfabit::adapters::shared