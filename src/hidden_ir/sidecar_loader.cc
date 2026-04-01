#include "dfabit/hidden_ir/sidecar_loader.h"

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

}  // namespace

dfabit::core::Status SidecarLoader::LoadFile(
    const std::string& path,
    std::vector<SidecarEntry>* entries) const {
  if (!entries) {
    return {dfabit::core::StatusCode::kInvalidArgument, "entries is null"};
  }

  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {dfabit::core::StatusCode::kNotFound, "failed to open sidecar file: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return LoadText(buffer.str(), entries);
}

dfabit::core::Status SidecarLoader::LoadText(
    const std::string& text,
    std::vector<SidecarEntry>* entries) const {
  if (!entries) {
    return {dfabit::core::StatusCode::kInvalidArgument, "entries is null"};
  }

  entries->clear();
  std::stringstream ss(text);
  std::string line;

  while (std::getline(ss, line)) {
    line = Trim(line);
    if (line.empty()) {
      continue;
    }

    auto parts = Split(line, ',');
    if (parts.size() < 3) {
      parts = Split(line, '|');
    }
    if (parts.size() < 3) {
      continue;
    }

    SidecarEntry entry;
    entry.symbol = parts[0];
    try {
      entry.stable_id = static_cast<std::uint64_t>(std::stoull(parts[1]));
    } catch (...) {
      entry.stable_id = 0;
    }
    entry.stage = parts[2];

    for (std::size_t i = 3; i < parts.size(); ++i) {
      const auto eq = parts[i].find('=');
      if (eq == std::string::npos) {
        continue;
      }
      entry.attributes.emplace(parts[i].substr(0, eq), parts[i].substr(eq + 1));
    }

    entries->push_back(std::move(entry));
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::hidden_ir