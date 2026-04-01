#include "dfabit/trace/writer.h"

#include <fstream>
#include <sstream>

namespace dfabit::trace {

static std::string EscapeJson(const std::string& s) {
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

dfabit::core::Status WriteJsonlLine(const Event& e, std::string* out) {
  if (!out) {
    return {dfabit::core::StatusCode::kInvalidArgument, "out is null"};
  }

  std::ostringstream ss;
  ss << "{";
  ss << "\"ts_ns\":" << e.ts_ns << ",";
  ss << "\"kind\":" << static_cast<int>(e.kind) << ",";
  ss << "\"run_id\":\"" << EscapeJson(e.run_id) << "\",";
  ss << "\"provider\":\"" << EscapeJson(e.provider) << "\",";
  ss << "\"event\":\"" << EscapeJson(e.event) << "\",";
  ss << "\"mode\":\"" << EscapeJson(e.mode) << "\",";
  ss << "\"stable_id\":" << e.stable_id << ",";
  ss << "\"stage\":\"" << EscapeJson(e.stage) << "\",";
  ss << "\"payload\":{";

  bool first = true;
  for (const auto& kv : e.payload) {
    if (!first) {
      ss << ",";
    }
    first = false;
    ss << "\"" << EscapeJson(kv.first) << "\":\"" << EscapeJson(kv.second) << "\"";
  }

  ss << "}}";
  *out = ss.str();
  return dfabit::core::Status::Ok();
}

class JsonlFileWriter final : public Writer {
 public:
  explicit JsonlFileWriter(std::string path) : path_(std::move(path)), ofs_(path_, std::ios::out) {}

  dfabit::core::Status Write(const Event& e) override {
    if (!ofs_.is_open()) {
      return {dfabit::core::StatusCode::kInternal, "failed to open trace file: " + path_};
    }

    std::string line;
    const auto st = WriteJsonlLine(e, &line);
    if (!st.ok()) {
      return st;
    }

    ofs_ << line << "\n";
    ofs_.flush();
    return dfabit::core::Status::Ok();
  }

  void Close() override {
    if (ofs_.is_open()) {
      ofs_.close();
    }
  }

 private:
  std::string path_;
  std::ofstream ofs_;
};

}  // namespace dfabit::trace