#include "dfabit/core/file_utils.h"

#include <filesystem>

namespace dfabit::core {

dfabit::core::Status EnsureDirectory(const std::string& path) {
  if (path.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "directory path is empty"};
  }

  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to create directory: " + path};
  }

  return dfabit::core::Status::Ok();
}

bool PathExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

bool IsRegularFile(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

}  // namespace dfabit::core