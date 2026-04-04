#pragma once

#include <string>

#include "dfabit/core/status.h"

namespace dfabit::core {

dfabit::core::Status EnsureDirectory(const std::string& path);
bool PathExists(const std::string& path);
bool IsRegularFile(const std::string& path);

}  // namespace dfabit::core