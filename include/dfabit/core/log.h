#pragma once
#include <string>

namespace dfabit::core {

enum class LogLevel { kInfo, kWarn, kError };

void Log(LogLevel lvl, const std::string& msg);

}  // namespace dfabit::core
