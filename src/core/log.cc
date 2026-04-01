#include "dfabit/core/log.h"
#include <iostream>

namespace dfabit::core {

void Log(LogLevel lvl, const std::string& msg) {
  switch (lvl) {
    case LogLevel::kInfo:  std::cerr << "[DFABIT][INFO] "; break;
    case LogLevel::kWarn:  std::cerr << "[DFABIT][WARN] "; break;
    case LogLevel::kError: std::cerr << "[DFABIT][ERROR] "; break;
  }
  std::cerr << msg << "\n";
}

}  // namespace dfabit::core
