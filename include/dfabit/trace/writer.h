#pragma once
#include <string>
#include "dfabit/core/status.h"
#include "dfabit/trace/event.h"

namespace dfabit::trace {

class Writer {
 public:
  virtual ~Writer() = default;
  virtual dfabit::core::Status Write(const Event& e) = 0;
  virtual void Close() = 0;
};

dfabit::core::Status WriteJsonlLine(const Event& e, std::string* out);

}  // namespace dfabit::trace
