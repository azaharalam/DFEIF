#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace dfabit::trace {

enum class EventKind {
  kSession = 0,
  kCompile = 1,
  kLoad = 2,
  kRun = 3,
  kSubgraph = 4,
  kOp = 5,
  kMetric = 6,
  kDiagnostic = 7,
  kDrop = 8
};

struct Event {
  std::uint64_t ts_ns = 0;
  EventKind kind = EventKind::kDiagnostic;
  std::string run_id;
  std::string provider;
  std::string event;
  std::string mode;
  std::uint64_t stable_id = 0;
  std::string stage;
  std::map<std::string, std::string> payload;
};

}  // namespace dfabit::trace