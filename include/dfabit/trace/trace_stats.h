#pragma once

#include <cstddef>

namespace dfabit::trace {

struct TraceStats {
  std::size_t event_count = 0;
  std::size_t bytes_written = 0;
  std::size_t flush_count = 0;
  std::size_t dropped_event_count = 0;
  std::size_t max_queue_depth = 0;
  double total_flush_latency_ms = 0.0;
};

}  // namespace dfabit::trace