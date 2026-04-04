#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dfabit/core/status.h"
#include "dfabit/trace/event.h"
#include "dfabit/trace/trace_stats.h"
#include "dfabit/trace/writer.h"

namespace dfabit::trace {

class BufferedTraceWriter final : public Writer {
 public:
  BufferedTraceWriter(std::unique_ptr<Writer> inner, std::size_t capacity);

  dfabit::core::Status Write(const Event& e) override;
  void Close() override;

  dfabit::core::Status Flush();
  const TraceStats& stats() const { return stats_; }

 private:
  dfabit::core::Status FlushPending();

  std::unique_ptr<Writer> inner_;
  std::size_t capacity_ = 0;
  std::vector<Event> pending_;
  TraceStats stats_;
};

}  // namespace dfabit::trace