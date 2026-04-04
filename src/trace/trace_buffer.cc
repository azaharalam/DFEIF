#include "dfabit/trace/trace_buffer.h"

#include <chrono>
#include <utility>

namespace dfabit::trace {

BufferedTraceWriter::BufferedTraceWriter(std::unique_ptr<Writer> inner, std::size_t capacity)
    : inner_(std::move(inner)), capacity_(capacity == 0 ? 1 : capacity) {
  pending_.reserve(capacity_);
}

dfabit::core::Status BufferedTraceWriter::Write(const Event& e) {
  if (!inner_) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "inner writer is null"};
  }

  pending_.push_back(e);
  if (pending_.size() > stats_.max_queue_depth) {
    stats_.max_queue_depth = pending_.size();
  }

  if (pending_.size() >= capacity_) {
    return FlushPending();
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status BufferedTraceWriter::Flush() {
  if (!inner_) {
    return {dfabit::core::StatusCode::kFailedPrecondition, "inner writer is null"};
  }
  return FlushPending();
}

void BufferedTraceWriter::Close() {
  if (inner_) {
    (void)FlushPending();
    inner_->Close();
  }
}

dfabit::core::Status BufferedTraceWriter::FlushPending() {
  if (pending_.empty()) {
    return dfabit::core::Status::Ok();
  }

  const auto start = std::chrono::steady_clock::now();

  for (const auto& event : pending_) {
    std::string line;
    const auto line_st = WriteJsonlLine(event, &line);
    if (!line_st.ok()) {
      ++stats_.dropped_event_count;
      continue;
    }

    const auto st = inner_->Write(event);
    if (!st.ok()) {
      ++stats_.dropped_event_count;
      continue;
    }

    ++stats_.event_count;
    stats_.bytes_written += line.size() + 1;
  }

  pending_.clear();
  ++stats_.flush_count;

  const auto end = std::chrono::steady_clock::now();
  stats_.total_flush_latency_ms +=
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::trace