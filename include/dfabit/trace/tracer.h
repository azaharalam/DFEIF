#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/core/status.h"
#include "dfabit/trace/event.h"
#include "dfabit/trace/trace_buffer.h"
#include "dfabit/trace/trace_stats.h"

namespace dfabit::trace {

class Tracer {
 public:
  Tracer() = default;

  dfabit::core::Status Open(
      const std::string& trace_path,
      const std::string& session_id,
      const std::string& run_id,
      const std::string& provider,
      const std::string& mode,
      std::size_t buffer_capacity);

  bool enabled() const { return writer_ != nullptr; }

  dfabit::core::Status Emit(
      EventKind kind,
      const std::string& event_name,
      std::uint64_t stable_id,
      const std::string& stage,
      std::unordered_map<std::string, std::string> payload = {});

  dfabit::core::Status EmitMetrics(
      const std::vector<dfabit::adapters::MetricSample>& metrics,
      const std::string& fallback_stage);

  dfabit::core::Status Flush();
  dfabit::core::Status Close();

  const TraceStats& stats() const { return stats_cache_; }

  dfabit::core::Status WriteStatsCsv(const std::string& path) const;

 private:
  std::uint64_t NowNs() const;

  std::unique_ptr<BufferedTraceWriter> writer_;
  std::string session_id_;
  std::string run_id_;
  std::string provider_;
  std::string mode_;
  TraceStats stats_cache_;
};

}  // namespace dfabit::trace