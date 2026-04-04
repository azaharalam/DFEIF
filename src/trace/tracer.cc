#include "dfabit/trace/tracer.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <utility>

#include "dfabit/trace/writer.h"

namespace dfabit::trace {

dfabit::core::Status Tracer::Open(
    const std::string& trace_path,
    const std::string& session_id,
    const std::string& run_id,
    const std::string& provider,
    const std::string& mode,
    std::size_t buffer_capacity) {
  if (trace_path.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "trace_path is empty"};
  }

  std::filesystem::create_directories(std::filesystem::path(trace_path).parent_path());

  auto writer = CreateJsonlFileWriter(trace_path);
  if (!writer) {
    return {dfabit::core::StatusCode::kInternal, "failed to create trace writer"};
  }

  writer_ = std::make_unique<BufferedTraceWriter>(std::move(writer), buffer_capacity);
  session_id_ = session_id;
  run_id_ = run_id;
  provider_ = provider;
  mode_ = mode;
  stats_cache_ = {};
  return dfabit::core::Status::Ok();
}

dfabit::core::Status Tracer::Emit(
    EventKind kind,
    const std::string& event_name,
    std::uint64_t stable_id,
    const std::string& stage,
    std::unordered_map<std::string, std::string> payload) {
  if (!writer_) {
    return dfabit::core::Status::Ok();
  }

  payload.emplace("session_id", session_id_);

  Event e;
  e.ts_ns = NowNs();
  e.kind = kind;
  e.run_id = run_id_;
  e.provider = provider_;
  e.event = event_name;
  e.mode = mode_;
  e.stable_id = stable_id;
  e.stage = stage;
  e.payload = std::map<std::string, std::string>(payload.begin(), payload.end());

  const auto st = writer_->Write(e);
  stats_cache_ = writer_->stats();
  return st;
}

dfabit::core::Status Tracer::EmitMetrics(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    const std::string& fallback_stage) {
  for (const auto& metric : metrics) {
    std::unordered_map<std::string, std::string> payload = metric.attributes;
    payload["value"] = std::to_string(metric.value);
    payload["unit"] = metric.unit;
    const auto st = Emit(
        EventKind::kMetric,
        metric.name,
        metric.stable_id,
        metric.stage.empty() ? fallback_stage : metric.stage,
        std::move(payload));
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status Tracer::Flush() {
  if (!writer_) {
    return dfabit::core::Status::Ok();
  }
  const auto st = writer_->Flush();
  stats_cache_ = writer_->stats();
  return st;
}

dfabit::core::Status Tracer::Close() {
  if (!writer_) {
    return dfabit::core::Status::Ok();
  }
  const auto st = Flush();
  writer_->Close();
  writer_.reset();
  return st;
}

dfabit::core::Status Tracer::WriteStatsCsv(const std::string& path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open trace stats csv: " + path};
  }

  ofs << "event_count,bytes_written,flush_count,dropped_event_count,max_queue_depth,total_flush_latency_ms\n";
  ofs << stats_cache_.event_count << ","
      << stats_cache_.bytes_written << ","
      << stats_cache_.flush_count << ","
      << stats_cache_.dropped_event_count << ","
      << stats_cache_.max_queue_depth << ","
      << stats_cache_.total_flush_latency_ms << "\n";

  return dfabit::core::Status::Ok();
}

std::uint64_t Tracer::NowNs() const {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace dfabit::trace