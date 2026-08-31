#include "dfabit/tools/builtin/program_analyzer_tool.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>

#include "dfabit/tools/tool_context.h"

namespace dfabit::tools::builtin {

namespace {

// Arithmetic intensity is MACs per byte moved. Both terms come from operator
// semantics and tensor shapes, so the intensity reported here is a static
// property of the graph. Real traffic is lower than the shape-derived figure
// wherever the backend reuses a tensor across operators, which makes this a
// lower bound rather than an estimate.
//
// Achieved throughput needs a measured latency and is therefore only computed
// when a runtime metric supplies one. Without it the tool reports intensity and
// boundness and leaves the efficiency columns empty, rather than substituting a
// figure nothing measured.

// Region boundaries as multiples of the ridge point. A hard cut at the ridge
// would make classification hypersensitive to error in the byte term, so a
// balanced band is kept either side of it.
constexpr double kMemoryBelow = 0.5;
constexpr double kComputeAbove = 2.0;

double PropertyAsDouble(const dfabit::api::Context& ctx, const char* key) {
  const auto raw = ctx.GetProperty(key);
  if (raw.empty()) return 0.0;
  try {
    return std::stod(raw);
  } catch (...) {
    return 0.0;
  }
}

std::string Classify(double intensity, double ridge) {
  if (ridge <= 0.0) return "unknown";
  if (intensity < ridge * kMemoryBelow) return "memory";
  if (intensity > ridge * kComputeAbove) return "compute";
  return "balanced";
}

double MetricValue(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    const std::string& name) {
  for (const auto& metric : metrics) {
    if (metric.name == name) return metric.value;
  }
  return 0.0;
}

std::string Csv(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    out += (c == ',' || c == '\n' || c == '"') ? '_' : c;
  }
  return out;
}

}  // namespace

std::string ProgramAnalyzerTool::name() const { return "program_analyzer"; }

dfabit::core::Status ProgramAnalyzerTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  filtered_ops_.clear();
  runtime_metrics_.clear();
  peak_macs_ = 0.0;
  peak_bw_bytes_ = 0.0;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  (void)compile_artifacts;
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  (void)compile_artifacts;
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  filtered_ops_ = ToolServices::FilterOps(*ctx, ctx->metadata_ops());

  // The only backend-specific inputs the analysis needs. The adapter publishes
  // them, so this tool holds no per-backend branch.
  peak_macs_ = PropertyAsDouble(*ctx, "peak_macs_per_s");
  peak_bw_bytes_ = PropertyAsDouble(*ctx, "peak_bw_bytes_per_s");
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  runtime_metrics_ = ToolServices::FilterMetrics(*ctx, runtime_artifacts.metrics);
  return dfabit::core::Status::Ok();
}

std::string ProgramAnalyzerTool::OutputDir(const dfabit::api::Context& ctx) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  const auto path = std::filesystem::path(base) / "tools" / "program_analyzer";
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  return path.string();
}

dfabit::core::Status ProgramAnalyzerTool::WritePerOpTable(
    const std::string& path) const {
  const double ridge =
      peak_bw_bytes_ > 0.0 ? peak_macs_ / peak_bw_bytes_ : 0.0;

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal,
            "failed to open program analysis table: " + path};
  }

  ofs << "stable_id,op_name,dialect,phase,macs,bytes,"
         "arithmetic_intensity,boundness\n";

  for (const auto& op : filtered_ops_) {
    // estimated_flops counts two operations per multiply-accumulate.
    const double macs = static_cast<double>(op.estimated_flops) / 2.0;
    const double bytes = static_cast<double>(op.estimated_bytes);
    const double intensity = bytes > 0.0 ? macs / bytes : 0.0;

    ofs << op.stable_id << ","
        << Csv(op.op_name) << ","
        << Csv(op.dialect) << ","
        << Csv(op.stage_tag) << ","
        << static_cast<long long>(macs) << ","
        << op.estimated_bytes << ","
        << intensity << ","
        << Classify(intensity, ridge) << "\n";
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::WriteSummary(
    const std::string& path) const {
  double total_macs = 0.0;
  double total_bytes = 0.0;
  std::map<std::string, double> macs_by_region;

  const double ridge =
      peak_bw_bytes_ > 0.0 ? peak_macs_ / peak_bw_bytes_ : 0.0;

  for (const auto& op : filtered_ops_) {
    // estimated_flops counts two operations per multiply-accumulate.
    const double macs = static_cast<double>(op.estimated_flops) / 2.0;
    const double bytes = static_cast<double>(op.estimated_bytes);
    total_macs += macs;
    total_bytes += bytes;
    const double intensity = bytes > 0.0 ? macs / bytes : 0.0;
    macs_by_region[Classify(intensity, ridge)] += macs;
  }

  const double model_intensity =
      total_bytes > 0.0 ? total_macs / total_bytes : 0.0;

  // Latency is the one input the graph cannot supply. When a paired
  // measurement is absent the throughput and efficiency rows are omitted
  // rather than filled with a default.
  const double latency_ms = MetricValue(runtime_metrics_, "latency_ms");

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal,
            "failed to open program analysis summary: " + path};
  }

  ofs << "metric,value,unit,note\n";
  ofs << "operators," << filtered_ops_.size() << ",count,\n";
  ofs << "total_macs," << static_cast<long long>(total_macs) << ",MAC,\n";
  ofs << "total_bytes," << static_cast<long long>(total_bytes) << ",B,"
      << "derived from operator semantics and tensor shapes\n";
  ofs << "arithmetic_intensity," << model_intensity << ",MAC/B,"
      << "lower bound; reuse is not modelled\n";
  ofs << "peak_macs_per_s," << peak_macs_ << ",MAC/s,from the adapter\n";
  ofs << "peak_bw_bytes_per_s," << peak_bw_bytes_ << ",B/s,from the adapter\n";
  ofs << "ridge_point," << ridge << ",MAC/B,\n";
  ofs << "boundness," << Classify(model_intensity, ridge) << ",class,\n";

  for (const auto& kv : macs_by_region) {
    const double share = total_macs > 0.0 ? kv.second / total_macs : 0.0;
    ofs << "mac_share:" << Csv(kv.first) << "," << share << ",fraction,\n";
  }

  if (latency_ms > 0.0 && peak_macs_ > 0.0) {
    const double achieved = total_macs / (latency_ms / 1000.0);
    ofs << "latency_ms," << latency_ms << ",ms,measured\n";
    ofs << "achieved_macs_per_s," << achieved << ",MAC/s,\n";
    ofs << "normalized_efficiency," << achieved / peak_macs_ << ",fraction,\n";
  } else {
    ofs << "normalized_efficiency,,fraction,"
        << "requires a measured latency; none was supplied\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status ProgramAnalyzerTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  if (filtered_ops_.empty()) return dfabit::core::Status::Ok();

  const auto dir = OutputDir(*ctx);
  auto st = WritePerOpTable(dir + "/program_analysis_ops.csv");
  if (!st.ok()) return st;
  return WriteSummary(dir + "/program_analysis_summary.csv");
}

std::unique_ptr<Tool> CreateProgramAnalyzerTool() {
  return std::make_unique<ProgramAnalyzerTool>();
}

}  // namespace dfabit::tools::builtin
