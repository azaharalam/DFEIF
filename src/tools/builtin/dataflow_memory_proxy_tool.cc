#include "dfabit/tools/builtin/dataflow_memory_proxy_tool.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "dfabit/tools/tool_context.h"

namespace dfabit::tools::builtin {

namespace {

// What this tool reports, and what it deliberately does not.
//
// Every figure below is derived from the compiled graph: tensor shapes, dtypes,
// and the def-use chain between operators. That makes them exact statements
// about the dataflow the compiler produced, and nothing more. In particular:
//
//   - Byte figures are logical tensor volume, not measured traffic. No
//     evaluated backend exposes memory traffic counters.
//   - Peak live bytes is the volume a sequential execution of the graph would
//     hold resident. It is meaningful where operators execute in order against
//     a single memory, which is the Edge TPU's model. It is NOT a statement
//     about on-wafer residency on Cerebras, where weights stream from external
//     memory and activations are partitioned across many cores, so no whole
//     tensor is ever resident. The value is still reported there because it
//     characterises the graph, but it must not be read as a hardware figure.
//   - Retained bytes counts values produced in the forward phase whose last
//     consumer is in the backward phase. That is the activation stash a
//     training step must keep somewhere, and it is the quantity gradient
//     checkpointing trades against recomputation.
//
// Where a backend reports its own memory accounting, the adapter publishes it
// as metrics and it appears alongside these estimates rather than being
// replaced by them.

struct TensorRef {
  double bytes = 0.0;
  std::size_t def_pos = 0;
  std::size_t last_use = 0;
  std::string phase;
  bool has_def = false;
};

double ResultBytes(const dfabit::metadata::OpDesc& op) {
  const auto it = op.attributes.find("result_bytes");
  if (it == op.attributes.end()) {
    // Backends that do not publish a separate result size fall back to the
    // whole-operator footprint, which over-counts when an operator reads more
    // than it writes.
    return static_cast<double>(op.estimated_bytes);
  }
  try {
    return std::stod(it->second);
  } catch (...) {
    return 0.0;
  }
}

std::string ModuleOf(const dfabit::metadata::OpDesc& op) {
  const auto it = op.attributes.find("module_path");
  if (it == op.attributes.end() || it->second.empty()) {
    return "<none>";
  }
  // Group at the enclosing module rather than the leaf, so a transformer block
  // aggregates instead of splintering across every projection inside it.
  const auto pos = it->second.rfind('.');
  return pos == std::string::npos ? it->second : it->second.substr(0, pos);
}

std::string Csv(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    out += (c == ',' || c == '\n') ? '_' : c;
  }
  return out;
}

}  // namespace

std::string DataflowMemoryProxyTool::name() const {
  return "dataflow_memory_proxy";
}

dfabit::core::Status DataflowMemoryProxyTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  filtered_ops_.clear();
  runtime_metrics_.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  (void)compile_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  (void)compile_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  filtered_ops_ = ToolServices::FilterOps(*ctx, ctx->metadata_ops());
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  runtime_metrics_ = ToolServices::FilterMetrics(*ctx, runtime_artifacts.metrics);
  return dfabit::core::Status::Ok();
}

std::string DataflowMemoryProxyTool::OutputDir(const dfabit::api::Context& ctx) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  const auto path =
      std::filesystem::path(base) / "tools" / "dataflow_memory_proxy";
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  return path.string();
}

dfabit::core::Status DataflowMemoryProxyTool::WritePerOpTable(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  // Build the def-use chain. A value is live from the operator that defines it
  // until the last operator that reads it.
  std::unordered_map<std::string, TensorRef> refs;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (const auto& out : ops[i].outputs) {
      if (out.name.empty()) continue;
      auto& r = refs[out.name];
      r.bytes = ResultBytes(ops[i]);
      r.def_pos = i;
      r.last_use = i;
      r.phase = ops[i].stage_tag;
      r.has_def = true;
    }
  }
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (const auto& in : ops[i].inputs) {
      const auto it = refs.find(in.name);
      if (it != refs.end() && it->second.has_def) {
        it->second.last_use = i;
      }
    }
  }

  std::unordered_map<std::string, std::size_t> observed;
  for (const auto& metric : runtime_metrics) {
    if (metric.stable_id != 0) {
      observed[std::to_string(metric.stable_id)]++;
    }
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal,
            "failed to open memory proxy table: " + path};
  }

  ofs << "stable_id,op_name,dialect,phase,module,operands,result_bytes,"
         "footprint_bytes,live_bytes_after,reuse_distance,retained_to_backward\n";

  std::unordered_map<std::string, bool> live;
  double live_bytes = 0.0;

  for (std::size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];

    for (const auto& out : op.outputs) {
      const auto it = refs.find(out.name);
      if (it == refs.end() || live[out.name]) continue;
      live[out.name] = true;
      live_bytes += it->second.bytes;
    }

    std::size_t reuse = 0;
    bool retained = false;
    if (!op.outputs.empty()) {
      const auto it = refs.find(op.outputs.front().name);
      if (it != refs.end() && it->second.has_def) {
        reuse = it->second.last_use - it->second.def_pos;
        // A value defined in the forward phase and last read in the backward
        // phase is stashed across the boundary. This is what a training step
        // must retain and what checkpointing recomputes instead.
        retained = it->second.phase == "forward" &&
                   it->second.last_use < ops.size() &&
                   ops[it->second.last_use].stage_tag == "backward";
      }
    }

    ofs << op.stable_id << ","
        << Csv(op.op_name) << ","
        << Csv(op.dialect) << ","
        << Csv(op.stage_tag) << ","
        << Csv(ModuleOf(op)) << ","
        << op.inputs.size() << ","
        << static_cast<long long>(ResultBytes(op)) << ","
        << op.estimated_bytes << ","
        << static_cast<long long>(live_bytes) << ","
        << reuse << ","
        << (retained ? 1 : 0) << "\n";

    for (const auto& name : {op.outputs.empty() ? std::string() : op.outputs.front().name}) {
      (void)name;
    }
    // Retire values whose last use is this operator.
    for (auto& kv : refs) {
      if (!kv.second.has_def || !live[kv.first]) continue;
      if (kv.second.last_use <= i) {
        live[kv.first] = false;
        live_bytes -= kv.second.bytes;
      }
    }
  }

  (void)observed;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::WriteSummary(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::unordered_map<std::string, TensorRef> refs;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (const auto& out : ops[i].outputs) {
      if (out.name.empty()) continue;
      auto& r = refs[out.name];
      r.bytes = ResultBytes(ops[i]);
      r.def_pos = i;
      r.last_use = i;
      r.phase = ops[i].stage_tag;
      r.has_def = true;
    }
  }
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (const auto& in : ops[i].inputs) {
      const auto it = refs.find(in.name);
      if (it != refs.end() && it->second.has_def) {
        it->second.last_use = i;
      }
    }
  }

  double peak_live = 0.0;
  std::size_t peak_at = 0;
  double live_bytes = 0.0;
  std::unordered_map<std::string, bool> live;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (const auto& out : ops[i].outputs) {
      const auto it = refs.find(out.name);
      if (it == refs.end() || live[out.name]) continue;
      live[out.name] = true;
      live_bytes += it->second.bytes;
    }
    if (live_bytes > peak_live) {
      peak_live = live_bytes;
      peak_at = i;
    }
    for (auto& kv : refs) {
      if (!kv.second.has_def || !live[kv.first]) continue;
      if (kv.second.last_use <= i) {
        live[kv.first] = false;
        live_bytes -= kv.second.bytes;
      }
    }
  }

  double retained = 0.0;
  std::size_t retained_count = 0;
  std::map<std::string, double> bytes_by_phase;
  std::map<std::string, double> bytes_by_module;
  double total_footprint = 0.0;
  double reuse_sum = 0.0;
  std::size_t reuse_n = 0;

  for (std::size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    total_footprint += static_cast<double>(op.estimated_bytes);
    bytes_by_phase[op.stage_tag.empty() ? "<none>" : op.stage_tag] +=
        static_cast<double>(op.estimated_bytes);
    bytes_by_module[ModuleOf(op)] += static_cast<double>(op.estimated_bytes);

    if (op.outputs.empty()) continue;
    const auto it = refs.find(op.outputs.front().name);
    if (it == refs.end() || !it->second.has_def) continue;

    reuse_sum += static_cast<double>(it->second.last_use - it->second.def_pos);
    ++reuse_n;

    if (it->second.phase == "forward" && it->second.last_use < ops.size() &&
        ops[it->second.last_use].stage_tag == "backward") {
      retained += it->second.bytes;
      ++retained_count;
    }
  }

  std::string dominant_phase = "<none>";
  double dominant_bytes = 0.0;
  for (const auto& kv : bytes_by_phase) {
    if (kv.second > dominant_bytes) {
      dominant_bytes = kv.second;
      dominant_phase = kv.first;
    }
  }

  std::string dominant_module = "<none>";
  double dominant_module_bytes = 0.0;
  for (const auto& kv : bytes_by_module) {
    if (kv.second > dominant_module_bytes) {
      dominant_module_bytes = kv.second;
      dominant_module = kv.first;
    }
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal,
            "failed to open memory proxy summary: " + path};
  }

  ofs << "metric,value,unit,note\n";
  ofs << "operators," << ops.size() << ",count,\n";
  ofs << "total_footprint_bytes," << static_cast<long long>(total_footprint)
      << ",B,logical tensor volume summed over operators\n";
  ofs << "peak_live_bytes," << static_cast<long long>(peak_live)
      << ",B,sequential-execution residency; not a hardware figure on "
         "partitioned or streaming backends\n";
  ofs << "peak_live_at_op," << peak_at << ",index,\n";
  ofs << "peak_live_op_name,"
      << (peak_at < ops.size() ? Csv(ops[peak_at].op_name) : std::string())
      << ",name,\n";
  ofs << "retained_to_backward_bytes," << static_cast<long long>(retained)
      << ",B,forward values whose last use is in the backward phase\n";
  ofs << "retained_to_backward_tensors," << retained_count << ",count,\n";
  ofs << "mean_reuse_distance,"
      << (reuse_n ? reuse_sum / static_cast<double>(reuse_n) : 0.0)
      << ",operators,distance from definition to last use\n";
  ofs << "dominant_phase," << Csv(dominant_phase) << ",name,\n";
  ofs << "dominant_phase_bytes," << static_cast<long long>(dominant_bytes) << ",B,\n";
  ofs << "dominant_module," << Csv(dominant_module) << ",name,\n";
  ofs << "dominant_module_bytes," << static_cast<long long>(dominant_module_bytes)
      << ",B,\n";
  ofs << "runtime_metrics_seen," << runtime_metrics.size() << ",count,\n";

  for (const auto& kv : bytes_by_phase) {
    ofs << "phase_bytes:" << Csv(kv.first) << ","
        << static_cast<long long>(kv.second) << ",B,\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status DataflowMemoryProxyTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  if (filtered_ops_.empty()) {
    return dfabit::core::Status::Ok();
  }

  const auto dir = OutputDir(*ctx);
  auto st = WritePerOpTable(dir + "/dataflow_memory_proxy_table.csv",
                            filtered_ops_, runtime_metrics_);
  if (!st.ok()) {
    return st;
  }
  return WriteSummary(dir + "/dataflow_memory_proxy_summary.csv",
                      filtered_ops_, runtime_metrics_);
}

std::unique_ptr<Tool> CreateDataflowMemoryProxyTool() {
  return std::make_unique<DataflowMemoryProxyTool>();
}

}  // namespace dfabit::tools::builtin
