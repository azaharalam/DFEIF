#include "dfabit/tools/builtin/semantic_attribution_tool.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "dfabit/tools/tool_context.h"

namespace dfabit::tools::builtin {

namespace {

// What this tool answers: what did the compiler make of the code that was
// written.
//
// Most operators in a compiled training graph correspond to no line the
// developer typed. Autodiff synthesises the backward pass, the optimizer is
// expanded inline, and metric computation is generated alongside. A tool that
// reports on the compiled graph is therefore reporting mostly on code nobody
// wrote, unless it can say where each operator came from.
//
// The compiler records that correspondence: each operator carries a location
// referring to the module it belongs to, the source-level operator it was
// lowered from, and a callsite chain. This tool reads it and reports the
// expansion: for each site in the model definition, how many operators it
// became and across which phases. Operators with no model-code frame are
// reported separately, since they are what the toolchain added on its own.
//
// The runtime column is present but will read zero on backends that do not tag
// runtime records with operator identity. Neither evaluated backend does: the
// Cerebras runtime reports aggregate throughput and the Edge TPU executes as a
// single fused operator. It is retained because the join is what the framework
// would use if a backend supplied such records, and reporting zero is more
// useful than omitting the column.

std::string Attr(const dfabit::metadata::OpDesc& op, const char* key) {
  const auto it = op.attributes.find(key);
  return it == op.attributes.end() ? std::string() : it->second;
}

std::string Csv(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    out += (c == ',' || c == '\n' || c == '"') ? '_' : c;
  }
  return out;
}

// The leaf of a module path is the specific layer; the parent is the block it
// belongs to. Reporting the parent groups a transformer block together instead
// of splintering it across every projection inside.
std::string ParentModule(const std::string& path) {
  if (path.empty()) return "<none>";
  const auto pos = path.rfind('.');
  return pos == std::string::npos ? path : path.substr(0, pos);
}

struct SiteStats {
  std::size_t total = 0;
  std::map<std::string, std::size_t> by_phase;
  std::map<std::string, std::size_t> aten_ops;
  double flops = 0.0;
  double bytes = 0.0;
};

}  // namespace

std::string SemanticAttributionTool::name() const {
  return "semantic_attribution";
}

dfabit::core::Status SemanticAttributionTool::OnRegister(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnInit(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  filtered_ops_.clear();
  runtime_metrics_.clear();
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnCompileBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  (void)compile_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnCompileEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts) {
  (void)compile_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  filtered_ops_ = ToolServices::FilterOps(*ctx, ctx->metadata_ops());
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnLoadBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnLoadEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnRunBegin(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  (void)runtime_artifacts;
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnRunEnd(
    dfabit::api::Context* ctx,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  runtime_metrics_ = ToolServices::FilterMetrics(*ctx, runtime_artifacts.metrics);
  return dfabit::core::Status::Ok();
}

std::string SemanticAttributionTool::OutputDir(const dfabit::api::Context& ctx) {
  const auto& base = ctx.run_context().config().output.base_output_dir;
  const auto path =
      std::filesystem::path(base) / "tools" / "semantic_attribution";
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  return path.string();
}

dfabit::core::Status SemanticAttributionTool::WriteAttributionTable(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::unordered_set<std::uint64_t> observed;
  for (const auto& metric : runtime_metrics) {
    if (metric.stable_id != 0) {
      observed.insert(metric.stable_id);
    }
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal,
            "failed to open semantic attribution table: " + path};
  }

  ofs << "stable_id,op_name,dialect,phase,module,block,aten_op,source_site,"
         "estimated_flops,estimated_bytes,runtime_attributed\n";

  for (const auto& op : ops) {
    const auto module = Attr(op, "module_path");
    ofs << op.stable_id << ","
        << Csv(op.op_name) << ","
        << Csv(op.dialect) << ","
        << Csv(op.stage_tag) << ","
        << Csv(module.empty() ? "<none>" : module) << ","
        << Csv(ParentModule(module)) << ","
        << Csv(Attr(op, "aten_op")) << ","
        << Csv(Attr(op, "src")) << ","
        << op.estimated_flops << ","
        << op.estimated_bytes << ","
        << (observed.count(op.stable_id) ? 1 : 0) << "\n";
  }
  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::WriteSummary(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops,
    const std::vector<dfabit::adapters::MetricSample>& runtime_metrics) const {
  std::map<std::string, SiteStats> by_site;
  std::map<std::string, std::size_t> by_phase;
  std::size_t with_module = 0;
  std::size_t with_source = 0;
  std::size_t attributed = 0;

  std::unordered_set<std::uint64_t> observed;
  for (const auto& metric : runtime_metrics) {
    if (metric.stable_id != 0) observed.insert(metric.stable_id);
  }

  for (const auto& op : ops) {
    const auto module = Attr(op, "module_path");
    const auto src = Attr(op, "src");
    if (!module.empty()) ++with_module;
    if (!src.empty()) ++with_source;
    if (observed.count(op.stable_id)) ++attributed;

    by_phase[op.stage_tag.empty() ? "<none>" : op.stage_tag]++;

    // Attribution has two levels, and collapsing them loses information.
    //
    // A forward operator carries a source line, because the developer's call
    // sits in its callsite chain. A backward or optimizer operator does not:
    // autodiff and optimizer expansion generate it, and no user callsite
    // appears anywhere in its chain. It still carries a module path though, so
    // it is not unattributed -- a gradient operator for
    // layers.11.self_attn.proj_q_dense_layer belongs to that module even
    // though no line of code produced it directly.
    //
    // Reporting those in one bucket alongside materialised constants would
    // hide that distinction. Sites are keyed by source line where one exists,
    // by module otherwise, and only operators with neither are unattributed.
    std::string key;
    if (!src.empty()) {
      key = src;
    } else if (!module.empty()) {
      key = "module:" + module;
    } else {
      key = "<unattributed>";
    }
    auto& stats = by_site[key];
    ++stats.total;
    stats.by_phase[op.stage_tag.empty() ? "<none>" : op.stage_tag]++;
    const auto aten = Attr(op, "aten_op");
    if (!aten.empty()) stats.aten_ops[aten]++;
    stats.flops += static_cast<double>(op.estimated_flops);
    stats.bytes += static_cast<double>(op.estimated_bytes);
  }

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal,
            "failed to open semantic attribution summary: " + path};
  }

  ofs << "metric,value,unit,note\n";
  ofs << "operators," << ops.size() << ",count,\n";
  ofs << "operators_with_module," << with_module << ",count,"
      << "recovered from compiler location records\n";
  ofs << "operators_with_source_site," << with_source << ",count,"
      << "first callsite outside framework internals\n";
  {
    std::size_t line_sites = 0;
    for (const auto& kv : by_site) {
      if (kv.first.rfind("module:", 0) != 0 && kv.first != "<unattributed>") {
        ++line_sites;
      }
    }
    ofs << "source_sites," << line_sites
        << ",count,distinct lines of model code\n";
  }
  ofs << "operators_attributed_by_module_only,"
      << (with_module > with_source ? with_module - with_source : 0)
      << ",count,generated by autodiff or optimizer expansion; carries a "
         "module but no source line\n";
  ofs << "operators_unattributed,"
      << (by_site.count("<unattributed>")
              ? by_site.at("<unattributed>").total
              : 0)
      << ",count,neither module nor source; compiler-materialised values\n";
  ofs << "runtime_attributed_operators," << attributed << ",count,"
      << "requires a backend that tags runtime records with operator identity\n";
  ofs << "runtime_metrics_seen," << runtime_metrics.size() << ",count,\n";

  for (const auto& kv : by_phase) {
    ofs << "phase_operators:" << Csv(kv.first) << "," << kv.second << ",count,\n";
  }

  // Expansion per source site, largest first. This is the figure that says
  // what one line of model code became after lowering.
  std::vector<std::pair<std::string, SiteStats>> sites(by_site.begin(),
                                                       by_site.end());
  std::sort(sites.begin(), sites.end(),
            [](const auto& a, const auto& b) { return a.second.total > b.second.total; });

  ofs << "\nsource_site,operators,forward,backward,optimizer,other,"
         "top_aten_op,estimated_flops,estimated_bytes\n";
  for (const auto& kv : sites) {
    const auto& s = kv.second;
    std::size_t other = s.total;
    for (const char* p : {"forward", "backward", "optimizer"}) {
      const auto it = s.by_phase.find(p);
      if (it != s.by_phase.end()) other -= it->second;
    }
    std::string top;
    std::size_t top_n = 0;
    for (const auto& a : s.aten_ops) {
      if (a.second > top_n) {
        top_n = a.second;
        top = a.first;
      }
    }
    const auto phase = [&s](const char* p) -> std::size_t {
      const auto it = s.by_phase.find(p);
      return it == s.by_phase.end() ? 0 : it->second;
    };
    ofs << Csv(kv.first) << "," << s.total << ","
        << phase("forward") << "," << phase("backward") << ","
        << phase("optimizer") << "," << other << ","
        << Csv(top) << ","
        << static_cast<long long>(s.flops) << ","
        << static_cast<long long>(s.bytes) << "\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status SemanticAttributionTool::OnShutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  if (filtered_ops_.empty()) {
    return dfabit::core::Status::Ok();
  }

  const auto dir = OutputDir(*ctx);
  auto st = WriteAttributionTable(dir + "/semantic_attribution_table.csv",
                                  filtered_ops_, runtime_metrics_);
  if (!st.ok()) {
    return st;
  }
  return WriteSummary(dir + "/semantic_attribution_summary.csv",
                      filtered_ops_, runtime_metrics_);
}

std::unique_ptr<Tool> CreateSemanticAttributionTool() {
  return std::make_unique<SemanticAttributionTool>();
}

}  // namespace dfabit::tools::builtin
