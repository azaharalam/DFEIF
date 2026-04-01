#include "dfabit/tools/tool_context.h"

#include <fstream>
#include <utility>

namespace dfabit::tools {

dfabit::policy::PolicyEngine ToolServices::BuildPolicyEngine(const dfabit::api::Context& ctx) {
  return dfabit::policy::PolicyEngine(ctx.run_context().config().policy);
}

std::vector<dfabit::metadata::OpDesc> ToolServices::FilterOps(
    const dfabit::api::Context& ctx,
    const std::vector<dfabit::metadata::OpDesc>& ops) {
  auto engine = BuildPolicyEngine(ctx);
  std::vector<dfabit::metadata::OpDesc> out;
  out.reserve(ops.size());

  for (const auto& op : ops) {
    if (engine.ShouldInstrument(op)) {
      out.push_back(op);
    }
  }

  return out;
}

std::vector<dfabit::adapters::MetricSample> ToolServices::FilterMetrics(
    const dfabit::api::Context& ctx,
    const std::vector<dfabit::adapters::MetricSample>& metrics) {
  auto engine = BuildPolicyEngine(ctx);
  std::vector<dfabit::adapters::MetricSample> out;
  out.reserve(metrics.size());

  for (const auto& metric : metrics) {
    if (metric.stable_id == 0 || engine.ShouldEmit(metric.stable_id)) {
      out.push_back(metric);
    }
  }

  return out;
}

dfabit::core::Status ToolServices::WriteMetricsCsv(
    const std::string& path,
    const std::vector<dfabit::adapters::MetricSample>& metrics) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open metrics csv: " + path};
  }

  ofs << "name,value,unit,stage,stable_id,attributes\n";
  for (const auto& metric : metrics) {
    ofs << metric.name << ","
        << metric.value << ","
        << metric.unit << ","
        << metric.stage << ","
        << metric.stable_id << ",";
    bool first = true;
    for (const auto& kv : metric.attributes) {
      if (!first) {
        ofs << ";";
      }
      first = false;
      ofs << kv.first << "=" << kv.second;
    }
    ofs << "\n";
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status ToolServices::WriteOpTableCsv(
    const std::string& path,
    const std::vector<dfabit::metadata::OpDesc>& ops) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {dfabit::core::StatusCode::kInternal, "failed to open op csv: " + path};
  }

  ofs << "stable_id,op_name,dialect,stage_tag,estimated_flops,estimated_bytes,input_count,output_count\n";
  for (const auto& op : ops) {
    ofs << op.stable_id << ","
        << op.op_name << ","
        << op.dialect << ","
        << op.stage_tag << ","
        << op.estimated_flops << ","
        << op.estimated_bytes << ","
        << op.inputs.size() << ","
        << op.outputs.size() << "\n";
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::tools