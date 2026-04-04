#include "dfabit/tools/register_builtin_tools.h"

#include "dfabit/tools/builtin/overhead_profiler_tool.h"
#include "dfabit/tools/builtin/portability_report_tool.h"
#include "dfabit/tools/builtin/semantic_attribution_tool.h"
#include "dfabit/tools/tool_registry.h"

namespace dfabit::tools {

dfabit::core::Status RegisterBuiltinTools() {
  if (!ToolRegistry::Instance().HasTool("portability_report")) {
    const auto st = ToolRegistry::Instance().Register(
        "portability_report",
        &dfabit::tools::builtin::CreatePortabilityReportTool);
    if (!st.ok()) {
      return st;
    }
  }

  if (!ToolRegistry::Instance().HasTool("overhead_profiler")) {
    const auto st = ToolRegistry::Instance().Register(
        "overhead_profiler",
        &dfabit::tools::builtin::CreateOverheadProfilerTool);
    if (!st.ok()) {
      return st;
    }
  }

  if (!ToolRegistry::Instance().HasTool("semantic_attribution")) {
    const auto st = ToolRegistry::Instance().Register(
        "semantic_attribution",
        &dfabit::tools::builtin::CreateSemanticAttributionTool);
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::tools