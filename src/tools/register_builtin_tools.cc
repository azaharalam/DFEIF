#include "dfabit/tools/register_builtin_tools.h"

#include "dfabit/tools/builtin/portability_report_tool.h"
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

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::tools