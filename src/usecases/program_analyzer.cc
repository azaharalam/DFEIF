#include "dfabit/usecases/program_analyzer.h"

#include "dfabit/analysis/report.h"
#include "dfabit/api/context.h"

namespace dfabit::usecases {

dfabit::core::Status ProgramAnalyzer::Run(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  const auto& out_path = ctx->run_context().config().output.program_analysis_csv_path;
  if (out_path.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "program_analysis_csv_path is empty"};
  }

  return dfabit::analysis::WriteProgramAnalysisCsv(out_path, ctx->ops());
}

}  // namespace dfabit::usecases