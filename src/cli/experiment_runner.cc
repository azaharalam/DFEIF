#include "dfabit/cli/experiment_runner.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include "dfabit/cli/runner.h"

namespace dfabit::cli {

namespace {

std::string TrialDir(
    const std::string& base,
    const std::string& experiment_name,
    const std::string& mode,
    int trial) {
  std::ostringstream ss;
  ss << "trial_" << std::setw(3) << std::setfill('0') << trial;
  return (std::filesystem::path(base) / experiment_name / mode / ss.str()).string();
}

}  // namespace

dfabit::core::Status ExperimentRunner::RunSpecs(
    const std::vector<ExperimentSpec>& specs,
    std::vector<ExperimentRunRecord>* records) const {
  if (!records) {
    return {dfabit::core::StatusCode::kInvalidArgument, "records is null"};
  }

  records->clear();

  for (const auto& spec : specs) {
    for (const auto& mode : spec.modes) {
      for (int trial = 1; trial <= spec.options.repeat; ++trial) {
        CliOptions run_options = spec.options;
        run_options.mode = mode;
        run_options.repeat = 1;
        run_options.output_dir =
            TrialDir(spec.options.output_dir, spec.name.empty() ? "default" : spec.name, mode, trial);

        ExperimentRunRecord record;
        record.experiment_name = spec.name.empty() ? "default" : spec.name;
        record.mode = mode;
        record.trial = trial;
        record.output_dir = run_options.output_dir;

        const auto st = Run(run_options);
        record.success = st.ok();
        record.message = st.ok() ? "ok" : st.message();
        records->push_back(std::move(record));
      }
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status ExperimentRunner::WriteIndexCsv(
    const std::string& path,
    const std::vector<ExperimentRunRecord>& records) const {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());

  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to open experiment index csv: " + path};
  }

  ofs << "experiment_name,mode,trial,output_dir,success,message\n";
  for (const auto& record : records) {
    ofs << record.experiment_name << ","
        << record.mode << ","
        << record.trial << ","
        << record.output_dir << ","
        << (record.success ? 1 : 0) << ","
        << record.message << "\n";
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::cli