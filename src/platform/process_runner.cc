#include "dfabit/platform/process_runner.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace dfabit::platform {

namespace {

std::string ScriptPath(
    const std::string& base_dir,
    const std::string& name) {
  return (std::filesystem::path(base_dir) / (".dfabit_" + name + ".sh")).string();
}

}  // namespace

dfabit::core::Status ProcessRunner::Run(
    const ProcessSpec& spec,
    ProcessResult* result) const {
  if (!result) {
    return {dfabit::core::StatusCode::kInvalidArgument, "result is null"};
  }
  if (spec.command.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "command is empty"};
  }

  const auto work_dir = spec.working_directory.empty() ? "." : spec.working_directory;
  std::filesystem::create_directories(work_dir);

  if (!spec.stdout_path.empty()) {
    std::filesystem::create_directories(std::filesystem::path(spec.stdout_path).parent_path());
  }
  if (!spec.stderr_path.empty()) {
    std::filesystem::create_directories(std::filesystem::path(spec.stderr_path).parent_path());
  }

  const auto script_path = ScriptPath(work_dir, spec.name.empty() ? "process" : spec.name);
  {
    std::ofstream ofs(script_path);
    if (!ofs.is_open()) {
      return {
          dfabit::core::StatusCode::kInternal,
          "failed to create process script: " + script_path};
    }
    ofs << "#!/usr/bin/env bash\n";
    ofs << "set -euo pipefail\n";
    ofs << "cd \"" << work_dir << "\"\n";
    ofs << spec.command << "\n";
  }

  const auto chmod_cmd = "chmod +x \"" + script_path + "\"";
  if (std::system(chmod_cmd.c_str()) != 0) {
    return {
        dfabit::core::StatusCode::kInternal,
        "failed to chmod process script: " + script_path};
  }

  std::string invoke = "\"" + script_path + "\"";
  if (!spec.stdout_path.empty()) {
    invoke += " > \"" + spec.stdout_path + "\"";
  }
  if (!spec.stderr_path.empty()) {
    invoke += " 2> \"" + spec.stderr_path + "\"";
  } else {
    invoke += " 2>/dev/null";
  }

  const auto start = std::chrono::steady_clock::now();
  const int rc = std::system(invoke.c_str());
  const auto end = std::chrono::steady_clock::now();

  result->exit_code = rc;
  result->elapsed_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  result->stdout_path = spec.stdout_path;
  result->stderr_path = spec.stderr_path;

  if (rc != 0) {
    return {
        dfabit::core::StatusCode::kInternal,
        "external process failed: " + spec.command};
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::platform