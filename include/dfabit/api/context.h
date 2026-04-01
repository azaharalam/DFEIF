#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dfabit/analysis/report.h"
#include "dfabit/core/framework_config.h"
#include "dfabit/core/run_context.h"
#include "dfabit/core/session.h"
#include "dfabit/metadata/op_desc.h"

namespace dfabit::api {

class Context {
 public:
  Context();
  explicit Context(dfabit::core::RunConfig run_cfg);

  void SetFrameworkConfig(dfabit::core::FrameworkConfig cfg);
  const dfabit::core::FrameworkConfig& framework_config() const { return framework_config_; }
  dfabit::core::FrameworkConfig& mutable_framework_config() { return framework_config_; }

  void SetRunConfig(dfabit::core::RunConfig cfg);
  const dfabit::core::RunContext& run_context() const { return run_context_; }
  dfabit::core::RunContext& mutable_run_context() { return run_context_; }

  const dfabit::core::Session& session() const { return session_; }
  dfabit::core::Session& mutable_session() { return session_; }

  void SetOps(std::vector<dfabit::analysis::OpRecord> ops);
  const std::vector<dfabit::analysis::OpRecord>& ops() const { return ops_; }

  void SetMetadataOps(std::vector<dfabit::metadata::OpDesc> ops);
  const std::vector<dfabit::metadata::OpDesc>& metadata_ops() const { return metadata_ops_; }

  void SetProperty(const std::string& key, std::string value);
  std::string GetProperty(const std::string& key) const;

 private:
  dfabit::core::FrameworkConfig framework_config_;
  dfabit::core::Session session_;
  dfabit::core::RunContext run_context_;
  std::vector<dfabit::analysis::OpRecord> ops_;
  std::vector<dfabit::metadata::OpDesc> metadata_ops_;
  std::unordered_map<std::string, std::string> properties_;
};

}  // namespace dfabit::api