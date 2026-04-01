#include "dfabit/api/context.h"

namespace dfabit::api {

Context::Context() : session_(framework_config_), run_context_(dfabit::core::RunConfig{}) {}

Context::Context(dfabit::core::RunConfig run_cfg)
    : session_(framework_config_), run_context_(std::move(run_cfg)) {}

void Context::SetFrameworkConfig(dfabit::core::FrameworkConfig cfg) {
  framework_config_ = std::move(cfg);
  session_.mutable_config() = framework_config_;
}

void Context::SetRunConfig(dfabit::core::RunConfig cfg) {
  run_context_ = dfabit::core::RunContext(std::move(cfg));
}

void Context::SetOps(std::vector<dfabit::analysis::OpRecord> ops) {
  ops_ = std::move(ops);
}

void Context::SetMetadataOps(std::vector<dfabit::metadata::OpDesc> ops) {
  metadata_ops_ = std::move(ops);
}

void Context::SetProperty(const std::string& key, std::string value) {
  properties_[key] = std::move(value);
}

std::string Context::GetProperty(const std::string& key) const {
  const auto it = properties_.find(key);
  return it == properties_.end() ? std::string() : it->second;
}

}  // namespace dfabit::api