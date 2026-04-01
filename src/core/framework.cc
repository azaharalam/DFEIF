#include "dfabit/core/framework.h"

#include <utility>

#include "dfabit/api/context.h"

namespace dfabit::core {

namespace {

RunConfig MergeRunConfig(const FrameworkConfig& framework_cfg, RunConfig run_cfg) {
  if (run_cfg.output.base_output_dir.empty()) {
    run_cfg.output.base_output_dir = framework_cfg.default_output.base_output_dir;
  }
  if (run_cfg.output.trace_jsonl_path.empty()) {
    run_cfg.output.trace_jsonl_path = framework_cfg.default_output.trace_jsonl_path;
  }
  if (run_cfg.output.report_dir.empty()) {
    run_cfg.output.report_dir = framework_cfg.default_output.report_dir;
  }
  if (run_cfg.output.program_analysis_csv_path.empty()) {
    run_cfg.output.program_analysis_csv_path = framework_cfg.default_output.program_analysis_csv_path;
  }

  if (run_cfg.backend.backend_name.empty()) {
    run_cfg.backend.backend_name = framework_cfg.default_backend.backend_name;
  }
  if (run_cfg.backend.provider_name.empty()) {
    run_cfg.backend.provider_name = framework_cfg.default_backend.provider_name;
  }
  if (run_cfg.backend.adapter_name.empty()) {
    run_cfg.backend.adapter_name = framework_cfg.default_backend.adapter_name;
  }

  if (!run_cfg.trace.enabled && framework_cfg.default_trace.enabled) {
    run_cfg.trace.enabled = framework_cfg.default_trace.enabled;
  }
  if (run_cfg.trace.buffer_capacity == 65536 && framework_cfg.default_trace.buffer_capacity != 65536) {
    run_cfg.trace.buffer_capacity = framework_cfg.default_trace.buffer_capacity;
  }
  if (!run_cfg.trace.flush_per_write && framework_cfg.default_trace.flush_per_write) {
    run_cfg.trace.flush_per_write = framework_cfg.default_trace.flush_per_write;
  }

  if (run_cfg.policy.mode == InstrumentationMode::kFull &&
      framework_cfg.default_policy.mode != InstrumentationMode::kFull) {
    run_cfg.policy.mode = framework_cfg.default_policy.mode;
  }
  if (run_cfg.policy.detail_level == DetailLevel::kFull &&
      framework_cfg.default_policy.detail_level != DetailLevel::kFull) {
    run_cfg.policy.detail_level = framework_cfg.default_policy.detail_level;
  }
  if (run_cfg.policy.include_ops.empty()) {
    run_cfg.policy.include_ops = framework_cfg.default_policy.include_ops;
  }
  if (run_cfg.policy.include_stages.empty()) {
    run_cfg.policy.include_stages = framework_cfg.default_policy.include_stages;
  }
  if (run_cfg.policy.include_dialects.empty()) {
    run_cfg.policy.include_dialects = framework_cfg.default_policy.include_dialects;
  }
  if (run_cfg.policy.sampling_ratio == 1.0) {
    run_cfg.policy.sampling_ratio = framework_cfg.default_policy.sampling_ratio;
  }
  if (run_cfg.policy.sampling_stride == 1) {
    run_cfg.policy.sampling_stride = framework_cfg.default_policy.sampling_stride;
  }
  if (run_cfg.policy.sampling_salt == 0) {
    run_cfg.policy.sampling_salt = framework_cfg.default_policy.sampling_salt;
  }

  for (const auto& kv : framework_cfg.attributes) {
    if (run_cfg.attributes.find(kv.first) == run_cfg.attributes.end()) {
      run_cfg.attributes.emplace(kv.first, kv.second);
    }
  }

  return run_cfg;
}

}  // namespace

dfabit::api::Context Framework::CreateContext() const {
  dfabit::api::Context ctx;
  ctx.SetFrameworkConfig(config_);
  return ctx;
}

dfabit::api::Context Framework::CreateContext(RunConfig run_cfg) const {
  dfabit::api::Context ctx(MergeRunConfig(config_, std::move(run_cfg)));
  ctx.SetFrameworkConfig(config_);
  return ctx;
}

}  // namespace dfabit::core