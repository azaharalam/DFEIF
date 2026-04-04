#include "dfabit/cli/runner.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dfabit/adapters/backend_adapter.h"
#include "dfabit/adapters/backend_registry.h"
#include "dfabit/adapters/register_builtin_adapters.h"
#include "dfabit/analysis/lightweight_fit.h"
#include "dfabit/analysis/overhead_engine.h"
#include "dfabit/analysis/reporting.h"
#include "dfabit/analysis/scalability_runner.h"
#include "dfabit/api/context.h"
#include "dfabit/core/framework_config.h"
#include "dfabit/tools/register_builtin_tools.h"
#include "dfabit/tools/tool_context.h"
#include "dfabit/tools/tool_manager.h"
#include "dfabit/tools/tool_registry.h"

namespace dfabit::cli {

namespace {

bool HasMetric(
    const std::vector<dfabit::adapters::MetricSample>& metrics,
    const std::string& name) {
  for (const auto& metric : metrics) {
    if (metric.name == name) {
      return true;
    }
  }
  return false;
}

dfabit::core::Status ConfigureContext(
    const CliOptions& options,
    dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  ctx->SetProperty("active_adapter", options.backend);

  if (!options.mlir_path.empty()) {
    ctx->SetProperty("mlir_module_path", options.mlir_path);
    ctx->mutable_run_context().SetAttribute("mlir_module_path", options.mlir_path);
  }

  if (!options.graph_path.empty()) {
    if (options.backend == "cerebras") {
      ctx->SetProperty("cerebras_graph_path", options.graph_path);
      ctx->mutable_run_context().SetAttribute("cerebras_graph_path", options.graph_path);
    } else if (options.backend == "sambanova") {
      ctx->SetProperty("sambanova_graph_path", options.graph_path);
      ctx->mutable_run_context().SetAttribute("sambanova_graph_path", options.graph_path);
    }
  }

  if (!options.sidecar_path.empty()) {
    if (options.backend == "cerebras") {
      ctx->SetProperty("cerebras_sidecar_path", options.sidecar_path);
      ctx->mutable_run_context().SetAttribute("cerebras_sidecar_path", options.sidecar_path);
    } else if (options.backend == "sambanova") {
      ctx->SetProperty("sambanova_sidecar_path", options.sidecar_path);
      ctx->mutable_run_context().SetAttribute("sambanova_sidecar_path", options.sidecar_path);
    }
  }

  if (!options.compile_report_path.empty()) {
    if (options.backend == "cerebras") {
      ctx->SetProperty("cerebras_compile_report_path", options.compile_report_path);
      ctx->mutable_run_context().SetAttribute(
          "cerebras_compile_report_path", options.compile_report_path);
    } else if (options.backend == "sambanova") {
      ctx->SetProperty("sambanova_compile_report_path", options.compile_report_path);
      ctx->mutable_run_context().SetAttribute(
          "sambanova_compile_report_path", options.compile_report_path);
    }
  }

  if (!options.runtime_log_path.empty()) {
    if (options.backend == "gpu_mlir") {
      ctx->SetProperty("runtime_metrics_path", options.runtime_log_path);
      ctx->mutable_run_context().SetAttribute("runtime_metrics_path", options.runtime_log_path);
    } else if (options.backend == "cerebras") {
      ctx->SetProperty("cerebras_runtime_log_path", options.runtime_log_path);
      ctx->mutable_run_context().SetAttribute(
          "cerebras_runtime_log_path", options.runtime_log_path);
    } else if (options.backend == "sambanova") {
      ctx->SetProperty("sambanova_runtime_log_path", options.runtime_log_path);
      ctx->mutable_run_context().SetAttribute(
          "sambanova_runtime_log_path", options.runtime_log_path);
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status BuildRunConfig(
    const CliOptions& options,
    dfabit::core::RunConfig* run_cfg) {
  if (!run_cfg) {
    return {dfabit::core::StatusCode::kInvalidArgument, "run_cfg is null"};
  }

  if (options.backend.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "backend is empty"};
  }

  if (options.output_dir.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "output_dir is empty"};
  }

  run_cfg->run_id = options.backend + "_run";
  run_cfg->model_name = options.backend + "_model";
  run_cfg->backend.backend_name = options.backend;
  run_cfg->backend.provider_name = options.backend;
  run_cfg->backend.adapter_name = options.backend;
  run_cfg->output.base_output_dir = options.output_dir;
  run_cfg->output.report_dir = (std::filesystem::path(options.output_dir) / "reports").string();
  run_cfg->output.trace_jsonl_path = (std::filesystem::path(options.output_dir) / "trace.jsonl").string();
  run_cfg->output.program_analysis_csv_path =
      (std::filesystem::path(options.output_dir) / "program_analysis.csv").string();
  run_cfg->policy.mode = dfabit::core::InstrumentationMode::kFull;
  run_cfg->policy.detail_level = dfabit::core::DetailLevel::kFull;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status AddBuiltinTools(
    const CliOptions& options,
    dfabit::tools::ToolManager* tool_manager) {
  if (!tool_manager) {
    return {dfabit::core::StatusCode::kInvalidArgument, "tool_manager is null"};
  }

  if (options.enable_portability_tool) {
    auto portability_tool =
        dfabit::tools::ToolRegistry::Instance().Create("portability_report");
    if (!portability_tool) {
      return {
          dfabit::core::StatusCode::kNotFound,
          "failed to create tool: portability_report"};
    }
    auto st = tool_manager->AddTool(std::move(portability_tool));
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status WriteBundleOutputs(
    const CliOptions& options,
    const dfabit::api::Context& ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts) {
  const auto reports_dir = std::filesystem::path(options.output_dir) / "reports";
  std::filesystem::create_directories(reports_dir);

  auto st = dfabit::tools::ToolServices::WriteOpTableCsv(
      (reports_dir / "metadata_ops.csv").string(),
      ctx.metadata_ops());
  if (!st.ok()) {
    return st;
  }

  st = dfabit::tools::ToolServices::WriteMetricsCsv(
      (reports_dir / "compile_metrics.csv").string(),
      compile_artifacts.metrics);
  if (!st.ok()) {
    return st;
  }

  st = dfabit::tools::ToolServices::WriteMetricsCsv(
      (reports_dir / "runtime_metrics.csv").string(),
      runtime_artifacts.metrics);
  if (!st.ok()) {
    return st;
  }

  {
    std::ofstream ofs((reports_dir / "run_summary.txt").string());
    if (!ofs.is_open()) {
      return {
          dfabit::core::StatusCode::kInternal,
          "failed to open run summary"};
    }

    ofs << "backend=" << options.backend << "\n";
    ofs << "output_dir=" << options.output_dir << "\n";
    ofs << "metadata_op_count=" << ctx.metadata_ops().size() << "\n";
    ofs << "compile_metric_count=" << compile_artifacts.metrics.size() << "\n";
    ofs << "runtime_metric_count=" << runtime_artifacts.metrics.size() << "\n";
    ofs << "graph_name=" << ctx.run_context().GetAttribute("graph_name") << "\n";
    ofs << "manifest_path=" << ctx.GetProperty("manifest_path") << "\n";
  }

  const bool has_overhead_inputs =
      HasMetric(runtime_artifacts.metrics, "baseline_latency_ms") &&
      HasMetric(runtime_artifacts.metrics, "instrumented_latency_ms");

  if (has_overhead_inputs) {
    dfabit::analysis::OverheadEngine overhead_engine;
    overhead_engine.AddMetricSamples(
        options.backend,
        "full",
        ctx.run_context().config().model_name,
        runtime_artifacts.metrics);

    dfabit::analysis::LightweightFitEngine fit_engine;
    dfabit::analysis::LightweightFitResult fit_result;
    st = fit_engine.Fit(overhead_engine.samples(), &fit_result);
    if (!st.ok()) {
      return st;
    }

    dfabit::analysis::ScalabilityRunner scalability_runner;
    scalability_runner.AddFromOverheadSamples("event_count", overhead_engine.samples());

    dfabit::analysis::Reporting reporting;
    const auto analysis_dir = (std::filesystem::path(options.output_dir) / "analysis").string();

    st = reporting.WriteOverheadBundle(analysis_dir, overhead_engine);
    if (!st.ok()) {
      return st;
    }

    st = reporting.WriteLightweightBundle(analysis_dir, fit_result);
    if (!st.ok()) {
      return st;
    }

    st = reporting.WriteScalabilityBundle(analysis_dir, scalability_runner);
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status ValidateOptions(const CliOptions& options) {
  if (options.backend.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "missing --backend"};
  }
  if (options.output_dir.empty()) {
    return {dfabit::core::StatusCode::kInvalidArgument, "missing --out"};
  }

  if (options.backend == "gpu_mlir") {
    if (options.mlir_path.empty()) {
      return {dfabit::core::StatusCode::kInvalidArgument, "gpu_mlir requires --mlir"};
    }
  } else if (options.backend == "cerebras") {
    if (options.graph_path.empty()) {
      return {dfabit::core::StatusCode::kInvalidArgument, "cerebras requires --graph"};
    }
  } else if (options.backend == "sambanova") {
    if (options.graph_path.empty()) {
      return {dfabit::core::StatusCode::kInvalidArgument, "sambanova requires --graph"};
    }
  } else {
    return {
        dfabit::core::StatusCode::kInvalidArgument,
        "unsupported backend: " + options.backend};
  }

  return dfabit::core::Status::Ok();
}

}  // namespace

dfabit::core::Status Run(const CliOptions& options) {
  auto st = ValidateOptions(options);
  if (!st.ok()) {
    return st;
  }

  st = dfabit::adapters::RegisterBuiltinAdapters();
  if (!st.ok()) {
    return st;
  }

  st = dfabit::tools::RegisterBuiltinTools();
  if (!st.ok()) {
    return st;
  }

  auto adapter =
      dfabit::adapters::BackendRegistry::Instance().Create(options.backend);
  if (!adapter) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "backend adapter not found: " + options.backend};
  }

  dfabit::core::RunConfig run_cfg;
  st = BuildRunConfig(options, &run_cfg);
  if (!st.ok()) {
    return st;
  }

  std::filesystem::create_directories(options.output_dir);
  std::filesystem::create_directories(run_cfg.output.report_dir);

  dfabit::api::Context ctx(run_cfg);
  st = ConfigureContext(options, &ctx);
  if (!st.ok()) {
    return st;
  }

  dfabit::tools::ToolManager tool_manager;
  st = AddBuiltinTools(options, &tool_manager);
  if (!st.ok()) {
    return st;
  }

  st = adapter->InitializeSession(&ctx);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnRegister(&ctx);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnInit(&ctx);
  if (!st.ok()) {
    return st;
  }

  dfabit::adapters::CompileArtifactSet compile_artifacts;
  dfabit::adapters::RuntimeArtifactSet runtime_artifacts;

  st = adapter->PrepareArtifacts(&ctx, &compile_artifacts, &runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnCompileBegin(&ctx, compile_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->CompileBegin(&ctx, compile_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->CompileEnd(&ctx, &compile_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnCompileEnd(&ctx, compile_artifacts);
  if (!st.ok()) {
    return st;
  }

  for (const auto& artifact : compile_artifacts.outputs) {
    if (artifact.kind == dfabit::adapters::ArtifactKind::kManifest &&
        !artifact.path.empty()) {
      st = adapter->LoadManifest(&ctx, artifact);
      if (!st.ok()) {
        return st;
      }
      break;
    }
  }

  st = tool_manager.OnLoadBegin(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->LoadBegin(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->LoadEnd(&ctx, &runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnLoadEnd(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnRunBegin(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->RunBegin(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->RunEnd(&ctx, &runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnRunEnd(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnShutdown(&ctx);
  if (!st.ok()) {
    return st;
  }

  st = WriteBundleOutputs(options, ctx, compile_artifacts, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = adapter->Shutdown(&ctx);
  if (!st.ok()) {
    return st;
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::cli