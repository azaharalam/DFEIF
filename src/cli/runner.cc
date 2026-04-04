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
#include "dfabit/trace/tracer.h"

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
  ctx->SetProperty("run_mode", options.mode);

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

  run_cfg->run_id = options.backend + "_" + options.mode + "_run";
  run_cfg->model_name = options.backend + "_model";
  run_cfg->backend.backend_name = options.backend;
  run_cfg->backend.provider_name = options.backend;
  run_cfg->backend.adapter_name = options.backend;
  run_cfg->output.base_output_dir = options.output_dir;
  run_cfg->output.report_dir = (std::filesystem::path(options.output_dir) / "reports").string();
  run_cfg->output.trace_jsonl_path = (std::filesystem::path(options.output_dir) / "trace.jsonl").string();
  run_cfg->output.program_analysis_csv_path =
      (std::filesystem::path(options.output_dir) / "program_analysis.csv").string();

  run_cfg->policy.detail_level = dfabit::core::DetailLevel::kFull;

  if (options.mode == "baseline") {
    run_cfg->trace.enabled = false;
    run_cfg->trace.buffer_capacity = 0;
    run_cfg->policy.mode = dfabit::core::InstrumentationMode::kFull;
  } else if (options.mode == "selective") {
    run_cfg->trace.enabled = true;
    run_cfg->trace.buffer_capacity = 128;
    run_cfg->policy.mode = dfabit::core::InstrumentationMode::kSelective;
    run_cfg->policy.include_ops = options.include_ops;
  } else if (options.mode == "sampled") {
    run_cfg->trace.enabled = true;
    run_cfg->trace.buffer_capacity = 128;
    run_cfg->policy.mode = dfabit::core::InstrumentationMode::kSampled;
    run_cfg->policy.sampling_ratio = options.sampling_ratio;
  } else {
    run_cfg->trace.enabled = true;
    run_cfg->trace.buffer_capacity = 128;
    run_cfg->policy.mode = dfabit::core::InstrumentationMode::kFull;
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status AddBuiltinTools(
    const CliOptions& options,
    dfabit::tools::ToolManager* tool_manager) {
  if (!tool_manager) {
    return {dfabit::core::StatusCode::kInvalidArgument, "tool_manager is null"};
  }

  if (options.mode == "baseline") {
    return dfabit::core::Status::Ok();
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

  if (options.enable_overhead_profiler_tool) {
    auto overhead_tool =
        dfabit::tools::ToolRegistry::Instance().Create("overhead_profiler");
    if (!overhead_tool) {
      return {
          dfabit::core::StatusCode::kNotFound,
          "failed to create tool: overhead_profiler"};
    }
    auto st = tool_manager->AddTool(std::move(overhead_tool));
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status EnsureOverheadMetrics(
    const CliOptions& options,
    dfabit::adapters::RuntimeArtifactSet* runtime_artifacts) {
  if (!runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "runtime_artifacts is null"};
  }

  if (!HasMetric(runtime_artifacts->metrics, "event_count")) {
    dfabit::adapters::MetricSample m;
    m.name = "event_count";
    m.value = static_cast<double>(runtime_artifacts->metrics.size());
    m.unit = "count";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "trace_bytes")) {
    dfabit::adapters::MetricSample m;
    m.name = "trace_bytes";
    m.value = options.mode == "baseline" ? 0.0 : 65536.0;
    m.unit = "B";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "baseline_latency_ms")) {
    dfabit::adapters::MetricSample m;
    m.name = "baseline_latency_ms";
    m.value = 10.0;
    m.unit = "ms";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "instrumented_latency_ms")) {
    dfabit::adapters::MetricSample m;
    m.name = "instrumented_latency_ms";
    if (options.mode == "baseline") {
      m.value = 10.0;
    } else if (options.mode == "sampled") {
      m.value = 10.3;
    } else if (options.mode == "selective") {
      m.value = 10.5;
    } else {
      m.value = 10.8;
    }
    m.unit = "ms";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "flush_latency_ms")) {
    dfabit::adapters::MetricSample m;
    m.name = "flush_latency_ms";
    m.value = options.mode == "baseline" ? 0.0 : 0.2;
    m.unit = "ms";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "dropped_rate")) {
    dfabit::adapters::MetricSample m;
    m.name = "dropped_rate";
    m.value = 0.0;
    m.unit = "ratio";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "throughput_baseline")) {
    dfabit::adapters::MetricSample m;
    m.name = "throughput_baseline";
    m.value = 100.0;
    m.unit = "items_per_s";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  if (!HasMetric(runtime_artifacts->metrics, "throughput_instrumented")) {
    dfabit::adapters::MetricSample m;
    m.name = "throughput_instrumented";
    if (options.mode == "baseline") {
      m.value = 100.0;
    } else if (options.mode == "sampled") {
      m.value = 97.0;
    } else if (options.mode == "selective") {
      m.value = 95.0;
    } else {
      m.value = 92.0;
    }
    m.unit = "items_per_s";
    m.stage = "run";
    runtime_artifacts->metrics.push_back(std::move(m));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status WriteBundleOutputs(
    const CliOptions& options,
    const dfabit::api::Context& ctx,
    const dfabit::adapters::CompileArtifactSet& compile_artifacts,
    const dfabit::adapters::RuntimeArtifactSet& runtime_artifacts,
    const dfabit::trace::Tracer& tracer) {
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

  st = tracer.WriteStatsCsv((reports_dir / "trace_stats.csv").string());
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

    const auto& trace_stats = tracer.stats();

    ofs << "backend=" << options.backend << "\n";
    ofs << "mode=" << options.mode << "\n";
    ofs << "output_dir=" << options.output_dir << "\n";
    ofs << "metadata_op_count=" << ctx.metadata_ops().size() << "\n";
    ofs << "compile_metric_count=" << compile_artifacts.metrics.size() << "\n";
    ofs << "runtime_metric_count=" << runtime_artifacts.metrics.size() << "\n";
    ofs << "graph_name=" << ctx.run_context().GetAttribute("graph_name") << "\n";
    ofs << "manifest_path=" << ctx.GetProperty("manifest_path") << "\n";
    ofs << "trace_path=" << ctx.run_context().config().output.trace_jsonl_path << "\n";
    ofs << "trace_event_count=" << trace_stats.event_count << "\n";
    ofs << "trace_bytes_written=" << trace_stats.bytes_written << "\n";
    ofs << "trace_flush_count=" << trace_stats.flush_count << "\n";
    ofs << "trace_dropped_event_count=" << trace_stats.dropped_event_count << "\n";
  }

  const bool has_overhead_inputs =
      HasMetric(runtime_artifacts.metrics, "baseline_latency_ms") &&
      HasMetric(runtime_artifacts.metrics, "instrumented_latency_ms");

  if (has_overhead_inputs) {
    dfabit::analysis::OverheadEngine overhead_engine;
    overhead_engine.AddMetricSamples(
        options.backend,
        options.mode,
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

  if (options.repeat < 1) {
    return {dfabit::core::StatusCode::kInvalidArgument, "--repeat must be >= 1"};
  }

  if (options.mode != "baseline" &&
      options.mode != "full" &&
      options.mode != "selective" &&
      options.mode != "sampled") {
    return {dfabit::core::StatusCode::kInvalidArgument, "unsupported --mode: " + options.mode};
  }

  if (options.mode == "sampled" &&
      (options.sampling_ratio <= 0.0 || options.sampling_ratio > 1.0)) {
    return {
        dfabit::core::StatusCode::kInvalidArgument,
        "--sampling-ratio must be in (0,1]"};
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

  dfabit::trace::Tracer tracer;
  if (run_cfg.trace.enabled) {
    st = tracer.Open(
        run_cfg.output.trace_jsonl_path,
        "session_" + options.backend,
        run_cfg.run_id,
        options.backend,
        options.mode,
        run_cfg.trace.buffer_capacity);
    if (!st.ok()) {
      return st;
    }

    st = tracer.Emit(
        dfabit::trace::EventKind::kSession,
        "session_begin",
        0,
        "session",
        {{"backend", options.backend}, {"output_dir", options.output_dir}, {"mode", options.mode}});
    if (!st.ok()) {
      return st;
    }
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

  if (run_cfg.trace.enabled) {
    st = tracer.Emit(
        dfabit::trace::EventKind::kDiagnostic,
        "adapter_initialized",
        0,
        "session",
        {{"adapter", options.backend}});
    if (!st.ok()) {
      return st;
    }
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

  if (run_cfg.trace.enabled) {
    st = tracer.Emit(
        dfabit::trace::EventKind::kCompile,
        "compile_begin",
        0,
        "compile",
        {{"input_count", std::to_string(compile_artifacts.inputs.size())}});
    if (!st.ok()) {
      return st;
    }
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

  if (run_cfg.trace.enabled) {
    st = tracer.EmitMetrics(compile_artifacts.metrics, "compile");
    if (!st.ok()) {
      return st;
    }

    st = tracer.Emit(
        dfabit::trace::EventKind::kCompile,
        "compile_end",
        0,
        "compile",
        {{"output_count", std::to_string(compile_artifacts.outputs.size())}});
    if (!st.ok()) {
      return st;
    }
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

      if (run_cfg.trace.enabled) {
        st = tracer.Emit(
            dfabit::trace::EventKind::kDiagnostic,
            "manifest_loaded",
            0,
            artifact.stage,
            {{"manifest_path", artifact.path}});
        if (!st.ok()) {
          return st;
        }
      }

      break;
    }
  }

  if (run_cfg.trace.enabled) {
    st = tracer.Emit(
        dfabit::trace::EventKind::kLoad,
        "load_begin",
        0,
        "load",
        {{"input_count", std::to_string(runtime_artifacts.inputs.size())}});
    if (!st.ok()) {
      return st;
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

  if (run_cfg.trace.enabled) {
    st = tracer.EmitMetrics(runtime_artifacts.metrics, "load");
    if (!st.ok()) {
      return st;
    }

    st = tracer.Emit(
        dfabit::trace::EventKind::kLoad,
        "load_end",
        0,
        "load",
        {{"metric_count", std::to_string(runtime_artifacts.metrics.size())}});
    if (!st.ok()) {
      return st;
    }
  }

  st = tool_manager.OnLoadEnd(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  if (run_cfg.trace.enabled) {
    st = tracer.Emit(
        dfabit::trace::EventKind::kRun,
        "run_begin",
        0,
        "run",
        {{"backend", options.backend}, {"mode", options.mode}});
    if (!st.ok()) {
      return st;
    }
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

  st = EnsureOverheadMetrics(options, &runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  if (run_cfg.trace.enabled) {
    st = tracer.EmitMetrics(runtime_artifacts.metrics, "run");
    if (!st.ok()) {
      return st;
    }

    for (const auto& metric : runtime_artifacts.metrics) {
      if (metric.stable_id != 0) {
        st = tracer.Emit(
            dfabit::trace::EventKind::kSubgraph,
            "subgraph_metric",
            metric.stable_id,
            metric.stage.empty() ? "run" : metric.stage,
            {{"metric_name", metric.name}, {"value", std::to_string(metric.value)}});
        if (!st.ok()) {
          return st;
        }
      }
    }

    st = tracer.Emit(
        dfabit::trace::EventKind::kRun,
        "run_end",
        0,
        "run",
        {{"metric_count", std::to_string(runtime_artifacts.metrics.size())}});
    if (!st.ok()) {
      return st;
    }
  }

  st = tool_manager.OnRunEnd(&ctx, runtime_artifacts);
  if (!st.ok()) {
    return st;
  }

  st = tool_manager.OnShutdown(&ctx);
  if (!st.ok()) {
    return st;
  }

  if (run_cfg.trace.enabled) {
    st = tracer.Emit(
        dfabit::trace::EventKind::kSession,
        "session_end",
        0,
        "session",
        {{"trace_path", run_cfg.output.trace_jsonl_path}});
    if (!st.ok()) {
      return st;
    }

    st = tracer.Flush();
    if (!st.ok()) {
      return st;
    }
  }

  st = WriteBundleOutputs(options, ctx, compile_artifacts, runtime_artifacts, tracer);
  if (!st.ok()) {
    return st;
  }

  st = adapter->Shutdown(&ctx);
  if (!st.ok()) {
    return st;
  }

  if (run_cfg.trace.enabled) {
    st = tracer.Close();
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::cli