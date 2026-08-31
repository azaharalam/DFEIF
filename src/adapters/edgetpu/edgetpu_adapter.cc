#include "dfabit/adapters/edgetpu/edgetpu_adapter.h"

#include "dfabit/adapters/edgetpu/edgetpu_flatbuffer.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dfabit/adapters/artifacts.h"
#include "dfabit/adapters/shared/runtime_log_parser.h"
#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"
#include "dfabit/metadata/stable_id.h"
#include "dfabit/platform/process_runner.h"

namespace dfabit::adapters::edgetpu {

// One row of the Edge TPU compiler's operator table:
//   Operator                       Count      Status
//   CONV_2D                        36         Mapped to Edge TPU
//   PAD                            2          Operation is otherwise supported...
struct OperatorMappingRecord {
  std::string op_name;       // CONV_2D
  int count = 0;             // 36
  std::string status;        // full status text
  bool mapped_to_tpu = false;
  std::uint64_t stable_id = 0;
};

// The scalar fields the compiler reports above the operator table. Sizes are
// normalized to bytes; the compiler prints them as "3.93MiB", "0.00B", etc.
struct EdgeTpuCompileSummary {
  std::string compiler_version;
  std::string input_model;
  std::string output_model;

  double input_size_bytes = 0.0;
  double output_size_bytes = 0.0;
  double on_chip_used_bytes = 0.0;
  double on_chip_remaining_bytes = 0.0;
  double off_chip_streamed_bytes = 0.0;

  int subgraph_count = 0;
  int total_operations = 0;
  double compile_time_ms = 0.0;

  bool has_summary = false;
};

// Parses `edgetpu_compiler -s` output (stdout or the *_edgetpu.log file).
//
// This is the richest compiler-visible information the Edge TPU exposes: which
// operators the compiler could place on the accelerator, which fell back to the
// CPU, how the model partitioned into subgraphs, and how parameter memory split
// between on-chip cache and off-chip streaming. Everything here is read from the
// compiler's own report -- nothing is estimated.
class EdgeTpuCompilerLogParser {
 public:
  dfabit::core::Status ParseFile(
      const std::string& path,
      EdgeTpuCompileSummary* summary,
      std::vector<OperatorMappingRecord>* operators) const;

  dfabit::core::Status ParseText(
      const std::string& text,
      EdgeTpuCompileSummary* summary,
      std::vector<OperatorMappingRecord>* operators) const;

  std::vector<MetricSample> ToMetricSamples(
      const EdgeTpuCompileSummary& summary,
      const std::vector<OperatorMappingRecord>& operators) const;

  // "3.93MiB" -> 4121097.0 ; "0.00B" -> 0.0 ; returns false if unparseable.
  static bool ParseByteSize(const std::string& text, double* out_bytes);
};

// Adapter for the Coral Edge TPU.
//
// Unlike the closed dataflow backends, the Edge TPU exposes a genuine compiler
// report: `edgetpu_compiler -s` states, per operator, whether the operator was
// placed on the accelerator or fell back to the CPU, how the model split into
// subgraphs, and how parameter memory divided between on-chip cache and
// off-chip streaming. That is real partition-level semantic information, so this
// adapter reports compile_report_available and partition_level_events as true
// when a log is present -- and false when it is not.
//
// What the platform does NOT expose is per-operator runtime timing: once
// compiled, the Edge TPU subgraph executes as a single opaque custom op. So
// op_level_events stays false, and byte movement is reported only as a static
// proxy derived from tensor shapes, never as a measured figure.
class EdgeTpuAdapter final : public BackendAdapter {
 public:
  EdgeTpuAdapter();

  std::string name() const override;
  std::string provider() const override;

  dfabit::core::Status InitializeSession(dfabit::api::Context* ctx) override;
  AdapterCapabilities DiscoverCapabilities(const dfabit::api::Context& ctx) const override;

  dfabit::core::Status PrepareArtifacts(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status LoadManifest(
      dfabit::api::Context* ctx,
      const ArtifactRef& manifest_artifact) override;

  dfabit::core::Status CompileBegin(
      dfabit::api::Context* ctx,
      const CompileArtifactSet& compile_artifacts) override;

  dfabit::core::Status CompileEnd(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts) override;

  dfabit::core::Status LoadBegin(
      dfabit::api::Context* ctx,
      const RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status LoadEnd(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status RunBegin(
      dfabit::api::Context* ctx,
      const RuntimeArtifactSet& runtime_artifacts) override;

  dfabit::core::Status RunEnd(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status SubgraphBegin(
      dfabit::api::Context* ctx,
      std::string subgraph_name,
      std::uint64_t stable_id) override;

  dfabit::core::Status SubgraphEnd(
      dfabit::api::Context* ctx,
      std::string subgraph_name,
      std::uint64_t stable_id) override;

  dfabit::core::Status CollectRuntimeMetrics(
      dfabit::api::Context* ctx,
      RuntimeArtifactSet* runtime_artifacts) override;

  dfabit::core::Status Shutdown(dfabit::api::Context* ctx) override;

 private:
  // Locates the compiler log that belongs to a given model. The compiler writes
  // <stem>_edgetpu.log next to the model it produced.
  static std::string DeriveCompilerLogPath(const std::string& model_path);

  // Runs edgetpu_compiler when asked to. Absent that, an existing log is used.
  dfabit::core::Status MaybeCompile(
      dfabit::api::Context* ctx,
      CompileArtifactSet* compile_artifacts);

  // Turns the operator table into a semantic model so tools that expect an op
  // list have one. Counts come from the compiler, not from guesses.
  void BuildModelFromOperators();
  void BuildModelFromGraph(
      const TfLiteGraph& graph,
      const std::string& source_path);

  EdgeTpuCompilerLogParser compiler_log_parser_;
  dfabit::adapters::shared::RuntimeLogParser runtime_log_parser_;
  dfabit::platform::ProcessRunner runner_;

  EdgeTpuCompileSummary compile_summary_;
  std::vector<OperatorMappingRecord> operators_;
  dfabit::metadata::ModelDesc model_;
};

namespace {

std::string Trim(std::string s) {
  std::size_t begin = 0;
  while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
    ++begin;
  }
  std::size_t end = s.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(begin, end - begin);
}

// Splits "Key: Value" once. Returns false when the line has no colon.
bool SplitKeyValue(const std::string& line, std::string* key, std::string* value) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return false;
  }
  *key = Trim(line.substr(0, colon));
  *value = Trim(line.substr(colon + 1));
  return true;
}

int ParseInt(const std::string& s, int fallback) {
  try {
    return std::stoi(s);
  } catch (...) {
    return fallback;
  }
}

std::string DetectPath(const dfabit::api::Context& ctx, const std::string& key) {
  const auto from_property = ctx.GetProperty(key);
  if (!from_property.empty()) {
    return from_property;
  }
  return ctx.run_context().GetAttribute(key);
}

bool FileExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

bool HaveCompilerBinary() {
  // `command -v` returns non-zero when absent; suppress all output.
  return std::system("command -v edgetpu_compiler > /dev/null 2>&1") == 0;
}

}  // namespace

bool EdgeTpuCompilerLogParser::ParseByteSize(const std::string& text, double* out_bytes) {
  if (!out_bytes) {
    return false;
  }

  const auto trimmed = Trim(text);
  if (trimmed.empty()) {
    return false;
  }

  // Leading numeric portion, then a unit suffix.
  std::size_t i = 0;
  while (i < trimmed.size() &&
         (std::isdigit(static_cast<unsigned char>(trimmed[i])) || trimmed[i] == '.' ||
          trimmed[i] == '-' || trimmed[i] == '+')) {
    ++i;
  }
  if (i == 0) {
    return false;
  }

  const double value = std::strtod(trimmed.substr(0, i).c_str(), nullptr);
  auto unit = Trim(trimmed.substr(i));
  for (auto& c : unit) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  double multiplier = 1.0;
  if (unit == "b" || unit.empty()) {
    multiplier = 1.0;
  } else if (unit == "kib") {
    multiplier = 1024.0;
  } else if (unit == "mib") {
    multiplier = 1024.0 * 1024.0;
  } else if (unit == "gib") {
    multiplier = 1024.0 * 1024.0 * 1024.0;
  } else if (unit == "kb") {
    multiplier = 1000.0;
  } else if (unit == "mb") {
    multiplier = 1000.0 * 1000.0;
  } else if (unit == "gb") {
    multiplier = 1000.0 * 1000.0 * 1000.0;
  } else {
    return false;
  }

  *out_bytes = value * multiplier;
  return true;
}

dfabit::core::Status EdgeTpuCompilerLogParser::ParseFile(
    const std::string& path,
    EdgeTpuCompileSummary* summary,
    std::vector<OperatorMappingRecord>* operators) const {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "failed to open edgetpu compiler log: " + path};
  }

  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return ParseText(buffer.str(), summary, operators);
}

dfabit::core::Status EdgeTpuCompilerLogParser::ParseText(
    const std::string& text,
    EdgeTpuCompileSummary* summary,
    std::vector<OperatorMappingRecord>* operators) const {
  if (!summary || !operators) {
    return {dfabit::core::StatusCode::kInvalidArgument, "null output arguments"};
  }

  *summary = EdgeTpuCompileSummary{};
  operators->clear();

  const dfabit::metadata::StableIdAssigner assigner;

  std::stringstream ss(text);
  std::string line;
  bool in_operator_table = false;

  while (std::getline(ss, line)) {
    const auto trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    // The operator table starts at its header row and runs to the end of the
    // section. Rows are "NAME  COUNT  STATUS..." with variable whitespace.
    if (trimmed.rfind("Operator", 0) == 0 &&
        trimmed.find("Count") != std::string::npos &&
        trimmed.find("Status") != std::string::npos) {
      in_operator_table = true;
      continue;
    }

    if (in_operator_table) {
      std::stringstream ls(trimmed);
      std::string name;
      std::string count_token;
      if (!(ls >> name >> count_token)) {
        continue;
      }

      // A row must have a numeric count; anything else is prose and ends the
      // table (e.g. "Compilation succeeded!").
      bool numeric = !count_token.empty();
      for (const char c : count_token) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
          numeric = false;
          break;
        }
      }
      if (!numeric) {
        in_operator_table = false;
        continue;
      }

      std::string status;
      std::getline(ls, status);
      status = Trim(status);

      OperatorMappingRecord rec;
      rec.op_name = name;
      rec.count = ParseInt(count_token, 0);
      rec.status = status;
      // The compiler says "Mapped to Edge TPU" for placed ops; every other
      // status is a fallback reason.
      rec.mapped_to_tpu = status.find("Mapped to Edge TPU") != std::string::npos;
      rec.stable_id = assigner.Assign("edgetpu_op", name);
      operators->push_back(std::move(rec));
      continue;
    }

    // The version banner has no colon: "Edge TPU Compiler version 16.0.384591198"
    {
      const std::string vkey = "Edge TPU Compiler version ";
      if (trimmed.rfind(vkey, 0) == 0) {
        summary->compiler_version = Trim(trimmed.substr(vkey.size()));
        summary->has_summary = true;
        continue;
      }
    }

    std::string key;
    std::string value;
    if (!SplitKeyValue(trimmed, &key, &value)) {
      continue;
    }

    if (key == "Input model" || key == "Input") {
      summary->input_model = value;
    } else if (key == "Output model" || key == "Output") {
      summary->output_model = value;
      summary->has_summary = true;
    } else if (key == "Input size") {
      ParseByteSize(value, &summary->input_size_bytes);
      summary->has_summary = true;
    } else if (key == "Output size") {
      ParseByteSize(value, &summary->output_size_bytes);
    } else if (key == "On-chip memory used for caching model parameters") {
      ParseByteSize(value, &summary->on_chip_used_bytes);
      summary->has_summary = true;
    } else if (key == "On-chip memory remaining for caching model parameters") {
      ParseByteSize(value, &summary->on_chip_remaining_bytes);
    } else if (key == "Off-chip memory used for streaming uncached model parameters") {
      ParseByteSize(value, &summary->off_chip_streamed_bytes);
    } else if (key == "Number of Edge TPU subgraphs") {
      summary->subgraph_count = ParseInt(value, 0);
      summary->has_summary = true;
    } else if (key == "Total number of operations") {
      summary->total_operations = ParseInt(value, 0);
      summary->has_summary = true;
    }
  }

  // The "Model compiled successfully in N ms." line has no colon, so it never
  // reaches the key/value branch above. Catch it with a direct scan.
  if (summary->compile_time_ms == 0.0) {
    const std::string key = "Model compiled successfully in ";
    const auto at = text.find(key);
    if (at != std::string::npos) {
      summary->compile_time_ms =
          std::strtod(text.substr(at + key.size()).c_str(), nullptr);
    }
  }

  return dfabit::core::Status::Ok();
}

std::vector<MetricSample> EdgeTpuCompilerLogParser::ToMetricSamples(
    const EdgeTpuCompileSummary& summary,
    const std::vector<OperatorMappingRecord>& operators) const {
  std::vector<MetricSample> out;

  const auto push = [&out](
                        const std::string& name,
                        double value,
                        const std::string& unit,
                        std::uint64_t stable_id) {
    MetricSample m;
    m.name = name;
    m.value = value;
    m.unit = unit;
    m.stage = "compile";
    m.stable_id = stable_id;
    m.attributes["source"] = "edgetpu_compiler_log";
    out.push_back(std::move(m));
  };

  if (summary.has_summary) {
    push("model_input_bytes", summary.input_size_bytes, "B", 0);
    push("model_output_bytes", summary.output_size_bytes, "B", 0);
    push("on_chip_param_bytes", summary.on_chip_used_bytes, "B", 0);
    push("on_chip_remaining_bytes", summary.on_chip_remaining_bytes, "B", 0);
    push("off_chip_streamed_bytes", summary.off_chip_streamed_bytes, "B", 0);
    push("edgetpu_subgraphs", static_cast<double>(summary.subgraph_count), "count", 0);
    push("total_operations", static_cast<double>(summary.total_operations), "count", 0);
    if (summary.compile_time_ms > 0.0) {
      push("compile_time_ms", summary.compile_time_ms, "ms", 0);
    }

    // Whether the whole parameter set fit on chip is the single most useful
    // derived fact here: off-chip streaming dominates latency when it happens.
    const double total_params =
        summary.on_chip_used_bytes + summary.off_chip_streamed_bytes;
    if (total_params > 0.0) {
      push(
          "on_chip_param_fraction",
          summary.on_chip_used_bytes / total_params,
          "ratio",
          0);
    }
  }

  int mapped_ops = 0;
  int fallback_ops = 0;

  for (const auto& op : operators) {
    MetricSample m;
    m.name = "operator_count";
    m.value = static_cast<double>(op.count);
    m.unit = "count";
    m.stage = "compile";
    m.stable_id = op.stable_id;
    m.attributes["operator"] = op.op_name;
    m.attributes["mapped_to_tpu"] = op.mapped_to_tpu ? "1" : "0";
    m.attributes["status"] = op.status;
    m.attributes["source"] = "edgetpu_compiler_log";
    out.push_back(std::move(m));

    if (op.mapped_to_tpu) {
      mapped_ops += op.count;
    } else {
      fallback_ops += op.count;
    }
  }

  if (!operators.empty()) {
    push("ops_mapped_to_tpu", static_cast<double>(mapped_ops), "count", 0);
    push("ops_fallback_to_cpu", static_cast<double>(fallback_ops), "count", 0);

    const int total = mapped_ops + fallback_ops;
    if (total > 0) {
      // The headline portability number for this backend: what fraction of the
      // model the accelerator could actually take.
      push(
          "tpu_mapping_ratio",
          static_cast<double>(mapped_ops) / static_cast<double>(total),
          "ratio",
          0);
    }
  }

  return out;
}

EdgeTpuAdapter::EdgeTpuAdapter() = default;

std::string EdgeTpuAdapter::name() const {
  return "edgetpu";
}

std::string EdgeTpuAdapter::provider() const {
  return "coral_edgetpu";
}

std::string EdgeTpuAdapter::DeriveCompilerLogPath(const std::string& model_path) {
  if (model_path.empty()) {
    return std::string();
  }

  const std::filesystem::path p(model_path);
  const auto dir = p.parent_path();
  auto stem = p.stem().string();

  // Strip a trailing "_edgetpu" so both foo.tflite and foo_edgetpu.tflite
  // resolve to the same base name.
  const std::string suffix = "_edgetpu";
  std::string base = stem;
  if (base.size() > suffix.size() &&
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
    base = base.substr(0, base.size() - suffix.size());
  }

  // Preference order matters. `edgetpu_compiler -s` prints the memory summary
  // (on-chip/off-chip split, subgraph count, total operations) to stdout, while
  // the auto-generated <base>_edgetpu.log holds only the operator table. Picking
  // the latter silently yields zeros for every memory metric, so any file that
  // looks like captured stdout is preferred over it.
  const std::string candidates[] = {
      (dir / (base + "_summary.log")).string(),
      (dir / (base + "_edgetpu_summary.log")).string(),
      (dir / (base + ".summary.log")).string(),
      (dir / (stem + ".log")).string(),
      (dir / (base + "_edgetpu.log")).string(),
  };

  for (const auto& candidate : candidates) {
    if (FileExists(candidate)) {
      return candidate;
    }
  }

  return std::string();
}

dfabit::core::Status EdgeTpuAdapter::InitializeSession(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }

  ctx->SetProperty("active_adapter", name());

  const auto model_path = DetectPath(*ctx, "edgetpu_model_path");
  auto log_path = DetectPath(*ctx, "edgetpu_compiler_log_path");
  if (log_path.empty()) {
    log_path = DeriveCompilerLogPath(model_path);
  }

  if (FileExists(log_path)) {
    ctx->SetProperty("edgetpu_compiler_log_resolved", log_path);
  }
  if (FileExists(model_path)) {
    ctx->SetProperty("edgetpu_model_resolved", model_path);
  }

  ctx->SetProperty("edgetpu_compiler_present", HaveCompilerBinary() ? "1" : "0");

  const auto runtime_log = DetectPath(*ctx, "edgetpu_runtime_log_path");
  if (FileExists(runtime_log)) {
    ctx->SetProperty("edgetpu_runtime_log_resolved", runtime_log);
  }

  return dfabit::core::Status::Ok();
}

AdapterCapabilities EdgeTpuAdapter::DiscoverCapabilities(
    const dfabit::api::Context& ctx) const {
  AdapterCapabilities caps;

  const bool has_log = !ctx.GetProperty("edgetpu_compiler_log_resolved").empty();
  const bool has_model = !ctx.GetProperty("edgetpu_model_resolved").empty();
  const bool has_runtime_log = !ctx.GetProperty("edgetpu_runtime_log_resolved").empty();
  const bool has_compiler = ctx.GetProperty("edgetpu_compiler_present") == "1";

  // The Edge TPU stack is TFLite based. There is no MLIR or LLVM IR to see.
  caps.visible_mlir = false;
  caps.visible_llvm = false;

  // The compiler report is a genuine, structured view of the partitioned graph.
  caps.visible_graph_ir = has_log;
  caps.compile_report_available = has_log;

  caps.runtime_log_available = has_runtime_log;
  caps.profiler_metrics_available = has_runtime_log;

  // Once compiled, the Edge TPU subgraph runs as one opaque custom op: there is
  // no per-operator runtime event to be had, at any instrumentation depth.
  caps.op_level_events = false;

  // Subgraph boundaries are reported by the compiler, so these are real.
  caps.subgraph_level_events = has_log;
  caps.partition_level_events = has_log;

  caps.custom_env_controls = has_compiler;

  caps.supported_stages = {"compile", "load", "run"};

  if (has_model) {
    caps.supported_artifact_types.emplace_back("tflite_model");
  }
  if (has_log) {
    caps.supported_artifact_types.emplace_back("compile_report");
    caps.supported_metric_names.emplace_back("operator_count");
    caps.supported_metric_names.emplace_back("ops_mapped_to_tpu");
    caps.supported_metric_names.emplace_back("ops_fallback_to_cpu");
    caps.supported_metric_names.emplace_back("tpu_mapping_ratio");
    caps.supported_metric_names.emplace_back("on_chip_param_bytes");
    caps.supported_metric_names.emplace_back("off_chip_streamed_bytes");
    caps.supported_metric_names.emplace_back("edgetpu_subgraphs");
  }
  if (has_runtime_log) {
    caps.supported_artifact_types.emplace_back("runtime_log");
    caps.supported_metric_names.emplace_back("latency_ms");
    caps.supported_metric_names.emplace_back("throughput");
  }

  return caps;
}

dfabit::core::Status EdgeTpuAdapter::PrepareArtifacts(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !compile_artifacts || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid PrepareArtifacts arguments"};
  }

  compile_artifacts->inputs.clear();
  compile_artifacts->outputs.clear();
  compile_artifacts->metrics.clear();
  runtime_artifacts->inputs.clear();
  runtime_artifacts->outputs.clear();
  runtime_artifacts->metrics.clear();

  const auto model_path = ctx->GetProperty("edgetpu_model_resolved");
  if (model_path.empty()) {
    return {
        dfabit::core::StatusCode::kInvalidArgument,
        "edgetpu requires --model pointing at a .tflite file"};
  }

  ArtifactRef model;
  model.kind = ArtifactKind::kBinary;
  model.name = "tflite_model";
  model.path = model_path;
  model.stage = "compile";
  compile_artifacts->inputs.push_back(std::move(model));

  const auto log_path = ctx->GetProperty("edgetpu_compiler_log_resolved");
  if (!log_path.empty()) {
    ArtifactRef log;
    log.kind = ArtifactKind::kCompileReport;
    log.name = "edgetpu_compiler_log";
    log.path = log_path;
    log.stage = "compile";
    log.attributes["format"] = "edgetpu_compiler";
    compile_artifacts->outputs.push_back(std::move(log));
  }

  const auto runtime_log = ctx->GetProperty("edgetpu_runtime_log_resolved");
  if (!runtime_log.empty()) {
    ArtifactRef rt;
    rt.kind = ArtifactKind::kRuntimeLog;
    rt.name = "edgetpu_runtime_log";
    rt.path = runtime_log;
    rt.stage = "run";
    runtime_artifacts->inputs.push_back(std::move(rt));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::LoadManifest(
    dfabit::api::Context* ctx,
    const ArtifactRef& manifest_artifact) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  if (!FileExists(manifest_artifact.path)) {
    return {
        dfabit::core::StatusCode::kNotFound,
        "manifest not found: " + manifest_artifact.path};
  }
  ctx->SetProperty("manifest_path", manifest_artifact.path);
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::MaybeCompile(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts) {
  const auto cmd = DetectPath(*ctx, "edgetpu_compile_cmd");
  if (cmd.empty()) {
    return dfabit::core::Status::Ok();
  }

  const auto work_dir = DetectPath(*ctx, "edgetpu_work_dir");
  const auto out_dir =
      (std::filesystem::path(ctx->run_context().config().output.base_output_dir) /
       "platform" / "edgetpu")
          .string();
  std::filesystem::create_directories(out_dir);

  dfabit::platform::ProcessSpec spec;
  spec.name = "edgetpu_compile";
  spec.command = cmd;
  spec.working_directory = work_dir.empty() ? "." : work_dir;
  spec.stdout_path = (std::filesystem::path(out_dir) / "compile.stdout").string();
  spec.stderr_path = (std::filesystem::path(out_dir) / "compile.stderr").string();

  dfabit::platform::ProcessResult result;
  const auto st = runner_.Run(spec, &result);

  MetricSample elapsed;
  elapsed.name = "compile_command_elapsed_ms";
  elapsed.value = result.elapsed_ms;
  elapsed.unit = "ms";
  elapsed.stage = "compile";
  compile_artifacts->metrics.push_back(std::move(elapsed));

  MetricSample code;
  code.name = "compile_command_exit_code";
  code.value = static_cast<double>(result.exit_code);
  code.unit = "code";
  code.stage = "compile";
  compile_artifacts->metrics.push_back(std::move(code));

  return st;
}

dfabit::core::Status EdgeTpuAdapter::CompileBegin(
    dfabit::api::Context* ctx,
    const CompileArtifactSet& compile_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)compile_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::CompileEnd(
    dfabit::api::Context* ctx,
    CompileArtifactSet* compile_artifacts) {
  if (!ctx || !compile_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid CompileEnd arguments"};
  }

  auto st = MaybeCompile(ctx, compile_artifacts);
  if (!st.ok()) {
    return st;
  }

  // A compile may have just produced the log we could not find earlier.
  if (ctx->GetProperty("edgetpu_compiler_log_resolved").empty()) {
    const auto derived =
        DeriveCompilerLogPath(ctx->GetProperty("edgetpu_model_resolved"));
    if (FileExists(derived)) {
      ctx->SetProperty("edgetpu_compiler_log_resolved", derived);
      ArtifactRef log;
      log.kind = ArtifactKind::kCompileReport;
      log.name = "edgetpu_compiler_log";
      log.path = derived;
      log.stage = "compile";
      log.attributes["format"] = "edgetpu_compiler";
      compile_artifacts->outputs.push_back(std::move(log));
    }
  }

  for (const auto& artifact : compile_artifacts->outputs) {
    if (artifact.kind != ArtifactKind::kCompileReport) {
      continue;
    }

    st = compiler_log_parser_.ParseFile(artifact.path, &compile_summary_, &operators_);
    if (!st.ok()) {
      continue;
    }

    auto metrics = compiler_log_parser_.ToMetricSamples(compile_summary_, operators_);
    compile_artifacts->metrics.insert(
        compile_artifacts->metrics.end(),
        std::make_move_iterator(metrics.begin()),
        std::make_move_iterator(metrics.end()));
  }

  // Prefer the graph as written over the compiler's summary table. The report
  // groups operators by type and carries no tensor shapes, so tensor lifetimes
  // cannot be recovered from it; the pre-compilation model still holds the full
  // operator list. When that file is not present beside the compiled one, the
  // summary table is the only source and the model degrades to type-level
  // granularity rather than failing.
  auto model_hint = ctx->GetProperty("edgetpu_model_resolved");
  const auto source_model =
      TfLiteFlatBufferReader::DeriveSourceModelPath(model_hint);

  bool built_from_graph = false;
  if (!source_model.empty()) {
    TfLiteGraph graph;
    TfLiteFlatBufferReader reader;
    if (reader.Read(source_model, &graph).ok() && !graph.ops.empty()) {
      BuildModelFromGraph(graph, source_model);
      built_from_graph = true;

      MetricSample params;
      params.name = "model_parameter_bytes";
      params.value = graph.parameter_bytes;
      params.unit = "B";
      params.stage = "compile";
      params.attributes["source"] = "tflite_model";
      compile_artifacts->metrics.push_back(std::move(params));

      ArtifactRef src;
      src.kind = ArtifactKind::kGraphIr;
      src.name = "edgetpu_source_model";
      src.path = source_model;
      src.stage = "compile";
      compile_artifacts->outputs.push_back(std::move(src));
    }
  }
  if (!built_from_graph) {
    BuildModelFromOperators();
  }

  // Peak compute and bandwidth for the device this adapter targets. These are
  // the only backend-specific numbers an analysis needs: a tool reads them
  // through the context and stays otherwise identical across backends.
  ctx->SetProperty("peak_macs_per_s", "2e12");     // 4 TOPS int8, 2 ops per MAC
  ctx->SetProperty("peak_bw_bytes_per_s", "8e9");  // 8 GB/s over USB 3.0

  ctx->SetMetadataOps(model_.ops);
  ctx->mutable_run_context().SetAttribute("graph_name", model_.graph_name);

  return dfabit::core::Status::Ok();
}

void EdgeTpuAdapter::BuildModelFromGraph(
    const TfLiteGraph& graph,
    const std::string& source_path) {
  model_.ops.clear();
  model_.backend_name = name();
  model_.graph_name = std::filesystem::path(source_path).stem().string();
  model_.model_name = model_.graph_name;

  // Mapping status is available per operator TYPE only, from the compiler
  // report. Each operator inherits its type's status: accurate when a type maps
  // uniformly, approximate when the compiler split a type across the boundary.
  // The report does not say which instances fell back, so no finer attribution
  // is possible from it.
  std::unordered_map<std::string, std::string> status_by_type;
  std::unordered_map<std::string, bool> mapped_by_type;
  for (const auto& op : operators_) {
    if (op.mapped_to_tpu ||
        status_by_type.find(op.op_name) == status_by_type.end()) {
      status_by_type[op.op_name] = op.status;
      mapped_by_type[op.op_name] = op.mapped_to_tpu;
    }
  }

  const dfabit::metadata::StableIdAssigner assigner;

  for (std::size_t i = 0; i < graph.ops.size(); ++i) {
    const auto& op = graph.ops[i];
    dfabit::metadata::OpDesc desc;
    desc.op_name = op.op_name;
    desc.dialect = "tflite";

    const auto mapped = mapped_by_type.find(op.op_name);
    desc.stage_tag = (mapped != mapped_by_type.end() && !mapped->second)
                         ? "cpu_fallback"
                         : "edgetpu";

    const auto status = status_by_type.find(op.op_name);
    if (status != status_by_type.end()) {
      desc.attributes["status"] = status->second;
    }
    desc.attributes["op_index"] = std::to_string(i);

    double result_bytes = 0.0;
    double footprint = 0.0;

    for (const auto index : op.inputs) {
      if (index < 0 ||
          static_cast<std::size_t>(index) >= graph.tensors.size()) {
        continue;
      }
      const auto& t = graph.tensors[index];
      footprint += t.bytes;
      // Constants stay out of the def-use chain. A weight is live for the whole
      // graph and would dominate any liveness figure, hiding the activation
      // behaviour that varies between models.
      if (t.is_constant) continue;
      dfabit::metadata::TensorDesc in;
      in.name = t.name;
      in.dtype = t.dtype;
      in.shape = t.shape;
      desc.inputs.push_back(std::move(in));
    }

    for (const auto index : op.outputs) {
      if (index < 0 ||
          static_cast<std::size_t>(index) >= graph.tensors.size()) {
        continue;
      }
      const auto& t = graph.tensors[index];
      footprint += t.bytes;
      result_bytes += t.bytes;
      dfabit::metadata::TensorDesc out;
      out.name = t.name;
      out.dtype = t.dtype;
      out.shape = t.shape;
      desc.outputs.push_back(std::move(out));
    }

    desc.estimated_bytes = static_cast<std::int64_t>(footprint);
    // Two operations per multiply-accumulate, matching the convention the
    // Cerebras adapter uses so a tool sees one unit across backends.
    desc.estimated_flops = static_cast<std::int64_t>(op.macs * 2.0);
    desc.attributes["macs_determined"] = op.macs_determined ? "1" : "0";
    desc.attributes["result_bytes"] =
        std::to_string(static_cast<long long>(result_bytes));

    desc.stable_id = assigner.Assign(
        "hidden_ir",
        model_.graph_name + "#" + std::to_string(i) + "|" + op.op_name +
            "|compile");

    model_.ops.push_back(std::move(desc));
  }
}

void EdgeTpuAdapter::BuildModelFromOperators() {
  model_.ops.clear();
  model_.backend_name = name();
  model_.graph_name = compile_summary_.input_model.empty()
                          ? std::string("edgetpu_model")
                          : compile_summary_.input_model;
  model_.model_name = model_.graph_name;

  for (const auto& op : operators_) {
    dfabit::metadata::OpDesc desc;
    desc.op_name = op.op_name;
    desc.dialect = "tflite";
    desc.stage_tag = op.mapped_to_tpu ? "edgetpu" : "cpu_fallback";
    desc.stable_id = op.stable_id;
    desc.attributes["count"] = std::to_string(op.count);
    desc.attributes["status"] = op.status;
    desc.attributes["mapped_to_tpu"] = op.mapped_to_tpu ? "1" : "0";
    model_.ops.push_back(std::move(desc));
  }
}

dfabit::core::Status EdgeTpuAdapter::LoadBegin(
    dfabit::api::Context* ctx,
    const RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::LoadEnd(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid LoadEnd arguments"};
  }

  MetricSample sample;
  sample.name = "graph_nodes";
  sample.value = static_cast<double>(model_.ops.size());
  sample.unit = "count";
  sample.stage = "load";
  runtime_artifacts->metrics.push_back(std::move(sample));
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::RunBegin(
    dfabit::api::Context* ctx,
    const RuntimeArtifactSet& runtime_artifacts) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)runtime_artifacts;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::RunEnd(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid RunEnd arguments"};
  }
  return CollectRuntimeMetrics(ctx, runtime_artifacts);
}

dfabit::core::Status EdgeTpuAdapter::SubgraphBegin(
    dfabit::api::Context* ctx,
    std::string subgraph_name,
    std::uint64_t stable_id) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)subgraph_name;
  (void)stable_id;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::SubgraphEnd(
    dfabit::api::Context* ctx,
    std::string subgraph_name,
    std::uint64_t stable_id) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  (void)subgraph_name;
  (void)stable_id;
  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::CollectRuntimeMetrics(
    dfabit::api::Context* ctx,
    RuntimeArtifactSet* runtime_artifacts) {
  if (!ctx || !runtime_artifacts) {
    return {dfabit::core::StatusCode::kInvalidArgument, "invalid arguments"};
  }

  for (const auto& artifact : runtime_artifacts->inputs) {
    if (artifact.kind != ArtifactKind::kRuntimeLog) {
      continue;
    }

    std::vector<dfabit::adapters::shared::RuntimeLogRecord> records;
    const auto st = runtime_log_parser_.ParseFile(artifact.path, &records);
    if (!st.ok()) {
      continue;
    }

    auto metrics = runtime_log_parser_.ToMetricSamples(records);
    runtime_artifacts->metrics.insert(
        runtime_artifacts->metrics.end(),
        std::make_move_iterator(metrics.begin()),
        std::make_move_iterator(metrics.end()));
  }

  return dfabit::core::Status::Ok();
}

dfabit::core::Status EdgeTpuAdapter::Shutdown(dfabit::api::Context* ctx) {
  if (!ctx) {
    return {dfabit::core::StatusCode::kInvalidArgument, "ctx is null"};
  }
  operators_.clear();
  model_.ops.clear();
  compile_summary_ = EdgeTpuCompileSummary{};
  return dfabit::core::Status::Ok();
}

std::unique_ptr<BackendAdapter> CreateEdgeTpuAdapter() {
  return std::make_unique<EdgeTpuAdapter>();
}

}  // namespace dfabit::adapters::edgetpu
