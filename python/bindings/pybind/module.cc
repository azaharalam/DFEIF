#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dfabit/analysis/report.h"
#include "dfabit/api/context.h"
#include "dfabit/core/framework_config.h"
#include "dfabit/metadata/op_desc.h"
#include "dfabit/metadata/tensor_desc.h"
#include "dfabit/usecases/program_analyzer.h"

namespace py = pybind11;

PYBIND11_MODULE(dfabit_native, m) {
  m.doc() = "DFABIT native core bindings";

  py::enum_<dfabit::core::InstrumentationMode>(m, "InstrumentationMode")
      .value("FULL", dfabit::core::InstrumentationMode::kFull)
      .value("SELECTIVE", dfabit::core::InstrumentationMode::kSelective)
      .value("SAMPLED", dfabit::core::InstrumentationMode::kSampled);

  py::enum_<dfabit::core::DetailLevel>(m, "DetailLevel")
      .value("IDS", dfabit::core::DetailLevel::kIds)
      .value("LITE", dfabit::core::DetailLevel::kLite)
      .value("FULL", dfabit::core::DetailLevel::kFull);

  py::class_<dfabit::core::OutputConfig>(m, "OutputConfig")
      .def(py::init<>())
      .def_readwrite("base_output_dir", &dfabit::core::OutputConfig::base_output_dir)
      .def_readwrite("trace_jsonl_path", &dfabit::core::OutputConfig::trace_jsonl_path)
      .def_readwrite("report_dir", &dfabit::core::OutputConfig::report_dir)
      .def_readwrite("program_analysis_csv_path", &dfabit::core::OutputConfig::program_analysis_csv_path);

  py::class_<dfabit::core::TraceConfig>(m, "TraceConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &dfabit::core::TraceConfig::enabled)
      .def_readwrite("buffer_capacity", &dfabit::core::TraceConfig::buffer_capacity)
      .def_readwrite("flush_per_write", &dfabit::core::TraceConfig::flush_per_write);

  py::class_<dfabit::core::PolicyConfig>(m, "PolicyConfig")
      .def(py::init<>())
      .def_readwrite("mode", &dfabit::core::PolicyConfig::mode)
      .def_readwrite("detail_level", &dfabit::core::PolicyConfig::detail_level)
      .def_readwrite("include_ops", &dfabit::core::PolicyConfig::include_ops)
      .def_readwrite("include_stages", &dfabit::core::PolicyConfig::include_stages)
      .def_readwrite("include_dialects", &dfabit::core::PolicyConfig::include_dialects)
      .def_readwrite("sampling_ratio", &dfabit::core::PolicyConfig::sampling_ratio)
      .def_readwrite("sampling_stride", &dfabit::core::PolicyConfig::sampling_stride)
      .def_readwrite("sampling_salt", &dfabit::core::PolicyConfig::sampling_salt);

  py::class_<dfabit::core::BackendConfig>(m, "BackendConfig")
      .def(py::init<>())
      .def_readwrite("backend_name", &dfabit::core::BackendConfig::backend_name)
      .def_readwrite("provider_name", &dfabit::core::BackendConfig::provider_name)
      .def_readwrite("adapter_name", &dfabit::core::BackendConfig::adapter_name);

  py::class_<dfabit::core::RunConfig>(m, "RunConfig")
      .def(py::init<>())
      .def_readwrite("run_id", &dfabit::core::RunConfig::run_id)
      .def_readwrite("model_name", &dfabit::core::RunConfig::model_name)
      .def_readwrite("output", &dfabit::core::RunConfig::output)
      .def_readwrite("trace", &dfabit::core::RunConfig::trace)
      .def_readwrite("policy", &dfabit::core::RunConfig::policy)
      .def_readwrite("backend", &dfabit::core::RunConfig::backend)
      .def_readwrite("attributes", &dfabit::core::RunConfig::attributes);

  py::class_<dfabit::core::FrameworkConfig>(m, "FrameworkConfig")
      .def(py::init<>())
      .def_readwrite("session_id", &dfabit::core::FrameworkConfig::session_id)
      .def_readwrite("default_output", &dfabit::core::FrameworkConfig::default_output)
      .def_readwrite("default_trace", &dfabit::core::FrameworkConfig::default_trace)
      .def_readwrite("default_policy", &dfabit::core::FrameworkConfig::default_policy)
      .def_readwrite("default_backend", &dfabit::core::FrameworkConfig::default_backend)
      .def_readwrite("attributes", &dfabit::core::FrameworkConfig::attributes);

  py::class_<dfabit::metadata::TensorDesc>(m, "TensorDesc")
      .def(py::init<>())
      .def_readwrite("name", &dfabit::metadata::TensorDesc::name)
      .def_readwrite("shape", &dfabit::metadata::TensorDesc::shape)
      .def_readwrite("dtype", &dfabit::metadata::TensorDesc::dtype)
      .def_readwrite("layout", &dfabit::metadata::TensorDesc::layout);

  py::class_<dfabit::metadata::OpDesc>(m, "OpDesc")
      .def(py::init<>())
      .def_readwrite("stable_id", &dfabit::metadata::OpDesc::stable_id)
      .def_readwrite("op_name", &dfabit::metadata::OpDesc::op_name)
      .def_readwrite("dialect", &dfabit::metadata::OpDesc::dialect)
      .def_readwrite("stage_tag", &dfabit::metadata::OpDesc::stage_tag)
      .def_readwrite("inputs", &dfabit::metadata::OpDesc::inputs)
      .def_readwrite("outputs", &dfabit::metadata::OpDesc::outputs)
      .def_readwrite("attributes", &dfabit::metadata::OpDesc::attributes)
      .def_readwrite("estimated_flops", &dfabit::metadata::OpDesc::estimated_flops)
      .def_readwrite("estimated_bytes", &dfabit::metadata::OpDesc::estimated_bytes);

  py::class_<dfabit::analysis::OpRecord>(m, "OpRecord")
      .def(py::init<>())
      .def_readwrite("op_id", &dfabit::analysis::OpRecord::op_id)
      .def_readwrite("op_name", &dfabit::analysis::OpRecord::op_name)
      .def_readwrite("layer", &dfabit::analysis::OpRecord::layer)
      .def_readwrite("mac_ops", &dfabit::analysis::OpRecord::mac_ops)
      .def_readwrite("total_bytes", &dfabit::analysis::OpRecord::total_bytes)
      .def_readwrite("measured_latency_ms", &dfabit::analysis::OpRecord::measured_latency_ms)
      .def_readwrite("kernel_id", &dfabit::analysis::OpRecord::kernel_id);

  py::class_<dfabit::api::Context>(m, "Context")
      .def(py::init<>())
      .def(py::init<dfabit::core::RunConfig>())
      .def("set_framework_config", &dfabit::api::Context::SetFrameworkConfig)
      .def("set_run_config", &dfabit::api::Context::SetRunConfig)
      .def("set_ops", &dfabit::api::Context::SetOps)
      .def("set_metadata_ops", &dfabit::api::Context::SetMetadataOps)
      .def("set_property", &dfabit::api::Context::SetProperty);

  py::class_<dfabit::usecases::ProgramAnalyzer>(m, "ProgramAnalyzer")
      .def(py::init<>())
      .def("run", [](dfabit::usecases::ProgramAnalyzer& self, dfabit::api::Context& ctx) {
        const auto st = self.Run(&ctx);
        if (!st.ok()) {
          throw std::runtime_error(st.message());
        }
      });
}