#include "dfabit/adapters/register_builtin_adapters.h"

#include "dfabit/adapters/backend_registry.h"
#include "dfabit/adapters/cerebras/cerebras_adapter.h"
#include "dfabit/adapters/gpu_mlir/gpu_mlir_adapter.h"
#include "dfabit/adapters/sambanova/sambanova_adapter.h"

namespace dfabit::adapters {

dfabit::core::Status RegisterBuiltinAdapters() {
  if (!BackendRegistry::Instance().HasAdapter("gpu_mlir")) {
    const auto st = BackendRegistry::Instance().Register(
        "gpu_mlir",
        &dfabit::adapters::gpu_mlir::CreateGpuMlirAdapter);
    if (!st.ok()) {
      return st;
    }
  }

  if (!BackendRegistry::Instance().HasAdapter("cerebras")) {
    const auto st = BackendRegistry::Instance().Register(
        "cerebras",
        &dfabit::adapters::cerebras::CreateCerebrasAdapter);
    if (!st.ok()) {
      return st;
    }
  }

  if (!BackendRegistry::Instance().HasAdapter("sambanova")) {
    const auto st = BackendRegistry::Instance().Register(
        "sambanova",
        &dfabit::adapters::sambanova::CreateSambaNovaAdapter);
    if (!st.ok()) {
      return st;
    }
  }

  return dfabit::core::Status::Ok();
}

}  // namespace dfabit::adapters