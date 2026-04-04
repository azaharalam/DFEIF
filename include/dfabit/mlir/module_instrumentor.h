#pragma once

#include <string>

#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"
#include "dfabit/mlir/module_loader.h"

namespace dfabit::mlir {

struct InstrumentedModuleSnapshot {
  std::string module_name;
  std::string stage;
  std::string source_path;
  std::string original_text;
  std::string instrumented_text;
};

class MlirModuleInstrumentor {
 public:
  dfabit::core::Status Instrument(
      const MlirModuleSnapshot& snapshot,
      const dfabit::metadata::ModelDesc& model,
      InstrumentedModuleSnapshot* out) const;

  dfabit::core::Status WriteSnapshot(
      const InstrumentedModuleSnapshot& snapshot,
      const std::string& original_path,
      const std::string& instrumented_path) const;
};

}  // namespace dfabit::mlir