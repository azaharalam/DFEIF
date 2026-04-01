#pragma once

#include <string>

#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"

namespace dfabit::mlir {

class MlirManifestExporter {
 public:
  dfabit::core::Status WriteJson(
      const dfabit::metadata::ModelDesc& model,
      const std::string& path) const;

  dfabit::core::Status ToJson(
      const dfabit::metadata::ModelDesc& model,
      std::string* out) const;
};

}  // namespace dfabit::mlir