#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"
#include "dfabit/metadata/stable_id.h"
#include "dfabit/mlir/module_loader.h"

namespace dfabit::mlir {

class MlirSemanticTagger {
 public:
  MlirSemanticTagger() = default;

  dfabit::core::Status BuildModelDescription(
      const MlirModuleSnapshot& snapshot,
      const std::string& backend_name,
      dfabit::metadata::ModelDesc* model) const;

  dfabit::core::Status TagStableIds(
      dfabit::metadata::ModelDesc* model) const;

 private:
  static std::string ExtractOpName(const std::string& line);
  static std::string InferDialect(const std::string& op_name);
  static std::vector<std::int64_t> ExtractShape(const std::string& line);
  static std::string ExtractDType(const std::string& line);
  static std::string ExtractFunctionName(const std::string& text);
};

}  // namespace dfabit::mlir