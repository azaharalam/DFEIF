#pragma once

#include <cstdint>

#include "dfabit/metadata/op_desc.h"
#include "dfabit/metadata/tensor_desc.h"

namespace dfabit::mlir {

class MlirCostModel {
 public:
  void Populate(dfabit::metadata::OpDesc* op) const;

 private:
  static std::int64_t NumElements(const dfabit::metadata::TensorDesc& tensor);
  static std::int64_t BytesPerElement(const std::string& dtype);
};

}  // namespace dfabit::mlir