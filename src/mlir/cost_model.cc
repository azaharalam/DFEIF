#include "dfabit/mlir/cost_model.h"

namespace dfabit::mlir {

void MlirCostModel::Populate(dfabit::metadata::OpDesc* op) const {
  if (!op) {
    return;
  }

  std::int64_t input_bytes = 0;
  std::int64_t output_bytes = 0;

  for (const auto& tensor : op->inputs) {
    input_bytes += NumElements(tensor) * BytesPerElement(tensor.dtype);
  }
  for (const auto& tensor : op->outputs) {
    output_bytes += NumElements(tensor) * BytesPerElement(tensor.dtype);
  }

  op->estimated_bytes = input_bytes + output_bytes;

  const auto op_name = op->op_name;
  if (op_name == "linalg.matmul" || op_name == "linalg.batch_matmul") {
    if (op->inputs.size() >= 2 &&
        !op->inputs[0].shape.empty() &&
        !op->inputs[1].shape.empty() &&
        op->inputs[0].shape.size() >= 2 &&
        op->inputs[1].shape.size() >= 2) {
      const auto m = op->inputs[0].shape[op->inputs[0].shape.size() - 2];
      const auto k = op->inputs[0].shape.back();
      const auto n = op->inputs[1].shape.back();
      if (m > 0 && n > 0 && k > 0) {
        op->estimated_flops = 2 * m * n * k;
        return;
      }
    }
  }

  if (op_name == "linalg.conv_2d_nhwc_hwcf" || op_name == "mhlo.convolution") {
    const std::int64_t out_elems =
        op->outputs.empty() ? 0 : NumElements(op->outputs.front());
    op->estimated_flops = out_elems > 0 ? 2 * out_elems : 0;
    return;
  }

  if (op_name == "arith.addf" || op_name == "arith.mulf" ||
      op_name == "arith.addi" || op_name == "arith.muli") {
    const std::int64_t out_elems =
        op->outputs.empty() ? 0 : NumElements(op->outputs.front());
    op->estimated_flops = out_elems;
    return;
  }

  const std::int64_t elems =
      op->outputs.empty() ? 0 : NumElements(op->outputs.front());
  op->estimated_flops = elems;
}

std::int64_t MlirCostModel::NumElements(const dfabit::metadata::TensorDesc& tensor) {
  if (tensor.shape.empty()) {
    return 0;
  }

  std::int64_t total = 1;
  for (const auto dim : tensor.shape) {
    if (dim <= 0) {
      return 0;
    }
    total *= dim;
  }
  return total;
}

std::int64_t MlirCostModel::BytesPerElement(const std::string& dtype) {
  if (dtype == "f16" || dtype == "bf16" || dtype == "i16" || dtype == "ui16") {
    return 2;
  }
  if (dtype == "f32" || dtype == "i32" || dtype == "ui32") {
    return 4;
  }
  if (dtype == "f64" || dtype == "i64" || dtype == "ui64") {
    return 8;
  }
  if (dtype == "i8" || dtype == "ui8") {
    return 1;
  }
  return 4;
}

}  // namespace dfabit::mlir