module {
  func.func @main(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = "linalg.matmul"(%arg0, %arg1) {dfabit.id = 5978602842982171947 : i64, dfabit.stage = "pre_lowering", dfabit.op = "linalg.matmul"} : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}