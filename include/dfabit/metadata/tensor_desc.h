#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dfabit::metadata {

struct TensorDesc {
  std::string name;
  std::vector<std::int64_t> shape;
  std::string dtype;
  std::string layout;
};

}  // namespace dfabit::metadata