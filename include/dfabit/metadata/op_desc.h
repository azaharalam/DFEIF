#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "dfabit/metadata/tensor_desc.h"

namespace dfabit::metadata {

struct OpDesc {
  std::uint64_t stable_id = 0;
  std::string op_name;
  std::string dialect;
  std::string stage_tag;
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
  std::unordered_map<std::string, std::string> attributes;
  std::int64_t estimated_flops = 0;
  std::int64_t estimated_bytes = 0;
};

}  // namespace dfabit::metadata