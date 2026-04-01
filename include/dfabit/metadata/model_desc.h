#pragma once

#include <string>
#include <vector>

#include "dfabit/metadata/op_desc.h"

namespace dfabit::metadata {

struct ModelDesc {
  std::string model_name;
  std::string backend_name;
  std::string graph_name;
  std::vector<OpDesc> ops;
};

}  // namespace dfabit::metadata