#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "dfabit/core/status.h"
#include "dfabit/metadata/model_desc.h"
#include "dfabit/metadata/stable_id.h"

namespace dfabit::hidden_ir {

struct GraphNodeRecord {
  std::string symbol;
  std::string op_name;
  std::string stage;
  std::string dtype;
  std::vector<std::int64_t> shape;
};

class GraphImporter {
 public:
  dfabit::core::Status ImportFromFile(
      const std::string& path,
      const std::string& backend_name,
      dfabit::metadata::ModelDesc* model) const;

  dfabit::core::Status ImportFromText(
      const std::string& text,
      const std::string& graph_name,
      const std::string& backend_name,
      dfabit::metadata::ModelDesc* model) const;
};

}  // namespace dfabit::hidden_ir