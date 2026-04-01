#include "dfabit/metadata/stable_id.h"

#include <sstream>

namespace dfabit::metadata {

std::uint64_t StableIdAssigner::Hash(std::string_view s) {
  constexpr std::uint64_t kOffset = 14695981039346656037ull;
  constexpr std::uint64_t kPrime = 1099511628211ull;
  std::uint64_t h = kOffset;
  for (const char c : s) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    h *= kPrime;
  }
  return h;
}

std::uint64_t StableIdAssigner::Assign(std::string_view scope, std::string_view key) const {
  std::string joined;
  joined.reserve(scope.size() + 1 + key.size());
  joined.append(scope.data(), scope.size());
  joined.push_back(':');
  joined.append(key.data(), key.size());
  return Hash(joined);
}

std::uint64_t StableIdAssigner::Assign(const OpDesc& op) const {
  std::ostringstream ss;
  ss << op.op_name << '|'
     << op.dialect << '|'
     << op.stage_tag << '|'
     << op.inputs.size() << '|'
     << op.outputs.size() << '|'
     << op.estimated_flops << '|'
     << op.estimated_bytes;
  return Assign("op", ss.str());
}

void StableIdRegistry::Insert(std::uint64_t stable_id, std::string symbol) {
  ids_[stable_id] = std::move(symbol);
}

std::string StableIdRegistry::Lookup(std::uint64_t stable_id) const {
  const auto it = ids_.find(stable_id);
  return it == ids_.end() ? std::string() : it->second;
}

}  // namespace dfabit::metadata