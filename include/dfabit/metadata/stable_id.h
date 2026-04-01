#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "dfabit/metadata/op_desc.h"

namespace dfabit::metadata {

class StableIdAssigner {
 public:
  std::uint64_t Assign(std::string_view scope, std::string_view key) const;
  std::uint64_t Assign(const OpDesc& op) const;
  static std::uint64_t Hash(std::string_view s);
};

class StableIdRegistry {
 public:
  void Insert(std::uint64_t stable_id, std::string symbol);
  std::string Lookup(std::uint64_t stable_id) const;
  std::size_t Size() const { return ids_.size(); }

 private:
  std::unordered_map<std::uint64_t, std::string> ids_;
};

}  // namespace dfabit::metadata