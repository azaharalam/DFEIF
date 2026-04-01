#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "dfabit/core/framework_config.h"

namespace dfabit::core {

class RunContext {
 public:
  RunContext() = default;
  explicit RunContext(RunConfig cfg) : config_(std::move(cfg)) {}

  const RunConfig& config() const { return config_; }
  RunConfig& mutable_config() { return config_; }

  void MarkStarted(std::uint64_t ts_ns) {
    start_ts_ns_ = ts_ns;
    started_ = true;
  }

  void MarkFinished(std::uint64_t ts_ns) {
    end_ts_ns_ = ts_ns;
    finished_ = true;
  }

  bool started() const { return started_; }
  bool finished() const { return finished_; }

  std::uint64_t start_ts_ns() const { return start_ts_ns_; }
  std::uint64_t end_ts_ns() const { return end_ts_ns_; }

  void SetAttribute(const std::string& key, std::string value) {
    config_.attributes[key] = std::move(value);
  }

  std::string GetAttribute(const std::string& key) const {
    const auto it = config_.attributes.find(key);
    return it == config_.attributes.end() ? std::string() : it->second;
  }

 private:
  RunConfig config_;
  bool started_ = false;
  bool finished_ = false;
  std::uint64_t start_ts_ns_ = 0;
  std::uint64_t end_ts_ns_ = 0;
};

}  // namespace dfabit::core