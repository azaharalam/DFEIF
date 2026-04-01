#pragma once
#include "dfabit/api/instrumentor.h"

namespace dfabit::usecases {

class ProgramAnalyzer final : public dfabit::api::Instrumentor {
 public:
  std::string name() const override { return "program_analyzer"; }
  dfabit::core::Status Run(dfabit::api::Context* ctx) override;
};

}  // namespace dfabit::usecases
