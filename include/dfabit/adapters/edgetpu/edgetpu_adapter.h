#pragma once

#include <memory>

#include "dfabit/adapters/backend_adapter.h"

namespace dfabit::adapters::edgetpu {

// Factory for the Coral Edge TPU adapter. The implementation, including the
// compiler-log parser, lives entirely in src/adapters/edgetpu/edgetpu_adapter.cc.
std::unique_ptr<BackendAdapter> CreateEdgeTpuAdapter();

}  // namespace dfabit::adapters::edgetpu
