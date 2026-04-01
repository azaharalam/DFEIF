#include <iostream>

#include "dfabit/adapters/backend_registry.h"

int main() {
  const auto names = dfabit::adapters::BackendRegistry::Instance().List();
  std::cerr << "dfabitctl: phase3 MLIR-visible path build OK\n";
  for (const auto& name : names) {
    std::cerr << "adapter=" << name << "\n";
  }
  return 0;
}