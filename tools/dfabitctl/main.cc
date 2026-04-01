#include <iostream>
#include <memory>
#include <vector>

#include "dfabit/adapters/backend_registry.h"

int main() {
  const auto names = dfabit::adapters::BackendRegistry::Instance().List();
  std::cerr << "dfabitctl: phase2 adapter layer build OK\n";
  std::cerr << "registered adapters: " << names.size() << "\n";
  return 0;
}