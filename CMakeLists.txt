cmake_minimum_required(VERSION 3.20)
project(dfabit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DFABIT_BUILD_TOOLS "Build dfabitctl CLI tools" ON)
option(DFABIT_BUILD_PYTHON "Build Python bindings if pybind11 is available" ON)
option(DFABIT_ENABLE_WERROR "Treat warnings as errors" OFF)

add_library(dfabit_core
  src/core/status.cc
  src/core/error.cc
  src/core/log.cc
  src/core/session.cc
  src/core/framework.cc
  src/trace/writer_jsonl.cc
  src/analysis/report_csv.cc
  src/usecases/program_analyzer.cc
  src/api/context.cc
  src/metadata/stable_id.cc
  src/policy/policy_engine.cc
  src/adapters/backend_adapter.cc
  src/adapters/backend_registry.cc
  src/adapters/shared/compile_report_parser.cc
  src/adapters/shared/runtime_log_parser.cc
  src/adapters/shared/artifact_correlator.cc
)

set_target_properties(dfabit_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(dfabit_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(dfabit_core PRIVATE /W4)
else()
  target_compile_options(dfabit_core PRIVATE -Wall -Wextra -Wpedantic)
  if(DFABIT_ENABLE_WERROR)
    target_compile_options(dfabit_core PRIVATE -Werror)
  endif()
endif()

if(DFABIT_BUILD_TOOLS)
  add_executable(dfabitctl tools/dfabitctl/main.cc)
  target_link_libraries(dfabitctl PRIVATE dfabit_core)
endif()

if(DFABIT_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND)
    pybind11_add_module(dfabit_native python/bindings/pybind/module.cc)
    target_link_libraries(dfabit_native PRIVATE dfabit_core)
    target_include_directories(dfabit_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  else()
    message(STATUS "pybind11 not found; skipping Python bindings (dfabit_native).")
  endif()
endif()