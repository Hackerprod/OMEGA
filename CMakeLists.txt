cmake_minimum_required(VERSION 3.18)
project(omega_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(NumPy REQUIRED)

pybind11_add_module(_kernels MODULE src/native/arnoldi.cpp)
target_include_directories(_kernels PRIVATE ${NumPy_INCLUDE_DIRS})
target_compile_definitions(_kernels PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

set_target_properties(
  _kernels
  PROPERTIES
    OUTPUT_NAME "_kernels"
    LIBRARY_OUTPUT_DIRECTORY "${SKBUILD_BINARY_DIR}/omega/core/native"
)
