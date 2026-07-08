cmake_minimum_required(VERSION 3.18)
project(puppet_parakeet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_lookup_result
    ERROR_VARIABLE _pybind11_lookup_error
  )
  if(_pybind11_lookup_result OR NOT pybind11_DIR)
    message(FATAL_ERROR
      "pybind11 not found for ${Python_EXECUTABLE}. "
      "Install it with: pip install pybind11\n"
      "${_pybind11_lookup_error}"
    )
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# PARAKEET_ROOT: parakeet.cpp build directory (contains libparakeet.so)
# PARAKEET_SRC: parakeet.cpp source tree (contains include/parakeet_capi.h)
set(PARAKEET_ROOT "" CACHE PATH "Path to parakeet.cpp build directory")
set(PARAKEET_SRC "" CACHE PATH "Path to parakeet.cpp source directory")

if(NOT PARAKEET_ROOT)
  message(FATAL_ERROR "Set -DPARAKEET_ROOT=/path/to/parakeet.cpp/build")
endif()

if(NOT PARAKEET_SRC)
  message(FATAL_ERROR "Set -DPARAKEET_SRC=/path/to/parakeet.cpp")
endif()

if(NOT EXISTS "${PARAKEET_ROOT}/libparakeet.so" AND NOT EXISTS "${PARAKEET_ROOT}/libparakeet.dylib")
  message(FATAL_ERROR "libparakeet not found in ${PARAKEET_ROOT}")
endif()

if(NOT EXISTS "${PARAKEET_SRC}/include/parakeet_capi.h")
  message(FATAL_ERROR "parakeet_capi.h not found in ${PARAKEET_SRC}/include")
endif()

pybind11_add_module(puppet_parakeet parakeet_binding.cpp)

target_include_directories(puppet_parakeet PRIVATE
  "${PARAKEET_SRC}/include"
  "${PARAKEET_SRC}/src"
  "${PARAKEET_SRC}/third_party/ggml/include"
)

target_link_directories(puppet_parakeet PRIVATE "${PARAKEET_ROOT}")
target_link_libraries(puppet_parakeet PRIVATE parakeet)

install(TARGETS puppet_parakeet LIBRARY DESTINATION puppet_parakeet)
