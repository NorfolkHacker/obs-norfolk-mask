cmake_minimum_required(VERSION 3.20)
project(norfolk-hacker-mask VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NHM_BUILD_PLUGIN "Build the OBS plugin" ON)
include(CTest)

add_library(nhm_core STATIC src/pose.cpp src/smoother.cpp src/layout.cpp)
target_include_directories(nhm_core PUBLIC src)
target_compile_options(nhm_core PRIVATE -Wall -Wextra)

if(BUILD_TESTING)
  add_executable(test_core tests/test_main.cpp tests/test_pose.cpp tests/test_smoother.cpp tests/test_layout.cpp)
  target_link_libraries(test_core PRIVATE nhm_core)
  add_test(NAME core COMMAND test_core)
endif()
