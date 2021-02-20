cmake_minimum_required(VERSION 3.16)
project(Halide_generator CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Halide REQUIRED)

add_executable(fc_generator main.cpp)
target_link_libraries(fc_generator PRIVATE Halide::Halide)
target_link_libraries(fc_generator PRIVATE stdc++fs)

add_library(fc_runner STATIC runner.cpp)
target_link_libraries(fc_runner PRIVATE Halide::Halide)