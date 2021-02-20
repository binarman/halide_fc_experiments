cmake_minimum_required(VERSION 3.16)
project(Halide_generator CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Halide REQUIRED)

add_executable(Halide_generator main.cpp)
target_link_libraries(Halide_generator PRIVATE Halide::Halide)