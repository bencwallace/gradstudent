# If linking against a release build of google benchmark, benchmarks should be built in release mode
cmake_minimum_required(VERSION 3.0.0)
project(gradstudent VERSION 0.1.0 LANGUAGES C CXX)

find_package(benchmark REQUIRED)

add_executable(benchmark main.cpp)
target_link_libraries(benchmark gradstudent)
target_link_libraries(benchmark benchmark::benchmark)
