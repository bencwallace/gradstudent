cmake_minimum_required(VERSION 3.0.0)
project(gradstudent-examples-lenet VERSION 0.1.0 LANGUAGES C CXX)

add_executable(lenet main.cpp)

find_package(TBB REQUIRED)

target_link_libraries(lenet gradstudent TBB::tbb)
target_include_directories(lenet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
