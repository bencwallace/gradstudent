cmake_minimum_required(VERSION 3.0.0)
project(gradstudent-examples-lenet VERSION 0.1.0 LANGUAGES C CXX)

find_package(TBB)
if (TBB)
  add_executable(lenet main.cpp)
  target_link_libraries(lenet gradstudent TBB::tbb)
  target_include_directories(lenet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
