cmake_minimum_required(VERSION 3.0.0)
project(gradstudent-examples-image-processing VERSION 0.1.0 LANGUAGES C CXX)

add_executable(image-processing main.cpp)
target_link_libraries(image-processing gradstudent)
target_include_directories(image-processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
