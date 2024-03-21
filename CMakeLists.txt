cmake_minimum_required(VERSION 3.0.0)
project(gradstudent VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fno-omit-frame-pointer -fsanitize=address")

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

add_subdirectory(src)
add_subdirectory(tests)
