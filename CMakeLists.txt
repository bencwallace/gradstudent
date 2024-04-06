cmake_minimum_required(VERSION 3.0.0)
project(gradstudent VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic")
set(DEBUG_FLAGS "-O0 -g -D_GLIBCXX_DEBUG")
set(SANITIZE_FLAGS "-fsanitize=address,leak,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS} ${SANITIZE_FLAGS}")

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)
include(CTest)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
