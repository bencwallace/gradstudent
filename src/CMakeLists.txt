set(SOURCE_FILES tensor.cpp utils.cpp)

add_library(tensor ${SOURCE_FILES})

target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
