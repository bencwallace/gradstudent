file(GLOB tensor_SRC "tensor/*.cpp")
set(SOURCE_FILES ${tensor_SRC} array.cpp kernels.cpp multi_index.cpp ops.cpp tensor_data.cpp)

add_library(tensor ${SOURCE_FILES})

target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
