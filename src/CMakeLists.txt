file(GLOB tensor_SRC "*.cpp" "tensor/*.cpp")
set(SOURCE_FILES ${tensor_SRC})

add_library(tensor ${SOURCE_FILES})

target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
