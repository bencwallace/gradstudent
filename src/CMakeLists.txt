file(GLOB_RECURSE tensor_SRC "*.cpp" "ops/*.cpp" "tensor/*.cpp" "internal/*.cpp")
set(SOURCE_FILES ${tensor_SRC})

add_library(gradstudent ${SOURCE_FILES})

target_include_directories(gradstudent PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(gradstudent PRIVATE ${PROJECT_SOURCE_DIR}/include/internal)
