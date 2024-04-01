file(GLOB tensor_SRC "*.cpp" "ops/*.cpp" "tensor/*.cpp")
set(SOURCE_FILES ${tensor_SRC})

add_library(gradstudent ${SOURCE_FILES})

target_link_libraries(gradstudent PUBLIC TBB::tbb)

target_include_directories(gradstudent PUBLIC /usr/include/boost)
target_include_directories(gradstudent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
