file(GLOB tensor_SRC "*.cpp" "tensor/*.cpp")

add_executable(runtests ${tensor_SRC})

target_link_libraries(runtests gradstudent GTest::gtest_main)

gtest_discover_tests(runtests)
