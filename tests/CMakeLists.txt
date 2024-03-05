file(GLOB tensor_SRC arrayTest.cpp "tensor/*.cpp")

add_executable(runtests ${tensor_SRC})

target_link_libraries(runtests tensor GTest::gtest_main)

gtest_discover_tests(runtests)
