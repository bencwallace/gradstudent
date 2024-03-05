add_executable(runtests test.cpp)

target_link_libraries(runtests tensor GTest::gtest_main)

gtest_discover_tests(runtests)
