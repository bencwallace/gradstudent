add_executable(runtests test.cpp)

target_link_libraries(runtests PRIVATE tensor)

add_test(NAME runtests COMMAND runtests)
