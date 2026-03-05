add_executable(test_tilize test_tilize.cpp)

target_include_directories(test_tilize PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_tilize PRIVATE
    tt_emule_lib
    Threads::Threads
)

add_test(NAME tilize COMMAND test_tilize)
