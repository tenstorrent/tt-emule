cmake_minimum_required(VERSION 3.16)
project(tt_emule CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Optionally enable ThreadSanitizer in Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=thread -g -O1)
    add_link_options(-fsanitize=thread)
endif()

# ---- Core emulator library ----
add_library(tt_emule_lib STATIC
    src/host_api.cpp
    src/kernel_runner.cpp
)

target_include_directories(tt_emule_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tt_emule_lib PUBLIC Threads::Threads)

# ---- Tests ----
enable_testing()
add_subdirectory(tests)
