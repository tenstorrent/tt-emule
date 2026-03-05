cmake_minimum_required(VERSION 3.16)
project(tt_emule CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

option(TT_EMULE_COMPAT_HEADERS
    "Build with tt-metal-compatible header aliases (ttkernel/ and tt_metal_compat.hpp)"
    OFF)

# Optionally enable ThreadSanitizer in Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=thread -g -O1)
    add_link_options(-fsanitize=thread)
endif()

# ---- Core emulator library ----
add_library(tt_emule_lib STATIC
    src/host_api.cpp
    src/kernel_runner.cpp
    src/jit_kernel.cpp
)

target_include_directories(tt_emule_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(tt_emule_lib PRIVATE
    TT_EMULE_JIT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw"
)

target_link_libraries(tt_emule_lib PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

# ---- Install rules ----
include(GNUInstallDirs)

install(TARGETS tt_emule_lib
    EXPORT tt-emule-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/tt_emule include/kernel_api
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TT_EMULE_COMPAT_HEADERS)
    install(DIRECTORY include/ttkernel
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(EXPORT tt-emule-targets
    FILE tt-emule-config.cmake
    NAMESPACE tt-emule::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-emule
)

# ---- Tests ----
enable_testing()
add_subdirectory(tests)
