cmake_minimum_required(VERSION 3.16)
project(tt_emule CXX)

if(NOT DEFINED PROJECT_IS_TOP_LEVEL)
    if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
        set(PROJECT_IS_TOP_LEVEL TRUE)
    else()
        set(PROJECT_IS_TOP_LEVEL FALSE)
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

option(TT_EMULE_COMPAT_HEADERS
    "Build with tt-metal-compatible header aliases (ttkernel/ and tt_metal_compat.hpp)"
    OFF)

option(TT_EMULE_ASAN "Build with AddressSanitizer (mutually exclusive with TSan)" OFF)

# Optionally enable ThreadSanitizer in Debug builds (suppressed when ASan is on)
if(PROJECT_IS_TOP_LEVEL AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT TT_EMULE_ASAN)
    add_compile_options(-fsanitize=thread -g -O1)
    add_link_options(-fsanitize=thread)
endif()

if(TT_EMULE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
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

# ---- INTERFACE header-only target (for consumers that only need includes) ----
add_library(tt_emule_headers INTERFACE)
add_library(tt-emule::headers ALIAS tt_emule_headers)

target_include_directories(tt_emule_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(tt_emule_headers PROPERTIES
    TT_EMULE_JIT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw"
    TT_EMULE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# ---- Install rules ----
include(GNUInstallDirs)

install(TARGETS tt_emule_lib tt_emule_headers
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
if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(tests)
endif()
