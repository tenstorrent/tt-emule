# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(tt_emule CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# tt-emule is consumed header-only: tt-metal and tt-umd link tt-emule::headers
# and JIT-compile kernels against include/jit_hw. There is no compiled library.

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

install(TARGETS tt_emule_headers
    EXPORT tt-emule-targets
)
install(DIRECTORY include/tt_emule
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT tt-emule-targets
    FILE tt-emule-config.cmake
    NAMESPACE tt-emule::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-emule
)

# ---- Header contract tests ----
include(CTest)
if(BUILD_TESTING)
    add_executable(tt_emule_l1_ptr_hints_test tests/unit/test_l1_ptr_hints.cpp)
    target_link_libraries(tt_emule_l1_ptr_hints_test PRIVATE tt-emule::headers)
    add_test(NAME tt_emule_l1_ptr_hints COMMAND tt_emule_l1_ptr_hints_test)
endif()
