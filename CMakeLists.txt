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

option(TT_EMULE_BUILD_TESTS "Build tt-emule focused tests" OFF)

if(TT_EMULE_BUILD_TESTS)
    enable_testing()

    set(TT_METAL_DIR "$ENV{TT_METAL_DIR}" CACHE PATH "Path to tt-metal checkout for upstream JIT headers")
    if(NOT TT_METAL_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tt-metal")
        set(TT_METAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tt-metal")
    endif()
    if(NOT EXISTS "${TT_METAL_DIR}/tt_metal/hostdevcommon/api/hostdevcommon/kernel_structs.h")
        message(FATAL_ERROR "TT_EMULE_BUILD_TESTS requires TT_METAL_DIR with tt-metal upstream headers")
    endif()

    add_executable(test_masked_reduction_nan
        tests/jit_hw/test_masked_reduction_nan.cpp
    )
    target_link_libraries(test_masked_reduction_nan PRIVATE tt-emule::headers)
    target_include_directories(test_masked_reduction_nan PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw
        ${TT_METAL_DIR}/tt_metal/api
        ${TT_METAL_DIR}/tt_metal/hostdevcommon/api
    )
    add_test(NAME test_masked_reduction_nan COMMAND test_masked_reduction_nan)

    add_executable(test_tilize_pack_offset
        tests/jit_hw/test_tilize_pack_offset.cpp
    )
    target_link_libraries(test_tilize_pack_offset PRIVATE tt-emule::headers)
    target_include_directories(test_tilize_pack_offset PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw
        ${TT_METAL_DIR}/tt_metal/api
        ${TT_METAL_DIR}/tt_metal/hostdevcommon/api
    )
    add_test(NAME test_tilize_pack_offset COMMAND test_tilize_pack_offset)

    add_executable(test_tensor_accessor_binding_token
        tests/jit_hw/test_tensor_accessor_binding_token.cpp
    )
    target_link_libraries(test_tensor_accessor_binding_token PRIVATE tt-emule::headers)
    target_include_directories(test_tensor_accessor_binding_token PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw
        ${TT_METAL_DIR}/tt_metal/api
        ${TT_METAL_DIR}/tt_metal/hostdevcommon/api
    )
    add_test(NAME test_tensor_accessor_binding_token COMMAND test_tensor_accessor_binding_token)

    add_executable(test_make_tensor_accessor_tuple
        tests/jit_hw/test_make_tensor_accessor_tuple.cpp
    )
    target_link_libraries(test_make_tensor_accessor_tuple PRIVATE tt-emule::headers)
    target_include_directories(test_make_tensor_accessor_tuple PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw
        ${TT_METAL_DIR}/tt_metal/api
        ${TT_METAL_DIR}/tt_metal/hostdevcommon/api
    )
    add_test(NAME test_make_tensor_accessor_tuple COMMAND test_make_tensor_accessor_tuple)

    add_executable(test_ziveli_compat
        tests/jit_hw/test_ziveli_compat.cpp
    )
    target_link_libraries(test_ziveli_compat PRIVATE tt-emule::headers)
    target_compile_definitions(test_ziveli_compat PRIVATE
        ARCH_WORMHOLE=1
        EMULE_SEM_BASE=0
    )
    target_include_directories(test_ziveli_compat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw
        ${TT_METAL_DIR}
        ${TT_METAL_DIR}/tt_metal/api
        ${TT_METAL_DIR}/tt_metal/hostdevcommon/api
    )
    add_test(NAME test_ziveli_compat COMMAND test_ziveli_compat)

    add_executable(test_ziveli_conv_chain
        tests/jit_hw/test_ziveli_conv_chain.cpp
    )
    target_link_libraries(test_ziveli_conv_chain PRIVATE tt-emule::headers)
    target_compile_definitions(test_ziveli_conv_chain PRIVATE
        ARCH_WORMHOLE=1
        EMULE_SEM_BASE=0
    )
    target_include_directories(test_ziveli_conv_chain PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jit_hw
        ${TT_METAL_DIR}
        ${TT_METAL_DIR}/tt_metal/api
        ${TT_METAL_DIR}/tt_metal/hostdevcommon/api
    )
    add_test(NAME test_ziveli_conv_chain COMMAND test_ziveli_conv_chain)
endif()

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
