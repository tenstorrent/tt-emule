add_subdirectory(eltwise_add)
add_subdirectory(matmul)
add_subdirectory(tilize)
add_subdirectory(dfb_passthrough)

if(TT_EMULE_COMPAT_HEADERS)
    add_subdirectory(compat)
endif()
