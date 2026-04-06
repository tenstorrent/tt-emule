#include "kernel_api/common.hpp"
#include "kernel_api/dfb_dataflow_api.hpp"
#include <cstdint>

void dfb_mc_producer_kernel_main() {
    uint32_t dfb_id      = get_arg_val<uint32_t>(0);
    uint32_t num_entries = get_arg_val<uint32_t>(1);
    uint32_t entry_words = get_arg_val<uint32_t>(2);
    uint32_t seed        = get_arg_val<uint32_t>(3);

    for (uint32_t i = 0; i < num_entries; ++i) {
        dfb_reserve_back(dfb_id, 1);

        uint32_t wr_ptr = dfb_get_write_ptr(dfb_id);
        uint32_t* dst = reinterpret_cast<uint32_t*>(
            static_cast<uintptr_t>(wr_ptr));

        for (uint32_t w = 0; w < entry_words; ++w)
            dst[w] = seed + i * entry_words + w;

        dfb_push_back(dfb_id, 1);
    }

    dfb_finish(dfb_id);
}
