#include "kernel_api/common.hpp"
#include "kernel_api/dfb_dataflow_api.hpp"
#include <cstdint>
#include <cstring>

void dfb_consumer_kernel_main() {
    uint32_t dfb_id      = get_arg_val<uint32_t>(0);
    uint32_t num_entries  = get_arg_val<uint32_t>(1);
    uint32_t entry_words  = get_arg_val<uint32_t>(2);
    uint32_t output_addr  = get_arg_val<uint32_t>(3);

    uint32_t* output = reinterpret_cast<uint32_t*>(
        static_cast<uintptr_t>(output_addr));

    for (uint32_t i = 0; i < num_entries; ++i) {
        dfb_wait_front(dfb_id, 1);

        uint32_t rd_ptr = dfb_get_read_ptr(dfb_id);
        const uint32_t* src = reinterpret_cast<const uint32_t*>(
            static_cast<uintptr_t>(rd_ptr));

        std::memcpy(&output[i * entry_words], src,
                    entry_words * sizeof(uint32_t));

        dfb_pop_front(dfb_id, 1);
    }
}
