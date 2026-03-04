// DM thread 1: write tiles from output CB to DRAM.
// rt_args: [dst_dram_offset, num_tiles, tile_size_bytes, noc_x, noc_y]
#include "kernel_api/kernel_includes.hpp"

void writer_kernel_main() {
    uint32_t dst_offset = get_arg_val<uint32_t>(0);
    uint32_t num_tiles  = get_arg_val<uint32_t>(1);
    uint32_t tile_size  = get_arg_val<uint32_t>(2);
    uint32_t noc_x      = get_arg_val<uint32_t>(3);
    uint32_t noc_y      = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = 16;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t tile_off = dst_offset + i * tile_size;

        cb_wait_front(cb_out0, 1);
        noc_async_write(get_read_ptr(cb_out0),
                        get_noc_addr(noc_x, noc_y, tile_off), tile_size);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
}
