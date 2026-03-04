// DM thread 0: read tiles from two DRAM buffers into input CBs.
// rt_args: [src0_dram_offset, src1_dram_offset, num_tiles, tile_size_bytes,
//           noc_x, noc_y]
//
// DRAM offsets fit in uint32_t (prototype DRAM ≤ 256 MB).
#include "kernel_api/kernel_includes.hpp"

void reader_kernel_main() {
    uint32_t src0_offset = get_arg_val<uint32_t>(0);
    uint32_t src1_offset = get_arg_val<uint32_t>(1);
    uint32_t num_tiles   = get_arg_val<uint32_t>(2);
    uint32_t tile_size   = get_arg_val<uint32_t>(3);
    uint32_t noc_x       = get_arg_val<uint32_t>(4);
    uint32_t noc_y       = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t tile_off0 = src0_offset + i * tile_size;
        uint32_t tile_off1 = src1_offset + i * tile_size;

        cb_reserve_back(cb_in0, 1);
        noc_async_read(get_noc_addr(noc_x, noc_y, tile_off0),
                       get_write_ptr(cb_in0), tile_size);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        cb_reserve_back(cb_in1, 1);
        noc_async_read(get_noc_addr(noc_x, noc_y, tile_off1),
                       get_write_ptr(cb_in1), tile_size);
        noc_async_read_barrier();
        cb_push_back(cb_in1, 1);
    }
}
