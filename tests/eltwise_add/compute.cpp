// Compute thread: add tiles from cb_in0 and cb_in1, push result to cb_out0.
// rt_args: [num_tiles]
#include "kernel_api/kernel_includes.hpp"

void compute_kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in0  = 0;
    constexpr uint32_t cb_in1  = 1;
    constexpr uint32_t cb_out0 = 16;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, 1);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
