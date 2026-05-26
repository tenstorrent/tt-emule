// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation shadow for api/core_local_mem.h.
//
// The real header chains through debug/sanitize headers that include dev_msgs.h
// which is not available in the x86 JIT environment.
//
// Include the experimental implementation (full CoreLocalMem<T,AddressType>),
// promote it to the global namespace, and specialise the global noc_traits_t
// so that the Noc class from api/dataflow/noc.h can use CoreLocalMem buffers.

#include "jit_hw/experimental/core_local_mem.h"
#include "jit_hw/api/dataflow/noc.h"  // ::noc_traits_t primary template and ::Noc

using experimental::CoreLocalMem;

// Specialise global ::noc_traits_t<CoreLocalMem<T,AddressType>>.
//
// The experimental version specialises experimental::noc_traits_t (a different
// class template in the experimental namespace), which is not found by the new
// ::Noc::async_{read,write,write_multicast}.
//
// In emulation CoreLocalMem::get_address() already returns a host pointer
// (translates L1 firmware offset → host via __emule_bridge_l1 for small addrs).
template <typename T, typename AddressType>
struct noc_traits_t<CoreLocalMem<T, AddressType>> {
    struct src_args_type       { uintptr_t offset_bytes = 0; };
    struct dst_args_type       { uintptr_t offset_bytes = 0; };
    struct dst_args_mcast_type { uintptr_t offset_bytes = 0; };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const CoreLocalMem<T, AddressType>& src, const Noc&,
                              const src_args_type& args) {
        return static_cast<uintptr_t>(src.get_address()) + args.offset_bytes;
    }
    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const CoreLocalMem<T, AddressType>& dst, const Noc&,
                              const dst_args_type& args) {
        return static_cast<uintptr_t>(dst.get_address()) + args.offset_bytes;
    }
    template <Noc::AddressType AT>
    static uintptr_t dst_addr_mcast(const CoreLocalMem<T, AddressType>& dst, const Noc&,
                                    const dst_args_mcast_type& args) {
        // Multicast in emulation: return the local host address; the multicast
        // write in __emule_multicast_write iterates all registered cores.
        return static_cast<uintptr_t>(dst.get_address()) + args.offset_bytes;
    }
};
