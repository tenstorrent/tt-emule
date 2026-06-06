// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stub of silicon's socket API. Multichip dispatch out
// of scope — provide enough surface that upstream kernels d2d_*/d2h_* ops parse.

#include <cstdint>

namespace tt::tt_metal::experimental {

struct SocketReceiverInterface {
    uint32_t read_ptr = 0;
    uint32_t write_ptr = 0;
    template <typename... Args> SocketReceiverInterface(Args&&...) {}
    template <typename... Args> void wait_for_data(Args&&...) {}
    template <typename... Args> void release(Args&&...) {}
    template <typename... Args> uint32_t get_read_addr(Args&&...) const { return 0; }
};

struct SocketSenderInterface {
    uint32_t read_ptr = 0;
    uint32_t write_ptr = 0;
    template <typename... Args> SocketSenderInterface(Args&&...) {}
    template <typename... Args> void wait_for_space(Args&&...) {}
    template <typename... Args> void send_data(Args&&...) {}
    template <typename... Args> uint32_t get_write_addr(Args&&...) const { return 0; }
};

}  // namespace tt::tt_metal::experimental

// Many upstream socket ops reference these unscoped (upstream kernels does
// `using tt::tt_metal::experimental::SocketReceiverInterface;` or just
// the bare names). Re-export to global scope.
using SocketReceiverInterface = tt::tt_metal::experimental::SocketReceiverInterface;
using SocketSenderInterface = tt::tt_metal::experimental::SocketSenderInterface;

// Free-function socket API used by upstream ops directly.
template <typename... Args>
inline SocketReceiverInterface create_receiver_socket_interface(Args&&...) { return {}; }
template <typename... Args>
inline SocketSenderInterface create_sender_socket_interface(Args&&...) { return {}; }
template <typename... Args> inline void create_receiver_socket(Args&&...) {}
template <typename... Args> inline void create_sender_socket(Args&&...) {}
template <typename... Args> inline void set_receiver_socket_page_size(Args&&...) {}
template <typename... Args> inline void set_sender_socket_page_size(Args&&...) {}
template <typename... Args> inline void socket_wait_for_pages(Args&&...) {}
template <typename... Args> inline void socket_pop_pages(Args&&...) {}
template <typename... Args> inline void socket_push_pages(Args&&...) {}
template <typename... Args> inline void socket_notify_sender(Args&&...) {}
template <typename... Args> inline void socket_notify_receiver(Args&&...) {}
template <typename... Args> inline void socket_reserve_pages(Args&&...) {}
template <typename... Args> inline void update_socket_config(Args&&...) {}
