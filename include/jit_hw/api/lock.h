// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

template <typename ReleaseFunc>
class Lock {
public:
    inline __attribute__((always_inline)) explicit Lock(ReleaseFunc release_func) : release_func_(release_func) {}
    inline __attribute__((always_inline)) ~Lock() { release_func_(); }

    Lock(const Lock&) = delete;
    Lock(Lock&&) = delete;
    Lock& operator=(const Lock&) = delete;
    Lock& operator=(Lock&&) = delete;

private:
    ReleaseFunc release_func_;
};
