// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// tt-emule JIT kernel patcher — the opaque public API.
//
// A pure source-to-source transform that prepares a device-kernel translation unit for x86 host
// JIT compilation: it applies the RISC-V/L1 rewrite rules (inline asm → host equivalents; 0-based
// L1 offset casts → rebased-at-deref via bridge_l1) and recursively inlines/normalizes the kernel's
// quoted project `#include`s (skipping headers that have an emule shadow, remapping escaping `../`
// includes so their patched copies stay inside the output dir). The transform stands alone: it takes
// its include roots as arguments and hands back the patched TU on disk. Compilation, caching, and
// dispatch are the caller's concern (the tt-metal emulation runner). Rule catalogue:
// docs/jit-l1-patch-pass.md.

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "tt_emule/detail/kernel_patcher.hpp"

namespace tt::emule {

// Patch the kernel TU at `src_path`, writing the patched TU to `out_path` and any patched shadow
// copies of its project headers into `out_path`'s directory (so `-I <that dir>` finds them first).
//   kernel_include_roots — the kernel-side `-I` roots (ttnn/, tt_metal/, …) used to resolve and
//                          patch non-same-dir project headers.
//   emule_shadow_roots   — the emule shadow roots (jit_hw, include); a header that exists under one
//                          of these is left to `-I` resolution and NOT patched/shadowed.
// Throws std::runtime_error on I/O failure.
inline void patch_kernel_source(
    const std::string& src_path,
    const std::string& out_path,
    const std::vector<std::string>& kernel_include_roots,
    const std::vector<std::string>& emule_shadow_roots) {
    const std::string out_dir = std::filesystem::path(out_path).parent_path().string();
    std::map<std::string, std::string> done;
    detail::preprocess_tu_recursive(src_path, out_path, out_dir, kernel_include_roots, emule_shadow_roots, done);
}

}  // namespace tt::emule
