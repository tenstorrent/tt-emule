// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Internals of the JIT kernel patch pass. NOT a public API — include
// "tt_emule/kernel_patcher.hpp" and call tt::emule::patch_kernel_source() instead.
// The rewrite-rule catalogue is documented in docs/jit-l1-patch-pass.md.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tt::emule::detail {

// Rewrite RISC-V-specific inline asm and raw-L1-pointer idioms so the kernel
// compiles for x86 host.  The rewrites:
//   1. `asm volatile("csrr %0, mhartid" : "=r"(V));` → `V = __processor_id;`
//      (x86 assembler rejects RISC-V CSR instructions; the runner sets the
//      __processor_id TLS before each kernel launch.)
//   2. `asm volatile("fence" ::: "memory");` or bare `asm volatile("fence");`
//      → `__sync_synchronize();` (Host memory barrier is the closest
//      emulation-side equivalent; the clobber list is optional — e.g. the
//      embedding_backward compute kernel's ARCH_BLACKHOLE cache-flush fence
//      omits it.)
//   3. `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` →
//      `reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>(N)))`
//      (Quasar kernels pass raw L1 firmware offsets as runtime args; x86 needs
//      translation through the per-thread __emule_bridge_l1 base pointer.)
// Reads from `src_path`, writes the patched source to `out_path`, and throws
// on any I/O failure.
// Regex-replace preserving the total line count (pads each replacement with the
// newlines its match consumed beyond it). The resulting line-alignment with the
// original kernel is what lets the `#line` directive below keep debug info — and
// thus ASAN backtraces — pointing at the real kernel file:line.
inline std::string emule_line_preserving_replace(
    const std::string& input, const std::regex& re, const std::string& fmt) {
    std::string out;
    auto pos = input.cbegin();
    for (std::sregex_iterator it(input.cbegin(), input.cend(), re), end; it != end; ++it) {
        const std::smatch& m = *it;
        out.append(pos, m[0].first);
        const std::string matched = m.str();
        const std::string rep = m.format(fmt);
        const long pad = std::count(matched.begin(), matched.end(), '\n') - std::count(rep.begin(), rep.end(), '\n');
        out += rep;
        for (long k = 0; k < pad; ++k) {
            out += '\n';
        }
        pos = m[0].second;
    }
    out.append(pos, input.cend());
    return out;
}

// Apply the x86 portability rewrites to one source string in place (RISC-V inline
// asm and L1-pointer/address reinterpret_casts that don't compile on the host).
// Line-spanning rewrites go through emule_line_preserving_replace to keep the
// body line-aligned with the original kernel (see there).
inline void apply_x86_rewrites(std::string& src) {
    static const std::regex mhartid_re(
        R"(asm\s+volatile\s*\(\s*"csrr\s+%0\s*,\s*mhartid"\s*:\s*"=r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)\s*;)");
    src = emule_line_preserving_replace(src, mhartid_re, "$1 = __emule_self->processor_id;");

    static const std::regex fence_re(R"(asm\s+volatile\s*\(\s*"fence"\s*(:::\s*"memory"\s*)?\)\s*;)");
    src = emule_line_preserving_replace(src, fence_re, "__sync_synchronize();");

    // ---- L1 offset model: translate a 0-based L1 offset to a host pointer at
    // the raw-deref site (docs/l1-emulation.md). get_write_ptr / get_read_ptr /
    // get_semaphore / get_arg_val now return offsets, so a cast to an L1 pointer
    // must rebase onto this fiber's bridge_l1. Rules are ordered so each site is
    // wrapped exactly once: P1 handles tt_l1_ptr casts and runs BEFORE the
    // get_arg_val/get_arg rules (so their output is not re-wrapped); P2 handles
    // attribute-less get_*_ptr casts, partitioned from P1 by a negative lookahead
    // on tt_l1_ptr. A missed site derefs a small offset → immediate SIGSEGV at
    // the real kernel line, surfaced for a targeted rule. ----

    // P1: cast to a tt_l1_ptr-attributed pointer (covers one- and two-step —
    // the deref cast re-adds the attribute). Operand allows one nested-paren level.
    // The `(?!\s*&)` skips an address-of operand — a `reinterpret_cast<tt_l1_ptr
    // T*>(&scalar)` re-interprets the bytes of a *stack* local (e.g. a pad value),
    // whose address is a host pointer, NOT a 0-based L1 offset; translating it would
    // corrupt it. (A bitwise `base & mask` operand does not start with & and still
    // matches.) The `get_(common_)?arg_addr` exclusion skips the rt-args accessors,
    // which return a host pointer (`&__emule_self->rt_args[i]`), not an L1 offset —
    // translating one would rebase a valid host pointer into garbage.
    static const std::regex l1_ptr_cast_re(
        R"(reinterpret_cast<\s*([^>]*\btt_l1_ptr\b[^>]*\*)\s*>\s*\(\s*(?!\s*&)(?!\s*get_(?:common_)?arg_addr\b)((?:[^()]|\([^()]*\))*?)\s*\))");
    src = emule_line_preserving_replace(
        src, l1_ptr_cast_re,
        "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)($2)))");

    static const std::regex l1_arg_ptr_re(
        R"(reinterpret_cast<([^>]+\*)>\s*\(\s*get_arg_val<uint32_t>\s*(\([^)]*\))\s*\))");
    src = emule_line_preserving_replace(
        src, l1_arg_ptr_re, "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr(get_arg_val<uint32_t>$2))");

    // Metal 2.0 named-arg pattern: reinterpret_cast<T*>(static_cast<uintptr_t>(get_arg(args::NAME)))
    static const std::regex l1_named_arg_ptr_re(
        R"(reinterpret_cast<([^>]+\*)>\s*\(\s*static_cast<uintptr_t>\s*\(\s*get_arg\s*\(\s*([^)]+)\s*\)\s*\)\s*\))");
    src = emule_line_preserving_replace(
        src,
        l1_named_arg_ptr_re,
        "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr(static_cast<uint32_t>(get_arg($2))))");

    // P2: an attribute-less cast (the negative lookahead excludes tt_l1_ptr,
    // which P1 already handled) whose operand is an L1 address from
    // get_write_ptr / get_read_ptr / get_semaphore — including the CircularBuffer
    // method form cb.get_write_ptr(). Operand is the call itself (one arg-paren
    // level); trailing arithmetic / nested args fall to the SIGSEGV net.
    static const std::regex l1_getptr_cast_re(
        R"(reinterpret_cast<\s*((?![^>]*\btt_l1_ptr\b)[^>]*\*)\s*>\s*\(\s*((?:[A-Za-z_]\w*\s*\.\s*)?(?:get_write_ptr|get_read_ptr|get_semaphore|get_tile_address)\s*(?:<[^>]*>)?\s*\([^()]*\))\s*\))");
    src = emule_line_preserving_replace(
        src, l1_getptr_cast_re,
        "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)($2)))");

    // P2b: the C-STYLE cast of an inline get_*_ptr — `(T*)(get_read_ptr(cb))` /
    // `(T*)(get_read_ptr(cb) + off)` (the all_to_all_dispatch writer reads token/mapping CBs
    // this way). P2 above only catches the reinterpret_cast spelling. Operand = the get_*_ptr
    // call (one arg-paren level) + optional trailing arithmetic (one nested-paren). The output
    // wraps in `(uint32_t)(...)` (no trailing `*`), which this rule's `\*`-terminated cast type
    // cannot re-match.
    static const std::regex l1_getptr_cstyle_re(
        R"(\(\s*([\w:][\w:\s]*\*)\s*\)\s*\(\s*((?:[A-Za-z_]\w*\s*\.\s*)?(?:get_write_ptr|get_read_ptr|get_semaphore|get_tile_address)\s*(?:<[^>]*>)?\s*\([^()]*\)(?:[^()]|\([^()]*\))*?)\s*\))");
    src = emule_line_preserving_replace(
        src, l1_getptr_cstyle_re,
        "($1)((uintptr_t)__emule_local_l1_to_ptr((uint32_t)($2)))");

    // P3: local L1<->L1 `memmove((void*)(dst), (void*)(src), n)` / memcpy — used by
    // tt::data_movement::common::tt_memmove. Both operands are 0-based L1 offsets
    // cast to void*; rebase each onto this fiber's L1. Narrowly anchored on the
    // memmove/memcpy call with two void*-cast args, so it never touches an unrelated
    // (void*) cast — a non-memmove L1 void* deref, if any, SIGSEGVs loudly and gets
    // its own targeted rule. One nested-paren level per operand.
    static const std::regex l1_mem_move_copy_re(
        R"(\b(memmove|memcpy)\s*\(\s*\(\s*void\s*\*\s*\)\s*\(\s*((?:[^()]|\([^()]*\))*?)\s*\)\s*,\s*\(\s*void\s*\*\s*\)\s*\(\s*((?:[^()]|\([^()]*\))*?)\s*\)\s*,)");
    src = emule_line_preserving_replace(
        src, l1_mem_move_copy_re,
        "$1((void*)__emule_local_l1_to_ptr((uint32_t)($2)), (void*)__emule_local_l1_to_ptr((uint32_t)($3)),");

    // P4 (precise, set-gated closure): translate casts of L1-address variables the
    // inline rules miss because the cast is attribute-less (P1 misses), not a
    // get_*_ptr call (P2 misses), C-style, over an arithmetic operand, or through a
    // #define'd pointer type. Two passes:
    //   (A) collect the L1-address variable set — vars assigned DIRECTLY from a
    //       producer (get_write_ptr / get_read_ptr / get_semaphore / get_tile_address,
    //       free or cb.-method form), then the transitive closure over
    //       `w = <collected-var> [+/- expr]` (e.g. vars_addr = means_addr + n).
    //   (B) per collected var v, translate a cast whose operand is v (optionally
    //       `v + arith`), in either the reinterpret_cast form (a non-tt_l1_ptr pointer
    //       target, or a `_l1_ptr`-suffixed macro type P1 can't see pre-macro) or the
    //       C-style `(T*)(...)` form.
    // Gating on the collected set is what keeps this from over-matching arbitrary casts:
    // a var not assigned from get_*_ptr is never collected, tt_l1_ptr reinterpret_casts
    // stay with P1 (negative lookahead), and get_arg_addr host pointers are never
    // collected. A collected var later reassigned to a non-L1 value is the only residual
    // risk — it fails loudly (SIGSEGV / the __emule_l1_translate OOB assert).
    {
        static const std::regex l1_addr_var_re(
            R"RE(\b(?:uint32_t|auto)\s+([A-Za-z_]\w*)\s*=\s*(?:[A-Za-z_]\w*\s*\.\s*)?(?:get_write_ptr|get_read_ptr|get_semaphore|get_tile_address|get_arg_val)\s*(?:<[^>]*>)?\s*\()RE");
        std::set<std::string> l1_addr_vars;
        for (std::sregex_iterator it(src.begin(), src.end(), l1_addr_var_re), end; it != end; ++it) {
            l1_addr_vars.insert((*it)[1].str());
        }
        // (A) transitive closure: `w = <collected-var> [+/- ...]` is itself an L1 address.
        bool grew = true;
        while (grew) {
            grew = false;
            for (const auto& c : std::vector<std::string>(l1_addr_vars.begin(), l1_addr_vars.end())) {
                const std::regex derived_re(
                    R"RE(\b(?:uint32_t|auto)\s+([A-Za-z_]\w*)\s*=\s*)RE" + c + R"RE(\b)RE");
                for (std::sregex_iterator it(src.begin(), src.end(), derived_re), end; it != end; ++it) {
                    if (l1_addr_vars.insert((*it)[1].str()).second) {
                        grew = true;
                    }
                }
            }
        }
        // (B) translate the collected vars' casts (operand = v, optionally v + arith
        // with one nested-paren level).
        for (const auto& v : l1_addr_vars) {
            // reinterpret_cast form: non-tt_l1_ptr pointer target (P1 owns tt_l1_ptr)
            // OR a `_l1_ptr`-suffixed macro type (opaque pre-macro, so P1 can't see it).
            const std::regex ric_re(
                R"RE(reinterpret_cast<\s*((?![^>]*\btt_l1_ptr\b)(?:[^>]*\*|\w*_l1_ptr))\s*>\s*\(\s*()RE" +
                v + R"RE(\b(?:[^()]|\([^()]*\))*?)\s*\))RE");
            src = emule_line_preserving_replace(
                src, ric_re,
                "reinterpret_cast<$1>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)($2)))");
            // C-style form `(<type>*)(v [+ arith])`. P1 never touches C-style casts, so
            // no tt_l1_ptr partition is needed. Runs after the reinterpret_cast form,
            // whose output ((uint32_t)(...) — no '*') this pattern cannot re-match.
            const std::regex cstyle_re(
                R"RE(\(\s*([\w:][\w:\s]*\*)\s*\)\s*\(\s*()RE" +
                v + R"RE(\b(?:[^()]|\([^()]*\))*?)\s*\))RE");
            src = emule_line_preserving_replace(
                src, cstyle_re,
                "($1)((uintptr_t)__emule_local_l1_to_ptr((uint32_t)($2)))");
            // C-style form with a bare (unparenthesised) operand `(<type>*)v` — e.g.
            // `patch_data = (volatile uint16_t*)intermed_l1_scratch`. Disjoint from the
            // parenthesised form above (which requires `(` after the cast) and from the
            // reinterpret_cast output.
            const std::regex cstyle_bare_re(
                R"RE(\(\s*([\w:][\w:\s]*\*)\s*\)\s*()RE" + v + R"RE()\b)RE");
            src = emule_line_preserving_replace(
                src, cstyle_bare_re,
                "($1)((uintptr_t)__emule_local_l1_to_ptr((uint32_t)($2)))");
        }
    }

    // P5 (per-site, interprocedural param-cast): two kernel helpers take a uint32_t
    // L1 address THROUGH a function parameter and reinterpret_cast<T*> it, so the
    // address is neither a tt_l1_ptr cast (P1) nor a locally get_*_ptr-assigned var
    // (P2/P4) — the value flows from cb.get_write_ptr() across a helper-call boundary
    // the intra-TU passes above can't follow. Each rule is anchored on the exact cast
    // (target type + the helper's documented L1-address operand) so it matches only
    // that one site; the translation happens at the cast, leaving the variable a
    // 0-based offset for its other uses (e.g. the sibling CoreLocalMem read path).
    // The full-suite run is the over-match gate.
    //   1) ttnn kernel_helper_functions/pad_tile.hpp fill_pad_tile(uint32_t l1_tile_ptr)
    //      — matmul in0/in1 padding readers (pad_last_ktile / pad_last_transposed_ktile).
    static const std::regex l1_pad_tile_re(
        R"(reinterpret_cast<\s*T\s*\*\s*>\s*\(\s*l1_tile_ptr\s*\))");
    src = emule_line_preserving_replace(
        src, l1_pad_tile_re,
        "reinterpret_cast<T*>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)(l1_tile_ptr)))");
    //   2) pad reader_pad_dims_rm_interleaved.cpp fill_with_val_async: curr_addr starts
    //      at the begin_addr param (= cb.get_write_ptr()) and is bumped per 2-byte store.
    static const std::regex l1_curr_addr_re(
        R"(reinterpret_cast<\s*uint16_t\s*\*\s*>\s*\(\s*curr_addr\s*\))");
    src = emule_line_preserving_replace(
        src, l1_curr_addr_re,
        "reinterpret_cast<uint16_t*>((uintptr_t)__emule_local_l1_to_ptr((uint32_t)(curr_addr)))");

    // reinterpret_cast<uint32_t>(ptr): the REVERSE direction — an L1 host pointer
    // narrowed to its device address. L1 offset model: the device address is the
    // 0-based offset, so subtract this fiber's bridge_l1 (was a bare truncation
    // under the old host-pointer-aliasing model). Arg allows one nested-paren
    // level; requiring '>' after uint32_t skips the pointer-typed
    // reinterpret_cast<T*> forms handled above.
    static const std::regex ptr_to_l1_addr_re(
        R"(reinterpret_cast<\s*(?:std::)?uint32_t\s*>\s*\(\s*((?:[^()]|\([^()]*\))*?)\s*\))");
    src = std::regex_replace(
        src, ptr_to_l1_addr_re,
        "static_cast<uint32_t>(reinterpret_cast<uintptr_t>($1) - reinterpret_cast<uintptr_t>(__emule_self->bridge_l1))");

    // tt-metal's sharded-layernorm dataflow util declares
    // `using RemoteNocCoords = RemoteNocCoord[N];`. The two-stage reduce path
    // instantiates it with N==0 for a core that has no second-stage workers,
    // forming a zero-length array RemoteNocCoord[0]. Silicon's kernel compiler
    // accepts that as a GNU extension; stock x86 clang rejects it as a
    // substitution failure ("zero-length arrays are not permitted in C++"),
    // even though the array is dead (the N==0 loop never runs and it is never
    // indexed). The instantiation is unavoidable because kernel_main() is not a
    // template, so the `if constexpr (use_two_stage_reduce)` dead branch is
    // still type-checked. Rewrite the array bound so N==0 yields a 1-element
    // dummy; behavior is identical on both arches (the element is never touched).
    static const std::regex zero_len_noc_coords_re(
        R"((using\s+RemoteNocCoords\s*=\s*RemoteNocCoord\s*\[)\s*N\s*(\]))");
    src = std::regex_replace(src, zero_len_noc_coords_re, "$1(N) == 0 ? 1 : (N)$2");

    // R2 (fabric header narrowing): a fabric/CCL kernel narrows a packet-header L1
    // *pointer* (pool-allocated, translated) to a device address via C-style
    // `(uint32_t)<header>` before handing it to the fabric client API. Under the offset
    // model that must yield the 0-based offset (ptr - bridge_l1), NOT the truncated host
    // pointer — the offset survives worker L1 mapped above 4 GB (the >4 GB galaxy), which
    // truncation does not; the emule fabric stub widens it back with __emule_local_l1_to_ptr
    // and keys the (chip-qualified) route table by it. Name-pattern-constrained to a
    // packet-header-pointer identifier (`*hdr*` / `*packet_header*` / `*header_ptr*` /
    // `*header_addr*`) — deliberately NOT bare `*header*`, which also matches
    // `current_cmd_header` (a CclCommandHeader *value*, not an L1 pointer). A blanket
    // `(uint32_t)EXPR` rule would hit ordinary int casts. An over-match of a non-pointer is loud, not silent
    // — reinterpret_cast<uintptr_t>(non_pointer) is ill-formed and fails the JIT compile.
    static const std::regex l1_fabric_hdr_narrow_re(
        R"(\(\s*uint32_t\s*\)\s*(\w*(?:hdr|packet_header|header_ptr|header_addr)\w*))");
    src = std::regex_replace(
        src, l1_fabric_hdr_narrow_re,
        "(uint32_t)(reinterpret_cast<uintptr_t>($1) - reinterpret_cast<uintptr_t>(__emule_self->bridge_l1))");
}

// Patch `src_path` into `out_path`, then recurse into the quoted project headers
// it #includes (a shared `*_common.hpp` can hold the offending casts too). Each
// patched header is written into `out_dir` under its include name; since the
// top-level patched_kernel.cpp also lives there, the compiler finds the patched
// copy before the original on the `-I kernel_dir` path. Includes resolved
// elsewhere (emule api/, system) or escaping `out_dir` are left alone; `done`
// guards cycles.
inline void preprocess_tu_recursive(
    const std::string& src_path,
    const std::string& out_path,
    const std::string& out_dir,
    const std::vector<std::string>& kernel_inc_roots,
    const std::vector<std::string>& emule_inc_roots,
    std::map<std::string, std::string>& done) {
    std::ifstream in(src_path);
    if (!in) {
        throw std::runtime_error("preprocess_kernel_source_for_x86: cannot read " + src_path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string src = ss.str();

    apply_x86_rewrites(src);

    const std::filesystem::path src_dir = std::filesystem::path(src_path).parent_path();
    const std::filesystem::path out_dir_canon = std::filesystem::weakly_canonical(out_dir);

    // Include directives to rewrite in `src` after the scan (a deep `../…` include
    // that escapes out_dir is remapped to an include-root-relative name so its
    // patched copy can live in out_dir). Applied post-loop to avoid invalidating the
    // iterator below.
    std::vector<std::pair<std::string, std::string>> inc_rewrites;
    static const std::regex include_re(R"RE(#[ \t]*include[ \t]*"([^"]+)")RE");
    for (std::sregex_iterator it(src.begin(), src.end(), include_re), end; it != end; ++it) {
        const std::string inc_name = (*it)[1].str();
        std::error_code ec;
        std::filesystem::path candidate = src_dir / inc_name;
        if (!std::filesystem::exists(candidate, ec)) {
            // Not a same-directory (quote-relative) include. If an emule shadow
            // header exists for this name (jit_hw / emule include), the compiler
            // resolves it there via -I order and it is already offset-correct — do
            // NOT patch or shadow it (a patched copy in out_dir would wrongly win
            // via quote-relative resolution). Otherwise resolve against the kernel
            // include roots (ttnn/, tt_metal/): a shared kernel helper in another
            // directory (e.g. ttnn/cpp/ttnn/kernel_lib/*.inl) carries the same raw
            // L1-deref idioms and must be patched too — same-dir recursion can't
            // reach it and the header itself must stay pristine.
            bool has_emule_shadow = false;
            for (const auto& er : emule_inc_roots) {
                if (std::filesystem::exists(std::filesystem::path(er) / inc_name, ec)) {
                    has_emule_shadow = true;
                    break;
                }
            }
            if (has_emule_shadow) {
                continue;
            }
            std::filesystem::path resolved;
            for (const auto& kr : kernel_inc_roots) {
                std::filesystem::path c2 = std::filesystem::path(kr) / inc_name;
                if (std::filesystem::exists(c2, ec)) {
                    resolved = c2;
                    break;
                }
            }
            if (resolved.empty()) {
                // System header, or unresolved — left to -fms-extensions.
                continue;
            }
            candidate = resolved;
        }
        const std::string canon = std::filesystem::weakly_canonical(candidate, ec).string();
        if (canon.empty()) {
            continue;
        }
        // Already patched under a "mirror name" (the out_dir-relative path its patched
        // copy lives at)? Point this include at that mirror. A header included via two
        // spellings (e.g. `cpp/ttnn/X` and `ttnn/X`) is patched once, under the first
        // spelling; the other spelling would otherwise fall through to the pristine
        // header, and #pragma once (path-keyed) cannot dedup patched-vs-pristine →
        // redefinition. Rewriting the alternate spelling to the mirror makes every
        // include resolve to the single patched copy (via -I out_dir).
        {
            auto prior = done.find(canon);
            if (prior != done.end()) {
                if (inc_name != prior->second) {
                    inc_rewrites.emplace_back(inc_name, prior->second);
                }
                continue;  // cycle / already patched
            }
        }
        std::string mirror_name = inc_name;  // out_dir-relative path the patched copy lives at
        std::filesystem::path out_inc = std::filesystem::weakly_canonical(
            std::filesystem::path(out_dir) / inc_name);
        std::string out_inc_str = out_inc.string();
        // A deep `../…` include (e.g. the softmax writer's
        // `../../../../../../kernel_helper_functions/pad_tile.hpp`) mirrors to a path
        // that escapes out_dir. Rather than leave it unpatched, remap it to an
        // include-root-relative name that stays inside out_dir, and rewrite the
        // directive in `src` so the compiler finds the patched copy (via -I out_dir)
        // instead of the pristine original.
        if (out_inc_str.compare(0, out_dir_canon.string().size(), out_dir_canon.string()) != 0) {
            std::string norm_name;
            for (const auto& kr : kernel_inc_roots) {
                const std::string krc = std::filesystem::weakly_canonical(kr, ec).string();
                if (!krc.empty() && canon.size() > krc.size() + 1 &&
                    canon.compare(0, krc.size(), krc) == 0 && canon[krc.size()] == '/') {
                    norm_name = canon.substr(krc.size() + 1);
                    break;
                }
            }
            if (norm_name.empty()) {
                continue;  // unresolvable escape — leave it to -fms-extensions
            }
            inc_rewrites.emplace_back(inc_name, norm_name);
            mirror_name = norm_name;
            out_inc = std::filesystem::weakly_canonical(std::filesystem::path(out_dir) / norm_name);
            out_inc_str = out_inc.string();
        }
        done[canon] = mirror_name;
        std::filesystem::create_directories(out_inc.parent_path(), ec);
        preprocess_tu_recursive(candidate.string(), out_inc_str, out_dir, kernel_inc_roots, emule_inc_roots, done);
    }
    // Apply the escaping-include remaps to `src` before it is written out. Plain
    // string replace of the quoted path (it contains regex-special chars); the path
    // stays on its original line, so line-count is preserved.
    for (const auto& [from, to] : inc_rewrites) {
        const std::string needle = "\"" + from + "\"";
        const std::string repl = "\"" + to + "\"";
        for (std::string::size_type pos = src.find(needle); pos != std::string::npos;
             pos = src.find(needle, pos + repl.size())) {
            src.replace(pos, needle.size(), repl);
        }
    }

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("preprocess_kernel_source_for_x86: cannot write " + out_path);
    }
    // Attribute the emitted body to the real kernel file so DWARF (and thus ASAN
    // backtraces) report `<real kernel>.cpp:<line>` rather than the generated temp
    // copy. The rewrites above are line-preserving, so line N here == line N there.
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(src_path, ec);
    out << "#line 1 \"" << (ec ? src_path : abs.string()) << "\"\n";
    out << src;
}

}  // namespace tt::emule::detail
