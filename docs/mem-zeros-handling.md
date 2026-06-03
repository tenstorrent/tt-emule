# MEM_ZEROS in emule

The `MEM_ZEROS` region is a firmware-reserved L1 area that silicon kernels
NOC-read from as a cheap source of zeros (the canonical "splat zeros into
this CB" idiom on Wormhole and Blackhole). This doc explains how emule
preserves the silicon contract without doing the wrong thing between
program runs.

## What MEM_ZEROS is on silicon

A small L1 region (512 bytes) at a per-arch fixed address:

| Arch | `MEM_ZEROS_BASE` | Source |
|---|---|---|
| Wormhole B0 | `0x3280` | `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` |
| Blackhole   | `0x32E0` | `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` |
| Quasar      | `0xE180` | `tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h` |

Computed in upstream's header cascade as `(MEM_MAILBOX_END + 31) & ~31` —
the next 32-byte-aligned address past the per-RISC mailbox region.

**Init**: BRISC firmware calls `wzeromem(MEM_ZEROS_BASE, MEM_ZEROS_SIZE)`
exactly **once at boot**, inside `risc_init()` (see
`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:237`). This runs before BRISC
enters its program-dispatch loop. There is no per-program re-zeroing
and no hardware write protection — the region just stays zero by
convention.

**Reader pattern**: kernels NOC-read from `MEM_ZEROS_BASE` to bulk-fill a
destination with zeros — cheap, NOC-pipelined, no CPU stalls. Common
sites in tt-metal:

- `ttnn/.../kernel_lib/l1_helpers.hpp` — `zero_tile(...)`
- `ttnn/.../kernel/dataflow/generate_reduce_scaler.hpp`
- `ttnn/.../operations/normalization/groupnorm/.../groupnorm_zero_fill.hpp::zero_whole_cb`
- `ttnn/.../operations/kv_cache/.../zero_cache_writer.cpp`
- `ttnn/.../operations/full/device/kernels/full_kernel_common.hpp::zero_buffer`

…plus conv halo, CCL, sliding-window, and several others. Blackhole uses
this idiom more heavily than Wormhole.

**Write protection**: none. Any RISC at any time can write to the region
and corrupt subsequent reads. The "zero" invariant is purely a
software-firmware convention, persistent until the next device reset.

## The silicon contract

> Under correct usage, `MEM_ZEROS_BASE..+MEM_ZEROS_SIZE` reads as zero
> from device boot to next device reset.

A faithful emulator must satisfy the same contract.

## Why the per-program memset was wrong

Earlier emule had a defensive memset in `Core::reset_l1_bump()` that
re-zeroed the MEM_ZEROS region between every program run (see PR #52
and the host-side mirror in `include/tt_emule/device.hpp`). This was
unlike silicon in three ways:

1. **Extra work silicon never does.** Silicon zeros once at boot.
   Re-zeroing on every `EnqueueProgram` is emule-only behavior.

2. **Masks latent kernel bugs.** If a buggy kernel writes to
   `MEM_ZEROS_BASE` on silicon, the next kernel reads its garbage and
   the bug surfaces (PCC failure / hang). With a per-program memset,
   the next program would silently re-zero the region and hide the
   bug. The "preemptive BH parity" goal — *catch bad memory accesses
   by real kernels* — requires emule to NOT paper over this.

3. **Could itself cause corruption.** PR #52's motivating bug was the
   memset clobbering user buffer space when `MEM_ZEROS_BASE` was wrong
   (`0xFFE00` inside the 1.43 MiB worker L1).

## How emule preserves the invariant today

emule's L1 mmap is `MAP_PRIVATE | MAP_ANONYMOUS`
(`include/tt_emule/device.hpp::mmap_region` and
`include/tt_emule/l1_pool.hpp`). Linux guarantees `MAP_ANONYMOUS` pages
are zero-initialized at first access. That single `mmap` is emule's
*"wzeromem at boot"* analog — performed once when the process maps L1,
never again.

The bump allocator (`Core::l1_alloc`, used by DFB fallback allocation)
is the only path that could overwrite the firmware-reserved L1 range
in emule. It is now gated behind `if (!dfb_impls.empty())` in
`tt_metal/impl/emulation/emulated_program_runner.cpp::allocate_dfbs_on_core`.
On Wormhole and Blackhole, `dataflow_buffers_on_core(...)` always
returns empty (DFBs are a Quasar-only feature), so the bump allocator
never runs — and the bytes at MEM_ZEROS_BASE retain their mmap-init
zeros forever.

No other code path in the integrated runner writes below
`l1_unreserved_base`:

- tt-metal allocator only hands out addresses ≥ `l1_unreserved_base`
- `init_core_cb_sync` writes at CB addresses (from the allocator)
- `init_core_semaphores` writes at `kernel_config_base + sem_offset`
- Runtime args go to `kernel_config_base + rta_offset`

All comfortably above the MEM_ZEROS region.

## Silicon ↔ emule mapping

| Phase | Silicon | emule |
|---|---|---|
| Init | BRISC `wzeromem` once at boot | Linux `MAP_ANONYMOUS` once at process start |
| Between programs | nothing (region stays zero) | nothing (region stays zero) |
| If a kernel writes there | corruption persists until device reset | corruption persists until process restart |
| Kernel NOC-read of `MEM_ZEROS_BASE` | returns zeros | returns zeros |

Same contract, same failure mode. The only difference is the time scale
of "boot": silicon's power cycle ↔ emule's process startup.

## Verification

A regression test asserts the invariant directly:

`tests/tt_metal/tt_metal/api/test_simple_l1_buffer.cpp::MeshDeviceFixture.EmuleMemZerosStaysZero`

NOC-reads `[MEM_ZEROS_BASE, +512)` on 8 worker cores per arch, both
before any program runs and after a real JIT kernel
(`SimpleTiledL1WriteCBRead`) — asserts every byte stays zero. Run
under both WH (`wormhole_N150.yaml`) and BH (`blackhole_P100.yaml`)
cluster descriptors.

## Quasar caveat

Quasar uses DFBs, which means the DFB fallback path in
`allocate_dfbs_on_core` DOES invoke `core->reset_l1_bump()` on Quasar.
The bump allocator currently starts at `0` and grows upward, which
on Quasar would eventually cross into `MEM_ZEROS_BASE = 0xE180` and
clobber the zeros.

When Quasar bring-up reaches this path, the fix is a single-call-site
change at the `core->reset_l1_bump()` call inside
`allocate_dfbs_on_core`: dispatch on `device->arch()` and reset the
bump above `MEM_ZEROS_BASE + MEM_ZEROS_SIZE` for Quasar. The doc
comment at the call site flags this. WH and BH continue to not need
any per-arch handling, because the bump allocator never runs on them.
