# Known Test Failures (tt-metal Emulated Mode)

These are known test failures in the `MeshDeviceFixture.*` test suite when running
under emulated mode (`TT_METAL_EMULE_MODE=1`). All are due to missing emulation
stubs for hardware-specific features. They are not regressions.

## 1. Stream Register Tests

**Tests:**
- `MeshDeviceFixture.TensixDirectedStreamRegWriteRead`
- `MeshDeviceFixture.TensixIncrementStreamRegWrite`
- `MeshDeviceFixture.TensixTestNocStreamRegs`
- `MeshDeviceFixture.IdleEthTestNocStreamRegs`

**Symptom:** JIT compilation fails with `fatal error: 'noc_overlay_parameters.h' file not found`.

**Root cause:** These kernels include `noc_overlay_parameters.h` and `stream_io_map.h`,
which define memory-mapped register addresses for the NOC stream overlay engine.
The emulator has no stubs for these headers because stream registers are a
hardware-specific concept (register I/O on the NOC overlay block). Emulating them
would require modeling the stream hardware state machine.

**Effort to fix:** High. Requires understanding the full stream overlay register
interface and deciding which subset to stub vs. functionally emulate.

## 2. Inline Write Tests

**Tests:**
- `MeshDeviceFixture.TensixInlineWrite4BAlignment`
- `MeshDeviceFixture.TensixInlineWriteDedicatedNoc`
- `MeshDeviceFixture.TensixInlineWriteDedicatedNocMisaligned`
- `MeshDeviceFixture.TensixInlineWriteDynamicNoc`

**Symptom:** JIT compilation fails with multiple errors:
- `no member named 'inline_dw_write' in 'experimental::Noc'`
- `use of undeclared identifier 'noc_mode'`
- `use of undeclared identifier 'DM_DYNAMIC_NOC'`

**Root cause:** These kernels use three features not yet emulated:
1. `noc->inline_dw_write()` -- a 4-byte inline NOC write that bypasses the normal
   async transaction path. On real hardware this is a single-cycle register write.
2. `noc_mode` -- a compile-time constant selecting between dedicated and dynamic
   NOC routing.
3. `DM_DYNAMIC_NOC` -- the enum value for dynamic NOC mode selection.

The emulator's `experimental::Noc` class implements `async_read`/`async_write` via
memcpy but does not model the inline write path or NOC mode selection, since all
emulated NOC operations use the same memcpy path regardless of routing mode.

**Effort to fix:** Low-medium. `inline_dw_write` can be stubbed as a 4-byte memcpy.
`noc_mode`/`DM_DYNAMIC_NOC` can be defined as no-op constants since the emulator
doesn't distinguish NOC routing modes.

## 3. Runtime Args Segfault

**Test:** `MeshDeviceFixture.TensixLegallyModifyRTArgsDataMovement`

**Symptom:** `SIGSEGV` at address `0x19520` (an L1 offset, not a valid host pointer).

**Root cause:** The kernel reads a runtime arg containing a raw L1 offset and
dereferences it directly as a pointer. On real hardware, L1 is memory-mapped
starting at address 0, so small offsets like `0x19520` are valid L1 addresses.
In emulation, L1 is mmap'd at a high host address (hint `0x40000000`), so the
raw offset is not a valid host pointer.

The kernel would need to convert the offset through `__emule_local_l1_ptr()` to
get a valid host pointer, but the test kernel was written for real hardware and
does not use this indirection.

**Effort to fix:** Medium. Requires either:
- (a) Intercepting raw L1 pointer dereferences in the JIT wrapper (e.g., via a
  custom allocator or pointer-rewriting pass), or
- (b) Patching the specific test kernel to use the emulation bridge function.

Option (b) is simple but doesn't generalize; option (a) is the correct long-term
fix but requires changes to the JIT memory model.

## 4. Active Ethernet Skip

**Test:** `MeshDeviceFixture.ActiveEthTestNocStreamRegs`

**Symptom:** Test is skipped (not failed).

**Root cause:** The emulated wormhole N150 cluster descriptor has no active
ethernet cores, so the test correctly skips itself. This is expected behavior,
not a bug.
