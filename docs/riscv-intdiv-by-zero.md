# Integer divide/modulo-by-zero in emule

The real Tensix cores are **RISC-V**, where integer divide/remainder by zero
is *defined and non-trapping*. emule JIT-compiles each kernel to **x86-64**,
where the same operation raises a `#DE` fault → `SIGFPE` → process abort.
This doc explains the gap and how emule's runtime closes it so JIT kernels
behave like silicon.

## What RISC-V defines

The RISC-V "M" extension specifies div/rem-by-zero with **no trap** and a
fixed result (RISC-V Unprivileged ISA, *Division Operations*):

| Op | Divisor 0 result |
|---|---|
| `DIVU` (unsigned div) | `2^XLEN − 1` (all ones) |
| `DIV` (signed div)    | `−1` (all ones, two's complement) |
| `REMU` / `REM`        | the dividend (unchanged) |

So for **both** signed and unsigned: **quotient = all-ones, remainder =
dividend**. Kernel code that divides by zero gets a defined value and keeps
running; it never faults.

## Why kernels legitimately divide by zero

A correct kernel can compute `x / 0` on **idle or degenerate cores** whose
result is never used. The motivating case is `binary_ng`:

`ttnn/.../eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`

```cpp
const uint32_t tiles_per_nd = D * N * C * Ht * Wt;        // dims from runtime args
const uint32_t offset_nd    = start_tile_id % tiles_per_nd;  // no zero-guard
```

The host program factory
(`ttnn/.../binary_ng/device/binary_ng_program_factory.cpp`, ~line 1029)
launches the reader on **every** core in the worker grid but only assigns
real work to a subset. Cores outside the work split get an **all-zero
`dummy_reader{0}`** runtime-arg vector → `D=N=C=Ht=Wt=0` → `tiles_per_nd =
0` → `start_tile_id % 0`. On silicon this is harmless: the modulo returns a
defined value and the core's tile loop runs zero iterations (the result is
dead). On x86 the `idiv` traps.

This is **not** a missing mock and **not** a runtime-args delivery bug — the
args are delivered correctly; the divisor is genuinely zero by design.

## The silicon contract

> A JIT kernel that performs an integer `/` or `%` by zero must not fault;
> it observes RISC-V's defined result (quotient all-ones, remainder =
> dividend) and continues.

## How emule satisfies it

A `SIGFPE` handler installed around kernel execution, in the emule runtime:

`tt-metal` companion → `tt_metal/impl/emulation/emulated_program_runner.cpp`

- **`emule_sigfpe_handler`** — on `SIGFPE` with `si_code == FPE_INTDIV`,
  reads the saved CPU register image (`ucontext_t`), writes RISC-V's defined
  result into the saved `RAX` (quotient = all-ones) and `RDX` (remainder =
  dividend), advances the saved `RIP` past the faulting instruction, and
  returns — so the faulting thread resumes as if the divide had produced the
  RISC-V value. Anything other than an integer divide-by-zero falls back to
  the default disposition (`SIG_DFL` + re-raise).
- **`emule_decode_divlen`** — a minimal length-decoder for the only x86 forms
  clang emits for C integer `/` and `%`: `F6 /6,/7` (8-bit) and `F7 /6,/7`
  (32/64-bit), with optional legacy + REX prefixes and ModRM/SIB/disp for a
  memory operand. It is *not* a general disassembler — just enough to measure
  one instruction so `RIP` can step over it. `REX.W` selects 64-bit operand
  size (remainder/dividend masking).
- **`EmuleSigfpeGuard`** — RAII installer (`sigaction` on construct, restore
  previous disposition on destruct), scoped to the body of `launch_cores()`
  so emule only intercepts `SIGFPE` while kernel code is running and never
  permanently alters the host process's signal disposition.

`SIGFPE` from `#DE` is **synchronous and thread-directed**, so one
process-wide handler correctly services whichever worker thread faults; no
per-thread setup is needed. The whole thing is guarded by
`#if defined(__x86_64__) && defined(__linux__)` (it depends on the host
register / `ucontext` layout).

## Result-value fidelity

For the idle-core case the result is dead, so any value would unblock the
run. The handler nonetheless writes RISC-V's *exact* values so that a kernel
which divides by zero and **uses** the result computes the same thing emule's
x86 backend would on silicon's RISC-V backend. (A kernel that genuinely
depends on a div-by-zero result is itself suspect, but emule should not
diverge from silicon there either.)

## Silicon ↔ emule mapping

| | Silicon (RISC-V Tensix) | emule (x86-64 host) |
|---|---|---|
| `x / 0` | quotient = all-ones, no trap | handler writes all-ones to `RAX` |
| `x % 0` | remainder = dividend, no trap | handler writes dividend to `RDX` |
| After the op | execution continues | handler steps `RIP` over the `idiv`, returns |
| Faulting unit | M-extension hardware | host CPU `#DE` → `SIGFPE` |

Same observable behavior; the only difference is the mechanism (defined HW
result vs. trap-and-fix-up).

## Verification

`ttnn.add`/`ttnn.mul` and all `binary_ng` tensor-tensor ops (including the
eltwise comparison ops from issue #75) ran a 64-core grid with idle cores
dividing by zero; before the handler every one aborted with *"Signal code:
Integer divide-by-zero (1)"*, and after it they complete with correct
results. Covered by the comparison-op entries in
`scripts/run_ttnn_pytests_{wormhole,blackhole}.sh`.
