#pragma once
// Stub: device-side debug print is not supported in emulation.
// Macros expand to no-ops so kernels that include this header compile cleanly.
#ifndef DPRINT
#define DPRINT (void)0 &&
#endif
#ifndef DPRINT_DATA0
#define DPRINT_DATA0(...)
#endif
#ifndef DPRINT_DATA1
#define DPRINT_DATA1(...)
#endif
