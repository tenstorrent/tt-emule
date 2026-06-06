// SPDX-License-Identifier: Apache-2.0
// Emule shadow for the silicon LLK math reduce API.
// Included transitively by upstream custom_tilize.h under #ifdef TRISC_MATH,
// which emule's program runner sets for all non-Quasar compute kernels.
// custom_tilize.h doesn't actually call any reduce APIs — provide an empty
// shadow to satisfy the include.
#pragma once
