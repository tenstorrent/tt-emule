// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shadow for the silicon LLK IO pack header. Real header lives under
// the silicon LLK IO tree and provides packer-side
// CB index/operand mapping. Emule's CB API doesn't need any of it — CB
// state lives in __emule_self->cbs (per-thread ctx array) and CB ops route through
// cb_sync_*. This empty file short-circuits the chain.
