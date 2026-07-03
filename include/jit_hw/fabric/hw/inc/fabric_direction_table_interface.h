// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// emule shadow: the real fabric_direction_table_interface.h reads device-L1 fabric routing tables that emule does
// not populate (no fabric firmware). emule resolves the destination chip via the
// cluster neighbor table in the teleport hook instead, so this is an empty stub.
