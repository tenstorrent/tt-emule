#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Inject tt-emule cmake args into tt-mlir's tt-metal ExternalProject_Add.

tt-mlir's third_party/CMakeLists.txt builds its own tt-metal via
ExternalProject_Add. To make D2M CI build tt-metal *against the PR's tt-emule*,
we append three CMAKE_ARGS to that ExternalProject definition before configure.

Idempotent: re-running on an already-patched file is a no-op.

Usage:
    inject_ttemule_into_mlir.py <path-to-third_party/CMakeLists.txt> <tt-emule-path>
"""

import re
import sys
from pathlib import Path


SENTINEL = "TT_EMULE_PATH"  # presence indicates the file is already patched
ANCHOR = "-DTT_USE_SYSTEM_SFPI=${TT_USE_SYSTEM_SFPI}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    cmake_path = Path(sys.argv[1])
    tt_emule_path = sys.argv[2]

    if not cmake_path.is_file():
        print(f"ERROR: not a file: {cmake_path}", file=sys.stderr)
        return 1

    text = cmake_path.read_text()

    if SENTINEL in text:
        print(f"  already patched ({SENTINEL} present), skipping")
        return 0

    if ANCHOR not in text:
        print(f"ERROR: anchor line not found: {ANCHOR!r}", file=sys.stderr)
        return 1

    # Preserve the indentation of the anchor line (4 spaces in tt-mlir today).
    m = re.search(r"^([ \t]*)" + re.escape(ANCHOR), text, flags=re.MULTILINE)
    if m is None:
        print(f"ERROR: anchor regex did not match", file=sys.stderr)
        return 1
    indent = m.group(1)

    injection = "\n".join(
        [
            f"{indent}-DTT_METAL_USE_TT_EMULE=ON",
            f"{indent}-DTT_METAL_EMULATION=ON",
            f"{indent}-DTT_EMULE_PATH={tt_emule_path}",
        ]
    )

    new_text = text.replace(
        ANCHOR,
        f"{ANCHOR}\n{injection}",
        1,
    )

    cmake_path.write_text(new_text)
    print(f"  injected tt-emule cmake args (TT_EMULE_PATH={tt_emule_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
