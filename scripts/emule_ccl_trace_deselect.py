# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# emule LoudBox CCL trace-parametrization deselect (issue #242).
#
# Trace capture (`enable_trace` / `trace_mode` = True) requires FAST dispatch, which the
# software emulator does not support — SDMeshCommandQueue::record_begin TT_THROWs "Not
# supported for slow dispatch". tt-metal #48548 carried this deselect as an emule-only
# conftest.py under the `blackhole_CI/box/{all_post_commit,nightly}` dirs, but the merged
# commit (a0f5d8e4) reverted those conftests (they were intentionally kept out of metal
# `main`). Without the hook the trace parametrizations run under emule and fail loudly,
# inflating the LoudBox FAIL count with configs emule can never execute.
#
# The emule harness re-adds the deselect as a pytest plugin (loaded via `-p`) instead of
# writing a conftest into the pinned tt-metal checkout — it deselects the exact param
# combinations the reverted conftest did, and is a no-op off emule (`TT_METAL_EMULE_MODE`
# unset) so it never touches upstream coverage. See scripts/run_ttnn_pytests_bh_loudbox.sh
# (the LoudBox box/{all_post_commit,nightly} suites are the only ones the conftest covered).

import os


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("TT_METAL_EMULE_MODE"):
        return
    kept = []
    deselected = []
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if params.get("enable_trace") is True or params.get("trace_mode") is True:
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
