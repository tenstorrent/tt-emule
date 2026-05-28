# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin: deterministic per-test seeding for D2M golden tests.

tt-mlir's `tools/builder/base/builder.py::_generate_random_tensor` calls
`torch.randn` / `torch.randint` without a per-test seed, so every CI run
generates different input tensors. For ops whose PCC threshold is tight
(layer-norm, ternary clamp, …), the same parametrization can pass on
one run and fail on the next — the basis of most issue #35 flakes.

This plugin registers an autouse fixture that seeds `torch`, `numpy`,
and `random` from a hash of `request.node.nodeid` before each test
function. Same test → same seed → same inputs across runs.

Loaded via `PYTEST_PLUGINS=d2m_seed_plugin` (which env-propagates
through `pytest --forked` workers). `run_d2m_regression.sh` is wired to
set that env var and prepend this directory to `PYTHONPATH`.
"""

from __future__ import annotations

import hashlib

import pytest


def _seed_from_nodeid(nodeid: str) -> int:
    # 32-bit fits torch.Generator().manual_seed and numpy.random.seed.
    return int(hashlib.sha256(nodeid.encode()).hexdigest()[:8], 16)


@pytest.fixture(autouse=True)
def _d2m_deterministic_seed(request):
    seed = _seed_from_nodeid(request.node.nodeid)

    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    yield
