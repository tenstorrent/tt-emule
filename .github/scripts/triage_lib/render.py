# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tiny string.Template renderer with a small set of filter-like helpers.

We deliberately avoid Jinja so this code can run on a minimal Python install
without any third-party deps beyond pyyaml + requests.

Templates use ``${name}`` and ``${name|filter}``.  Supported filters:
- ``|length`` — replaces with the length of the value (list or string).
- ``|bullets`` — joins a list of strings as ``- item\\n`` bullets.

The renderer ignores unknown placeholders rather than raising, so partially
filled-in templates (e.g. ``${ai_summary?}``) degrade gracefully.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)(?:\|([a-zA-Z_]+))?(\?)?\}")


def render(template: str, ctx: Mapping[str, Any]) -> str:
    """Render ``template`` with ``ctx``.

    Missing names: raise ``KeyError`` unless the placeholder ends in ``?``,
    in which case it is silently replaced with the empty string.
    """

    def _sub(match: re.Match[str]) -> str:
        name, filt, optional = match.group(1), match.group(2), match.group(3)
        if name not in ctx:
            if optional:
                return ""
            raise KeyError(f"template references unknown name: {name!r}")
        value = ctx[name]
        if filt is None:
            return str(value)
        if filt == "length":
            return str(len(value))
        if filt == "bullets":
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"|bullets requires a list, got {type(value).__name__}")
            return "\n".join(f"- {item}" for item in value)
        raise ValueError(f"unknown filter: {filt!r}")

    return _PLACEHOLDER.sub(_sub, template)
