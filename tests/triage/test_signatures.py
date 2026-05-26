# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from triage_lib.signatures import normalize, signature


def test_normalize_strips_workspace_paths():
    a = normalize("/__w/tt-emule/tt-emule/tt-metal/foo.cpp:42:7")
    b = normalize("/opt/runner/_work/tt-emule/tt-emule/tt-metal/foo.cpp:42:7")
    c = normalize("/home/runner/work/tt-emule/tt-emule/tt-metal/foo.cpp:42:7")
    assert a == b == c == "<WS>/tt-metal/foo.cpp:L:C"


def test_normalize_strips_hex_addresses():
    n = normalize("crashed at 0x7f3ab2cdef00 with code")
    assert "0xADDR" in n
    assert "0x7f3ab2cdef00" not in n


def test_signature_stable_across_runs():
    cap = {"symbol": "__emule_cb_waited_pages"}
    assert signature("MISSING_SYMBOL", cap) == signature("MISSING_SYMBOL", cap)


def test_signature_stable_across_runners():
    a = signature("BUILD_FAILURE", {"file": "/__w/tt-emule/tt-emule/foo/bar.cpp"})
    b = signature("BUILD_FAILURE", {"file": "/home/runner/work/tt-emule/tt-emule/foo/bar.cpp"})
    assert a == b


def test_signature_depends_on_captures():
    a = signature("MISSING_SYMBOL", {"symbol": "__emule_a"})
    b = signature("MISSING_SYMBOL", {"symbol": "__emule_b"})
    assert a != b


def test_signature_depends_on_category():
    a = signature("MISSING_SYMBOL", {"symbol": "__emule_a"})
    b = signature("BUILD_FAILURE", {"symbol": "__emule_a"})
    assert a != b


def test_signature_format():
    s = signature("CAT", {"x": "y"})
    assert s.startswith("trg-")
    assert len(s) == len("trg-") + 12
