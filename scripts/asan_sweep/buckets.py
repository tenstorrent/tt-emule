#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Which manifest entries each ASAN sweep bucket runs, and who owns it.

Buckets are grouped by CODE OWNER, not by size: when a bucket goes red, its name
should be enough to know which team to hand it to. The owners below are the
tt-metal CODEOWNERS rules for each entry's test path — they are near-disjoint
sets of people (mmfusedreduce shares no members with convolutions or sdpa), which
is why matmul/reduce/fused are kept apart from conv/sdpa rather than merged into
one "ops" bucket.

Sizes are deliberately uneven as a result. `eltwise` is ~63% of the test set and
is the run's critical path; splitting it is a size decision, not an ownership one,
so it stays whole until someone wants the wall-clock back.

Consumed by the workflow's `plan` step, which turns this into the job matrix, and
by the report job, which checks every bucket reported in. `validate()` is what
stops a newly added tt-metal entry from silently falling out of the sweep.
"""

# Each bucket: id (job name + marker), owner (who to ping), entries (manifest slugs).
BUCKETS = [
    {
        "id": "eltwise",
        "owner": "metalium-developers-eltwise",
        "entries": [
            "ttnn-eltwise-group-1",
            "ttnn-eltwise-group-2",
            "ttnn-eltwise-group-3",
            "ttnn-eltwise-group-4",
        ],
    },
    {
        "id": "matmul-reduce-fused",
        "owner": "metalium-developers-mmfusedreduce",
        "entries": [
            "ttnn-matmul-group",
            "ttnn-reduce-group",
            "ttnn-fused-group-1",
            "ttnn-fused-group-2",
        ],
    },
    {
        "id": "conv-sdpa",
        # Two teams, but they overlap (pavlejosipovic, skrsticTT are in both) —
        # unlike mmfusedreduce, which shares nobody with either.
        "owner": "metalium-developers-convolutions + metalium-developers-sdpa",
        "entries": [
            "ttnn-conv-group",
            "ttnn-sdpa-group",
            "ttnn-indexer-score-group",
        ],
    },
    {
        "id": "data-movement",
        "owner": "metalium-developers-ops-data-movement",
        "entries": ["ttnn-data-movement-group"],
    },
    {
        "id": "pool",
        "owner": "metalium-developers-convolutions",
        "entries": ["ttnn-pool-group"],
    },
    {
        "id": "misc",
        "owner": "metalium-developers-ttnn-core",
        "entries": ["ttnn-misc-ops-group", "core-ttnn-unit-test-group"],
    },
]


def bucket_ids():
    return [b["id"] for b in BUCKETS]


def validate(manifest_slugs):
    """Compare the bucket table against what the manifest actually contains.

    Returns (unassigned, stale): entries the manifest has that no bucket claims,
    and entries a bucket claims that the manifest no longer has. The first is the
    dangerous one — an unassigned entry would just never be swept, quietly
    shrinking coverage while every bucket still reported success.
    """
    claimed = {e for b in BUCKETS for e in b["entries"]}
    present = set(manifest_slugs)
    return sorted(present - claimed), sorted(claimed - present)


def _label(b):
    """Job name: the bucket, plus the owner when that adds anything. `eltwise
    (eltwise)` reads as a mistake, so the parenthetical is dropped when the owner
    is just the bucket name again."""
    owner = b["owner"].replace("metalium-developers-", "")
    return b["id"] if owner == b["id"] else f"{b['id']} ({owner})"


def matrix():
    """The GitHub Actions job matrix: one entry per bucket.

    `only` is passed through as SWEEP_ONLY, so each job runs exactly its own
    entries rather than a round-robin slice — which is what lets the job name
    describe the contents.
    """
    return {
        "include": [
            {
                "id": b["id"],
                "label": _label(b),
                "only": ",".join(b["entries"]),
            }
            for b in BUCKETS
        ]
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "matrix":
        print(json.dumps(matrix()))
    else:
        for b in BUCKETS:
            print(f"{b['id']:<22} {b['owner']}")
            for e in b["entries"]:
                print(f"    {e}")
