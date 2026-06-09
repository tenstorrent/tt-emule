#!/usr/bin/env python3
# Per-file collection pre-filter for the BH post-commit per-file sweep.
#
# For each unique file in expanded_manifest.yaml, run
#   pytest --collect-only --no-header --tb=line <file>
# under the emule env. Categorize:
#   RUN   — exit 0, parsed N tests collected
#   DROP  — exit non-zero with ImportError / ModuleNotFoundError
#   EMPTY — exit 5 (no tests collected) — treat as RUN with collected=0
#
# Output: snapshots/bh_sanity/file_classification.yaml
#   {run: [{file, collected}], drop: [{file, missing_module, reason}], empty: [...]}
#
# Designed to be safely re-runnable: skips files that already have a stable
# classification in the existing YAML.

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

EXPANDED = Path("/localdev/arminale/tt-emule/snapshots/bh_sanity/expanded_manifest.yaml")
OUT_PATH = Path("/localdev/arminale/tt-emule/snapshots/bh_sanity/file_classification.yaml")
TT_METAL = Path("/localdev/arminale/tt-metal")
PYTEST = "/opt/ttmlir-toolchain/venv/bin/pytest"

# Patterns
COLLECTED_RE = re.compile(r"(\d+)\s+tests? collected")
NO_TESTS_RE = re.compile(r"no tests ran", re.I)
MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
IMPORT_ERROR_RE = re.compile(r"ImportError: (.+)")


def collect_one(rel_path: str) -> dict:
    """Run pytest --collect-only on one file; return classification dict."""
    abs_path = TT_METAL / rel_path
    if not abs_path.is_file():
        return {"file": rel_path, "status": "MISSING", "reason": "file does not exist on disk"}
    t0 = time.monotonic()
    r = subprocess.run(
        [PYTEST, "--collect-only", "--no-header", "--tb=line", "-q", str(abs_path)],
        cwd=str(TT_METAL),
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - t0
    text = r.stdout + r.stderr

    m = MISSING_MODULE_RE.search(text)
    if m:
        return {
            "file": rel_path,
            "status": "DROP",
            "missing_module": m.group(1),
            "reason": "ModuleNotFoundError during collection",
            "elapsed": round(elapsed, 2),
        }
    if "ImportError" in text and r.returncode != 0:
        im = IMPORT_ERROR_RE.search(text)
        return {
            "file": rel_path,
            "status": "DROP",
            "missing_module": "(unknown — see reason)",
            "reason": (im.group(1) if im else "ImportError during collection")[:160],
            "elapsed": round(elapsed, 2),
        }
    if r.returncode == 5 or NO_TESTS_RE.search(text):
        return {
            "file": rel_path,
            "status": "EMPTY",
            "collected": 0,
            "elapsed": round(elapsed, 2),
        }
    if r.returncode != 0:
        return {
            "file": rel_path,
            "status": "DROP",
            "missing_module": "(non-import error)",
            "reason": f"pytest exit {r.returncode}: " + (text[-200:].replace("\n", " ")),
            "elapsed": round(elapsed, 2),
        }
    cm = COLLECTED_RE.search(text)
    n = int(cm.group(1)) if cm else 0
    return {
        "file": rel_path,
        "status": "RUN",
        "collected": n,
        "elapsed": round(elapsed, 2),
    }


def main():
    # Set up emule env in this process so subprocesses inherit it.
    cluster_examples = str(TT_METAL / "tt_metal/third_party/umd/tests/cluster_descriptor_examples")
    build_dir = str(TT_METAL / "build_emule")
    env = {
        "TT_METAL_DIR": str(TT_METAL),
        "BUILD_DIR": build_dir,
        "CLUSTER_EXAMPLES": cluster_examples,
        "PYTHONPATH": f"{TT_METAL}/ttnn:{TT_METAL}/tools:{build_dir}/lib:{TT_METAL}",
        "LD_LIBRARY_PATH": f"{build_dir}/lib",
        "TT_METAL_HOME": str(TT_METAL),
        "TT_METAL_RUNTIME_ROOT": str(TT_METAL),
        "TT_METAL_EMULE_MODE": "1",
        "TT_METAL_SLOW_DISPATCH_MODE": "1",
        "MESH_DEVICE": "P100",
        "TT_METAL_MOCK_CLUSTER_DESC_PATH": f"{cluster_examples}/blackhole_P100.yaml",
    }
    for k, v in env.items():
        os.environ[k] = v
    # Preserve existing PATH but prepend pytest's directory
    os.environ["PATH"] = f"{Path(PYTEST).parent}:{os.environ.get('PATH', '')}"

    with EXPANDED.open() as f:
        data = yaml.safe_load(f)
    all_files = data["all_files"]
    # entry membership for each file
    entry_for = {}
    for slug, entry in data["entries"].items():
        for item in entry["files"]:
            entry_for.setdefault(item["file"], []).append(slug)

    # Reuse previous classification if file unchanged
    prior = {}
    if OUT_PATH.is_file():
        with OUT_PATH.open() as f:
            prior_data = yaml.safe_load(f) or {}
        for bucket in ("run", "drop", "empty", "missing"):
            for r in (prior_data.get(bucket) or []):
                prior[r["file"]] = r

    results = []
    t_total = time.monotonic()
    for i, rel in enumerate(all_files, 1):
        if rel in prior and "status" in prior[rel]:
            # Cached
            results.append(prior[rel])
            print(f"[{i:3d}/{len(all_files)}] cached  {prior[rel].get('status'):6s}  {rel}")
            continue
        r = collect_one(rel)
        r["entries"] = entry_for.get(rel, [])
        results.append(r)
        print(f"[{i:3d}/{len(all_files)}] {r['status']:6s}  ({r.get('elapsed','?')}s)  {rel}"
              + (f"  → missing {r['missing_module']}" if r.get("missing_module") else "")
              + (f"  → collected {r['collected']}" if r.get("collected") is not None and r['status']=='RUN' else ""))

    elapsed_total = time.monotonic() - t_total

    # Bucket the results
    buckets = {"run": [], "drop": [], "empty": [], "missing": []}
    for r in results:
        s = r["status"]
        key = {"RUN": "run", "DROP": "drop", "EMPTY": "empty", "MISSING": "missing"}.get(s, "drop")
        buckets[key].append(r)

    summary = {
        "expanded_manifest": str(EXPANDED),
        "pytest_bin": PYTEST,
        "total_files": len(all_files),
        "elapsed_total_s": round(elapsed_total, 1),
        "run_count": len(buckets["run"]),
        "drop_count": len(buckets["drop"]),
        "empty_count": len(buckets["empty"]),
        "missing_count": len(buckets["missing"]),
        "total_collected_tests": sum(r.get("collected", 0) for r in buckets["run"]) + sum(r.get("collected", 0) for r in buckets["empty"]),
    }

    out = {**summary, **buckets}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, width=200)

    print()
    print("=" * 60)
    print(f"TOTAL FILES:           {summary['total_files']}")
    print(f"  RUN:                 {summary['run_count']}")
    print(f"  DROP:                {summary['drop_count']}")
    print(f"  EMPTY:               {summary['empty_count']}")
    print(f"  MISSING:             {summary['missing_count']}")
    print(f"TOTAL COLLECTED TESTS: {summary['total_collected_tests']}")
    print(f"Elapsed:               {summary['elapsed_total_s']:.0f}s")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
