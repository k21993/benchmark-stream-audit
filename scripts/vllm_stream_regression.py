#!/usr/bin/env python3
"""Run captured implementations or verify saved conformance results offline."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_stream_audit.vllm_regression import (
    SOURCES, run_probes, verify_replay, verify_result,
)

RECORDED = ROOT / "experiments/vllm-regression/evidence/differential-20261007/result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "replay"):
        run = sub.add_parser(action)
        run.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--result", type=Path, default=RECORDED)
    args = parser.parse_args()
    try:
        if args.action in ("run", "replay"):
            output = args.output.resolve()
            if output.exists():
                raise ValueError("output exists; use a new directory")
            protected_paths = [ROOT / name for name in (
                "bundle", "source", "reports", "profiles", "policies", ".git")]
            protected_paths.extend(ROOT / "experiments/vllm-regression/evidence" / directory
                                   for directory, _ in SOURCES.values())
            for protected in protected_paths:
                if output == protected or protected in output.parents or output in protected.parents:
                    raise ValueError("output would enter or contain preserved inputs")
            result = run_probes(ROOT)
            verify_result(ROOT, result)
            output.mkdir(parents=True, exist_ok=False)
            (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result["summary"], indent=2))
            if args.action == "replay":
                verify_replay(ROOT, result, json.loads(RECORDED.read_text()))
                print("Executed 28 pinned-source probes; all recorded outcomes and errors matched.")
                return 0
            return 1 if any(row["failed_checks"] for row in result["rows"]) else 0
        result = json.loads(args.result.read_text())
        verify_result(ROOT, result)
        print("Verified 28 saved probes, source hashes, input bytes and conformance checks; no clients executed.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"stream regression failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
