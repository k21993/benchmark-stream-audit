#!/usr/bin/env python3
"""Capture a pinned vLLM revision or check its chat streaming behavior."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_stream_audit.revision_check import capture, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("capture", "check"):
        command = sub.add_parser(action)
        command.add_argument("--revision", required=True)
        command.add_argument("--output", required=True, type=Path)
        if action == "check":
            command.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "capture":
            result = capture(ROOT, args.revision, args.output)
            print(f"Captured {result['commit']}; {len(result['files'])} files.")
            return 0
        result = check(ROOT, args.source, args.revision, args.output)
        print(json.dumps({"status": result["status"], **result["summary"]}, indent=2))
        return {"conformant": 0, "nonconformant": 1, "execution_error": 2}[result["status"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"revision check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
