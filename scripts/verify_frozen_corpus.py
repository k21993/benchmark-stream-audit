#!/usr/bin/env python3
"""Check the saved evidence hashes without running a benchmark client."""

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify() -> None:
    names = set()
    for line in (ROOT / "EVIDENCE_SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in names:
            raise ValueError(f"unsafe or duplicate evidence path: {name}")
        names.add(name)
        resolved = (ROOT / path).resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError(f"evidence path escapes repository: {name}")
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
            raise ValueError(f"evidence changed: {name}")
    if not names:
        raise ValueError("empty evidence manifest")


def main() -> int:
    try:
        verify()
    except (OSError, ValueError) as exc:
        print(f"Evidence verification failed: {exc}", file=sys.stderr)
        return 1
    print("Saved evidence hashes verified; no benchmark clients executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
