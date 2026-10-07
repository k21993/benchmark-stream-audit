#!/usr/bin/env python3
"""Verify saved evidence and run the regression tests."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
commands = [
    ["scripts/verify_frozen_corpus.py"],
    ["scripts/vllm_stream_regression.py", "verify"],
    ["scripts/vllm_fix_cli.py", "verify"],
    ["-m", "unittest", "discover", "-s", "tests", "-v"],
]
for args in commands:
    result = subprocess.run([sys.executable, "-B"] + args, cwd=ROOT, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
print("All saved evidence and regression checks passed.")
