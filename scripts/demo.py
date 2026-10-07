#!/usr/bin/env python3
"""Show one failure and its correction from verified saved native CLI evidence."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_stream_audit.vllm_cli import check_export
from scripts.verify_frozen_corpus import verify as verify_integrity
from scripts.vllm_fix_cli import DEFAULT_RESULT, verify as verify_candidate


def main() -> int:
    try:
        verify_integrity()
        verify_candidate(DEFAULT_RESULT)
        baseline = ROOT / "experiments/vllm-cli/evidence/mac-20261006"
        journal = [json.loads(line) for line in
                   (baseline / "server_journal.jsonl").read_text().splitlines()]
        before = json.loads((baseline / "text_then_error/native.json").read_text())
        after = json.loads((DEFAULT_RESULT / "text_then_error/native.json").read_text())
        row = check_export(before, journal, "text_then_error")
        if row["explicit_error_false_successes"] != 5:
            raise ValueError("saved baseline no longer reproduces the expected failure")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Demo verification failed: {exc}", file=sys.stderr)
        return 1
    print("Saved native CLI evidence; no vLLM client executed.\n")
    print("Case: text arrives, then an explicit server error (5 requests).")
    print("Policy: an explicit error means failure, even after partial text.\n")
    print("                         Baseline b6d8e8a   PR #57519 1e74a69")
    print(f"Completed / failed       {before['completed']} / {before['failed']}               "
          f"{after['completed']} / {after['failed']}")
    print(f"Completed output tokens  {before['total_output_tokens']}                  "
          f"{after['total_output_tokens']}")
    print("Retained text            Hello world         Hello world\n")
    print("The baseline counted five explicit failures as successes.")
    print("The captured fix preserves partial text and the server error while")
    print("excluding the partial output from completed-request statistics.\n")
    for path in (baseline, DEFAULT_RESULT):
        print(f"Evidence: {path.relative_to(ROOT)}/text_then_error/native.json")
    print("These are synthetic cases at different pinned revisions.")
    print("They do not measure model speed or production failure rates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
