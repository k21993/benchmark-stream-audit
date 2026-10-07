#!/usr/bin/env python3
"""Run or verify native CLI evidence for the captured upstream stream-error fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark_stream_audit.vllm_cli import (
    CASES, COUNT, check_export, command, events, fixture, prepare, write_json,
)
from scripts.vllm_cli_trial import process

COMMIT = "1e74a69c71604c3c3ec6cc3cce9dbe8b702bfcda"
SOURCE = REPO / "experiments/vllm-regression/evidence/pr57519-cli-source-20261007"
DEFAULT_RESULT = REPO / "experiments/vllm-regression/evidence/cli-fix-20261007"


def source_hashes() -> dict:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if manifest["commit"] != COMMIT:
        raise ValueError("candidate source revision changed")
    hashes = {}
    for row in manifest["files"]:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe source path")
        digest = hashlib.sha256((SOURCE / path).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"candidate source corrupted: {path}")
        hashes[str(path)] = digest
    endpoint = "vllm/benchmarks/lib/endpoint_request_func.py"
    expected_paths = {endpoint, "vllm/benchmarks/serve.py",
                      "vllm/entrypoints/cli/main.py",
                      "vllm/entrypoints/cli/benchmark/main.py",
                      "vllm/benchmarks/datasets/datasets.py"}
    if set(hashes) != expected_paths:
        raise ValueError("candidate source population changed")
    if hashes.get(endpoint) != "4dc3f56168973789110090a9136d88c94a242d365d8b1c0287b7c56667c67364":
        raise ValueError("endpoint differs from previously reviewed PR head")
    return hashes


def provenance(python: str, source: Path, root: Path, label: str) -> dict:
    source = source.resolve()
    for name, args in (("head", ["rev-parse", "HEAD"]),
                       ("status", ["status", "--porcelain", "--untracked-files=normal"])):
        process(["git", "-C", str(source)] + args, root, f"{label}-{name}", 20)
    if (root / f"{label}-head.stdout").read_text().strip() != COMMIT:
        raise ValueError("checkout is not the captured PR head")
    if (root / f"{label}-status.stdout").read_text().strip():
        raise ValueError("candidate checkout is not pristine")
    hashes = source_hashes()
    for path, digest in hashes.items():
        if hashlib.sha256((source / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"candidate checkout source differs: {path}")
    probe = (
        "import importlib.util,importlib.metadata,json,platform,sys; "
        "s=importlib.util.find_spec('vllm'); "
        "print(json.dumps({'origin':s.origin if s else None,'python':sys.version,"
        "'platform':platform.platform(),'version':importlib.metadata.version('vllm')}))"
    )
    process([python, "-c", probe], root, f"{label}-package", 20)
    package = json.loads((root / f"{label}-package.stdout").read_text())
    if Path(package["origin"]).resolve() != source / "vllm/__init__.py":
        raise ValueError("Python imports vLLM from a different checkout")
    return {"commit": COMMIT, "source_root": str(source), "source_sha256": hashes,
            "package": package, "tracked_and_untracked_source_clean": True}


def check_candidate(native: dict, journal: list[dict], case: str) -> dict:
    """Require failure reasons and preserved partial text, beyond baseline checks."""
    row = check_export(native, journal, case)
    if "error" in case:
        if row["native_failed"] != COUNT:
            raise ValueError(f"{case}: explicit error counted as success")
        error = next(event["error"] for event in events(case)
                     if isinstance(event, dict) and "error" in event)
        for observed in native["errors"]:
            if json.loads(observed) != error:
                raise ValueError(f"{case}: server error reason lost")
        text = "Hello world" if case == "text_then_error" else ""
        if native["generated_texts"] != [text] * COUNT:
            raise ValueError(f"{case}: failed partial text lost or changed")
        row["partial_output_tokens_local"] = 10 if text else 0
        row["partial_count_source"] = "verified local whole-text tokenizer; excluded from completed totals"
    return row


def verified_rows(root: Path) -> list[dict]:
    journal = [json.loads(line) for line in (root / "server_journal.jsonl").read_text().splitlines()]
    for row in journal:
        auxiliary = (row.get("kind") == "auxiliary" and row.get("method") == "GET"
                     and row.get("path") == "/metrics" and "case" not in row)
        measured = (row.get("case") in CASES and row.get("kind") in
                    {"request", "write_before", "write_after", "http_eof", "disconnect"})
        if not auxiliary and not measured:
            raise ValueError("unexpected fixture request")
    return [check_candidate(json.loads((root / case / "native.json").read_text()), journal, case)
            for case in CASES]


def verify(root: Path, require_manifest: bool = True) -> dict:
    if require_manifest:
        listed = set()
        for line in (root / "SHA256SUMS").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or relative in listed:
                raise ValueError("unsafe or duplicate evidence checksum path")
            listed.add(relative)
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"candidate evidence corrupted: {relative}")
        actual = {str(path.relative_to(root)) for path in root.rglob("*")
                  if path.is_file() and path.name != "SHA256SUMS"}
        if actual != listed:
            raise ValueError("candidate evidence checksum population changed")
    result = json.loads((root / "verification.json").read_text())
    rows = verified_rows(root)
    provenance_record = json.loads((root / "provenance.json").read_text())
    before, after = provenance_record["before"], provenance_record["after"]
    if before != after or before["commit"] != COMMIT or before["source_sha256"] != source_hashes():
        raise ValueError("source provenance does not reconcile")
    if before.get("tracked_and_untracked_source_clean") is not True:
        raise ValueError("source was not recorded pristine")
    if (result.get("commit") != COMMIT or result.get("execution_layer") != "full_cli"
            or result.get("cases") != rows or result.get("attempted") != 30
            or result.get("completed") != 15 or result.get("failed") != 15
            or result.get("explicit_error_false_successes") != 0
            or result.get("completed_output_tokens") != 45
            or result.get("partial_output_tokens_local") != 10):
        raise ValueError("candidate result does not reconcile with native exports")
    for case in CASES:
        record = json.loads((root / case / "cli.command.json").read_text())
        if record["exit_code"] != 0 or record["timed_out"]:
            raise ValueError("CLI did not finish successfully")
        argv = record["argv"]
        # Paths and port vary; require the same command contract in their recorded positions.
        expected = command(argv[0], Path(argv[argv.index("--result-dir") + 1]).parent,
                           case, argv[argv.index("--base-url") + 1])
        if argv != expected:
            raise ValueError("recorded CLI flags changed")
    tokenizer = json.loads((root / "tokenizer-check.stdout").read_text())
    if len(tokenizer["whole_text_ids"]) != 2:
        raise ValueError("partial text tokenizer count changed")
    return result


def summary(rows: list[dict]) -> dict:
    return {"schema_version": 1, "commit": COMMIT, "execution_layer": "full_cli",
            "attempted": 30, "completed": sum(r["native_completed"] for r in rows),
            "failed": sum(r["native_failed"] for r in rows),
            "explicit_error_false_successes": sum(r["explicit_error_false_successes"] for r in rows),
            "completed_output_tokens": sum(r["total_output_tokens"] for r in rows),
            "partial_output_tokens_local": sum(r.get("partial_output_tokens_local", 0) for r in rows),
            "cases": rows,
            "limits": ["synthetic chat corpus; no production prevalence or model performance",
                       "captured PR head, not rebased onto main or a controlled one-patch baseline",
                       "input ordinal plus fixture ID; native export has no request IDs",
                       "server clocks measure emission, not client receipt",
                       "peak choices-event counters are not model-token throughput",
                       "no practitioner effort or demand evaluation"]}


def seal(root: Path) -> None:
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"))


def run(root: Path, python: str, source: Path) -> dict:
    prepare(root, REPO, python)
    plan = json.loads((root / "plan.json").read_text())
    plan.update(commit=COMMIT, setup_cap_seconds=1800,
                source_contract={"source_sha256": source_hashes(),
                                 "association": "input ordinal plus fixture ID; native exports have no IDs"})
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "running", "cli_gate_passed": False})
    started = time.monotonic()
    before = provenance(python, source, root, "before")
    write_json(root / "provenance.json", {"before": before})
    process([python, "-c", "import importlib.metadata,json; print(json.dumps(sorted("
             "({'name':d.metadata['Name'],'version':d.version} for d in "
             "importlib.metadata.distributions()),key=lambda x:x['name'])))"],
            root, "dependencies", 30)
    process([python, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--help"],
            root, "cli-help", 120)
    process([python, "-c", "import json,sys; from transformers import AutoTokenizer; "
             "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True); "
             "ids=t('Hello world',add_special_tokens=False).input_ids; "
             "print(json.dumps({'whole_text_ids':ids})); assert len(ids)==2",
             str(root / "tokenizer")], root, "tokenizer-check", 30)
    with fixture(root / "server_journal.jsonl") as base_url:
        for case in CASES:
            process(command(python, root, case, base_url), root / case, "cli", 120)
    after = provenance(python, source, root, "after")
    write_json(root / "provenance.json", {"before": before, "after": after})
    rows = verified_rows(root)
    result = summary(rows)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(root / "verification.json", result)
    verify(root, require_manifest=False)
    write_json(root / "status.json", {"status": "verified", "cli_gate_passed": True})
    seal(root)
    verify(root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--python")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.action == "run" and not all((args.output, args.python, args.source_root)):
        parser.error("run requires --output, --python and --source-root")
    started_run = False
    try:
        if args.action == "verify":
            result = verify(args.result)
        else:
            root = args.output.resolve()
            for name in ("bundle", "source", "reports", "profiles", "policies", ".git", "evaluation"):
                protected = REPO / name
                if root == protected or protected in root.parents or root in protected.parents:
                    raise ValueError("output overlaps protected evidence")
            if root == SOURCE or SOURCE in root.parents or root in SOURCE.parents:
                raise ValueError("output overlaps captured source")
            if root.exists():
                raise ValueError("output already exists; choose a new directory")
            started_run = True
            result = run(root, os.path.abspath(args.python), args.source_root)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        if started_run and root.exists():
            write_json(root / "status.json", {"status": "blocked", "cli_gate_passed": False,
                                             "reason": str(exc)})
        print(f"Candidate CLI verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified captured fix CLI: {result['completed']} completed / {result['failed']} failed; "
          f"false successes={result['explicit_error_false_successes']}; "
          f"completed tokens={result['completed_output_tokens']}; "
          f"retained partial tokens (local)={result['partial_output_tokens_local']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
