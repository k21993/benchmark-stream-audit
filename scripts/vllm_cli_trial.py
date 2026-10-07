#!/usr/bin/env python3
"""Prepare or run a six-case trial using an existing pristine vLLM environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark_stream_audit.vllm_cli import (
    CASES, COMMIT, check_export, command, fixture, prepare, write_json,
)


def environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if (key.upper().endswith(("_PROXY", "_API_KEY", "_TOKEN"))
                or key in ("PYTHONPATH", "PYTHONHOME", "VLLM_RUST_FRONTEND_PATH")):
            env.pop(key, None)
    env.update(VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               VLLM_USE_RUST_BENCH="0", NO_PROXY="127.0.0.1,localhost,::1",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def process(argv: list[str], root: Path, label: str, timeout: float) -> dict:
    """Persist every bounded command; kill its process group if it times out."""
    started = time.monotonic()
    timed_out = False
    with (root / f"{label}.stdout").open("w") as out, (root / f"{label}.stderr").open("w") as err:
        with subprocess.Popen(argv, cwd=root, env=environment(), stdout=out, stderr=err,
                              start_new_session=True) as child:
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise
    result = {"argv": argv, "exit_code": code, "timed_out": timed_out,
              "elapsed_seconds": time.monotonic() - started}
    write_json(root / f"{label}.command.json", result)
    if code != 0 or timed_out:
        raise ValueError(f"{label} stopped (exit {code}); see {root / (label + '.stderr')}")
    return result


def provenance(python: str, source: Path, root: Path, label: str) -> dict:
    source = source.resolve()
    if not (source / ".git").exists():
        raise ValueError("source root must be a real Git checkout at the pinned commit")
    for name, args in (("head", ["rev-parse", "HEAD"]),
                       ("status", ["status", "--porcelain", "--untracked-files=normal"])):
        process(["git", "-C", str(source)] + args, root, f"{label}-{name}", 20)
    if (root / f"{label}-head.stdout").read_text().strip() != COMMIT:
        raise ValueError("checkout is not the pinned commit; current-version runs need a separate profile")
    if (root / f"{label}-status.stdout").read_text().strip():
        raise ValueError("checkout contains changes; pristine CLI gate cannot pass")
    files = {
        "vllm/benchmarks/lib/endpoint_request_func.py": "source/endpoint_request_func.py",
        "vllm/benchmarks/serve.py": "source/vllm__benchmarks__serve.py",
        "vllm/entrypoints/cli/main.py": "source/vllm__entrypoints__cli__main.py",
        "vllm/entrypoints/cli/benchmark/main.py": "source/vllm__entrypoints__cli__benchmark__main.py",
    }
    hashes = {}
    for path, bundled in files.items():
        data = (source / path).read_bytes()
        if data != (REPO / "bundle/vllm" / bundled).read_bytes():
            raise ValueError(f"upstream source differs: {path}")
        hashes[path] = hashlib.sha256(data).hexdigest()
    dataset = "vllm/benchmarks/datasets/datasets.py"
    hashes[dataset] = hashlib.sha256((source / dataset).read_bytes()).hexdigest()
    if hashes[dataset] != "6f2239d9d27fe439f255bef3bc8df1f484608fdd31b316fd21c32dd2d3f769f7":
        raise ValueError("custom dataset parser differs from the reviewed pinned source")
    probe = (
        "import importlib.util,importlib.metadata,json,platform,sys; "
        "s=importlib.util.find_spec('vllm'); "
        "print(json.dumps({'origin':s.origin if s else None,'python':sys.version,"
        "'platform':platform.platform(),'version':importlib.metadata.version('vllm')}))"
    )
    process([python, "-c", probe], root, f"{label}-package", 20)
    package = json.loads((root / f"{label}-package.stdout").read_text())
    if not package.get("origin") or Path(package["origin"]).resolve() != source / "vllm/__init__.py":
        raise ValueError("selected Python does not resolve vLLM to the pristine checkout")
    return {"commit": COMMIT, "source_root": str(source), "source_sha256": hashes,
            "package": package, "tracked_and_untracked_source_clean": True}


def execute(root: Path, python: str, source: Path, allow_macos: bool = False) -> dict:
    start = time.monotonic()
    deadline = start + 1800
    if platform.system() != "Linux" and not (allow_macos and platform.system() == "Darwin"):
        raise ValueError("no ready supported Linux environment: stop before dependency/native build setup")
    if platform.machine() not in ("x86_64", "aarch64", "arm64"):
        raise ValueError("architecture not covered by this trial's environment gate")
    before = provenance(python, source, root, "before")
    write_json(root / "provenance.json", {"before": before})
    process([python, "-c", "import importlib.metadata,json; "
             "print(json.dumps(sorted(({'name':d.metadata['Name'],'version':d.version} "
             "for d in importlib.metadata.distributions()),key=lambda x:x['name'])))"],
            root, "dependencies", 30)
    process([python, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--help"],
            root, "cli-help", 120)
    # Verify the tokenizer through the real installed library before running CLI.
    probe = (
        "import json,sys; from transformers import AutoTokenizer; "
        "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True); "
        "ids=t('Hello world',add_special_tokens=False).input_ids; "
        "print(json.dumps({'whole_text_ids':ids})); assert len(ids)==2"
    )
    process([python, "-c", probe, str(root / "tokenizer")], root, "tokenizer-check", 30)
    with fixture(root / "server_journal.jsonl") as base_url:
        for case in CASES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("30-minute execution cap reached")
            process(command(python, root, case, base_url), root / case, "cli", min(120, remaining))
    after = provenance(python, source, root, "after")
    write_json(root / "provenance.json", {"before": before, "after": after})
    if before != after:
        raise ValueError("source or package provenance changed during trial")
    journal = [json.loads(line) for line in (root / "server_journal.jsonl").read_text().splitlines()]
    rows = []
    for case in CASES:
        native = json.loads((root / case / "native.json").read_text())
        rows.append(check_export(native, journal, case))
    false_successes = sum(row["explicit_error_false_successes"] for row in rows)
    result = {"schema_version": 1, "status": "verified", "cli_gate_passed": True,
              "scope": "actual pinned vLLM Python CLI against synthetic loopback fixture",
              "commit": COMMIT, "platform": platform.platform(), "attempted": 30, "cases": rows,
              "explicit_error_false_successes": false_successes,
              "classification": ("reproduced" if false_successes == 10 else
                                 "not_reproduced" if false_successes == 0 else "mixed"),
              "elapsed_seconds": time.monotonic() - start,
              "limits": ["synthetic corpus; no production prevalence or model performance",
                         "native arrays associate by input ordinal, not an exported response ID",
                         "server clocks describe emission, not client receipt",
                         "peak choices-event counter is not independently measured model token throughput",
                         "no comparison of practitioner diagnosis effort performed"]}
    write_json(root / "verification.json", result)
    (root / "verification.md").write_text(
        "# vLLM native CLI trial\n\n"
        f"Verified 30 requests at `{COMMIT}`. Explicit-error false successes: **{false_successes}**.\n\n"
        "| Case | Completed | Failed | Oracle | Peak choices events/s |\n"
        "|---|---:|---:|---|---:|\n" + "".join(
            f"| {r['case']} | {r['native_completed']} | {r['native_failed']} | {r['oracle_outcome']} | {r['peak_choices_events_per_second']} |\n"
            for r in rows) + "\n" + "\n".join(f"- {x}" for x in result["limits"]) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", required=True, type=Path, help="new directory; never overwritten")
    parser.add_argument("--python", default=sys.executable, help="existing environment's Python executable")
    parser.add_argument("--source-root", type=Path, help="pristine pinned checkout used by editable installation")
    parser.add_argument("--allow-macos", action="store_true",
                        help="opt in to an explicitly authorized, already built native Mac environment")
    args = parser.parse_args()
    if args.action == "run" and args.source_root is None:
        parser.error("run requires --source-root")
    root = args.output.resolve()
    # Resolving a venv's Python symlink would bypass that environment.
    python = os.path.abspath(args.python) if "/" in args.python else args.python
    try:
        prepare(root, REPO, python)
    except (ValueError, OSError) as exc:
        print(f"Cannot prepare: {exc}", file=sys.stderr)
        return 2
    if args.action == "prepare":
        print(f"Prepared six cases / 30 requests in {root}; no vLLM CLI executed.")
        return 0
    write_json(root / "status.json", {"status": "running", "cli_gate_passed": False})
    try:
        result = execute(root, python, args.source_root, args.allow_macos)
    except KeyboardInterrupt:
        write_json(root / "status.json", {"status": "interrupted", "cli_gate_passed": False})
        return 130
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        write_json(root / "status.json", {"status": "blocked", "cli_gate_passed": False,
                                         "reason": str(exc)})
        print(f"CLI gate remains blocked: {exc}", file=sys.stderr)
        return 1
    write_json(root / "status.json", {"status": "verified", "cli_gate_passed": True})
    print(f"Verified native CLI evidence: {result['classification']}; "
          f"explicit-error false successes={result['explicit_error_false_successes']}. See {root / 'verification.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
