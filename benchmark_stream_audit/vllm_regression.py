"""Owned conformance probes for captured vLLM request-function implementations."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

from .oracle import classify_events
from .vllm_cli import CASES, events


SOURCES = {
    "main": ("upstream-20261007", "08567505b09324f891797f4a13c1bd73fc75a827"),
    "pr57519": ("pr57519-20261007", "1e74a69c71604c3c3ec6cc3cce9dbe8b702bfcda"),
}
ENDPOINT = "vllm/benchmarks/lib/endpoint_request_func.py"
HELPERS = "tests/benchmarks/test_endpoint_request_func_timing.py"
PROBES = (*CASES, "finish_then_error")
LAYOUTS = ("per_event", "coalesced")
LAYER = "complete endpoint module with upstream fake session"


def source_record(repo: Path, variant: str) -> dict:
    directory, revision = SOURCES[variant]
    root = repo / "experiments/vllm-regression/evidence" / directory
    manifest = json.loads((root / "source_manifest.json").read_text())
    if manifest.get("commit", manifest.get("head_commit")) != revision:
        raise ValueError("source revision differs")
    hashes = {}
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source manifest path")
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"captured source changed: {relative}")
        hashes[row["path"]] = actual
    return {"revision": revision, "directory": str(root.relative_to(repo)),
            "source_sha256": hashes}


def probe_events(case: str) -> list:
    if case == "finish_then_error":
        return events("normal")[:-1] + events("error_only")
    return events(case)


def contract(case: str) -> dict:
    normalized = []
    text = ""
    error = None
    usage = None
    count = 0
    for event in probe_events(case):
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            error = event["error"]
            normalized.append({"type": "sse_error"})
        if event.get("choices"):
            count += 1
            choice = event["choices"][0]
            delta = choice["delta"]
            if delta.get("content"):
                text += delta["content"]
                normalized.append({"type": "sse_content", "text": delta["content"]})
            if choice.get("finish_reason"):
                normalized.append({"type": "sse_finish", "reason": choice["finish_reason"]})
        if event.get("usage"):
            usage = event["usage"]
    normalized.append({"type": "http_eof"})
    return {"outcome": classify_events(normalized), "text": text,
            "error": error, "usage": usage, "choices_events": count}


def wire_chunks(case: str, layout: str) -> list[bytes]:
    if layout not in LAYOUTS:
        raise ValueError("unknown layout")
    chunks = [("data: " + (event if isinstance(event, str) else
               json.dumps(event, separators=(",", ":"))) + "\n\n").encode()
              for event in probe_events(case)]
    return chunks if layout == "per_event" else [b"".join(chunks)]


def assess(case: str, output: dict) -> list[str]:
    expected = contract(case)
    failures = []
    if type(output.get("success")) is not bool or output["success"] != (expected["outcome"] == "completed"):
        failures.append("policy_outcome")
    if output.get("generated_text") != expected["text"]:
        failures.append("retained_text")
    if expected["error"]:
        try:
            reason = json.loads(output.get("error", ""))
        except (ValueError, TypeError):
            reason = None
        if reason != expected["error"]:
            failures.append("server_error_reason")
    elif output.get("error"):
        failures.append("unexpected_error")
    if expected["usage"] and expected["outcome"] == "completed":
        if (output.get("output_tokens") != expected["usage"]["completion_tokens"] or
                output.get("prompt_len") != expected["usage"]["prompt_tokens"]):
            failures.append("normal_usage")
    if len(output.get("itl", [])) != max(0, expected["choices_events"] - 1):
        failures.append("choices_timing_count")
    try:
        times = [output["latency"], output["ttft"], output["start_time"], *output["itl"]]
        if not all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times):
            raise ValueError
        if not math.isclose(output["latency"], output["ttft"] + sum(output["itl"]), abs_tol=1e-7):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        failures.append("timing_consistency")
    return failures


def fake_session(repo: Path):
    path = repo / "experiments/vllm-regression/evidence/upstream-20261007" / HELPERS
    names = {"_FakeContent", "_FakeResponse", "_FakeSession"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("upstream fake-session helper inventory differs")
    namespace = {"asyncio": asyncio, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_FakeSession"]


def run_probes(repo: Path) -> dict:
    records = {variant: source_record(repo, variant) for variant in SOURCES}
    session_class = fake_session(repo)
    rows = []
    for variant, source in records.items():
        path = repo / source["directory"] / ENDPOINT
        name = "stream_audit_captured_" + variant
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            for case in PROBES:
                for layout in LAYOUTS:
                    chunks = wire_chunks(case, layout)
                    request = module.RequestFuncInput(
                        prompt="Hello world", api_url="http://fixture/v1/chat/completions",
                        prompt_len=7, output_len=32, model="fixture-model")
                    output = asdict(asyncio.run(module.async_request_openai_chat_completions(
                        request, session_class([(0.0, chunk) for chunk in chunks]))))
                    rows.append({"variant": variant, "case": case, "layout": layout,
                                 "wire_hex": [chunk.hex() for chunk in chunks],
                                 "expected": contract(case), "output": output,
                                 "failed_checks": assess(case, output)})
        finally:
            sys.modules.pop(name, None)
    if records != {variant: source_record(repo, variant) for variant in SOURCES}:
        raise ValueError("source changed during execution")
    return {"schema_version": 1, "layer": LAYER,
            "python": sys.version, "platform": platform.platform(), "sources": records,
            "rows": rows, "summary": summarize(rows),
            "limits": ["No current CLI, real HTTP, aggregation, model or GPU execution.",
                       "PR head is captured without rebasing onto current main.",
                       "Only three upstream helper classes are AST-selected; full pytest suite not run.",
                       "Event-read coalescing is tested; arbitrary byte fragmentation is not.",
                       "Synthetic unit probes do not establish production prevalence or novelty."]}


def summarize(rows: list[dict]) -> dict:
    return {variant: {"attempted": sum(row["variant"] == variant for row in rows),
                      "conformant": sum(row["variant"] == variant and not row["failed_checks"] for row in rows),
                      "false_successes": sum(row["variant"] == variant and row["expected"]["outcome"] == "failed"
                                             and row["output"]["success"] for row in rows)}
            for variant in SOURCES}


def verify_result(repo: Path, result: dict) -> None:
    if result.get("schema_version") != 1 or result.get("layer") != LAYER:
        raise ValueError("unsupported or mislabeled evidence layer")
    if result["sources"] != {variant: source_record(repo, variant) for variant in SOURCES}:
        raise ValueError("saved source provenance differs")
    expected_keys = {(variant, case, layout) for variant in SOURCES
                     for case in PROBES for layout in LAYOUTS}
    observed = []
    for row in result["rows"]:
        key = (row["variant"], row["case"], row["layout"])
        if key not in expected_keys:
            raise ValueError("unexpected probe")
        observed.append(key)
        if (row["wire_hex"] != [c.hex() for c in wire_chunks(row["case"], row["layout"])] or
                row["expected"] != contract(row["case"]) or
                row["failed_checks"] != assess(row["case"], row["output"])):
            raise ValueError("probe bytes, contract or checks differ")
    if len(observed) != len(expected_keys) or set(observed) != expected_keys:
        raise ValueError("missing or duplicate probes")
    if result["summary"] != summarize(result["rows"]):
        raise ValueError("summary differs")


def verify_replay(repo: Path, result: dict, recorded: dict) -> None:
    """Require the pinned implementations to reproduce the recorded behavior."""
    verify_result(repo, result)
    verify_result(repo, recorded)
    expected_rows = {(row["variant"], row["case"], row["layout"]): row
                     for row in recorded["rows"]}
    for row in result["rows"]:
        key = row["variant"], row["case"], row["layout"]
        expected = expected_rows[key]
        if row["failed_checks"] != expected["failed_checks"]:
            raise ValueError(f"{key}: conformance checks changed")
        for field in ("success", "error", "generated_text", "output_tokens", "prompt_len"):
            if row["output"][field] != expected["output"][field]:
                raise ValueError(f"{key}: {field} changed")
