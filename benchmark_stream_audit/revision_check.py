"""Capture immutable upstream sources and check the owned chat contract."""

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
from urllib.request import urlopen

from .vllm_regression import (
    ENDPOINT, HELPERS, LAYOUTS, PROBES, assess, contract, fake_session,
    source_record, wire_chunks,
)

FILES = (ENDPOINT, HELPERS, "LICENSE")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_directory(repo: Path, output: Path):
    output = output.resolve()
    for name in ("bundle", "source", "reports", "profiles", "policies", ".git",
                 "experiments"):
        protected = (repo / name).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("output would enter or contain preserved inputs")
    output.mkdir(parents=True, exist_ok=False)
    return output


def capture(repo: Path, revision: str, output: Path):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")
    output = new_directory(repo, output)
    manifest = {"repository": "vllm-project/vllm", "commit": revision,
                "captured_at": utc_now(), "files": []}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/vllm-project/vllm/{revision}/{name}"
        with urlopen(url, timeout=30) as response:
            data = response.read()
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"].append({"path": name, "url": url, "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_source(source: Path, revision: str):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")
    manifest = json.loads((source / "source_manifest.json").read_text())
    if manifest.get("commit") != revision:
        raise ValueError("source revision differs from requested revision")
    rows = manifest["files"]
    if len(rows) != len(FILES) or {r["path"] for r in rows} != set(FILES):
        raise ValueError("source inventory differs")
    for row in rows:
        path = (source / row["path"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("source path escapes snapshot")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"] or len(data) != row["bytes"]:
            raise ValueError(f"captured source changed: {row['path']}")
    return manifest


def check(repo: Path, source: Path, revision: str, output: Path):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output overlaps source snapshot")
    output = new_directory(repo, output)
    result = {"schema_version": 1, "revision": revision, "checked_at": utc_now(),
              "layer": "complete endpoint module with frozen upstream fake session",
              "python": sys.version, "platform": platform.platform(),
              "status": "execution_error", "rows": [], "errors": [],
              "limits": ["Chat request function only; no native CLI or model execution.",
                         "Fake session is frozen at the original baseline revision.",
                         "Error after finish follows the corpus policy.",
                         "Coalesced reads do not test arbitrary byte fragmentation."]}
    name = "stream_audit_candidate"
    try:
        result["source"] = verify_source(source, revision)
        result["helper_source"] = source_record(repo, "main")
        session_class = fake_session(repo)
        spec = importlib.util.spec_from_file_location(name, source / ENDPOINT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for case in PROBES:
            for layout in LAYOUTS:
                chunks = wire_chunks(case, layout)
                row = {"case": case, "layout": layout,
                       "wire_hex": [chunk.hex() for chunk in chunks],
                       "expected": contract(case)}
                try:
                    request = module.RequestFuncInput(
                        prompt="Hello world", api_url="http://fixture/v1/chat/completions",
                        prompt_len=7, output_len=32, model="fixture-model")
                    row["output"] = asdict(asyncio.run(
                        module.async_request_openai_chat_completions(
                            request, session_class([(0.0, chunk) for chunk in chunks]))))
                    if str(row["output"].get("error", "")).startswith("Traceback (most recent call last):"):
                        raise RuntimeError(row["output"]["error"])
                    row["failed_checks"] = assess(case, row["output"])
                    row["status"] = "nonconformant" if row["failed_checks"] else "conformant"
                except Exception as exc:
                    row.update(status="execution_error", error=f"{type(exc).__name__}: {exc}")
                result["rows"].append(row)
        if result["source"] != verify_source(source, revision):
            raise ValueError("source changed during execution")
        if result["helper_source"] != source_record(repo, "main"):
            raise ValueError("helper source changed during execution")
        statuses = {row["status"] for row in result["rows"]}
        result["status"] = ("execution_error" if "execution_error" in statuses else
                            "nonconformant" if "nonconformant" in statuses else "conformant")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        sys.modules.pop(name, None)
    rows = result["rows"]
    result["summary"] = {
        "planned": len(PROBES) * len(LAYOUTS), "attempted": len(rows),
        "conformant": sum(r["status"] == "conformant" for r in rows),
        "nonconformant": sum(r["status"] == "nonconformant" for r in rows),
        "execution_errors": sum(r["status"] == "execution_error" for r in rows) + len(result["errors"]),
        "false_successes": sum(r["expected"]["outcome"] == "failed" and
                               r.get("output", {}).get("success") is True for r in rows),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
