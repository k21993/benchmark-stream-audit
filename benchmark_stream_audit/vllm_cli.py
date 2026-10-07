"""Bounded loopback fixtures and checks for native, pinned vLLM CLI exports.

Fixture shapes follow the preserved study; this module never imports vLLM.
Server emission clocks are not client latency or model-token observations.
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time


COMMIT = "b6d8e8afd985f5711eee68e343d2ce908d166488"
COUNT = 5
CASES = ("normal", "error_only", "role_then_error", "text_then_error",
         "one_content_chunk", "three_content_chunks")


def choice(delta: dict, finish: str | None = None) -> dict:
    return {"id": "fixture-chat", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def events(case: str) -> list:
    role = choice({"role": "assistant"})
    text = choice({"content": "Hello world"})
    error = {"error": {"message": "injected fixture failure",
                       "type": "server_error", "code": "fixture_error"}}
    terminal = [choice({}, "stop"),
                {"id": "fixture-chat", "object": "chat.completion.chunk",
                 "created": 0, "model": "fixture-model", "choices": [],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                           "total_tokens": 10}}, "[DONE]"]
    return {
        "normal": [role, text] + terminal,
        "error_only": [error, "[DONE]"],
        "role_then_error": [role, error, "[DONE]"],
        "text_then_error": [text, error, "[DONE]"],
        "one_content_chunk": [text] + terminal,
        "three_content_chunks": [choice({"content": part})
                                 for part in ("Hel", "lo", " world")] + terminal,
    }[case]


def prompt(case: str, index: int) -> str:
    return f"Say Hello world. Audit case {case} request {index}."


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(python: str, root: Path, case: str, base_url: str) -> list[str]:
    return [python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai-chat", "--base-url", base_url,
            "--endpoint", f"/{case}/v1/chat/completions", "--model", "fixture-model",
            "--tokenizer", str(root / "tokenizer"), "--tokenizer-mode", "hf",
            "--dataset-name", "custom", "--dataset-path", str(root / case / "input.jsonl"),
            "--custom-output-len", "32", "--skip-chat-template", "--disable-shuffle",
            "--no-oversample", "--num-prompts", str(COUNT), "--num-warmups", "0",
            "--ready-check-timeout-sec", "0", "--request-rate", "inf",
            "--request-id-prefix", "",
            "--max-concurrency", "1", "--seed", "0", "--disable-tqdm",
            "--save-result", "--save-detailed", "--result-dir", str(root / case),
            "--result-filename", "native.json"]


def prepare(root: Path, repo: Path, python: str) -> None:
    """Create a new trial directory, refusing existing paths and evidence paths."""
    root = root.resolve()
    for protected in (repo / "bundle", repo / "source"):
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("trial output must be outside preserved evidence")
    root.mkdir(parents=True, exist_ok=False)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(
        (repo / "bundle/vllm/artifacts/fixture_tokenizer.json").read_bytes())
    write_json(tokenizer / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
        "model_max_length": 4096, "clean_up_tokenization_spaces": False})
    write_json(tokenizer / "special_tokens_map.json", {"unk_token": "[UNK]"})
    for case in CASES:
        directory = root / case
        directory.mkdir()
        (directory / "input.jsonl").write_text("".join(
            json.dumps({"prompt": prompt(case, i), "output_tokens": 32}) + "\n"
            for i in range(COUNT)))
    write_json(root / "plan.json", {
        "schema_version": 1, "scope": "prepared commands; no CLI run",
        "commit": COMMIT, "requests_per_case": COUNT,
        "setup_cap_seconds": 3600, "execution_cap_seconds": 1800,
        "process_timeout_seconds": 120,
        "commands": {case: command(python, root, case, "{LOOPBACK_URL}")
                     for case in CASES},
        "oracle": {case: "failed" if "error" in case else "completed" for case in CASES},
        "source_contract": {
            "datasets_path": "vllm/benchmarks/datasets/datasets.py",
            "datasets_sha256": "6f2239d9d27fe439f255bef3bc8df1f484608fdd31b316fd21c32dd2d3f769f7",
            "association": "input ordinal and fixture request ID; native export has no IDs",
            "peak_counter": "choices-event timestamp buckets, checked from native arrays"}})
    write_json(root / "status.json", {"status": "prepared", "cli_gate_passed": False})


@contextmanager
def fixture(journal_path: Path):
    lock = threading.Lock()

    def record(kind: str, **values):
        with lock, journal_path.open("a") as stream:
            stream.write(json.dumps({"kind": kind,
                                     "server_monotonic_ns": time.monotonic_ns(),
                                     "server_wall_time_ns": time.time_ns(), **values}) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            record("auxiliary", method="GET", path=self.path)
            self.send_error(404)

        def do_POST(self):
            case = self.path.split("/")[1]
            if case not in CASES or self.path != f"/{case}/v1/chat/completions":
                record("auxiliary", method="POST", path=self.path)
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_id = self.headers.get("x-request-id")
            record("request", case=case, request_id=request_id, payload=payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for index, event in enumerate(events(case)):
                    value = event if isinstance(event, str) else json.dumps(event, separators=(",", ":"))
                    wire = ("data: " + value + "\n\n").encode()
                    record("write_before", case=case, request_id=request_id,
                           event_index=index, event=event, wire_hex=wire.hex())
                    self.wfile.write(wire)
                    self.wfile.flush()
                    record("write_after", case=case, request_id=request_id, event_index=index)
                    time.sleep(.002)
                record("http_eof", case=case, request_id=request_id)
            except (BrokenPipeError, ConnectionResetError):
                record("disconnect", case=case, request_id=request_id)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: expected finite nonnegative number")
    return value


def check_export(native: dict, journal: list[dict], case: str) -> dict:
    """Reject incomplete/corrupt exports before interpreting client behavior."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"{case}: {message}")

    require(native.get("num_prompts") == COUNT, "wrong exported population")
    require(native.get("backend") == "openai-chat", "wrong backend")
    arrays = ("errors", "generated_texts", "output_lens", "input_lens", "start_times",
              "ttfts", "itls", "latencies")
    for field in arrays:
        require(isinstance(native.get(field), list) and len(native[field]) == COUNT,
                f"missing or truncated native {field}")
    require(all(isinstance(x, str) for x in native["errors"] + native["generated_texts"]),
            "invalid text/error fields")
    successes = [not error for error in native["errors"]]
    completed = sum(successes)
    require(native.get("completed") == completed and native.get("failed") == COUNT - completed,
            "error population does not reconcile with aggregate")
    duration = number(native.get("duration"), "duration")
    require(duration > 0, "missing measurement window")
    for field in ("output_lens", "input_lens", "start_times", "ttfts", "latencies"):
        for value in native[field]:
            number(value, field)
    for intervals in native["itls"]:
        require(isinstance(intervals, list), "invalid ITL list")
        for value in intervals:
            number(value, "ITL")
    span = max(start + latency for start, latency in zip(native["start_times"], native["latencies"])) - min(native["start_times"])
    require(span <= duration + 1e-7, "client event span exceeds benchmark window")
    require(all(float(x).is_integer() for x in native["output_lens"] + native["input_lens"]),
            "nonintegral token lengths")
    require(native.get("total_output_tokens") == sum(native["output_lens"]), "token total mismatch")
    require(native.get("total_input_tokens") == sum(x for x, ok in zip(native["input_lens"], successes) if ok),
            "input token total mismatch")
    for field, numerator in (("request_throughput", completed),
                             ("output_throughput", sum(native["output_lens"]))):
        require(math.isclose(number(native.get(field), field), numerator / duration, rel_tol=1e-7),
                f"{field} denominator mismatch")

    rows = [row for row in journal if row.get("case") == case]
    requests = [row for row in rows if row["kind"] == "request"]
    require(len(requests) == COUNT, "unexpected requests (readiness/warmup/retry or missing request)")
    require([row.get("request_id") for row in requests] == [str(i) for i in range(COUNT)],
            "input ordinal/request ID association differs")
    for index, request in enumerate(requests):
        body = request["payload"]
        require(body.get("model") == "fixture-model" and body.get("stream") is True
                and body.get("stream_options", {}).get("include_usage") is True
                and body.get("max_completion_tokens") == 32,
                "request flags differ from declared profile")
        require(body.get("messages") == [{"role": "user", "content": [
                    {"type": "text", "text": prompt(case, index)}]}],
                "input prompt/order changed")
        writes = [row for row in rows if row["kind"] == "write_before" and row.get("request_id") == str(index)]
        require([row["event"] for row in writes] == events(case), "emission sequence differs")
        for i, row in enumerate(writes):
            event = row["event"]
            encoded = event if isinstance(event, str) else json.dumps(event, separators=(",", ":"))
            require(row.get("event_index") == i and row.get("wire_hex") == ("data: " + encoded + "\n\n").encode().hex(),
                    "emission bytes/order differ")
        after = [row["event_index"] for row in rows if row["kind"] == "write_after" and row.get("request_id") == str(index)]
        require(after == list(range(len(events(case)))), "fixture write did not complete")
        require(sum(row["kind"] == "http_eof" and row.get("request_id") == str(index) for row in rows) == 1,
                "fixture did not terminate cleanly")
        require(not any(row["kind"] == "disconnect" and row.get("request_id") == str(index) for row in rows),
                "transport disconnected; explicit error case is confounded")

    expected_failure = "error" in case
    if not expected_failure:
        require(completed == COUNT and native["generated_texts"] == ["Hello world"] * COUNT,
                "normal control failed")
        require(native["output_lens"] == [3] * COUNT and native["input_lens"] == [7] * COUNT,
                "usage control changed")
    if case == "error_only":
        require(completed == 0, "error-only negative control changed")
    for index, ok in enumerate(successes):
        if not ok:
            require(native["output_lens"][index] == 0, "failed output entered successful token population")
        elif case in ("role_then_error", "text_then_error"):
            require(native["generated_texts"][index] == ("" if case == "role_then_error" else "Hello world"),
                    "successful export differs from emitted content")
            require(native["output_lens"][index] == (0 if case == "role_then_error" else 2),
                    "partial-output tokenizer fallback changed")
        require(math.isclose(native["ttfts"][index] + sum(native["itls"][index]),
                             native["latencies"][index], abs_tol=1e-7), "client event window does not reconcile")

    # Same bucket definition as the pinned code, calculated from native arrays.
    buckets = {}
    starts = [value for value, ok in zip(native["start_times"], successes) if ok]
    for index, ok in enumerate(successes):
        if ok:
            timestamp = native["start_times"][index] + native["ttfts"][index]
            for interval in [0] + native["itls"][index]:
                timestamp += interval
                bucket = int(timestamp - min(starts))
                buckets[bucket] = buckets.get(bucket, 0) + 1
    peak = max(buckets.values(), default=0)
    require(number(native.get("max_output_tokens_per_s"), "peak counter") == peak,
            "peak choices-event counter differs from native arrays")
    return {"case": case, "attempted": COUNT, "native_completed": completed,
            "native_failed": COUNT - completed,
            "oracle_outcome": "failed" if expected_failure else "completed",
            "explicit_error_false_successes": completed if expected_failure else 0,
            "total_output_tokens": native["total_output_tokens"],
            "peak_choices_events_per_second": peak, "duration_seconds": duration,
            "partial_text_exports": native["generated_texts"] if expected_failure else [],
            "association": "input ordinal plus fixture ID; native export has no request IDs",
            "evidence": {"native": f"{case}/native.json", "journal": "server_journal.jsonl"}}
