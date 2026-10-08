"""Check trial evidence rejection and actual HTTP fixture bytes without vLLM.

Native-shaped values below are test doubles, never full CLI experiment evidence.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_stream_audit.vllm_cli import (
    CASES, check_export, events, fixture, prepare, prompt,
)


def evidence(case, failed=False):
    error = case == "error_only" or failed
    text = "" if error or case == "role_then_error" else "Hello world"
    length = 0 if error or case == "role_then_error" else 2 if case == "text_then_error" else 3
    intervals = len([event for event in events(case) if isinstance(event, dict) and event.get("choices")]) - 1
    ttft = 0 if error else .001
    itls = [] if error else [.002] * intervals
    native = {"backend": "openai-chat", "num_prompts": 5,
              "completed": 0 if error else 5, "failed": 5 if error else 0,
              "errors": ["injected error" if error else ""] * 5,
              "generated_texts": [text] * 5, "output_lens": [length] * 5,
              "input_lens": [7] * 5, "total_input_tokens": 0 if error else 35,
              "total_output_tokens": length * 5, "duration": .1,
              "start_times": [10 + i * .015 for i in range(5)], "ttfts": [ttft] * 5,
              "itls": [itls] * 5, "latencies": [ttft + sum(itls)] * 5,
              "max_output_tokens_per_s": 0 if error else 5 * (1 + len(itls)),
              "request_throughput": 0 if error else 50,
              "output_throughput": length * 50}
    journal = []
    for index in range(5):
        base = {"case": case, "request_id": str(index)}
        journal.append({**base, "kind": "request", "payload": {
            "model": "fixture-model", "stream": True, "stream_options": {"include_usage": True},
            "max_completion_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt(case, index)}]}]}})
        for i, event in enumerate(events(case)):
            text_event = event if isinstance(event, str) else json.dumps(event, separators=(",", ":"))
            journal.append({**base, "kind": "write_before", "event_index": i, "event": event,
                            "wire_hex": ("data: " + text_event + "\n\n").encode().hex()})
            journal.append({**base, "kind": "write_after", "event_index": i})
        journal.append({**base, "kind": "http_eof"})
    return native, journal


class NativeExportTests(unittest.TestCase):
    def test_reproduction_and_correct_failure_are_distinguished(self):
        """The checker must distinguish a reproduced bug from a correct failure."""
        for case in CASES:
            native, journal = evidence(case)
            result = check_export(native, journal, case)
            self.assertEqual(result["explicit_error_false_successes"],
                             5 if case in ("role_then_error", "text_then_error") else 0)
        native, journal = evidence("text_then_error", failed=True)
        self.assertEqual(check_export(native, journal, "text_then_error")["native_failed"], 5)

    def test_corrupt_export_cannot_close_cli_gate(self):
        """Corrupt exports must not satisfy the CLI evidence gate."""
        native, journal = evidence("one_content_chunk")
        mutations = {"total_output_tokens": 999, "failed": 1, "output_throughput": 999,
                     "errors": [""], "duration": float("nan"),
                     "max_output_tokens_per_s": 15, "latencies": [.01] * 5}
        for field, value in mutations.items():
            changed = copy.deepcopy(native)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_export(changed, journal, "one_content_chunk")

    def test_extra_requests_missing_terminal_and_wrong_identity_are_rejected(self):
        """Extra requests, missing EOF or wrong IDs invalidate the measured population."""
        native, journal = evidence("role_then_error")
        for mutation in ("extra", "eof", "identity", "bytes", "flag"):
            changed = copy.deepcopy(journal)
            if mutation == "extra":
                changed.append(changed[0])
            elif mutation == "eof":
                changed = [x for x in changed if x["kind"] != "http_eof"]
            elif mutation == "identity":
                changed[0]["request_id"] = "warmup"
            elif mutation == "bytes":
                changed[1]["wire_hex"] = "00"
            else:
                changed[0]["payload"]["stream"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                check_export(native, changed, "role_then_error")


class FixtureTests(unittest.TestCase):
    def test_loopback_server_emits_declared_bytes_and_journals_real_requests(self):
        """The fixture must emit and record the exact bytes used by the checks."""
        # Loopback requests do not need system proxy discovery.
        opener = build_opener(ProxyHandler({}))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.jsonl"
            with fixture(path) as base:
                for case in CASES:
                    body = json.dumps({"messages": [{"role": "user", "content": prompt(case, 0)}]}).encode()
                    req = Request(base + f"/{case}/v1/chat/completions", data=body,
                                  headers={"Content-Type": "application/json", "x-request-id": "0"})
                    with opener.open(req, timeout=5) as response:
                        stream = response.read().decode()
                    values = [line.removeprefix("data: ") for line in stream.splitlines() if line]
                    parsed = [value if value == "[DONE]" else json.loads(value) for value in values]
                    self.assertEqual(parsed, events(case))
            journal = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(sum(x["kind"] == "request" for x in journal), 6)
            self.assertEqual(sum(x["kind"] == "http_eof" for x in journal), 6)

    def test_prepare_cannot_overwrite_evidence_or_an_existing_trial(self):
        """Preparing a trial must not overwrite saved evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trial"
            prepare(root, ROOT, sys.executable)
            self.assertFalse(json.loads((root / "status.json").read_text())["cli_gate_passed"])
            with self.assertRaises(FileExistsError):
                prepare(root, ROOT, sys.executable)
            with self.assertRaises(ValueError):
                prepare(ROOT / "bundle/new-trial", ROOT, sys.executable)

    @unittest.skipUnless(sys.platform == "darwin", "Mac environment stop condition")
    def test_run_on_mac_records_blocker_without_importing_vllm(self):
        """Unsupported environments must stop before importing the benchmark client."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trial"
            run = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/vllm_cli_trial.py"),
                                  "run", "--output", str(root), "--source-root", temporary],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "blocked")
            self.assertFalse(status["cli_gate_passed"])
            self.assertFalse((root / "cli-help.command.json").exists())

    @unittest.skipUnless(sys.platform == "darwin", "explicit Mac environment opt-in")
    def test_mac_optin_still_requires_real_pristine_source(self):
        """Mac opt-in must not bypass source provenance."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trial"
            run = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/vllm_cli_trial.py"),
                                  "run", "--output", str(root), "--source-root", temporary,
                                  "--allow-macos"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            status = json.loads((root / "status.json").read_text())
            self.assertIn("real Git checkout", status["reason"])
            self.assertFalse(status["cli_gate_passed"])
            self.assertFalse((root / "cli-help.command.json").exists())


if __name__ == "__main__":
    unittest.main()
