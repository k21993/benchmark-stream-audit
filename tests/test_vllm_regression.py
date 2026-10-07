"""Verify saved differential evidence without importing upstream clients."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_stream_audit.vllm_regression import assess, verify_result

RESULT = ROOT / "experiments/vllm-regression/evidence/differential-20261007/result.json"


class StreamConformanceTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads(RESULT.read_text())

    def test_saved_source_comparison_preserves_actual_conformance_failures(self):
        """Saved results must retain both observed failures and passes."""
        verify_result(ROOT, self.result)
        self.assertEqual(self.result["summary"]["main"],
                         {"attempted": 14, "conformant": 6, "false_successes": 6})
        self.assertEqual(self.result["summary"]["pr57519"],
                         {"attempted": 14, "conformant": 14, "false_successes": 0})

    def test_error_only_failure_without_server_reason_is_not_conformance(self):
        """A failed flag is insufficient if the server reason was lost."""
        rows = [row for row in self.result["rows"]
                if row["variant"] == "main" and row["case"] == "error_only"]
        for row in rows:
            self.assertFalse(row["output"]["success"])
            self.assertEqual(assess(row["case"], row["output"]), ["server_error_reason"])

    def test_failed_text_must_remain_available_after_correct_classification(self):
        """Correct failure classification must not discard generated text."""
        row = next(row for row in self.result["rows"]
                   if row["variant"] == "pr57519" and row["case"] == "text_then_error")
        output = copy.deepcopy(row["output"])
        output["generated_text"] = ""
        self.assertEqual(assess("text_then_error", output), ["retained_text"])

    def test_transport_read_grouping_keeps_semantics_separate_from_clock_values(self):
        """HTTP read grouping must not change result semantics."""
        grouped = {}
        for row in self.result["rows"]:
            key = row["variant"], row["case"]
            values = {field: row["output"][field] for field in (
                "success", "error", "generated_text", "output_tokens", "prompt_len")}
            if key in grouped:
                self.assertEqual(grouped[key], values)
            grouped[key] = values

    def test_corrupt_inputs_checks_provenance_and_missing_rows_are_rejected(self):
        """Corrupt metadata or missing probes must not pass verification."""
        for mutation in ("wire", "contract", "checks", "source", "missing", "duplicate", "summary"):
            result = copy.deepcopy(self.result)
            if mutation == "wire":
                result["rows"][0]["wire_hex"][0] = "00"
            elif mutation == "contract":
                result["rows"][0]["expected"]["outcome"] = "failed"
            elif mutation == "checks":
                next(row for row in result["rows"] if row["failed_checks"])["failed_checks"] = []
            elif mutation == "source":
                result["sources"]["main"]["revision"] = "wrong"
            elif mutation == "missing":
                result["rows"].pop()
            elif mutation == "duplicate":
                result["rows"][-1] = result["rows"][0]
            else:
                result["summary"]["main"]["false_successes"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_result(ROOT, result)

    def test_request_function_probes_cannot_be_reclassified_as_full_cli_evidence(self):
        """Request-function evidence must not be presented as full CLI evidence."""
        self.result["layer"] = "full_cli"
        with self.assertRaisesRegex(ValueError, "mislabeled"):
            verify_result(ROOT, self.result)

    def test_runner_cannot_overwrite_results_or_enter_preserved_inputs(self):
        """Replays must not overwrite earlier results or source snapshots."""
        before = RESULT.read_bytes()
        for output in (RESULT.parent, ROOT / "bundle/new-regression",
                       ROOT / "experiments/vllm-regression/evidence/upstream-20261007/new"):
            command = [sys.executable, "-B", str(ROOT / "scripts/vllm_stream_regression.py"),
                       "run", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2, result.stderr)
            if output != RESULT.parent:
                self.assertFalse(output.exists())
        self.assertEqual(RESULT.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
