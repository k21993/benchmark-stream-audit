"""Guard preserved failure diagnostics and native accounting in the captured fix."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.vllm_fix_cli import DEFAULT_RESULT, check_candidate, verified_rows, verify


class CapturedFixCliTests(unittest.TestCase):
    def setUp(self):
        self.journal = [json.loads(line) for line in
                        (DEFAULT_RESULT / "server_journal.jsonl").read_text().splitlines()]
        self.partial = json.loads((DEFAULT_RESULT / "text_then_error/native.json").read_text())

    def test_saved_full_cli_excludes_failed_partial_text_from_successful_tokens(self):
        """Failed partial output must remain outside completed-token totals."""
        result = verify(DEFAULT_RESULT)
        self.assertEqual((result["completed"], result["failed"]), (15, 15))
        self.assertEqual(result["completed_output_tokens"], 45)
        self.assertEqual(result["partial_output_tokens_local"], 10)

    def test_lost_partial_text_error_reason_or_failed_output_tokens_are_rejected(self):
        """Failure records must retain text and errors without adding completed tokens."""
        mutations = (("generated_texts", ""), ("errors", "generic failure"),
                     ("output_lens", 2))
        for field, replacement in mutations:
            with self.subTest(field=field):
                native = copy.deepcopy(self.partial)
                native[field][0] = replacement
                with self.assertRaises(ValueError):
                    check_candidate(native, self.journal, "text_then_error")

    def test_saved_evidence_corruption_is_rejected_before_interpretation(self):
        """Changed evidence must be rejected before interpreting results."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            shutil.copytree(DEFAULT_RESULT, root)
            (root / "text_then_error/native.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                verify(root)

    def test_changed_cli_flags_cannot_be_relabelled_as_the_same_experiment(self):
        """Changed flags would describe a different experiment."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            shutil.copytree(DEFAULT_RESULT, root)
            path = root / "normal/cli.command.json"
            record = json.loads(path.read_text())
            record["argv"][record["argv"].index("--num-warmups") + 1] = "1"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "flags changed"):
                verify(root, require_manifest=False)

    def test_metrics_reads_do_not_hide_unexpected_or_extra_model_requests(self):
        """Metrics reads must not hide extra model calls or unexpected traffic."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            shutil.copytree(DEFAULT_RESULT, root)
            self.assertEqual(len(verified_rows(root)), 6)
            path = root / "server_journal.jsonl"
            original = path.read_text()
            for row in ({"kind": "auxiliary", "method": "POST", "path": "/metrics"},
                        {"kind": "auxiliary", "method": "GET", "path": "/other"},
                        next(r for r in self.journal if r["kind"] == "request")):
                path.write_text(original + json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    verified_rows(root)


if __name__ == "__main__":
    unittest.main()
