"""Keep new-revision conformance separate from replay and setup failures."""

from dataclasses import make_dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from benchmark_stream_audit.revision_check import FILES, capture, check

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a' * 40
SAVED = ROOT / 'experiments/vllm-regression/evidence/differential-20261007/result.json'


class RevisionCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        files = []
        for name in FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# source fixture\n')
            files.append({'path': name, 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        (self.source / 'source_manifest.json').write_text(json.dumps(
            {'commit': REVISION, 'files': files}))
        self.output = self.root / 'result'

    def execute(self, variant='pr57519', error=None, wrapped=False):
        rows = [r for r in json.loads(SAVED.read_text())['rows'] if r['variant'] == variant]
        output_type = make_dataclass('Output', list(rows[0]['output']))
        effects = [output_type(**r['output']) for r in rows]
        if error:
            if wrapped:
                effects[0].error = 'Traceback (most recent call last):\nTypeError: changed API'
            else:
                effects[0] = error
        module = SimpleNamespace(RequestFuncInput=Mock(),
                                 async_request_openai_chat_completions=AsyncMock(side_effect=effects))
        with patch('benchmark_stream_audit.revision_check.importlib.util.module_from_spec',
                   return_value=module), patch(
                       'benchmark_stream_audit.revision_check.importlib.util.spec_from_file_location',
                       return_value=SimpleNamespace(loader=Mock())):
            return check(ROOT, self.source, REVISION, self.output)

    def test_candidate_must_conform_instead_of_reproduce_baseline_bugs(self):
        """Known historical bugs must fail a new-revision check."""
        result = self.execute('main')
        self.assertEqual(result['status'], 'nonconformant')
        self.assertEqual(result['summary']['false_successes'], 6)
        self.assertEqual(len(json.loads((self.output / 'result.json').read_text())['rows']), 14)

    def test_conformant_candidate_passes_without_matching_old_clocks(self):
        """Corrected behavior passes without comparison to baseline outcomes."""
        result = self.execute()
        self.assertEqual(result['status'], 'conformant')
        self.assertEqual(result['summary']['conformant'], 14)

    def test_api_failure_is_saved_separately_and_other_probes_continue(self):
        """An API mismatch must not be counted as a stream-classification failure."""
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                self.output = self.root / f'api-failure-{wrapped}'
                result = self.execute(error=TypeError('changed request API'), wrapped=wrapped)
                self.assertEqual(result['status'], 'execution_error')
                self.assertEqual(result['summary']['execution_errors'], 1)
                self.assertEqual(result['summary']['nonconformant'], 0)
                self.assertEqual(result['summary']['conformant'], 13)
                self.assertTrue((self.output / 'result.json').is_file())

    def test_tampered_or_wrong_revision_sources_are_rejected_before_import(self):
        """Source integrity failures retain diagnostics without executing source."""
        (self.source / FILES[0]).write_text('raise RuntimeError("must not execute")')
        result = check(ROOT, self.source, REVISION, self.output)
        self.assertEqual(result['status'], 'execution_error')
        self.assertEqual(result['rows'], [])
        self.assertIn('captured source changed', result['errors'][0])
        result = check(ROOT, self.source, 'b' * 40, self.root / 'wrong-revision')
        self.assertIn('revision differs', result['errors'][0])

    def test_existing_outputs_and_preserved_sources_cannot_be_overwritten(self):
        """A revision run must never rewrite historical evidence or earlier runs."""
        self.execute()
        before = (self.output / 'result.json').read_bytes()
        for output in (self.output, self.source / 'nested', ROOT / 'experiments/new'):
            with self.assertRaises((OSError, ValueError)):
                check(ROOT, self.source, REVISION, output)
        self.assertEqual((self.output / 'result.json').read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'commit SHA'):
            capture(ROOT, 'main', self.root / 'mutable')
        self.assertFalse((self.root / 'mutable').exists())


if __name__ == '__main__':
    unittest.main()
