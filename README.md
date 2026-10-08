# Benchmark Stream Audit

Saved evidence and regression checks for vLLM streaming errors.

The baseline benchmark client counted some failed streams as successful.
The captured fix recognizes the error, preserves partial text, and excludes
failed output from completed-request statistics.

## Run

Python 3.10 or newer. No dependencies, model or GPU needed.

```sh
git clone https://github.com/k21993/benchmark-stream-audit.git
cd benchmark-stream-audit
python3 -B scripts/demo.py
python3 -B scripts/check_all.py
```

The demo reads saved evidence. The checks verify hashes, native reports,
28 historical request-function probes and 27 regression tests. One test uses a
local HTTP fixture. These commands do not run vLLM or call external services.

## Example

Five requests received `Hello world`, followed by an explicit server error.

| Native report | Baseline | Captured fix |
|---|---:|---:|
| Completed / failed | 5 / 0 | 0 / 5 |
| Completed-output tokens | 10 | 0 |
| Partial text retained | Yes | Yes |

Across all six native CLI cases, the baseline reports 25 completed / 5 failed.
The captured fix reports 15 completed / 15 failed, with zero false successes.
Normal controls pass in both runs.

The proposed fix comes from [vLLM PR #57519](https://github.com/vllm-project/vllm/pull/57519).
The native baseline is commit `b6d8e8afd985f5711eee68e343d2ce908d166488`.
The captured fix is commit `1e74a69c71604c3c3ec6cc3cce9dbe8b702bfcda`.
The request-function comparison also covers captured main
`08567505b09324f891797f4a13c1bd73fc75a827`.

## Coverage

- Error-only streams and errors after a role or text frame.
- Exact server errors and retained partial text.
- Failed output excluded from completed-output token totals.
- Normal completion and one versus three content chunks.
- Error after finish/usage and coalesced reads at the request-function layer.

Error after finish is classified as failed by this corpus's declared policy.
These request-function probes cover the chat backend only. Exact error-object
checks strengthen the proposed fix's existing error-message assertion.

## Replay the pinned sources

To execute the 28 probes again, install the locked request-function dependencies
in a separate environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements/replay.txt
.venv/bin/python -B scripts/vllm_stream_regression.py replay --output reruns/replay
```

Use a new output directory for each run. Replay passes only when every probe
matches the recorded outcomes, errors, text and usage. It expects the baseline's
known failures and the proposed fix's corrected behavior. New timestamps are
checked for consistency, not equality with the old run.

CI runs saved-evidence checks without dependencies and executes these pinned
sources in a separate replay job. Both jobs cover Linux/macOS and Python 3.10/3.14.
Replay uses fake HTTP sessions, with no model, GPU or external service calls.
It does not test new upstream revisions or rerun the native CLI.
Failed CI replays retain `result.json` as an artifact for 14 days when available.

## Check another revision

Capture an explicit commit, then execute the same 14 chat probes against it:

```sh
REVISION=8352b2427f704652da1910f0d53f3fdcc2fa2b0d
.venv/bin/python -B scripts/check_vllm_revision.py capture --revision "$REVISION" --output reruns/candidate-source
.venv/bin/python -B scripts/check_vllm_revision.py check --revision "$REVISION" --source reruns/candidate-source --output reruns/candidate-check
```

Capture downloads the endpoint module, timing tests and license from GitHub.
Check executes that module locally using the locked dependencies above and the
original frozen fake-session helpers. Source hashes and the requested commit
are checked before execution. Use fresh output directories; old evidence is preserved.

Exit 0 means all probes conform, 1 means conformance failures, and 2 means an
input or execution failure. Results retain per-probe inputs, outputs and errors.
Import/API exceptions and upstream traceback errors are reported separately
from contract failures. This command does not expect historical bugs to pass.
It does not update historical replay or run automatically against moving main.

The October 7, 2026 check of `8352b2427f70` completed all 14 probes: 6 conformed,
8 failed the contract, including 6 false successes; no execution errors occurred.
The endpoint module and timing tests were unchanged from captured main
`08567505b093`. This is request-function evidence, not a new native CLI run.

`requirements/replay.in` lists direct dependencies. `requirements/replay.txt`
locks all dependencies and distribution hashes. To regenerate it with uv:

```sh
uv pip compile requirements/replay.in --universal --python-version 3.10 --generate-hashes --no-annotate --output-file requirements/replay.txt
```

Code is in `benchmark_stream_audit/` and `scripts/`. Tests are in `tests/`.
Saved reports, journals, commands and source snapshots are in `experiments/`.
`EVIDENCE_SHA256SUMS` covers the published evidence. Public source inventories
include the modules needed for these checks; recorded probe inputs and outputs
are unchanged.

## Limits

These are synthetic cases at fixed revisions. They do not establish production
failure rates, model speed or behavior in a current release. The baseline and
fix use different revisions. Native request arrays are joined by input order;
the exports do not expose independent request IDs. Server clocks measure
emission, not client receipt. Peak counters count choices events, not measured
model-token generation times.

The candidate's first harness rejected routine `GET /metrics` reads after all
CLI commands finished successfully. Its blocked status is retained. The saved
outputs passed verification after allowing those reads; no requests were rerun.

## License

Project code: [MIT](LICENSE). Copied vLLM source keeps its Apache 2.0 license
and notices. See [NOTICE](NOTICE).
