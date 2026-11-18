# docs/DECISIONS.md
# Architecture & Design Decisions

## D1 – Flat step list, not a graph
Steps are stored as a sequential list (Step.index) rather than a DAG.
LangGraph's execution is already linearised at record time; replaying in
order is simpler and the UI is easier to render.  If true graph branching
is needed in Phase 3+ we can add a `parent_step_id` column.

## D2 – Event split from Step
Splitting events (llm_call | tool_call | state) out of Step allows the
replay engine to match on `(node_name, input_hash, occurrence)` rather
than sequential index.  This means replay is robust to prompt-length
changes that shift token counts but don't change the logical call.

## D3 – input_hash computed after redaction
Hashing happens *after* PII/secret stripping.  Consequence: a prompt
containing a customer e-mail and the same prompt with the e-mail redacted
hash to the same value.  This is intentional – it means replay still works
even if the redaction regex changes slightly between runs.

## D4 – No Celery/Redis in MVP
Background work uses FastAPI BackgroundTasks.  Good enough for single-node
demo.  Celery would add operational complexity for no benefit at this scale.

## D5 – SQLite default, Postgres opt-in via DATABASE_URL
SQLite is zero-config for local dev and CI.  `check_same_thread=False` is
set so FastAPI's thread-pool workers can share a connection.  Postgres
support is free because SQLAlchemy abstracts the dialect.

## D6 – EvalLabel separate from Trace
`injected_fault` and `ground_truth_step` live in `EvalLabel`, not `Trace`.
This guarantees that `investigate.py` cannot accidentally read ground-truth
labels (enforced by a test that checks the module's import list).

## D7 – run_id for idempotency
The SDK accepts a caller-supplied `run_id` (UUID4 by default).  Retried
ingest calls are no-ops: the existing Trace row is returned.  This is
critical for at-least-once delivery in fault-injection benchmarks.

## D8 – UTC-aware datetimes everywhere
All `DateTime` columns use `timezone=True` and
`datetime.datetime.now(datetime.timezone.utc)`.  Avoids DST confusion and
makes cross-timezone comparisons safe.

## D9 – Pure-Python stub LLM for demo
The demo agent uses a deterministic Python function instead of a live LLM.
This means the demo runs with no API key, CI is cheap, and boundary
conditions are testable without flakiness.  Swapping in a real LLM
requires only changing `_stub_llm()`.

## D10 – Generated tests assert on behaviour, not LLM text
Tests mock tool functions via `unittest.mock` and assert on *side effects*
(e.g. `issue_refund` not called), not on the LLM's exact response text.
This keeps tests stable across model versions.

## Known limitations (Phase 1)
- No streaming ingest endpoint yet (SDK writes directly to DB).
- No auth on the FastAPI endpoints.
- Trace viewer has no pagination; will be slow with many traces.
- The demo agent is synchronous; LangGraph's async streaming is not yet hooked.
- Timestamps embedded inside strings: Regex normalization in `strip_volatile_fields` strips standard full ISO 8601 timestamps (`YYYY-MM-DDTHH:MM:SS...`) and UUIDs from strings. Non-standard date/time strings (e.g. localized dates like `18/11/2026`, epoch millisecond integers, or natural language times like "2 hours ago") embedded in prompt text will not be stripped and will alter the computed `input_hash`. Conversely, if an agent legitimately depends on the exact sub-second timestamp string to make a decision, regex substitution to `<VOLATILE_TIMESTAMP>` collapses those differences.
