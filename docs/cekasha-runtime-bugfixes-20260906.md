# Runtime failure and auxiliary route repairs

## Scope

This candidate repairs verified adapter/control-flow failures without changing
context compression, session history, credentials, live profiles, or provider
HTTP 400 handling. It does not diagnose the underlying network ConnectError.
The installed runtime and primary source branch are not promotion targets until
the owner approves a independently reviewed candidate.

## Contracts

- Async Antigravity vision wraps the existing synchronous bridge client. Its
  process, command, abort and close ownership are retained; no OpenAI-only
  `api_key` attribute is assumed.
- Explicit OpenRouter auxiliary output caps reach the request. An absent cap
  remains absent. This is not a new context/compression strategy.
- `blocked` is distinct from `done` in goal parsing, persistent goals, loops,
  and both CLI/tool Kanban handoff surfaces.
- A failed worker result is not judged as progress. Initial failures skip the
  continuation loop; later failures preserve the original structured result
  through the quiet CLI exit contract. Existing terminal/run fencing wins.
- An unavailable or malformed continuation judge stops and blocks the open
  task instead of spending further worker turns. Existing handoff-gate error
  semantics are unchanged: this is not a port of the unmerged fail-open PR.
- A card with explicit provider and model receives an internal launch snapshot.
  Primary-worker cross-provider fallback is disabled for that pinned launch;
  its continuation and handoff judges use that same route, with auxiliary
  fallback disabled. Later card edits apply to later launches, not this run.
- The existing `__KANBAN_CARD_MODEL_REQUIRED__` profile sentinel is validated
  before process spawn. Missing pins are rejected there, not sent as a model
  name to a provider. Ordinary profile inheritance and model-only overrides
  remain compatible. No live profile is migrated by this patch.
- `call_llm(..., allow_fallback=False)` is call-scoped and opt-in. Task-specific
  budget/timeout settings remain, but auxiliary task endpoint/model/API mode
  cannot replace the explicit route. Provider-internal SDK retries and account
  selection within that provider are not replaced by this control.

## Upstream provenance

The comparison target was NousResearch/hermes-agent main at
`6f4a822d9fa15735853a6292d48ac6ab371765c3`.

- PR #99725: selectively port the explicit OpenRouter output-cap gate from
  `b26a1eae8f`; the final PR commit `13c3958df5` contains its tests. Exclude
  affordable-credit compression retry changes.
- PR #101278: selectively port the blocked verdict and consumer correction
  from `1bd9fce6cb622846ecdf003754e8d2f7388d1883` and
  `5c6cbbc1be5b46f92f8a536d6f1bd45683bfcf70`, retaining our terminal fencing and
  trusted continuation provenance.
- PR #83746 was not merged at comparison time and is not applied here.

## Verification and promotion gate

Use `scripts/run_tests.sh` with an isolated test runtime. Affected suites cover
auxiliary adapters, real SDK requests with mock HTTP transports, vision routing,
goal/loop verdicts, worker spawn, route snapshots, structured errors and terminal
execution fences. Live inference/account tests are intentionally excluded.

Record final exact-SHA test results and independent review in the handoff. Only
then request owner approval to update local main and the installed checkout.
No push, deployment, profile mutation or application restart is part of this
candidate implementation.
