# Bounded working context and lossless searchable archive

Status: implementation in progress, not a released capability.
Implementation base: `0be448cacf4198f5ef044f3c308d585c466b3a89` (`cekasha/main`).
No production migration or runtime activation is authorized by this document.

## Product contract

A conversation retains its identity and complete raw history. Provider requests
use a bounded, versioned working context rather than loading the entire archive.
The context contains structured task state, hierarchical summaries, exact anchors,
recent complete turns, bounded archive retrieval, and unresolved tool effects.
Archive retrieval returns source provenance. Summaries are derived data, never
the sole record of the conversation.

Context capacity is checked on the final provider-shaped request, including tool
schemas and output reserve. The current byte/token heuristic is an estimate, not
a proof for arbitrary tokenizers or multimodal inputs. Unsupported counting must
not be described as an unconditional fit guarantee.

Owner decision (2026-09-04): preserve compatibility on routes without reliable
counting. Explicitly distinguish verified and estimated budgets. Estimated routes
use headroom and bounded overflow recovery; absence of a known window is not
evidence that the request fits. Do not disable these providers by default.

Compaction builds a candidate away from the session-open path and publishes it
atomically against its source revision. Incomplete summaries and stale candidates
are rejected. Unmatched tool calls must not be summarized away or replayed as new
effects. Publication is episodic, preserving the cached prefix between episodes.

Opening a conversation must not require an archive scan, agent initialization,
or completion of a compaction job. Existing Desktop pagination/windowing is reused.
Archive storage and indexing must not monopolize the hot control-plane database.
Migration preserves raw payloads and verifies identities/counts/hashes before
switching readers; it is resumable and never rewrites a live multi-GB database
in one request.

## Implementation sequence and evidence gates

1. Durable execution admission and terminal receipts (implemented).
2. Versioned context snapshots, bounded candidate validation, atomic publication,
   and resume from snapshot plus a fenced post-snapshot tail.
3. Searchable raw archive separation, artifact references, hierarchical summaries,
   structured state and exact anchors; maintain existing recovery APIs.
4. Independent session-open/control path, worker resource isolation, lazy legacy
   migration, and large-history end-to-end tests.
5. Exact-candidate independent review, recall/cache-cost evaluation, and explicit
   owner approval before integration or runtime activation.

Passing tests for step 1 does not establish steps 2–5.

## Versioned manifest bridge (step 2 in progress)

In-place compaction now atomically publishes a versioned manifest of durable
working-row IDs and a post-commit watermark. The manifest contains no copied
message bodies. Single-session model reload and resume use primary-key probes
for these references plus an indexed post-watermark tail, pinned in one SQLite
read transaction. Multi-segment legacy lineage and sessions without a manifest
still use the legacy reader; opening those remains part of the later migration
and control-plane work.

A small transactional tail-reference table records only active post-watermark
appends. This avoids scanning later inactive imports and requires no new index
build over the existing messages table. Head invalidation cascades to its tail;
publication clears the now-covered tail in the same transaction. Manifest format
v2 identifies this completeness contract; older manifests use legacy reads until
republication rather than assuming a newly created empty tail index is complete.

The materialized working projection is capped at 8,192 rows and 16 MiB of stored
projection fields, measured before body materialization. These are storage/read
safety bounds, not provider token limits or total-process RSS guarantees. An
oversized candidate rolls back the whole publication; an oversized appended tail
returns an explicit error, never a partial transcript. Subsequent worker-side
recovery and independent history browsing must handle that state before release.

Legacy membership-changing writes invalidate the current manifest transactionally.
Content and presentation updates read through the row references. Consequently
these are versioned **membership manifests**, not immutable historical payload
snapshots yet. Existing `archive_and_compact` still materializes its projection
and clones its concurrent tail; replacing that behavior with immutable raw-event
references and cold storage is required, not waived by this bridge.

Verification includes byte-for-byte replay sidecars, restart, atomic rollback,
lost publication lease, sibling rewrite during a pinned read, legacy read-only
stores, oversized tails, and bounded query work with 10,000 synthetic archived
rows. This is not evidence of multi-GB control-plane latency or completed cold
archive migration.

## Durable admission increment

`context_compaction_jobs` stores hashes and receipts, not transcript bodies.
Its key is `(conversation, source fingerprint, strategy fingerprint)`, deliberately
excluding the process-local generation. Strategy includes engine identity,
configured summarizer route, context policy and focus. Automatic callers sharing
one logical conversation join or defer instead of executing twice.

`running` transitions to `committed`, `no_progress`, `timed_out` or `aborted`.
These outcomes suppress identical automatic work after a process restart.
`cooldown`, `deferred_lock` and `native_delegated` are retryable nonexecution
outcomes. A different source/strategy or explicit force rearms a terminal job;
force cannot steal a live job. An abandoned lease expires to a terminal state,
not into another automatic attempt. A unique owner token fences receipt writes.
The existing compression lock/commit fence still controls transcript publication.

Journal admission uses the existing short activity-write contention budget and
fails closed without modifying the transcript. Failure to write a completion
receipt does not mask cancellation or a successfully returned result. The
unfinished receipt remains conservative until expiration or explicit recovery.
This first increment retains the existing legacy untagged compression-call
contract; these callers and direct engine compaction entry points still need to
converge on the snapshot publication service in step 2.

## Budget certainty prerequisite

The final-wire gate separates `fits` from `allows_dispatch`. A missing context
window produces `fits=None` / `unknown_window` while compatibility policy still
admits the request. A known window produces `estimated_fit` or
`estimated_overflow`, retaining the configured output reserve and safety margin.
Debug diagnostics and fit errors carry this distinction without request bodies
or credentials. This does not add retries or change the existing bounded
pressure-recovery policy.

All current final-wire counts remain explicitly `estimated`. Model metadata,
a configured context window and a successful previous request do not establish
an authoritative count of a new full request. No current route is labeled
`verified`; adding that label requires an adapter-level complete counting and
context-window contract, including tool schemas and multimodal accounting.
This prerequisite is not the bounded snapshot validator or a universal fit
guarantee, and does not replace the remaining work in steps 2–5.

## Opening acknowledgement increment

Desktop's primary chat and tile now both request deferred model-history loading
and omit the duplicate WebSocket transcript; their authenticated REST page read
remains separate. Previously the tile omitted returned messages but still waited
for synchronous history materialization. Deferred resumes read compact runtime
metadata without flushing usage writes or joining the persisted system prompt.
The existing materialization safety guard runs in the history worker, before
reopening or reading transcript bodies; eager and watch clients retain their
synchronous contract. Profile DB ownership remains with the worker until it
closes the dedicated handle. Both guard paths explicitly bind the owning profile
while resolving policy, then restore the caller's prior context. A background
thread must not substitute launch-profile limits for the target profile's limits.
Real separate-profile configuration and SQLite tests cover both a stricter owner
and an owner that disabled the guard, plus foreign-context restoration.

Tests hold the guard behind a real synchronization barrier and prove that the
acknowledgement returns first, including when the eventual result is oversized.
They do not establish complete independent browsing: failed history preparation
still discards the runtime handle, and the UI still needs a distinct recoverable
preparation state. Profile store construction, legacy title/adoption fallback,
lineage resolution and workspace metadata also remain on the acknowledgement
path; bounded resource isolation and large-archive latency are not yet proven.

## Required verification beyond the current increment

- Restart and multiple real connections/processes: one admitted source/strategy;
  no stale-owner publication or receipts.
- Candidate rejection preserves the prior snapshot and raw event identities.
- Concurrent appends and interrupted tool groups survive compaction exactly.
- Repeated compactions do not clone the retained raw tail.
- Final request budget includes transport-specific fields and multimodal costs;
  oversized mandatory input returns an explicit actionable failure.
- Archive search and user-visible history retain exact provenance after migration.
- Opening, stopping and navigating other chats remain responsive during heavy
  compaction/indexing; measure tail latency against large synthetic archives.
- Recall with recovery and cache reuse do not regress against the checked-in
  compaction evaluation corpus.
- Migration interruption, storage failure and downgrade have explicit tested
  behavior. Installed state, provider credentials and live sessions are not test
  fixtures.
