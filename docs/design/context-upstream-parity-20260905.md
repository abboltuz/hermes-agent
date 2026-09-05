# Upstream context-management restoration

## Scope and source pins

Restore the agent's context-management policy, not the repository, to upstream.
Preserve the owner's unrelated Desktop, provider, persistence, message-provenance,
and terminal-execution changes. No Kanban workflow is used for this task.

- Candidate base: `72365ee68b51a929b7da55f29a6c01f70a454fcd`.
- Upstream policy pin: `baf7ceeaceacf206b8cc10736db19dc9884f55f8` (`origin/main`).
- Pre-decomposition integration reference: `561b053f794a1781868bb032029d589c67708119`.
- Common ancestor for scoped change attribution: `1fe0f2f3ac9748ce799272eb93bee2937b5ab802`.
- Task branch: `cekasha/context-upstream-parity-20260905`.

Upstream decomposed the conversation loop and several utility modules after the
integration reference. Its context policy is imported at the main pin; integration
hunks are forward-ported into this checkout's existing module layout. This is not
a wholesale main merge or a claim that the entire custom tree equals upstream.

## Removed local context policy

- Remove `agent/compression_v3.py`, including final-wire admission, request budget
  receipts, and the custom projection-refusal retry/recovery path.
- Remove final-wire admission from ordinary requests and iteration-limit summary
  requests. Provider errors follow upstream's bounded recovery paths.
- Remove custom compression trigger strings, hard-pressure timeout overrides,
  concurrent-winner join loops, and summary-route fallback retries.
- Remove custom recovery-row identity bookkeeping from the live flush path.

Existing recovery/archive tables and read APIs remain for historical compatibility.
No stored chat data is deleted, no user configuration is rewritten, and no live
conversation is modified during implementation or tests.

## Upstream policy and integration

| Area | Source / adaptation |
| --- | --- |
| Summary, tail selection, cooldowns, no-op handling, token accounting | Main-pinned `context_compressor.py`; typed-message adapter described below |
| Micro-compaction | Main-pinned `micro_compaction.py`; retain semantic fields when user carriers merge |
| Attempt ownership, cancellation, deadline, atomic publication | Main-pinned `conversation_compression.py` and `compression_facade.py` |
| Main-loop/preflight pressure, real usage anchors, provider overflow recovery | Scoped upstream integration hunks in `conversation_loop.py` and `turn_context.py` |
| Auxiliary summary progress and stall handling | Scoped upstream hunks in `auxiliary_client.py`, preserving local provider bridges and shared-client ownership |
| Model context metadata | Main-pinned `model_metadata.py` |
| Native Responses compaction | Upstream automatic threshold and route capability lifecycle; retain local provenance filtering |
| Tool-prefix continuity across a context boundary | Upstream MCP refresh/persist/restore helpers in this checkout's existing `tools/mcp_tool.py` |
| Durable context state | Upstream tail-duplicate flags, lease refresh at publication, accidental-end classification, merge-max failure cooldown, and persisted recovery deadline |

Import seams adapted to the pre-decomposition layout:

- `tools.mcp_tool_agent` -> `tools.mcp_tool`.
- `tools.file_tools_read_tracking` -> `tools.file_tools`.
- Add upstream `serialized_messages_bytes` and `stale_thinking_reaches_wire` to
  `agent/message_sanitization.py`.
- Add upstream `_classify_tool_call_orphans` to `agent/agent_runtime_helpers.py`.
- Preserve the existing `context_compressor.tool_result_id_variants` import
  without importing upstream's absent temporary `hermes_cli.plugin_compat` module.

The recovery deadline adds one nullable, declaratively ensured session column.
It does not introduce a new archive or replace existing schema/data migrations.

## Preservation boundary

Typed message provenance remains authoritative: explicit unknown, invalid,
notification, and delivery-mirror messages cannot become human task anchors.
Unstamped legacy compressor callers retain upstream's content-marker heuristics.
This compatibility adapter does not replace upstream tail sizes, budgets, summary
prompts, attempt limits, or retry policy. Summary serialization preserves neutral
actor labels and excludes notifications/mirrors. Human anchors retain semantic
fields when merged; standalone todo scaffolding remains hidden.

The primary source checkout and installed checkout are not implementation targets.
Provider bridges, terminal fences, Desktop resume/history implementation, and
durable message schema/projection remain in the candidate. The only gateway/TUI
edits remove obsolete `trigger=` arguments from manual/automatic compression calls.

## Verification and promotion

All Python tests run through `scripts/run_tests.sh`, with per-file subprocesses,
sanitized credentials, and isolated test homes. A task-local ignored `.venv`
references already-installed Python 3.11 test/runtime libraries read-only; it does
not modify the installed runtime or its dependencies. Tests import the candidate's
modules, not the installed source tree.

Regression coverage includes upstream attempt ownership, timeout, stall recovery,
usage anchors, lean single-aux-call behavior, real SQLite rotation/in-place commits,
lease refresh, native model switching, and the existing custom provider,
provenance, terminal fence, resume, and rewind tests. Older legacy-mode assertions
are pinned explicitly as upstream does; removed v3 policy tests are not retained
as acceptance criteria for a policy that no longer exists.

Verified on 2026-09-05 (macOS, Python 3.11.15):

```sh
scripts/run_tests.sh tests/agent/test_compression*.py tests/agent/test_compressor*.py tests/agent/test_compaction*.py tests/agent/test_context_compressor*.py tests/agent/test_micro_compaction.py tests/agent/test_compress_context_progress_timeout.py tests/agent/test_preflight_compression*.py tests/agent/test_post_compression_trim.py tests/run_agent/test_compression*.py tests/run_agent/test_in_place_compaction.py tests/run_agent/test_413_compression.py -q -j 3
```

Exit 0: 59 files, **654 passed, 0 failed**, 81.2 seconds.

```sh
scripts/run_tests.sh tests/agent/test_auxiliary*.py tests/agent/test_antigravity_bridge.py tests/agent/test_cursor_bridge_client_contract.py tests/agent/test_cursor_bridge_transport_contract.py tests/agent/test_message_provenance.py tests/agent/transports/test_message_provenance_projection.py tests/test_message_provenance_persistence.py tests/run_agent/test_run_agent.py tests/run_agent/test_kanban_terminal_execution_fence.py tests/agent/test_model_metadata*.py tests/agent/test_portal_tags.py tests/agent/test_prompt_cache_scope.py tests/agent/test_turn_context*.py tests/run_agent/test_native_compaction*.py tests/run_agent/test_infinite_compaction_loop.py tests/run_agent/test_post_tool_compression_attempt_cap.py tests/run_agent/test_preflight_compression_cap_e2e.py tests/tools/test_refresh_agent_mcp_tools.py tests/agent/test_usage_anchor.py tests/agent/test_fast_compression_lane.py tests/agent/test_lean_single_aux_call.py tests/state/test_compression_lease_refresh_before_publish.py tests/agent/test_reference_handoff_active_turn.py tests/tui_gateway/test_composite_carrier_rewind.py tests/hermes_state/test_composite_carrier_rewind.py tests/tui_gateway/test_session_resume_db_ownership.py tests/test_tui_gateway_server_crash_history.py -q -j 3
```

Exit 0: 53 files, **1,186 passed, 0 failed**, 78.6 seconds.

`git diff --check` passes. Ruff `F821,F822,F823` across changed Python files
reports only three baseline findings, independently reproduced against the base
blobs: `RateLimitState` in `agent/agent_init.py`, `ActivityProvenance` in
`gateway/run.py`, and `_pending_reaction_notes` in `tui_gateway/server.py`.
They are not changed here; this is not a claim that the complete baseline lints
cleanly. The full repository suite, packaged Desktop rebuild, and live-chat soak
are not covered by these focused/affected test runs.

Primary source remains clean at `72365ee68b`; installed runtime remains clean at
`c3895376ca`. Live explicit compression settings are retained, including threshold
0.5, target ratio 0.2, head 3, tail 20, and max attempts 3; no explicit tail mode is
set, so upstream's lean default applies after activation.

Promotion requires a clean committed candidate and exact-SHA approval. Runtime
observation after promotion is separate from these offline checks; no claim of a
successful live soak or repaired existing chat follows from unit tests alone.

## Independent review corrections

The first exact candidate, `49e3952d9a`, received CHANGES_REQUESTED. The correction
keeps the same policy pin and addresses all three findings:

- Preserve upstream in-flight task replay for typed trusted cron/delegation turns,
  including repeated handoffs, carrier merges, and post-compression continuation
  checks. Actor/trust fields survive; these tasks are never classified as human
  intent. Untrusted plugin/webhook events and completed tasks are not replayed.
- Restore upstream outbound stale tool-image eviction in both ordinary and
  iteration-limit summary requests. Request-capture tests keep the newest three
  image-bearing tool rows while asserting every original transcript row is exact.
- Remove an imported endpoint-normalization change outside context scope.
  Same-provider omitted-URL refresh retains custom routing and derives native
  compression capability from that retained endpoint. Cross-provider unresolved
  endpoints retain the original fail-without-mutation contract.

After corrections, the core context command above plus
`tests/run_agent/test_switch_model*.py` passes: **67 files, 692 passed, 0 failed**,
82.3 seconds. The two request-capture regressions also pass in isolation. Their
initial fixture failures (invalid image data, then an unspecified non-vision model)
were corrected with a valid PNG and explicitly mocked vision capabilities; they
were not product-code failures. Ruff F821/F822/F823 passes for all correction files.

The complete preservation/affected command above was then rerun successfully:
**53 files, 1,188 passed, 0 failed**, 75.9 seconds. Together, the two disjoint
post-correction runs cover **120 files and 1,880 passing tests**.

An additional pre-correction state run passed **289 tests across 6 files**:
`test_hermes_state`, compression locks, busy retry, read-only preflight,
`get_messages_include_compacted`, and `append_messages_batch`. These state files
were not changed by the review corrections.

Upstream advanced to `377118af86` during this work. The pinned compressor/facade/
micro-compaction policy files have not changed in that range. A newer oversized
`@file` ingress-pointer feature in `agent/context_references.py` is outside this
pinned restoration; this report does not claim every newer context-related
feature is included.

The second exact candidate, `8bc749884e`, resolved the image and routing findings
but received CHANGES_REQUESTED for one remaining continuation consumer:
`reference_handoff_would_drive_next_model_call` still treated standalone typed
task replay as reference-only. It now uses the same trusted in-flight predicate.
The repeated-handoff regression exercises the actual
`_should_skip_model_call_for_reference_handoff` boundary and asserts no appended
row, no restoration of an older turn-start instruction, and an unchanged
transcript. Untrusted plugin/webhook events still do not drive continuation.

Final correction verification:

```sh
scripts/run_tests.sh tests/agent/test_compression_typed_inflight_task.py tests/agent/test_reference_handoff_active_turn.py tests/agent/test_context_compressor_zero_user_provenance.py tests/agent/test_compressor_actionable_tail_anchor.py tests/run_agent/test_run_agent.py -q -j 3
```

Exit 0: **5 files, 345 passed, 0 failed**, 54.3 seconds. Diff-check and Ruff
F821/F822/F823 on the final correction files pass. This is a rerun of affected
tests within the 1,880-test set, not 345 additional distinct tests.
