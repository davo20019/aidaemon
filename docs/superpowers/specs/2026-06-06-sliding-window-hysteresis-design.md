# Sliding-Window Hysteresis Design

**Status:** Draft — Phase 0 approved for implementation; Phase 1 gated on the
Phase 0 attribution result. Last revised 2026-06-06.

## Goal

Stop the history region of the prompt from being rewritten between LLM calls,
so llama.cpp prefix-cache reuse extends past the (now stable) system prompt.
Live measurement showed blocks of consecutive large requests re-evaluating
~13K tokens (~31s each) with the reusable prefix stuck at the system-prompt
size (`n_past = 9816`).

## What is proven and what is not

Proven (llama-server log): within the bad blocks, consecutive large prompts
share exactly a 9,816-token prefix and diverge immediately after it. The
size is consistent with the serialized system-prompt-and-tools region, but
that equality cannot be confirmed by Rust-side token estimates — they do not
reproduce llama.cpp's chat template or tokenizer. Phase 0 correlates
structural regions (which hash changed), not exact token boundaries; as an
optional one-off during the attribution run, the boundary can be measured
exactly via llama-server's `/apply-template` + `/tokenize` endpoints.
Something early in the prompt changes on every call.

Not proven: which transform causes it. The log line previously used as
evidence (`Adaptive sliding window: trimmed old conversation pairs`) fires
whenever any messages were removed — `pre_collapse_len` is captured before
age-based tool collapse, so the counter conflates collapse and window trim
and says nothing about whether `keep_from` moved. Candidate causes, none yet
attributed:

- positional `keep_from` shifting as the event-fetch window slides
- `identity_preserve_indices` content matches drifting in and out of the
  fetched history (they bypass `keep_from`)
- pre-boundary content mutation without `keep_from` movement: Prior-1 tool
  result summarization, old-assistant truncation
  (`MAX_OLD_ASSISTANT_CONTENT_CHARS`), and repeated tool-error collapse.
  (The execution checkpoint is appended after the boundary and cannot flip
  `prefix_hash_pre_boundary`; it is tracked as tail growth, not listed here.)
- the `[Current Task]` boundary marker moving between turns
- session-summary churn at message index one
- tool-schema refitting changing serialized tool definitions
- **message-zero (system prompt) churn**: message zero embeds a per-minute
  UTC timestamp and an injected session-context block
  (`system_prompt.rs:757`), so `prefix_hash_system` changes at least once a
  minute regardless of the history region. This is a prerequisite, not a
  Phase 1 target: the anchor pins the history trim and cannot stabilize
  message zero. The cache-stable-system-prompt work must land first, and
  Phase 0 must confirm `prefix_hash_system` is stable across the evaluated
  calls before the anchor can extend reuse past it.

This design therefore has two phases. Phase 1 ships only after Phase 0
attributes the churn, and only the parts that the evidence supports.

## Phase 0 — Observability

Add build-phase and provider-call instrumentation behind normal
`info!`/`debug!` logging:

1. **Provider-call prefix fingerprint** (`info!`), once per LLM phase,
   computed in `llm_phase` after its security-message injection and
   force-text tool selection, immediately before `call_llm_with_recovery`.
   This is the final payload for the normal successful primary attempt, not
   a mid-build approximation. The boundary is the last user message whose
   content equals the active `user_text`; final `Vec<Value>` messages do not
   carry persisted message ids or `turn_id`, so the provider-call
   fingerprint does not pretend those fields are available. Fields:
   - `prefix_hash_system` — hash of message zero only (must be constant
     after the cache-stable system-prompt work)
   - `prefix_hash_pre_boundary` — hash of the complete message objects in
     `[1..boundary)`, including `tool_calls`, `name`, and `tool_call_id`; the
     current interaction's tool chain is excluded so normal tail growth does
     not flip the hash
   - `boundary_pos` and total message count
   - `tool_defs_hash` — hash of the actual effective tool definitions passed
     to the normal provider attempt (empty in force-text mode)
   - `force_text` — boolean tag; force-text iterations empty the tool set
     and change the server-side prompt shape, so attribution analyzes them
     separately instead of mistaking them for schema-refit churn
   - `session_summary_hash`, emitted separately to make index-one churn
     immediately visible
   Hashes use SHA-256 (already a dependency) over canonical JSON with object
   keys recursively sorted. Hashes never include raw message content in logs.
2. **Stage hashes** (`debug!`) of the pre-boundary region after each
   mutating stage, in actual pipeline order: age-based collapse, window
   trim, duplicate removal, JSON conversion (which performs Prior-1 tool
   summarization and old-assistant truncation), `[Current Task]` marker
   insertion, repeated tool-error collapse, history fitting
   (`fit_messages_with_source_quotas`), session-summary insertion, and
   execution checkpoint insertion (the checkpoint sits after the boundary —
   its stage hash tracks tail growth for completeness, never pre-boundary
   attribution) — each instrumented separately so a
   stable `keep_from` with an unstable `prefix_hash_pre_boundary` is
   attributed to content mutation, not misread as window trim. The marker
   is inserted before history fitting and the summary after it; the stage
   list reflects that. Note there are **two distinct summary insertion
   points**: history fitting itself can insert a
   `[Conversation summary: …]` near index one during fitting
   (`context_window.rs:301`), and message build inserts `[Session Summary]`
   at index one after system-prompt insertion
   (`message_build_phase.rs:1002`). Instrument both; hooking only the later
   one mis-attributes fitting-stage summary churn. Intermediate stages use prompt-equivalent complete
   message objects, not role-and-content-only projections.
   Mutators that run *after* message build — empty-response recovery,
   message-zero system insertion, and the `llm_phase` security-message
   injection / force-text tool selection — are not separately stage-hashed;
   they are reflected in the final provider-call fingerprint (item 1). If a
   break is observed there with all build stage hashes stable, it is
   attributed to this post-build region by elimination, and the `force_text`
   tag disambiguates the force-text case. Add a dedicated hash around the
   security-injection step only if elimination proves insufficient during
   the run.
3. **Window-decision log** (`info!`) in message build, where persisted
   `Message` metadata is still available. Fields: current `turn_id`,
   boundary message id, oldest fetched persisted message id, oldest kept
   persisted message id, `keep_from`, identity-preserve bypass count,
   `history_limit`
   (`40.max(iteration*3)` capped at 120, `message_build_phase.rs:224`),
   fetched message count, and whether the current-user-message injection or
   the safe-collapse warn path fired — fetch-window slide is a prime
   suspect and this ties `keep_from` movement to fetch mechanics directly.
4. **Split the misleading log**: report age-based collapse and window trim
   as separate counters, and log `keep_from` movement explicitly
   (`old_keep_from`, `new_keep_from`, oldest-kept message id).
5. **Attribution run**: a real multi-iteration task plus a multi-turn
   session, executed with a single active session and no concurrent
   sessions, so daemon-log and `scripts/cache-eval.sh` entries join by
   order (llama-server runs `--parallel 1`, serializing its log; the
   server's `task` ids are internal and never used as join keys). Filter
   daemon logs by `session_id` before joining. The run is valid only if the
   primary provider succeeds without retries or cascade fallback; either
   condition can produce multiple server requests for one LLM phase and
   invalidates an order-only join. Each turn in the run must use distinct
   user-message text — the boundary is matched by content, and duplicate
   `user_text` across turns mis-identifies it and pollutes the hashes.

Phase 0 exit criterion. All three conditions must hold to proceed to Phase 1
anchor work:

1. **System-prompt stability gate.** `prefix_hash_system` is constant across
   the calls being evaluated. If it churns (per-minute timestamp,
   session-context injection), message zero breaks the prefix before the
   history region and the anchor cannot help — fix system-prompt churn first
   and re-run attribution.
2. **Cause attribution.** Attributed `keep_from` movement *alone* accounts
   for at least half of the large re-evaluations in the attribution run (or
   the majority of blocks where `reused_prefix` sits at the system-prompt
   boundary). Identity-preserve drift does **not** count toward this
   threshold: those entries bypass `keep_from` by design and Phase 1 does
   not fix them (see anchor semantics below), so attributing the gate to
   drift would authorize an anchor that does not address the measured cause.
   Drift is measured separately and may motivate a distinct follow-up.
3. **Sample floor.** The run observed at least 20 large requests (evaluated
   ≥ 20% of prompt); below that the attribution percentage is not
   statistically meaningful and the run must be extended.

If the dominant cause is something else (identity drift, marker, summary,
content mutation, schema refit), design for that instead — do not ship the
anchor on spec faith.

## Phase 1 — Per-session window anchor (conditional on Phase 0)

Pin the trim boundary to a per-session anchor: the `Message.id` of the
oldest normally retained pair-starting user message. Identity-preserve
bypasses may still include older messages and are logged separately. The
anchor persists across iterations and across user turns in the same session.
It moves only when forced, and cuts deep when it does (hysteresis).

State: `window_anchors: Arc<RwLock<HashMap<String, AnchorState>>>` on
`Agent`, keyed by session id. `AnchorState` holds:

- the anchor message id — always the persisted id of a **user-role message
  that begins a pair** (the `keep_from` position is drawn from
  `old_user_positions`, so this holds by construction); synthetic or
  non-persisted messages (injected current-user copies, pinned memories)
  are never eligible — if the candidate id is not a persisted history
  message, skip anchoring for that build and behave exactly as today
- `cap_tokens = available_budget * WINDOW_ENTRY_BUDGET_PCT / 100` computed
  at establishment time, compared against the skeleton token sum of the
  anchored pairs (token sum, never pair count)
- validity fields: the model name, a hash of the **base system prompt**
  (the stable configuration-derived prompt only — explicitly excluding the
  per-minute timestamp, session-context block, and any other per-turn
  section; hashing the fully rendered dynamic prompt would mismatch every
  turn and force a `config_changed` re-anchor on every build), a
  hash of the sorted tool-name set, a hash of pinned-memory ids and contents,
  the enforced policy context budget, and the **resolved total model context
  budget** (the `model_context_budget` result — a per-model budget or
  default-budget config change with an unchanged model name must also
  invalidate the snapshot) — if any differ at build time, the snapshot is
  stale (selected-model change, dynamic MCP tool injection, pinned-memory
  change, policy change, or config change); re-anchor with
  `reason="config_changed"` **at the normal 30% entry rule**, consistent
  with loss and establishment — the deep 15% cut is reserved for genuine
  growth pressure. Request-time cascade fallback occurs after message build
  and is not treated as an anchor-state model change. A changed system
  prompt, tool set, or pinned-memory prefix breaks the serialized prefix
  anyway, so re-anchoring adds no further cache loss.
- `last_used_at`, used for opportunistic cleanup of entries not touched for
  more than two hours.

In-memory only — cleared per session by the idle-gap reset (>2h). Phase 1
adds explicit map eviction to `Agent::clear_session`
(`runtime/models.rs:131`), which currently clears persistent state only.
Anchor-map writes opportunistically prune entries whose `last_used_at` is
older than two hours. Anchors are lost on daemon restart; a daemon-only
restart against a still-warm llama-server therefore causes one accepted
re-establishment break. Spawned sub-agents run under distinct session ids
and get independent anchors. The build phase reaches the map through
`AgentServices`; `MessageBuildCtx` is unchanged.

Build-phase flow changes the `keep_from` computation and adds an
anchor-preservation contract to final context fitting. Age-based collapse,
duplicate removal, and the idle-gap reset are otherwise unchanged:

1. **Anchored path.** If the session has an anchor, its validity fields
   match, and its message id appears in fetched history before the
   boundary, `keep_from` is that position. Estimate the kept old pairs with
   the same skeleton arithmetic used today and compare against
   `cap_tokens` from the **snapshot stored with the anchor** — not a
   freshly computed budget — so tool-schema refits cannot trigger false
   overflows. If within the cap, use the anchor as-is, with no trim log
   output.
2. **Re-anchor on overflow.** If the anchored pairs exceed the snapshot
   cap, recompute the window with a 15% target — a deep cut that frees
   headroom for many iterations of growth — store the new anchor and cap
   snapshot, and log `reason="overflow"`. Exceeding the structural cap
   (`WINDOW_MAX_ANCHORED_PAIRS`) triggers the same 15% re-anchor with
   `reason="structural_cap"`.
3. **Re-anchor on loss.** If the anchor id is absent from the fetch or sits
   at/after the boundary, recompute at the normal 30% rule, store, and log
   `reason="anchor_lost"`. The first build for a session does the same with
   `reason="established"`.
4. **Degenerate fallback.** If no pairs fit the target, the window is zero
   and `keep_from` is the boundary — today's existing behavior. Remove any
   existing anchor and do not create a new one; the current-boundary message
   is never used as an anchor. No loop, no panic.
5. **Anchor-aware history fitting.** History fitting
   (`fit_messages_with_source_quotas` — distinct from the final tool-defs
   refit, which never drops messages) must not silently remove messages
   from the anchored pre-boundary region. If the current
   effective message budget (including policy enforcement and current tool
   definitions) cannot preserve that region, report
   `reason="budget_pressure"`, recompute once at the 15% target, and rerun
   fitting. If the deep-cut result still cannot fit, fall back to the
   existing fitter without an anchor for that build **and remove the stored
   anchor** — otherwise the next iteration repeats the same failed retry.
   This is the only permitted one-build retry; it prevents tool-schema
   growth or policy-budget shrink from silently defeating the stored anchor.

   *Implementation note — protected-range mapping (riskiest step).* The
   converted JSON messages carry no persisted id (`message_build_phase.rs:725`
   emits `{role, content, ...}` only), so there is nothing to "resolve"
   against after conversion. The build must therefore carry an explicit
   provenance channel: a parallel `Vec<Option<String>>` of source message
   ids (`Message.id` is a String UUID) aligned 1:1 with the message vec
   (`None` for synthetic/injected entries — marker, fitting-stage summary,
   injected current-user copy), maintained in lockstep through every
   transform that inserts, removes, or reorders messages **up to history
   fitting**. The range is derived **immediately before
   `fit_messages_with_source_quotas`** — the only stage that drops history
   messages under budget pressure — by scanning the sidecar for the slot
   whose id equals `oldest_kept_msg_id`, and passed to the fitter as an
   explicit `[protect_from..boundary_pos)` range; the fitter never infers
   it. Resolving "after all transforms" is impossible: system-prompt
   insertion, the build's summary insertion, the checkpoint, and directives
   all run *after* fitting — they shift indices but only insert, never
   remove pre-boundary history, so the sidecar does not need to extend past
   the fitting call. A unit test must assert the sidecar stays aligned
   (same length, same order) from fetch through the fitting call;
   misalignment silently protects the wrong range and is the primary
   failure mode of this step.

Re-anchor logs, including `budget_pressure`, carry structured fields:
`session_id`, `reason`,
`old_anchor_id`, `new_anchor_id`, `old_keep_from`, `new_keep_from`,
`anchored_pair_tokens`, `cap_tokens`.

Constants in `sliding_window.rs`: `WINDOW_ENTRY_BUDGET_PCT = 30` and
`WINDOW_REANCHOR_BUDGET_PCT = 15`, replacing the hardcoded 30 in
`calculate_window_size`, which gains a target-percent parameter. The 15%
value is an initial hysteresis target, not a measured optimum; Phase 0
growth data may tune it before Phase 1 implementation.

### Pair caps

Today's hard 5-pair cap would force a re-anchor every turn once history
exceeds five pairs, defeating cross-turn pinning. The anchored path instead
enforces the token cap plus a generous structural cap
(`WINDOW_MAX_ANCHORED_PAIRS = 40` — roughly 8x the old hard cap: large
enough that cross-turn pinning is not throttled, small enough to bound the
risk of the content-only token estimator under-counting many tiny pairs,
whose role/template framing it ignores). Re-anchor paths still apply the
existing `min(5, fit)` rule when computing a new window, matching today's
behavior at establishment time. Consequence to accept: an anchored window
that grew toward 40 pairs collapses to ≤5 on the re-anchor — one large cache
break at re-anchor time, which is the intended hysteresis (rare deep cut)
rather than the per-turn churn it replaces.

### Anchor semantics under downstream transforms

The anchor primarily pins where the window trim cuts. Downstream transforms
— duplicate removal, message-order fixups, empty-response recovery,
fresh-context isolation, and the `[Current Task]` marker — may still alter
the serialized prefix without changing that trim decision; they are accepted
cache boundaries that the Phase 0 fingerprint log makes attributable. History
fitting (`fit_messages_with_source_quotas`) is different: it may alter
unanchored tail content, but it must emit `budget_pressure` rather than
silently remove any message from the anchored pre-boundary region. In particular:

- `identity_preserve_indices` entries below `keep_from` bypass the trim and
  can drift with the fetch window. Phase 0 measures how often; if it is a
  dominant break source, a follow-up may disable the bypass below the
  anchor (identity facts belong in pinned memories), but that is out of
  scope here.
- The `[Current Task]` marker moves to the newest user message each turn,
  breaking the prefix near the tail of the previous turn. That break is
  bounded (hundreds of tokens re-evaluated, not the whole history) and
  acceptable; cross-turn anchoring still prevents the expensive head
  breaks when the session summary and pinned prefix are unchanged. The
  cross-turn benefit is therefore conditional: "stable historical region,
  bounded tail break," not byte-identical prefix extension. Summary or
  pinned-memory churn remains an earlier cache boundary and is measured
  separately.

## Scope

- Phase 0: prefix fingerprint logging (region sub-hashes, fetch-mechanics
  fields, stage hashes at debug level), split trim counters, controlled
  single-session attribution run.
- Phase 1 (conditional): per-session anchor with cap snapshot, deep
  re-anchor on overflow, structural cap 40 on the anchored path, anchor
  preservation during final fitting (with the provenance sidecar), and
  anchor cleanup on idle gap and session clear.
- Prerequisite, out of scope here: stabilizing message zero
  (cache-stable-system-prompt work). Phase 1 does not begin until
  `prefix_hash_system` is confirmed stable in the Phase 0 run.
- Leave history fetching, age-based collapse, duplicate removal, session
  summaries, tool-schema fitting, and the `[Current Task]` marker unchanged.

## Tests

A shared helper (new module `src/agent/loop/prefix_fingerprint.rs`) makes the
hash tests deterministic and is the same code the
provider-call fingerprint uses:
`canonical_prefix(messages, user_text) ->
(hash_system, hash_pre_boundary, boundary_pos)`. It canonicalizes complete
JSON message objects by recursively sorting object keys, hashes message zero
alone, then hashes messages `[1..boundary)`, where the boundary is the last
user message matching `user_text`. Persisted ids remain in the separate
window-decision diagnostics. The execution checkpoint and current
interaction's tool chain sit at or after the boundary, so ordinary
per-iteration tail growth does not flip either prefix hash.

Phase 0:
- Fingerprint log emits the `canonical_prefix` fields (which include
  `hash_system`) plus `tool_defs_hash` and `session_summary_hash`; hashes
  are identical across two provider attempts whose inputs are identical.
- Complete-message hashing changes when a pre-boundary `tool_calls`, `name`,
  or `tool_call_id` field changes even if role and content are unchanged.
- Window-decision diagnostics emit persisted boundary/oldest-kept ids without
  requiring those internal fields to be sent to the provider.
- Force-text iterations emit `force_text=true` with an empty
  `tool_defs_hash`; a normal iteration emits `force_text=false` — the tag is
  what lets attribution analyze these separately.

Phase 1:
- Multi-iteration, same turn: builds at iteration 1 and 2 with tool results
  appended and identical `user_text` produce the same oldest-kept message id
  and identical `canonical_prefix` hash (not just equal `keep_from`).
- Cross-turn: turn N+1 build keeps the anchor from turn N when within cap
  (validates the 5-pair-cap bypass).
- Overflow: anchored pairs exceeding the cap snapshot re-anchor to the 15%
  target and update the stored anchor.
- Tool-budget shrink without message growth does NOT re-anchor (cap
  snapshot makes overflow independent of refits) unless final fitting would
  remove an anchored pre-boundary message, in which case
  `reason="budget_pressure"` re-anchors once.
- Validity-field mismatch (model name, tool-set hash, pinned-memory hash,
  enforced policy budget, **or resolved `model_context_budget`** changed)
  re-anchors with `reason="config_changed"` at the 30% rule (not the 15%
  deep cut). A per-model or default-budget config change with an unchanged
  model name still invalidates the snapshot.
- Provenance sidecar stays aligned (equal length, same order) with the
  message vec across the full transform pipeline; the protected range
  derived from `oldest_kept_msg_id` covers exactly the anchored
  pre-boundary region.
- Event-window slide, both cases: (a) anchor id still fetched but at a
  shifted index — resolves by id, `keep_from` holds at that message;
  (b) oldest pairs dropped from fetch while the anchor id remains — the
  anchored path still holds; only when the id itself vanishes does
  `reason="anchor_lost"` fire.
- Anchor lost: stored id absent from fetch re-anchors at 30% without panic.
- Exceeding `WINDOW_MAX_ANCHORED_PAIRS` re-anchors at the 15% target with
  `reason="structural_cap"`.
- A failed `budget_pressure` retry (deep cut still does not fit) falls back
  unanchored AND removes the stored anchor — the next build establishes
  fresh instead of repeating the failed retry.
- Zero-pair establishment stores no anchor; zero-pair fallback removes an
  existing anchor.
- Idle gap clears the stored anchor map entry (extend the existing
  idle-gap test to assert map state).
- `Agent::clear_session` removes the session's anchor, and opportunistic TTL
  pruning removes untouched entries older than two hours.
- Final fitting either preserves every anchored pre-boundary message or
  emits `budget_pressure`; it never silently drops part of the anchored
  region.
- No window-trim log line on the anchored path.
- Update existing `calculate_window_size` tests for the target-percent
  parameter; existing message-build and system-prompt tests pass.

## Evaluation

Phase 0 attribution is the primary evaluation: every large re-evaluation in
the controlled single-session run must be matched to a named cause via the
fingerprint logs. `scripts/cache-eval.sh` supplies the llama.cpp side; its
`task` ids are server-internal, so the join is by log order within the
isolated run (single active session, `--parallel 1` server), with daemon
logs filtered by `session_id` first.

Phase 1 success, measured the same way after implementation:
- No blocks of consecutive large requests with `reused_prefix` stuck at the
  system-prompt size attributable to window-trim movement.
- The large-request cache-hit rate (evaluated < 20% of prompt) is at least
  70% and at least 15 percentage points above the pre-change baseline
  (98 of 182, or 53.8%, in the original measurement run). Record the model,
  context size, tool set, server flags, and sample count with both baseline
  and post-change results.

Known remaining cache boundaries, expected and acceptable: session-summary
churn at index one, tool-schema refitting, `[Current Task]` marker movement
(bounded tail break), identity-preserve drift (measured in Phase 0), and
forced re-anchors themselves.
