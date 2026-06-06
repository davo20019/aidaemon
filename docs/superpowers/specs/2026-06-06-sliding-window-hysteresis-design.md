# Sliding-Window Hysteresis Design

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
that equality has not been independently measured against the provider's
chat template; Phase 0 confirms the exact boundary by token-counting the
serialized prefix. Something early in the prompt changes on every call.

Not proven: which transform causes it. The log line previously used as
evidence (`Adaptive sliding window: trimmed old conversation pairs`) fires
whenever any messages were removed — `pre_collapse_len` is captured before
age-based tool collapse, so the counter conflates collapse and window trim
and says nothing about whether `keep_from` moved. Candidate causes, none yet
attributed:

- positional `keep_from` shifting as the event-fetch window slides
- `identity_preserve_indices` content matches drifting in and out of the
  fetched history (they bypass `keep_from`)
- the `[Current Task]` boundary marker moving between turns
- session-summary churn at message index one
- tool-schema refitting changing serialized tool definitions

This design therefore has two phases. Phase 1 ships only after Phase 0
attributes the churn, and only the parts that the evidence supports.

## Phase 0 — Observability

Add build-phase instrumentation, behind normal `info!`/`debug!` logging:

1. **Prefix fingerprint log** (`info!`), once per build, computed on the
   final `messages` `Vec<Value>` immediately before the LLM call — not
   mid-pipeline, so summary insertion, the `[Current Task]` marker,
   duplicate removal, and final fitting are all reflected. The boundary is
   the position of the current user message (matched by `turn_id`, falling
   back to content). Fields:
   - `prefix_hash_system` — hash of message zero only (must be constant
     after the cache-stable system-prompt work)
   - `prefix_hash_pre_boundary` — hash of role + content for messages
     `[1..boundary)`, the expensive region; the current interaction's tool
     chain is excluded so normal tail growth does not flip the hash
   - `boundary_pos`, `boundary_msg_id`, oldest included persisted message
     id, `keep_from`, count of identity-preserve bypasses below `keep_from`
   - `tool_defs_hash` — hashed separately, since tool definitions serialize
     into the prompt but live outside `messages`
2. **Stage hashes** (`debug!`) of the pre-boundary region after each
   mutating stage — age-based collapse, window trim, duplicate removal,
   marker + summary insertion, final context fitting — to localize which
   transform changed the prefix once the final fingerprint detects a break.
3. **Fetch mechanics fields** on the fingerprint log: `history_limit`
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
   daemon logs by `session_id` before joining.

Phase 0 exit criterion: proceed to Phase 1 anchor work only if attributed
`keep_from` movement and/or identity-preserve drift account for at least
half of the large re-evaluations in the attribution run (or the majority of
blocks where `reused_prefix` sits at the measured system-prompt boundary).
If the dominant cause is something else (marker, summary, schema refit),
design for that instead — do not ship the anchor on spec faith.

## Phase 1 — Per-session window anchor (conditional on Phase 0)

Pin the trim boundary to a per-session anchor: the `Message.id` of the
oldest kept history message. The anchor persists across iterations and
across user turns in the same session. It moves only when forced, and cuts
deep when it does (hysteresis).

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
- validity fields: the model name, a hash of the base system prompt, and a
  hash of the sorted tool-name set — if any differ at build time, the
  snapshot is stale (model switch via router fallback, dynamic MCP tool
  injection, config change); re-anchor with `reason="config_changed"`. A
  changed tool set or system prompt breaks the serialized prefix anyway, so
  this re-anchor costs nothing extra.

In-memory only — cleared per session by the idle-gap reset (>2h), by
`Agent::clear_session` (`runtime/models.rs:131`), and lost on restart (the
server-side KV cache is stale then anyway). Spawned sub-agents run under
distinct session ids and get independent anchors. The build phase reaches
the map through `AgentServices`; `MessageBuildCtx` is unchanged.

Build-phase flow (replaces only the `keep_from` computation; age-based
collapse, duplicate removal, and the idle-gap reset are unchanged):

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
   snapshot, and log `reason="overflow"`.
3. **Re-anchor on loss.** If the anchor id is absent from the fetch or sits
   at/after the boundary, recompute at the normal 30% rule, store, and log
   `reason="anchor_lost"`. The first build for a session does the same with
   `reason="established"`.
4. **Degenerate fallback.** If no pairs fit the target, the window is zero
   and `keep_from` is the boundary — today's existing behavior; no loop, no
   panic.

Re-anchor logs carry structured fields: `session_id`, `reason`,
`old_anchor_id`, `new_anchor_id`, `old_keep_from`, `new_keep_from`,
`anchored_pair_tokens`, `cap_tokens`.

Constants in `sliding_window.rs`: `WINDOW_ENTRY_BUDGET_PCT = 30` and
`WINDOW_REANCHOR_BUDGET_PCT = 15`, replacing the hardcoded 30 in
`calculate_window_size`, which gains a target-percent parameter.

### Pair caps

Today's hard 5-pair cap would force a re-anchor every turn once history
exceeds five pairs, defeating cross-turn pinning. The anchored path instead
enforces the token cap plus a generous structural cap
(`WINDOW_MAX_ANCHORED_PAIRS = 40` — roughly 8x the old hard cap: large
enough that cross-turn pinning is not throttled, small enough to bound the
risk of the content-only token estimator under-counting many tiny pairs,
whose role/template framing it ignores). Re-anchor paths still apply the
existing `min(5, fit)` rule when computing a new window, matching today's
behavior at establishment time.

### Anchor semantics under downstream transforms

The anchor pins exactly one decision: where the window trim cuts. Downstream
transforms — duplicate removal, message-order fixups, final context fitting,
empty-response recovery, fresh-context isolation, the `[Current Task]`
marker — may still alter the serialized prefix. They do not invalidate the
anchor; they are accepted cache boundaries that the Phase 0 fingerprint log
makes attributable. In particular:

- `identity_preserve_indices` entries below `keep_from` bypass the trim and
  can drift with the fetch window. Phase 0 measures how often; if it is a
  dominant break source, a follow-up may disable the bypass below the
  anchor (identity facts belong in pinned memories), but that is out of
  scope here.
- The `[Current Task]` marker moves to the newest user message each turn,
  breaking the prefix near the tail of the previous turn. That break is
  bounded (hundreds of tokens re-evaluated, not the whole history) and
  acceptable; cross-turn anchoring still prevents the expensive head
  breaks. The cross-turn benefit is "stable head, bounded tail break", not
  byte-identical prefix extension.

## Scope

- Phase 0: prefix fingerprint logging (region sub-hashes, fetch-mechanics
  fields, stage hashes at debug level), split trim counters, controlled
  single-session attribution run.
- Phase 1 (conditional): per-session anchor with cap snapshot, deep
  re-anchor on overflow, structural cap 40 on the anchored path, anchor
  cleanup on idle gap and session clear.
- Leave history fetching, age-based collapse, duplicate removal, session
  summaries, tool-schema fitting, and the `[Current Task]` marker unchanged.

## Tests

A shared helper makes the hash tests deterministic and is the same code the
fingerprint log uses: `canonical_prefix(messages, user_text) ->
(hash_system, hash_pre_boundary, oldest_msg_id, boundary_pos)` — message
zero hashed alone, then role + content for messages `[1..boundary)`, where
the boundary is the current user message. The execution checkpoint and the
current interaction's tool chain sit at or after that boundary, so
per-iteration tail growth never flips either hash.

Phase 0:
- Fingerprint log emits the `canonical_prefix` fields plus
  `prefix_hash_system` and `tool_defs_hash`; hashes are identical across
  two builds whose inputs are identical.

Phase 1:
- Multi-iteration, same turn: builds at iteration 1 and 2 with tool results
  appended and identical `user_text` produce the same oldest-kept message id
  and identical `canonical_prefix` hash (not just equal `keep_from`).
- Cross-turn: turn N+1 build keeps the anchor from turn N when within cap
  (validates the 5-pair-cap bypass).
- Overflow: anchored pairs exceeding the cap snapshot re-anchor to the 15%
  target and update the stored anchor.
- Tool-budget shrink without message growth does NOT re-anchor (cap
  snapshot makes overflow independent of refits).
- Validity-field mismatch (model name or tool-set hash changed) re-anchors
  with `reason="config_changed"`.
- Event-window slide, both cases: (a) anchor id still fetched but at a
  shifted index — resolves by id, `keep_from` holds at that message;
  (b) oldest pairs dropped from fetch while the anchor id remains — the
  anchored path still holds; only when the id itself vanishes does
  `reason="anchor_lost"` fire.
- Anchor lost: stored id absent from fetch re-anchors at 30% without panic.
- Idle gap clears the stored anchor map entry (extend the existing
  idle-gap test to assert map state).
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
- The large-request cache-hit rate (evaluated < 20% of prompt) rises
  clearly above the pre-change baseline (98 of 182 at time of writing).

Known remaining cache boundaries, expected and acceptable: session-summary
churn at index one, tool-schema refitting, `[Current Task]` marker movement
(bounded tail break), identity-preserve drift (measured in Phase 0), and
forced re-anchors themselves.
