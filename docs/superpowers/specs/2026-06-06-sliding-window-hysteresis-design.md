# Sliding-Window Hysteresis Design

## Goal

Stop the history region of the prompt from being rewritten between LLM calls,
so llama.cpp prefix-cache reuse extends past the (now stable) system prompt.
Live measurement showed blocks of consecutive large requests re-evaluating
~13K tokens (~31s each) with the reusable prefix stuck at the system-prompt
size (`n_past = 9816`).

## What is proven and what is not

Proven (llama-server log): within the bad blocks, consecutive large prompts
share exactly the system-prompt prefix and diverge immediately after it.
Something in the history region changes on every call.

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

1. **Prefix fingerprint log**, once per build: `session_id`, `iteration`,
   oldest included persisted message id, `keep_from` position, count of
   identity-preserve bypasses below `keep_from`, and a hash of the
   serialized messages from index zero up to the current-interaction
   boundary. Identical hash across iterations = stable prefix; a changed
   hash plus the other fields identifies which region moved.
2. **Split the misleading log**: report age-based collapse and window trim
   as separate counters, and log `keep_from` movement explicitly
   (`old_keep_from`, `new_keep_from`, oldest-kept message id).
3. Re-run a real multi-iteration task and a multi-turn session; join the
   fingerprint logs with `scripts/cache-eval.sh` output by timestamp order
   to attribute each large re-evaluation to a cause.

Phase 0 exit criterion: each observed cache break in the test run is
attributed to a named cause. If `keep_from` movement or identity-preserve
drift is confirmed as a dominant cause, proceed to Phase 1. If the dominant
cause is something else (marker, summary, schema refit), design for that
instead — do not ship the anchor on spec faith.

## Phase 1 — Per-session window anchor (conditional on Phase 0)

Pin the trim boundary to a per-session anchor: the `Message.id` of the
oldest kept history message. The anchor persists across iterations and
across user turns in the same session. It moves only when forced, and cuts
deep when it does (hysteresis).

State: `window_anchors: Arc<RwLock<HashMap<String, AnchorState>>>` on
`Agent`, keyed by session id, where `AnchorState` holds the anchor message
id and a snapshot of the token cap computed when the anchor was set.
In-memory only — cleared per session by the idle-gap reset (>2h), on any
explicit session-clear path, and lost on restart (the server-side KV cache
is stale then anyway). Spawned sub-agents run under distinct session ids and
get independent anchors. The build phase reaches the map through
`AgentServices`; `MessageBuildCtx` is unchanged.

Build-phase flow (replaces only the `keep_from` computation; age-based
collapse, duplicate removal, and the idle-gap reset are unchanged):

1. **Anchored path.** If the session has an anchor and its message id
   appears in fetched history before the boundary, `keep_from` is that
   position. Estimate the kept old pairs with the same skeleton arithmetic
   used today and compare against the cap **snapshot stored with the
   anchor** — not a freshly computed budget — so tool-schema refits cannot
   trigger false overflows. If within the cap, use the anchor as-is, with
   no trim log output.
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
(`WINDOW_MAX_ANCHORED_PAIRS = 40`) as a guard against pathological tiny-pair
accumulation that the content-only token estimator under-counts. Re-anchor
paths still apply the existing `min(5, fit)` rule when computing a new
window, matching today's behavior at establishment time.

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

- Phase 0: prefix fingerprint logging, split trim counters, attribution run.
- Phase 1 (conditional): per-session anchor with cap snapshot, deep
  re-anchor on overflow, structural cap 40 on the anchored path, anchor
  cleanup on idle gap and session clear.
- Leave history fetching, age-based collapse, duplicate removal, session
  summaries, tool-schema fitting, and the `[Current Task]` marker unchanged.

## Tests

Phase 0:
- Fingerprint log emits oldest-kept message id and prefix hash; hash is
  identical across two builds whose inputs are identical.

Phase 1:
- Multi-iteration, same turn: builds at iteration 1 and 2 with tool results
  appended and identical `user_text` produce the same oldest-kept message id
  and identical serialized prefix up to the current-interaction boundary
  (hash comparison, not just `keep_from`).
- Cross-turn: turn N+1 build keeps the anchor from turn N when within cap
  (validates the 5-pair-cap bypass).
- Overflow: anchored pairs exceeding the cap snapshot re-anchor to the 15%
  target and update the stored anchor.
- Tool-budget shrink without message growth does NOT re-anchor (cap
  snapshot makes overflow independent of refits).
- Anchor survives event-window slide: anchor id still fetched but at a
  shifted index resolves by id, not position.
- Anchor lost: stored id absent from fetch re-anchors at 30% without panic.
- Idle gap clears the stored anchor map entry (extend the existing
  idle-gap test to assert map state).
- No window-trim log line on the anchored path.
- Update existing `calculate_window_size` tests for the target-percent
  parameter; existing message-build and system-prompt tests pass.

## Evaluation

Phase 0 attribution is the primary evaluation: every large re-evaluation in
a real multi-iteration, multi-turn run must be matched to a named cause via
the fingerprint logs. `scripts/cache-eval.sh` supplies the llama.cpp side;
its `task` ids are server-internal, so the join is by timestamp order
against the daemon log, not by id.

Phase 1 success, measured the same way after implementation:
- No blocks of consecutive large requests with `reused_prefix` stuck at the
  system-prompt size attributable to window-trim movement.
- The large-request cache-hit rate (evaluated < 20% of prompt) rises
  clearly above the pre-change baseline (98 of 182 at time of writing).

Known remaining cache boundaries, expected and acceptable: session-summary
churn at index one, tool-schema refitting, `[Current Task]` marker movement
(bounded tail break), identity-preserve drift (measured in Phase 0), and
forced re-anchors themselves.
