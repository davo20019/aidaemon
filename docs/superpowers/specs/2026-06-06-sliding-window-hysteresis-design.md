# Sliding-Window Hysteresis Design

## Goal

Stop the adaptive sliding window from moving its trim boundary on every LLM
call. A moving boundary rewrites the head of the history region and breaks
llama.cpp prompt-cache reuse even though message zero is now stable (see
`2026-06-06-cache-stable-system-prompt-design.md`). Live measurement showed
blocks of consecutive agent-loop iterations re-evaluating ~13K tokens (~31s
each) with the reusable prefix stuck at the system-prompt size, correlated
with `Adaptive sliding window: trimmed old conversation pairs` firing every
iteration.

## Design

Pin the trim boundary to a per-session anchor: the `Message.id` of the oldest
kept history message. The anchor persists across iterations and across user
turns in the same session, so consecutive prompts extend the cached prefix
instead of rewriting it. The anchor only moves when it must, and when it
moves it cuts deep so the next move is far away (hysteresis).

State: a new `window_anchors: Arc<RwLock<HashMap<String, String>>>` field on
`Agent`, mapping session id to anchor message id. In-memory only — cleared
per session by the existing idle-gap reset (>2h) and lost on restart, when
the server-side KV cache is stale anyway. The build phase reaches it through
`AgentServices`; `MessageBuildCtx` is unchanged.

Build-phase flow (replaces only the `keep_from` computation in
`run_message_build_phase`; age-based collapse, duplicate removal, the
new-task aggressive trim, and the idle-gap reset are unchanged):

1. **Anchored path.** If the session has an anchor and its message id appears
   in fetched history before the boundary, `keep_from` is that position.
   Estimate the kept old pairs with the same skeleton arithmetic used today;
   if they fit within the 30% entry cap, use the anchor as-is. No log output
   on this path.
2. **Re-anchor on overflow.** If the anchored pairs exceed the 30% cap,
   recompute the window with a 15% target — a deep cut that frees enough
   headroom for many iterations of growth — store the new anchor, and log
   `reason="overflow"` with old and new positions.
3. **Re-anchor on loss.** If the anchor id is absent from the fetch (the
   event window slid past it) or sits at/after the boundary, recompute at
   the normal 30% rule, store, and log `reason="anchor_lost"`. The first
   build for a session does the same with `reason="established"`.

Constants in `sliding_window.rs`: `WINDOW_ENTRY_BUDGET_PCT = 30` and
`WINDOW_REANCHOR_BUDGET_PCT = 15`. `calculate_window_size` gains a
target-percent parameter.

### Deliberate behavior change: the 5-pair cap

The anchored path enforces only the token cap, not today's hard 5-pair cap.
Otherwise cross-turn pinning would force a re-anchor every turn once history
exceeds five pairs, defeating the feature. The pair cap still applies
whenever a new anchor is computed. The token cap is the real constraint;
pair count was a proxy for it.

The existing new-task aggressive trim (word-overlap check that keeps only
the most recent pair) runs upstream and takes precedence. When it fires, the
message set changes, the prefix breaks once — a legitimate cache boundary —
and the anchor re-establishes on the next build.

## Scope

- Pin `keep_from` to a per-session anchor message id, stored on `Agent`.
- Re-anchor deep (15%) on budget overflow; re-anchor normal (30%) on anchor
  loss or first build.
- Skip the 5-pair cap on the anchored path only.
- Clear the anchor on idle-gap reset.
- Leave history fetching, age-based collapse, duplicate removal, the
  new-task trim, session summaries, and tool-schema fitting unchanged.

## Tests

- Two consecutive builds with history growing only at the tail produce the
  same `keep_from` head: the old-pairs region of build two starts with the
  same message as build one.
- A build whose anchored pairs exceed the 30% cap re-anchors to the 15%
  target and updates the stored anchor.
- A build whose stored anchor id is absent from fetched history re-anchors
  at 30% without panicking.
- The existing idle-gap test still passes, and the idle-gap path clears the
  stored anchor.
- Existing message-build and system-prompt tests pass.

## Evaluation

Run `scripts/cache-eval.sh` before and after on a real multi-iteration task.
Success means:

- No blocks of consecutive large requests with `reused_prefix` stuck at the
  system-prompt size while `evaluated` stays in the tens of thousands.
- The large-request cache-hit rate (evaluated < 20% of prompt) rises clearly
  above the pre-change baseline (98 of 182 at time of writing).

Known remaining cache boundaries, expected and acceptable: session-summary
churn at message index one, tool-schema refitting under context pressure,
the new-task aggressive trim, and forced re-anchors themselves. The eval
should attribute breaks to these causes via the daemon log before treating
them as failures of the anchor.
