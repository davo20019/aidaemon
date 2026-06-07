# Cross-Turn Prefix Stability (Phase 1)

**Status:** Draft — pending review
**Depends on:** Phase 0 observability (shipped; see
`2026-06-06-sliding-window-hysteresis-design.md`)
**Supersedes:** the within-task `keep_from` anchor from the hysteresis spec's
original Phase 1. Turn-anchored fetch and archived-render stability provide
within-task stability structurally, and the message-zero freeze is the
prerequisite that spec's §"dormant cross-turn persistence" named for
cross-turn cache extension.

## Why this phase, and not the original anchor

The Phase 0 attribution run (2026-06-06, 10 turns, gemma-4-26b via local
llama-server, single session) measured the cause of prefix-cache breaks:

| Cause | Attributable breaks | Cost |
|---|---|---|
| Cross-turn system-prompt churn | 9/15 (60%) | full ~22–24k re-prefill at turn start, ~45–50 s each |
| `age_collapse` content mutation | 3/15 | mid-task full or multi-k re-evals in long tasks |
| Tool-def refit | 2/15 | fixed separately (force-text now retains tool defs) |
| `keep_from` movement alone | 1/15 (7%) | the original Phase 1 target |

Within-task reuse is already healthy (100–800 token re-evals). The prize is
the turn-start re-prefill, and the secondary win is killing the fetch-window
slide that mutates collapse inputs mid-task.

## Requirements (settled during design review)

1. Scope: both pillars together — **A:** message-zero freeze/split,
   **B:** monotonic turn rendering with turn-anchored fetch.
2. Volatile per-message context moves to a **task context tail** message;
   the stable core keeps no memory-derived content.
3. Core recompilation is **event-driven** (content-hash invalidation), never
   time-driven.
4. Rendering is pure over `(turn contents, render policy/version)` — **not**
   over age class. A turn transitions once (Current → Archived); the archived
   form never degrades further. Deep whole-turn eviction replaces the
   Prior-1/Prior-2 ladder as the budget mechanism.
5. The render cache is an optimization and an assertion mechanism, not the
   source of truth. Debug/test builds re-render and assert byte equality.
6. Exit criteria are measured with the Phase 0 instrumentation, which stays
   untouched. Absolute thresholds are set **after** the first instrumented
   re-run (see §Exit criteria).

## Payload layout

Per task (turn) N, iteration 1:

```
[0]    system  STABLE CORE      identity, persona, behavioral rules,
                                orchestrator/direct-mode guidance,
                                Available Specialists, channel/privacy rule
                                set, skills availability catalog (names +
                                one-line descriptions only).
                                Bytes change only on core-input change.
[1..k] history ARCHIVED TURNS   byte-stable renderings, whole turns, from
                                the eviction anchor through turn N-1.
[k+1]  system  TASK CONTEXT     [Current Date & Time], session context
       (tail)  TAIL             block, session summary (moves here from
                                message index 1), query-ranked facts and
                                procedures, matched skill content,
                                current-speaker/people context, resume
                                checkpoint. Compiled once per task.
[k+2..] CURRENT TURN            current user message, then append-only
                                assistant/tool exchanges and system
                                directives as iterations proceed.
```

The tail sits **before** the current turn. Every within-task iteration is a
strict prefix extension (appends only). Nothing about turn N's payload is a
prefix promise for turn N+1 at or after the tail position — see the
invariants below for what is promised.

### Cross-turn prefix invariants

Turn N final payload:  `core → archived[..N-1] → tail[N] → current[N]`
Turn N+1 first payload: `core → archived[..N-1, N] → tail[N+1] → current[N+1]`

Divergence necessarily occurs where `tail[N]` is replaced by `archived[N]`.
The promised invariants are:

1. **Core + archived turns through N-1 are byte-identical** between turn N's
   final payload and turn N+1's first payload (absent a logged core
   invalidation or eviction).
2. **Turn N's archived rendering is byte-stable from turn N+1 onward** — it
   never changes again (absent a content_fp change from a late write, which
   is logged).
3. **Within a task, each iteration's payload is a strict prefix extension**
   of the previous iteration's payload, **absent a logged prefix mutation**.
   Three retained recovery mechanisms legitimately rewrite current-turn
   bytes and are explicitly excluded from the invariant: repeated-tool-error
   collapse (`collapse_repeated_tool_errors`), history-fitting overflow
   trimming, and the empty-response retry rebuild. Each must emit
   `Prefix mutation reason=<mechanism>` when it fires so every within-task
   re-evaluation is attributable. Tests assert strict prefix extension on
   mutation-free paths AND assert the log line fires on each mutator path.
   (History-fitting overflow should become rare: whole-turn eviction at the
   60% low-water mark leaves headroom that the per-iteration fitter
   previously had to create.)
4. **Turn-start re-evaluation is bounded by the prior task's tail + current
   region** (archived[N] + tail[N+1] + new user message), not by an assumed
   constant. The bound is measured, then a threshold is set.

## Pillar A — stable core / task context tail

### Split

`build_system_prompt_for_message` splits into:

- `build_core_prompt(core_inputs) -> (component_hashes, aggregate_hash, bytes)`
- `build_context_tail(per_task_inputs) -> bytes`

### Core inputs and hashing

Core inputs are the **actual content inputs**, not proxies (a config revision
counter both over-invalidates and misses externally sourced changes):

- base prompt template (per role/mode),
- canonicalized tool roster (names + schemas, sorted by name),
- canonicalized skills availability catalog (name + description + enabled,
  sorted; matched skill *content* is tail-side),
- canonicalized specialist registry,
- channel/privacy rule set for the session's visibility class,
- persona/identity configuration.

Each component is canonicalized (stable ordering for any unordered
collection) and hashed separately; the aggregate hash is the hash of the
component hashes. On mismatch, the changed component is named factually in
the log line: `Core prompt invalidated component=<name>`.

**Anti-pattern — per-turn dynamic tool gating.** Selecting the tool roster
per message (MCP-trigger style, or any query-dependent gating) makes the
roster a per-turn input and invalidates the core every turn — the
`component=tool_roster` log would expose it, but it structurally defeats
Pillar A. The compatible shape is a **static-but-smaller roster** (Pillar C
slims descriptions; membership stays stable per session). Future payload
passes must not reintroduce per-turn roster variance.

`build_core_prompt` must be deterministic: no timestamps, no randomized or
map-iteration ordering, no environment-dependent formatting. This is
enforced by golden-byte tests and by the debug re-render assertion (§Pillar
B cache, same mechanism).

### Cache

`core_prompts: HashMap<session_id, (aggregate_hash, component_hashes, bytes)>`
on `Agent`. Per task bootstrap: recompute input hashes (cheap; no render),
reuse bytes on match, re-render and log on mismatch. In-memory only; after a
daemon restart the core is re-derived — deterministically, so if inputs are
unchanged the bytes (and therefore the server-side prefix cache) still match.

### Tail

Everything memory-derived or per-message lives here: the per-task timestamp
(unchanged freshness semantics — the prompt is already compiled once per
task today), session context block, session summary, query-ranked relevant
facts/procedures, matched skill content, people/current-speaker context,
resume checkpoint. Storing a fact, creating an episode, or matching a skill
never invalidates the core.

### Provider serialization requirement

The logical layout must survive into the final provider payload:

- `openai_compatible` (primary; llama.cpp): system messages pass through
  inline and in order. Mid-payload system messages are already exercised in
  production by the existing tail directives. Verify with a payload-shape
  test.
- `anthropic_native`, `google_genai`: these adapters extract system content
  into top-level parameters. The mapping for core + mid-payload system
  messages must be **defined and deterministic** (same logical messages →
  same serialized payload bytes). Cache wins on those providers are out of
  scope; deterministic mapping is not.

## Pillar B — turn-anchored fetch and monotonic rendering

### Canonical history source and turn identity (prerequisite)

The primary history source is the event store, and event hydration does not
carry turn identity today: `ConversationTurn::into_message` sets
`turn_id: None` explicitly (`src/events/conversation_turn.rs:190`), so a
turn-keyed fetch cannot be built on hydrated events as they stand. This
phase therefore requires:

- `turn_id` added to the canonical conversation event payloads
  (user message, assistant response, tool call/result) and propagated
  through projection/hydration into `Message.turn_id`;
- a single **canonical monotonic sequence** for ordering: the event store's
  insertion sequence, propagated through hydration. Timestamps are forbidden
  as an ordering key anywhere in the fetch or render path (event queries
  that currently order by timestamp are migrated for this path). The
  sequence is unique, so no tie-breaking rule is needed; the plan names the
  concrete column and adds the supporting index
  `(session_id, turn_seq, msg_seq)`.
- **Migration/fallback:** events written before `turn_id` existed hydrate
  with `turn_id = NULL` and fall under the legacy rule below (excluded from
  reconstruction; covered by the session summary). No retroactive grouping
  is attempted.

### Fetch

Replace the message-count window (`history_limit`, splits turns, slides
every build) with: fetch all messages with turn start sequence in
`[anchor_turn .. current_turn]`, ordered by
**(turn start sequence, message sequence)** as defined above. The anchor is
the only fetch parameter; nothing slides unless the anchor moves.

**Anchor initialization (no anchor in memory — first build of a session, or
after daemon restart):** walk turns backward from the most recent,
accumulating provider-equivalent estimated tokens per archived rendering,
and stop at the last whole turn that fits within the **60% low-water mark**
of the context budget; that turn becomes the anchor. The walk is a bounded,
paged query (never an unbounded session load), and the same low-water rule
used by eviction governs it, so a restart lands on the same anchor a
long-running daemon would have converged to.

Edge cases (defined, not discovered):

- **Legacy `turn_id = NULL` messages:** the anchor floor is the earliest
  turn_id-stamped message. Pre-turn_id history is not reconstructed into the
  payload; the session summary covers it. (One-time cost at upgrade.)
- **Incomplete/crashed turns** (no final assistant message): archived form
  renders the user message, a deterministic `[task interrupted]` placeholder,
  and tool summaries.
- **Late writes** (a message appended under an already-archived turn_id,
  e.g., a background notifier): the turn's content_fp changes → cache
  mismatch → re-render once, prefix break localized to that turn, logged.
- **Missing anchor turn** (evicted/pruned from the DB): anchor advances to
  the next existing turn; logged.

### Rendering

`render_turn(turn_messages, mode, renderer_version) -> Vec<Value>` is pure:

- `mode = Current`: full messages, append-only.
- `mode = Archived`: the single permanent form. Survivorship is explicit:
  - the user message text survives **in full**;
  - the final assistant reply survives, truncated by the existing
    deterministic rule (`MAX_OLD_ASSISTANT_CONTENT_CHARS`). **Selection
    rule:** the final reply is the last assistant-role record in the turn's
    sequence with non-empty content — tool-call-bearing assistant records
    contribute only their content (tool_calls stripped by the existing
    orphan rule); recovery retries and checkpoints are assistant/system
    records earlier in the sequence and lose by position; a turn whose last
    assistant record has empty content (tool-only or crashed turn) renders
    the deterministic `[task interrupted]`/no-reply placeholder;
  - tool results survive as deterministic summaries;
  - messages matching the identity-critical detector
    (`text_relates_to_critical_identity`) survive **verbatim** — identity
    safety cannot rest on tool summaries alone. Identity regression tests
    (the existing security/identity integration suites) must pass against
    archived-form history.
- Purity rules as for the core: no timestamps, no map-iteration order, no
  environment-dependent formatting.

The Prior-1/Prior-2 age ladder, the message-count trim, the
`current_user_injected` synthetic-user path (a whole-turn fetch cannot lose
the current user message), and the index-based identity-preserve bypass are
all absorbed by this mechanism and deleted.

### Render cache

`HashMap<turn_id, CachedRender { content_fp, renderer_version, mode, bytes }>`
per session, in-memory.

- `content_fp` is the hash of the **canonical serialization of the complete
  ordered render input** — every message in sequence order with all fields
  (role, content, tool_name, tool_call_id, tool_calls_json, annotations,
  and the sequence position itself). The only excluded fields are an
  explicit denylist of never-rendered fields (`embedding`, `importance`,
  `created_at`; finalized in the plan). Anything not on the denylist is in
  the fingerprint by default — omissions fail closed (spurious re-render),
  never open (stale bytes).
- Lookup requires `content_fp + renderer_version + mode` to match; any
  mismatch re-renders and replaces.
- Debug/test builds always re-render and assert byte equality against the
  cached entry — nondeterminism is a test failure, not a silent cache break.
- Logging: cache hits at `debug!` (or sampled); misses and fp mismatches at
  `info!` (rare, attributable events).

### Eviction

When the estimated payload exceeds the context budget: evict oldest whole
turns until the estimate is at or below a low-water mark (60% of budget),
advance the anchor, and log through the existing `Window decision` line.
Estimation uses **provider-equivalent serialized renderings** (token
estimates over the final rendered JSON content) with a safety margin (10%),
not raw message lengths. Eviction is the only operation that rewrites the
prefix head; it is rare and deep by construction. The anchor is in-memory;
persisting it is deferred unless warm-across-restart behavior becomes a
requirement.

## Pillar C — payload reduction

Stability (Pillars A/B) makes re-evaluation rare; this pillar makes every
remaining evaluation — and the steady-state context load — smaller. Measured
composition of the ~23k payload includes ~9.7k of tool schemas and ~4.5k of
`## Tools` prose that substantially restates them, plus ~3.9k of verbose
descriptions across ~10 rarely-used `manage_*` admin tools.

1. **Dedupe `## Tools` prose against the schemas.** The prose shrinks to a
   brief selection guide (when to reach for which tool, cross-tool rules);
   schemas remain the single source of truth for per-tool semantics.
2. **Slim the fat `manage_*` schema descriptions** to the same density as
   the frequently-used tools.

Target: ~23k → ~16–17k per call. Side effects this enables elsewhere in the
design: real headroom under the 28k policy budget (the near-zero slack today
is what drives history-fitting and eviction pressure), so the 60% low-water
mark and the "eviction is rare and deep" property hold with margin.

**Behavioral guard (this is the only pillar that can change model
behavior):** tool-selection quality on small local models depends on this
prose. The reduction ships behind the existing tool-selection integration
suite plus a short live smoke (representative tasks exercising terminal,
file tools, memory tools, web tools), and the prose is reduced to a guide —
not removed.

**Sequencing:** Pillar C lands **first**. It is small, independent, causes a
single one-time `tool_defs_hash` break, and re-baselines payload size before
the stability work measures itself — avoiding mid-phase attribution noise.

## Observability

Phase 0 instrumentation is untouched — it is the measurement harness for
this phase. New lines:

- `Core prompt invalidated component=<name>` (info)
- `Turn render cache hit` (debug/sampled) / `miss` / `fp_mismatch` (info)
- `Prefix mutation reason=<mechanism>` (info) — emitted by each retained
  within-task mutator (repeated-tool-error collapse, history-fitting
  overflow, empty-response retry rebuild)

After this phase, a `prefix_hash_system` cross-turn flip without a matching
`Core prompt invalidated` line is a bug by definition.

`AIDAEMON_DUMP_LLM_REQUESTS` complements the fingerprints: the hashes say
*that* a region flipped; the request dump says *which bytes*. Diffing
consecutive dumped requests is the expected diagnosis path for any
unattributed flip (it is how the turn 1→2 skill-toggle churn was found),
and the body-level integration assertions in §Testing can reuse the same
serialization.

## Testing

- **Unit:** core-hash canonicalization (input reordering → same hash;
  single-component change → correct `component` named); `render_turn`
  golden-byte tests per mode including incomplete-turn and identity-critical
  fixtures; cache fp/version/mode mismatch behavior.
- **Integration — prefix assertions run on canonical serialized request
  bodies, not internal message Vecs.** Internal-message comparisons cannot
  prove byte stability: prefix reuse depends on the request body after
  adapter conversion, and `anthropic_native` merges adjacent roles and
  hoists every system message. Each adapter's request builder is exercised
  directly (unit seam over the body-construction function), and the
  cross-iteration/cross-turn invariants are asserted on those bodies:
  1. within a task, iteration k+1's body is a strict prefix extension of
     iteration k's (mutation-free path), and each mutator path emits its
     `Prefix mutation` line instead;
  2. across two turns, core + archived[..N-1] is byte-identical in the
     bodies and archived[N] is byte-stable in a third turn;
  3. storing a fact between turns changes the tail region only (core bytes
     identical);
  4. a skills-catalog change between turns produces exactly one
     `Core prompt invalidated component=skills` and new core bytes;
  5. per-adapter determinism: same logical messages → identical serialized
     body bytes (incl. the Anthropic system-hoist/role-merge mapping).
- **Identity regression:** existing identity/security integration tests run
  against archived-form history.
- **Live:** re-run the 10-turn attribution protocol with
  `scripts/cache-attribution.py`.

## Exit criteria

1. All four cross-turn prefix invariants hold in integration tests.
2. Live re-run: criterion 1 (within-task system stability) stays PASS, and
   every cross-turn `prefix_hash_system` flip pairs with a logged core
   invalidation.
3. Live re-run: median turn-start evaluated tokens reduced versus the
   Phase 0 baseline (~22.3k median). **≥80% reduction is the target** (the
   expected bound — archived[N] + tail + user message — is a few thousand
   tokens of a ~22k payload), not a pass/fail gate: the first valid
   instrumented re-run establishes the measured bound, and the regression
   threshold used for gating thereafter is set from that measurement.
4. Pillar C: median per-call payload at or below ~17k tokens, with the
   tool-selection integration suite green and the live smoke showing no
   tool-selection regressions.

## Out of scope

- Durable render/anchor persistence (Approach 1 of the design discussion) —
  revisit only if warm llama.cpp KV across daemon restarts becomes an
  explicit requirement.
- Prompt-cache optimization for Anthropic/Google adapters (deterministic
  serialization only).
- Queued-message durability across restarts (separate issue, noted in the
  Phase 0 run report).
- Multi-session contention on the shared llama-server slot (`--parallel 1`
  assumption stands; interleaved sessions still evict each other's KV).
