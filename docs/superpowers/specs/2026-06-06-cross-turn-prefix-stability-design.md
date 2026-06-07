# Cross-Turn Prefix Stability & Payload Reduction (Phase 1)

**Status:** Draft — pending review
**Covers:** prefix stability (Pillars A/B) and payload reduction (Pillar C,
lands first)
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

1. Scope: three pillars in one phase — **A:** message-zero freeze/split,
   **B:** monotonic turn rendering with turn-anchored fetch, **C:** payload
   reduction (added during review; mandatory, lands **first**, carries its
   own exit gate).
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
6. Exit criteria are measured with the Phase 0 instrumentation, which is
   **extended, never redefined**: existing fields keep their semantics
   (additions in §Observability — `tail_hash`, `prefix_hash_archived`;
   `session_summary_hash` retires by reporting empty). Absolute thresholds
   are set **after** the first instrumented re-run (see §Exit criteria).

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
[k+2..] CURRENT TURN            current user message, then persisted
                                assistant/tool exchanges (stable region),
                                followed by a TRANSIENT SUFFIX of
                                regenerated execution checkpoints and
                                one-shot system directives.
```

The tail sits **before** the current turn. Within a task, the **stable
region** (everything through the persisted current-turn messages) extends
monotonically across iterations, while the **transient suffix**
(checkpoints, one-shot directives) may be replaced each iteration — see
invariant 3. Nothing about turn N's payload is a prefix promise for turn
N+1 at or after the tail position — see the invariants below for what is
promised.

### Cross-turn prefix invariants

Turn N final payload:  `core → archived[..N-1] → tail[N] → current[N]`
Turn N+1 first payload: `core → archived[..N-1, N] → tail[N+1] → current[N+1]`

Divergence necessarily occurs where `tail[N]` is replaced by `archived[N]`.

Throughout these invariants, "prefix" and "byte-identical" refer to the
**ordered prompt/message representation after adapter conversion** (and,
ultimately, the rendered chat-template/token sequence the server compares) —
never to serialized JSON request bodies, which cannot be prefix-extensions
of each other by construction (closing delimiters move). The promised
invariants are:

1. **Core + archived turns through N-1 are byte-identical** between turn N's
   final payload and turn N+1's first payload (absent a logged core
   invalidation or eviction).
2. **Turn N's archived rendering is byte-stable from turn N+1 onward** — it
   never changes again (absent a content_fp change from a late write, which
   is logged).
3. **Within a task, the payload decomposes into a stable region and a
   transient suffix, and the stable region extends monotonically.**
   - The **stable region** — core, archived turns, task tail, and persisted
     current-turn messages — is a strict prefix extension of the previous
     iteration's stable region, absent a logged prefix mutation.
   - The **transient suffix** — execution checkpoints (regenerated each
     iteration, not persisted; `message_build_phase.rs:1248`) and one-shot
     system directives (`message_build_phase.rs:1293`) — is replaceable by
     design: it disappears next iteration and new persisted messages occupy
     its positions. Strict whole-sequence prefix extension is therefore NOT
     promised even on clean paths.
   - **Within-task re-evaluation is bounded by the previous iteration's
     transient suffix plus newly appended messages.** Checkpoints and
     directives stay small for this reason; their size is part of the
     measured bound.

   Three retained recovery mechanisms additionally rewrite **stable-region**
   bytes and are excluded from the invariant: repeated-tool-error collapse
   (`collapse_repeated_tool_errors`), history-fitting overflow trimming, and
   the empty-response retry rebuild. Each must emit
   `Prefix mutation reason=<mechanism>` when it fires so every stable-region
   re-evaluation is attributable. **History-fitting is scoped to the
   current-turn region only.** Fitting that trims, drops, or summarizes an
   archived turn is forbidden by construction: archived turns are
   whole-turn-evicted at the anchor (§Eviction), and eviction never reaches
   inside a turn, so an archived rendering can only change via a logged
   `content_fp` mismatch (invariant 2), never via fitting. Whole-turn
   eviction at the low-water mark also leaves the current-turn headroom the
   per-iteration fitter previously had to manufacture, so current-region
   fitting itself becomes rare.
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

**Canonical order is the emission order, not just the hash order.** The same
name-sorted tool roster that feeds the hash must be the order in which tools
are emitted into the provider payload's **tool array**. If the hash sorts by
name but emission preserves source order, a source-order change leaves the
aggregate core hash unchanged while flipping `tool_defs_hash` and the
rendered prefix — a cache break with no `Core prompt invalidated` line,
which exit criterion 2 would then read as a bug. Canonicalization is
therefore a property of the rendered output, asserted by the golden-byte
tests, not an internal-only step before hashing. The `## Tools` prose is
**not** bound to the name-sorted order: it lives inside the core and needs
only determinism (covered by the core golden-byte hash), so Pillar C's
selection guide is free to group tools logically (file tools together,
memory tools together).

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

**Single summary insertion point.** Two summary paths exist today: the
build-stage `[Session Summary]` at index one
(`message_build_phase.rs:1230`) and the fit-stage `[Conversation summary: …]`
near index one (`fit_messages_with_source_quotas`,
`context_window.rs:200,310`). The session summary is part of the task tail
and lives there only — the **fit-stage insertion is deleted**. Left in place
it injects `[Conversation summary]` into the archived-turns region near index
one and silently defeats Pillar A's tail stability. `fit_messages_*` retains
its message-fitting role for the current-turn region (see §Pillar B
eviction / invariant 3) but no longer takes or emits a summary argument.

**Resume checkpoint is a move, not a copy.** It is injected into the core
system prompt today (`system_prompt.rs:768`); this phase relocates it to the
tail. Flagged so a later pass does not "optimize" it back into core, which
would make the core per-resume-state and defeat Pillar A.

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
  (user message, assistant response, tool call/result) **and to the
  task-completion record** (`EventType::TaskEnd`) and propagated through
  projection/hydration into `Message.turn_id`. The completion record must be
  turn-attributable because the archived render's terminal-state placeholder
  (§Rendering) is derived from it — without `turn_id` on it the renderer
  cannot attribute the outcome to the turn.
- **Completion status is an explicit render input, and preserves the real
  outcome.** `TaskEnd` carries a `TaskStatus` of `Completed`, `Cancelled`, or
  `Failed` (`src/events/mod.rs:209`); collapsing all three into "completed"
  would render a failed or cancelled tool-only turn as `[completed: …]`. The
  per-turn render input therefore carries
  `terminal_state ∈ {completed, failed, cancelled, interrupted}` — the first
  three mirror `TaskStatus`; `interrupted` is the absence of any `TaskEnd`
  record (crash). `content_fp` (§Render cache) includes `terminal_state`, so
  a tool-only turn that later acquires a completion record — or whose outcome
  changes — re-renders rather than serving a stale entry, precisely in the
  no-text-reply case where the message set alone is ambiguous.
- a single **canonical monotonic ordering** built from the event store's
  existing insertion id (`events.id`; no new atomic counter):
  - `msg_seq` = the message's own `events.id`, ordering messages **within** a
    turn; a late write's id is large, so it deterministically sorts last
    inside its turn — exactly its append position. This needs no new storage
    (it is the existing row id).
  - `turn_seq` = the **lowest `events.id` in the turn**, used to order and
    range whole turns. (Not "the user message's id": scheduled goal
    dispatches and background re-engagement turns do not start with a user
    message.) This is a **cross-row value** — a message's `turn_seq` lives
    on a *different* row (the turn's first event row, reached via
    `turn_id`) — so it **cannot** be produced by a
    SQLite expression index over the message row. The design requires it
    materialized: either denormalized onto every event row at write time
    (stamp the turn's start id when the row is inserted) or a
    `turn_id → start events.id` mapping joined at fetch time. The plan picks
    between those two; a pure expression index is explicitly ruled out. A
    late write inherits its turn's `turn_seq` through `turn_id`, so it groups
    and ranges with that turn, not at the session tail.

  Timestamps are forbidden as an ordering key anywhere in the fetch or render
  path (event queries that currently order by timestamp are migrated for this
  path). Insertion ids are unique, so no tie-breaking rule is needed; the
  supporting index is `(session_id, turn_seq, msg_seq)`.
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
and stop at the last whole turn that fits within the **archived-region
budget** (defined under §Eviction: the low-water fraction of what remains
after non-evictable reservations); that turn becomes the anchor. The walk is
a bounded, paged query, never an unbounded session load.

Reconstruction is **not** exact: a running session legitimately grows from
the low-water mark toward the eviction threshold without moving its anchor,
so a restart that rebuilds to low-water lands on a deeper anchor than the
pre-restart one. This phase explicitly accepts **one boundary change (one
full re-prefill) per daemon restart per session**. Persisting the anchor is
the named upgrade path if restart warmness later matters (see Out of scope).

Edge cases (defined, not discovered):

- **Legacy `turn_id = NULL` messages:** the anchor floor is the earliest
  turn_id-stamped message. Pre-turn_id history is not reconstructed into the
  payload; the session summary covers it. (One-time cost at upgrade.)
- **Incomplete/crashed turns** (no final assistant message): archived form
  renders the user message, the `terminal_state`-selected placeholder
  (§Rendering — `[task interrupted]` when no `TaskEnd` record exists, or the
  `failed`/`cancelled` form when one does), and tool summaries.
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
    rule:** the **last assistant-role record with non-empty content** wins,
    regardless of any later empty assistant record. Tool-call-bearing
    assistant records contribute only their content (tool_calls stripped by
    the existing orphan rule); recovery retries and checkpoints lose by
    position. If **no** assistant record in the turn has non-empty content,
    the placeholder is selected by `terminal_state` (§prerequisite):
    `completed` → `[completed: N tool steps, no text reply]`, `failed` →
    `[failed: N tool steps, no text reply]`, `cancelled` →
    `[cancelled: N tool steps, no text reply]`, and `interrupted` (no
    `TaskEnd` record) → `[task interrupted]`. The status string is fixed
    per state so the placeholder is deterministic;
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
  and the sequence position itself), **plus the turn's `terminal_state`**
  (§prerequisite) so the no-text-reply completed/interrupted distinction is
  captured. The only excluded fields are an
  explicit denylist of never-rendered fields (`embedding`, `importance`,
  `created_at`; finalized in the plan). `Message.id` is never rendered
  either but is deliberately **retained** in the fingerprint: it is
  DB-stable, uniquely identifies the row, and with the synthetic-user path
  deleted it cannot vary between rebuilds — fail-closed keeps it harmless
  and it strengthens identity of the input set. Anything not on the
  denylist is in the fingerprint by default — omissions fail closed
  (spurious re-render), never open (stale bytes).
- Lookup requires `content_fp + renderer_version + mode` to match; any
  mismatch re-renders and replaces.
- Debug/test builds always re-render and assert byte equality against the
  cached entry — nondeterminism is a test failure, not a silent cache break.
- Logging: cache hits at `debug!` (or sampled); misses and fp mismatches at
  `info!` (rare, attributable events).

### Eviction

Budgeting is computed over the **evictable region only**. The
archived-region budget is:

```
archived_budget = (context_budget
                   − core − tool definitions − task tail estimate
                   − current-turn reserve − output reserve)
                  × (1 − safety_margin)
low_water = 60% of archived_budget
```

When the archived-region estimate exceeds `archived_budget`: evict oldest
whole turns until at or below `low_water`, advance the anchor, and log
through the existing `Window decision` line. Estimation uses
**provider-equivalent serialized renderings** (token estimates over the
final rendered content) with a 10% safety margin, not raw message lengths.

**Degenerate case:** if the non-evictable region alone meets or exceeds the
context budget, the payload carries **zero archived turns** (current turn +
core + tail only), a `warn!` names the overflowing components, and any
further reduction (tail compaction, schema slimming) is a Pillar C concern —
the eviction mechanism never truncates inside a turn.

Eviction is the only operation that rewrites the prefix head; it is rare
and deep by construction. The anchor is in-memory; persisting it is
deferred unless warm-across-restart behavior becomes a requirement.

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

Phase 0 instrumentation is retained and extended — it is the measurement
harness for this phase. New lines:

- `Core prompt invalidated component=<name>` (info)
- `Turn render cache hit` (debug/sampled) / `miss` / `fp_mismatch` (info)
- `Prefix mutation reason=<mechanism>` (info) — emitted by each retained
  within-task mutator (repeated-tool-error collapse, history-fitting
  overflow, empty-response retry rebuild)

**New fingerprint region: the tail must be separable in live logs.** Under
this design the task tail sits inside the fingerprint's pre-boundary region,
so `prefix_hash_pre_boundary` flips on every turn **by design** (expected
tail replacement) — making expected tail churn and unexpected
archived-region instability indistinguishable, which would demote the
invariant-1/2 guarantees to test-only properties. The provider-call
fingerprint therefore gains two fields:

- `tail_hash` — the tail message located by its deterministic marker and
  hashed separately. The marker is a shared constant between the tail
  builder and `prefix_fingerprint.rs` (the plan defines it, e.g.
  `TASK_CONTEXT_TAIL_MARKER`), the same arrangement that keeps
  `SESSION_SUMMARY_MARKER` from drifting today;
- `prefix_hash_archived` — the pre-boundary region **excluding** the tail.

`prefix_hash_pre_boundary` keeps its existing semantics for cross-phase
comparability. The live diagnosis rule becomes: a `prefix_hash_archived`
flip without a matching eviction (`Window decision`), `Prefix mutation`
line, or render-cache `fp_mismatch` (late write or late completion record into an archived turn) is
an archived-region bug; a tail-only flip is expected.

The `session_summary_hash` index-1 special case retires when the summary
moves into the tail: the field reports empty, and the summary participates
in `tail_hash`. `scripts/cache-attribution.py` is updated to parse the new
fields and to attribute tail-only flips as expected.

After this phase, a `prefix_hash_system` cross-turn flip without a matching
`Core prompt invalidated` line is a bug by definition.

`AIDAEMON_DUMP_LLM_REQUESTS` complements the fingerprints: the hashes say
*that* a region flipped; the request dump says *which bytes*. The routine
path is hash attribution via the rules above; diffing consecutive dumped
requests is the diagnosis path for any flip the hashes leave
**unattributed** (it is how the turn 1→2 skill-toggle churn was found).
The body-level integration assertions in §Testing can reuse the same
serialization.

## Testing

- **Unit:** core-hash canonicalization (input reordering → same hash;
  single-component change → correct `component` named); `render_turn`
  golden-byte tests per mode including incomplete-turn and identity-critical
  fixtures; cache fp/version/mode mismatch behavior.
- **Integration — what "prefix" is asserted on.** A serialized JSON body
  can never be a strict byte-prefix extension of a shorter one (arrays and
  objects close with delimiters; appending a message rewrites the suffix),
  so full-body byte-prefix assertions are wrong by construction. The
  prefix invariants are asserted over the **provider's ordered
  prompt/message representation after adapter conversion** — element-wise
  equality of the converted message sequence (and, for llama.cpp where
  available, the rendered chat-template/token sequence). Full serialized
  bodies are tested **only for determinism** (same logical input →
  identical bytes). Each adapter's request builder is exercised directly
  (unit seam over the body-construction function).

  Cross-turn cache-prefix assertions apply **only to the
  OpenAI-compatible/llama.cpp adapter** — `anthropic_native` and
  `google_genai` hoist system content (including the task tail) into
  top-level parameters, so a changed tail legitimately rewrites their
  payload head every turn; for those two, only deterministic conversion is
  asserted (cache wins there are out of scope, per §Pillar A).

  1. within a task, iteration k+1's **stable region** extends iteration k's
     element-wise (mutation-free path), with the transient suffix
     (checkpoints, one-shot directives) identified and excluded by the test
     harness; each mutator path emits its `Prefix mutation` line instead;
  2. across two turns (OpenAI-compatible adapter), core + archived[..N-1]
     elements are identical and archived[N] is stable in a third turn;
  3. storing a fact between turns changes the tail element only (core
     element identical);
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
2. Live re-run: criterion 1 (within-task system stability) stays PASS;
   every cross-turn `prefix_hash_system` flip pairs with a logged
   `Core prompt invalidated` line; and every `prefix_hash_archived` flip
   pairs with a logged eviction (`Window decision`), `Prefix mutation`
   line, or render-cache `fp_mismatch` (late write or late completion record into an archived turn —
   spec-compliant per §Fetch edge cases). Tail-only flips (`tail_hash`
   changed, `prefix_hash_archived` stable) are expected and pass.
   `tool_defs_hash` is cross-turn stable **within every attribution run**;
   the only sanctioned change is the Pillar C deployment itself, which lands
   **between** runs, so no run ever contains the break and
   `cache-attribution.py` must not tolerate an in-run flip (the tool roster
   is a core input; any in-run flip is a per-turn-gating regression, per the
   §Pillar A anti-pattern). **Force-text turns are not a special case on the primary
   adapter:** force-text retains the tool definitions in the payload and only
   disables calling via `tool_choice=none` (`llm_phase.rs:300`,
   "tool defs retained for prefix stability"), so `tool_defs_hash` and the
   rendered prefix stay stable across a force-text turn. (`anthropic_native`
   strips tool defs when `tool_choice=none`, but cross-turn cache assertions
   do not apply to that adapter — §Testing.)
3. Baselines are sequenced with Pillar C: the Phase 0 baseline (~22.3k
   median payload, ~22.3k median turn-start re-eval) applies to Pillar C
   only. **After Pillar C lands, a fresh attribution re-run establishes the
   post-C baseline** (expected ~16–17k), and Pillars A/B are measured
   against **that**, not Phase 0 — otherwise the absolute re-eval bound
   (archived[N] + tail + user ≈ 4–5k) would read as ~70–75% against a
   payload that no longer exists. **≥80% reduction of median turn-start
   evaluated tokens versus the post-C baseline is the target**, not a
   pass/fail gate: the first valid post-A/B re-run establishes the measured
   bound, and the regression threshold used for gating thereafter is set
   from that measurement.
4. Pillar C gate (against the Phase 0 baseline): median per-call payload at
   or below ~17k tokens, with the tool-selection integration suite green
   and the live smoke showing no tool-selection regressions.

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
