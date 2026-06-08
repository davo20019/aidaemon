# Memory: Reliable Relational Recall — Design

> Status: **design proposed, revised after code-verified review → ready for plan.**
> Date: 2026-06-08.
> Revision note: Bug 2 narrowed to derived recall; Bug 3 people-injection claim corrected
> (graph is fetched but not rendered in owner DMs); Fix 1 widened to close the tail `CRITICAL FACTS`
> leak (Fix 1+2 must ship together); Fix 2 given concrete `CoreInputs` wiring + cost accounting;
> open questions on children source-of-truth and the deterministic arms resolved.
> **Generalization revision:** Fix 2 reframed from a partner/children block to a **salience-selected
> core *profile*** (auto-salience + user pin override, bounded, session-frozen) so it fixes the *class*
> — any high-salience fact dropping out of recall — not just the wife/children instance.
> Companion to [`memory-graph-analysis.md`](../../memory-graph-analysis.md) and
> [`memory-system-design-notes.md`](../../memory-system-design-notes.md) (the polyglot-memory brainstorm).
> This spec narrows that long-term direction in response to a **concrete, reproduced failure**.

---

## The failure that triggered this

Real transcript (owner DM, People system enabled):

```
who is my wife?
Here are the core relationship details I have pinned:
• partner: June 30, 1990
• partner: Aracely Zambrano
• children: August 20, 2020
• children: Bella

who's bella loor mom?
I don't have information about Bella's mother in my records.

who is bella's mom?           (asked again, later)
Aracely Zambrano is Bella's mom.        ✓ correct

who is aracely?
Aracely Zambrano is your partner. Birthday June 30, 1990.   ✓ correct
```

The agent **can** reason "Bella → you → Aracely" (it does so for "who is bella's mom?"), but the
same class of question intermittently fails or returns a garbled bullet dump. This is not a
missing-data problem and not (primarily) a missing-graph problem.

## Root cause (verified in code)

There are **two answer paths**, and the phrasing of the question decides which one runs.

### Bug 1 — A deterministic guardrail short-circuits the LLM with a bad dump

`src/agent/policy/recall_guardrails.rs`:

- `detect_critical_fact_query` (lines ~120-131) matches phrases like `"who is my wife"`,
  `"who is my spouse"`, `"who are my children"` → returns `CriticalFactQuery::CoreRelationships`.
- `extract_critical_fact_summary` (lines ~136-259) reads the **flat `facts` slice**, maps keys to
  labels via `relationship_label_for_key` (wife/husband/spouse/partner → `partner`;
  daughter/son/children → `children`), and formats `"{label}: {value}"`.
- `deterministic_reply_for_critical_query` (lines ~276-284) emits
  `"Here are the core relationship details I have pinned:\n- ..."` — **bypassing the LLM entirely.**

Because values in the flat facts are shredded (name in one row, birthdate in another, all keyed
`partner`/`children`), the dump renders dates as if they were people: `• partner: June 30, 1990`.

`"who is bella's mom?"` matches **none** of the guardrail keywords, so it falls through to the
normal LLM path, the facts land in context, and the LLM reasons correctly. **The guardrail meant to
guarantee recall is the thing producing the worst answers.**

### Bug 2 — Per-turn fact retrieval is non-deterministic (hurts *derived* recall)

`src/state/sqlite/facts.rs:769-902` (`get_relevant_facts_for_channel`):

- Embeds the **short user message** ("who is my wife") each turn and scores every fact (O(n) scan).
- Score = cosine + lexical (≤0.55) + freshness boost (≤0.15); must clear a tight `0.3` threshold;
  only the top `max_facts` survive. `default_max_facts = 20` (`config.rs:1019`).

Facts near the 0.3 boundary pass on one turn and fail the next → "I don't have information" one
moment, correct answer later.

**Scope correction (verified in code):** this lottery does **not** govern the direct relationship
questions in the transcript. The owner's identity/profile facts that feed the tail `CRITICAL FACTS`
block come from a **deterministic category fetch** (`system_prompt.rs:283-305`: iterates `identity`,
`personal`, `family`, … via `get_facts(Some(cat))`), *not* semantic scoring. So "who is my wife?"
returning *"I don't have information"* is attributable to **Bug 1** (the short-circuit) and to the
shredded formatting of that deterministic block — not to a 0.3 threshold miss on that turn.

Bug 2 bites the **derived / non-keyword** questions — "who's bella's mom?" — which fall through to
the LLM path and depend on whatever the semantic `facts` retriever surfaces that turn. There, the
most important relationships are subject to the same lottery as trivia, which is what makes that
question intermittent.

### Bug 3 — The structured entity layer already exists, and the broken path ignores it

`route_people_fact` (`src/memory/manager.rs:823-962`) already routes `category == "people"` facts
into the `people` + `person_facts` tables — a real entity layer (Person record + PersonFacts linked
by `person_id` FK). That is *why* "who is aracely?" is clean: it resolves a Person. But:

- The guardrail dump (Bug 1) reads the **flat `facts` table**, not the people graph.
- Children are stored as flat facts, **not linked** to the owner Person.
- There are **no explicit edges** between entities (owner —partner→ Aracely; owner —child→ Bella),
  so nothing can traverse.

In owner DMs the people graph is **fetched** (`get_all_people()` + owner `person_facts` loaded into
`MemoryContext`), but — **verified, this was overstated in an earlier draft** — it is **not actually
rendered** into the owner's prompt. `build_system_prompt_with_memory` only renders person facts under
`## Current Speaker Context`, gated on `memory.current_person.is_some()` (`skills/mod.rs:1056-1081`,
the `take(10)` there is for a **non-owner speaker**, not the owner list). In an owner DM
`current_person` is `None`, so the only people-derived text that survives is the **People Privacy
Rules** header (`skills/mod.rs:1044-1054`). The structured graph is paid for and then dropped.

Net: relationship recall in owner DMs rides entirely on (1) the Bug 1 guardrail, (2) the variable
semantic `facts` retriever (Bug 2), and (3) the shredded, deterministic `CRITICAL FACTS` tail block.
None of these reads the clean people graph — which is exactly the case for Fix 2.

## Does the planned graph direction fix this?

Partially, and not on its own. The long-term polyglot/edge direction is the right **destination** for
Goal 4 (associative recall), but it sits one layer *below* the proximate causes here. A perfect edge
table sandwiched between a dump-everything guardrail (Bug 1) and a coin-flip retriever (Bug 2) still
produces this transcript. So this spec splits the work:

- **Phase 0 (this spec's priority):** stop the embarrassment now with two surgical, low-risk fixes on
  data we can eyeball.
- **Phase 1 (durable):** explicit relationship edges + traversal, the Goal-4 layer — built on the
  clean foundation Phase 0 establishes. Detailed later; scoped but not fully specified here.

---

## Phase 0 — Design

### Fix 1: Remove the deterministic relationship dump

Stop `CriticalFactQuery::CoreRelationships` from short-circuiting the LLM. Relationship questions
flow through the normal LLM path, which already answers correctly when the facts are present.

- Remove the `CoreRelationships` branch from `detect_critical_fact_query` (or make it a no-op that
  returns `None`), and remove `deterministic_reply_for_critical_query`'s relationship arm.
- **Also close the second leak path (critical):** `detect_critical_fact_query` is *not* the only
  consumer of the shredded data. `extract_critical_fact_summary` populates `summary.relationships`
  from the flat `facts` table independently (`recall_guardrails.rs:179-192`), and
  `build_critical_facts_prompt_block` renders it into the **tail** at `system_prompt.rs:618` under a
  `═══ CRITICAL FACTS — USE THESE EXACT VALUES ═══` header that instructs the LLM *"Do NOT substitute,
  paraphrase, or infer."* (`recall_guardrails.rs:291-294, 303-307`). After the short-circuit removal
  this block still injects `• partner: June 30, 1990` / `• partner: Aracely Zambrano` every owner turn
  — now *strengthened* by the "exact values" instruction, and **directly contradicting Fix 2's clean
  block** if both are present. So Fix 1 MUST also stop `relationships` from flowing into that block:
  drop the `relationships` arm of `build_critical_facts_prompt_block` and stop
  `extract_critical_fact_summary` populating `summary.relationships` (delegate relationships entirely
  to Fix 2). Leave the `personal_facts` / name arms of that block untouched.
- **Generalize the tail-block cleanup, don't special-case relationships:** because Fix 2 becomes the
  single cache-resident home for *all* stable owner-identity context, the tail `CRITICAL FACTS`
  *context-injection* block is now redundant for its **`personal_facts` arm too**, not only
  `relationships`. Both arms migrate to Fix 2; the volatile tail block stops injecting owner-identity
  context entirely (avoids double-emission and the "exact values, do not infer" instruction fighting
  the LLM). This keeps the fix at the level of the *mechanism* (deterministic dump beats LLM) rather
  than the relationship instance.
- **Keep the scalar question-answer short-circuits** (`OwnerName` / `AssistantName` in
  `detect_critical_fact_query`): those answer a direct "what's my name?" with a scalar lookup, no
  shredding, and are a separate mechanism from context injection (answers Open Question #4 — only the
  multi-row shredded arms are structurally broken). They may later fold into the let-the-LLM-answer
  principle, but that is out of scope here.
- Net effect: "who is my wife?" now takes the same path as "who is bella's mom?" — which works.

Because Fix 1's tail-block removal deletes the *only* relationship text that exists today, **Fix 1 and
Fix 2 must ship together** (or in strict order Fix 2 → Fix 1). Shipping Fix 1 alone would briefly
leave owner DMs with *no* relationship context and re-expose Bug 2 on the LLM path; Fix 2 is what
restores reliable presence.

### Fix 2: A deterministic, cache-resident "core profile" block (salience-selected)

Guarantee the owner's **highest-salience facts** are always present and reliable, independent of the
semantic-threshold lottery — and place them where they are **cached**, not re-paid every turn.
Relationships (partner, children) are the *triggering* case, but Fix 2 is deliberately **not** limited
to them: it is a general "what the agent must never forget about me" block that also covers name, job,
location, allergies, pets, parents — anything salient. We fix the *class* (Bug 2 can drop *any*
important fact), not just the wife/children instance.

**Selection rule (resolved — "auto-salience + user pin override"):** membership =
`(explicitly pinned) ∪ (top-N by salience)`, capped at **N ≈ 30**, computed over the **union of two
sources**: the owner's `person_facts` (the clean graph) **and** the owner's identity/profile flat
`facts`. Salience per fact:

```
salience = pinned ? 1.0
         : 0.6 · norm(recall_count)   // facts the user keeps asking about have proven important
         + 0.3 · category_weight      // identity / family / health / relationships weighted high
         + 0.1 · recency
```

- **Pinned wins (force-in); unpinned can be force-out** — direct user control on top of automatic
  capture (addresses the "no control / unpredictable memory" complaint from the research notes).
- **Bounded** by N so the cached core stays small no matter how many facts accumulate over years.
- **Rendered deterministically**, grouped for readability (an "About you" block: identity lines, then
  relationships with linked name+birthday, then other high-salience facts). Synonym normalization so
  "wife"/"spouse"/"husband" resolve to the `partner` role and "kids"/"daughter"/"son" to `children`.

**Cache-stability guard — snapshot at session start (critical):** salience inputs like `recall_count`
change *every turn*; fed live, they would thrash the core component hash and destroy the cache. So
membership + ordering are **computed once at session start and frozen for the session** (slow-changing
inputs only, within a session). Still always-current — it recomputes next session — and prefix-stable
within the session. The per-turn hash is over the *frozen snapshot*, so it changes only when a
pin/relationship actually changes.

**Pin mechanism (small additive piece):** the `Fact` model has no `pinned` field today. Add a
`pinned` flag (a `pinned INTEGER DEFAULT 0` column on `facts`/`person_facts`, or a small `pinned_facts`
table) plus a `manage_memories(action='pin'|'unpin')` action. **Stage it:** the auto-salience half
ships with **no schema change** (uses existing `recall_count` / `category` / `updated_at`); the pin
override is a small follow-on within Phase 0.

**Where it goes — the caching decision:** inject it into the **session-static CORE prompt**
(`src/agent/runtime/core_prompt.rs` → `render_core_prompt`), NOT the volatile per-turn tail.

Rationale (verified against the prefix-stability architecture on this branch):

- The CORE prompt is cached per `session_id`, keyed by a component-hash aggregate
  (`core_cache_decision`, `core_prompt.rs:157`). On a HIT the bytes are reused — it is the cacheable
  prefix.
- The frozen core-profile snapshot is **session-stable** (Fix 2's whole point), so it belongs in the
  cached prefix, not the per-turn tail where the current people/facts injection lives.
- Cost: paid **once per session** as a cache write (bounded — N≈30 facts × ~10-20 tokens), then a
  cache read on every later turn → effectively free per-turn.
- It only busts the session cache when a pinned/high-salience fact actually changes —
  `core_cache_decision` already diffs component hashes and re-renders only on change. Rare and correct.
- This removes the most important facts from the 20-fact semantic lottery → **kills Bug 2's variance**
  for *all* core-profile facts (relationships, identity, health, …), not just relationships.

**Prefill impact:** small, bounded net-new volume (≤ N≈30 short lines, cached after turn one). Note the
correction from Bug 3: owner DMs **fetch but do not render** the people graph today, so Fix 2 is
genuinely adding this text to the prompt for the first time — not merely relocating existing tail text.
It is bounded by N and lives in the cached prefix, so the per-turn token cost is ~zero after the first
turn.

**Prefix-stability constraint:** the block must render deterministically (stable field order, stable
formatting) so identical inputs produce identical bytes — consistent with the existing
`core_prompt_renders_identically_for_identical_inputs` invariant. Adding it changes the core
component-hash aggregate; ensure the new component participates in `core_cache_decision` so a
relationship change correctly invalidates and re-renders.

**Concrete `CoreInputs` wiring (do not leave to the implementer):** `CoreInputs`
(`core_prompt.rs:18-110`) today carries **only application-structural** components — `base_template`,
`tool_roster`, `skills_catalog`, `specialists`, `channel_rules`, `persona`. **None is per-user
memory.** Fix 2 makes CORE depend on mutable per-owner data for the first time, so spell out:

1. **New component `core_profile`**, added to the `entries` array in
   `CoreInputs::component_hashes()` and to `render_core_prompt`. Note `ComponentHashes` is a
   **fixed-size array** (`COMPONENT_COUNT = 6`, `core_prompt.rs:54`): bump to `7` and add the entry —
   not just "ensure it participates." The existing `diff`/`aggregate` tests must be updated for the
   new count.
2. **Gating:** populate the component only when `user_role == UserRole::Owner` **and**
   `should_inject_personal_memory()` **and** `config.people.enabled`. For any non-owner session the
   component hashes to **empty** — CORE is keyed by `session_id`, so an unguarded block would leak the
   owner's profile into a non-owner's cached prefix. Add a test that a non-owner session renders no
   core-profile block.
3. **Sub-agents:** spawned executors (`depth > 0`) should **not** receive this block (no owner-DM
   recall context); gate it off for non-root agents.
4. **Pre-fetch + frozen salience:** read the **union source** (owner `person_facts` ∪ owner
   identity/profile flat `facts`) into the inputs *before* `assemble_core_inputs`, mirroring the
   existing async pre-fetch pattern for skills/specialists. Reconcile the two timing requirements
   explicitly: **salience inputs (`recall_count`, recency) are snapshotted at session start and frozen**
   (this is what stops per-turn thrash), while the **fact/relationship/pin set is read per turn** so a
   genuine mid-session edit (user pins a fact, adds a child via `manage_people`) propagates and busts
   the cache exactly once. The per-turn read must be the same cheap owner query already run, not a new
   heavy scan.

**Cost note (tighten the "paid once per session" claim):** the *render + token write* is once per
session (until a real edit); the *owner read + hash* is **per-turn** (that's how cache invalidation
detects an edit). Because salience scoring uses the **frozen** snapshot, recall-count drift does **not**
change the hash — only a genuine pin/relationship/identity edit does. Net DB cost is unchanged from
today — `system_prompt.rs:283-309` already fetches identity facts every owner turn — so the win is
**prompt tokens**, not the query. State it that way to avoid implying the DB read is eliminated.

### Phase 0 data flow (after)

```
user: "who is my wife?"
  → (Fix 1) no guardrail short-circuit
  → normal LLM path
  → system prompt CORE prefix already contains (cached) deterministic core-PROFILE block,
    salience-selected (not hard-coded to relationships):
        ## About you
        Name: David Loor · Location: … · Employer: …      ← identity (also salience-selected)
        Partner: Aracely Zambrano (b. 1990-06-30)
        Children: Bella (b. 2012-05-17), Cami (b. 2020-08-20)
        Allergy: penicillin   [pinned]                    ← arbitrary high-salience / pinned fact
  → LLM answers naturally: "Your wife is Aracely Zambrano."
user: "who's bella's mom?"
  → same cached block present every turn
  → LLM reasons Bella → you → Aracely → "Aracely, your partner, is Bella's mom."
user: "am I allergic to anything?"
  → same block → "Yes — penicillin."   (the *class* fix: not a relationship, still reliable)
```

Deterministic because the block is always there (cache-resident, salience-selected), not because of a
hard-coded reply for one question type.

---

## Phase 1 — Durable relational layer (scoped, not fully specified here)

Once Phase 0 is in, promote relationships from string keys to **explicit, queryable edges** so
associative recall generalizes beyond the bounded core-profile block (which guarantees the top-N but
can't traverse arbitrary connections):

- Explicit edges: `owner —partner_of→ Aracely`, `owner —child_of→ Bella/Cami` (and inverse). Convert
  children from orphan flat facts into owner-linked entities.
- Relation-typed retrieval: a relationship question resolves a role → entity deterministically, then
  expands 1 hop for context, instead of semantic-cosine-on-flat-facts.
- Co-parent / derived relations by **query-time traversal** (partner-of-owner ∧ parent-of-child),
  not materialized assumptions — avoids asserting biological/step relationships the data doesn't
  support. The LLM phrases the connective answer.

This is the entity/edge layer from the companion brainstorm. `sqlite-vec` (the original "step 1")
moves to "when recall latency actually bites" — it does nothing for this failure class and is not on
the critical path for relational recall.

---

## Testing

- **Unit (Fix 1):** `detect_critical_fact_query` returns `None` for "who is my wife/spouse/husband",
  "who are my children", **"do i have daughters?"/"do i have kids?"** (also mapped to
  `CoreRelationships` today); `OwnerName`/`AssistantName` queries still classify. Update the existing
  `detects_critical_fact_queries` / `deterministic_reply_uses_critical_facts` tests in
  `recall_guardrails.rs` for the removed arm.
- **Unit (Fix 1, tail leak):** the tail `CRITICAL FACTS` *context-injection* block no longer emits
  owner-identity context — **neither** the `relationships` arm (no `partner:`/`children:` line, no
  `partner: <date>` bullet given shredded facts) **nor** the `personal_facts` arm. Asserts both leak
  paths are closed and that owner-identity context is delegated entirely to Fix 2 (no double-emission).
- **Unit (Fix 2):** core-profile block renders deterministically (byte-identical for identical
  inputs); role synonyms (wife/spouse/husband → partner) resolve to the partner entity; empty graph
  renders an empty/omitted block, not an error.
- **Unit (Fix 2 generality — proves it's not narrowed to relationships):** given a mix of facts,
  selection picks the **top-N by salience across categories** (a high-`recall_count` non-relationship
  fact — e.g. an allergy or employer — is included; a low-salience trivia fact is excluded); the block
  is capped at N; a **pinned** fact is force-included even with low salience, and an **unpinned**
  high-salience fact is force-excluded.
- **Unit (Fix 2 snapshot/cache):** incrementing `recall_count` mid-session does **not** change the
  rendered block or its component hash (frozen snapshot); changing membership (pin/add relationship)
  **does** change it exactly once.
- **Cache invariants:** the new core component participates in the aggregate hash; unchanged
  relationships → cache HIT (block byte-stable across turns); a changed relationship → exactly one
  re-render. Extends the existing `core_prompt.rs` cache-decision tests.
- **Integration (mock LLM):** in an owner DM seeded with a partner + 2 children **and** a
  non-relationship high-salience fact (e.g. a pinned allergy), "who is my wife?", "who is my spouse?",
  "who's <child>'s mom?", **and "am I allergic to anything?"** all surface the relevant fact in the
  system prompt and let the LLM answer — across repeated turns (no variance). The allergy case is the
  generality proof: a non-relationship recall is just as reliable. **Assert on the system-prompt bytes**
  (the core-profile block is present), not only on mock LLM output.
- **Seed via the graph, not flat facts:** integration tests must seed with `upsert_person` +
  person-facts (and, for the fallback path, owner identity-category flat facts + a pinned fact), so
  Fix 2's actual source of truth and the pin override are both exercised — not just `upsert_fact`.
- **Cache bust:** update the partner via `manage_people`; assert `core_cache_decision` reports the
  `core_profile` component changed on the next turn (and only that component).
- **Regression:** "who is aracely?" (direct entity lookup) still works. Note `text_relates_to_critical_identity`
  still matches `wife`/`children` for personal-memory tool-scoping (separate from the removed
  short-circuit) — confirm that tool-scoping behavior is unchanged.

## Out of scope

- `sqlite-vec` / ANN index (deferred; not on this critical path).
- The full polyglot memory build (typed numeric records, artifacts, reflection loop, decay).
- Re-extraction / backfill of historically shredded flat facts — Phase 1 entity work may revisit;
  not required for Phase 0 since the people graph is already the source of truth.

## Open questions

- [x] **Fix 2 source of truth for children** — **resolved: people graph first, flat identity facts as
      fallback.** Resolve from `people` records whose `relationship` is child-like (`child`, `son`,
      `daughter`); fall back to owner identity-category flat facts (`children`, `daughter_name`, …) so
      existing un-linked data still renders before Phase 1 does the owner-linking. (Caveat from Bug 3:
      children are likely *still* flat facts today, so the fallback is the path that actually fires
      until Phase 1.)
- [x] **Non-relationship `CriticalFactQuery` arms** — **resolved: keep `OwnerName` / `AssistantName`.**
      They are scalar lookups with no shredding; only `CoreRelationships` is structurally broken. The
      "dump beats LLM" pathology is specific to multi-row shredded values, not the whole mechanism.
- [ ] Exact placement/heading of the core-profile block within `render_core_prompt` ordering.
      (Leaning: after `channel_rules`, before the skills catalog — reads as an "about you" block
      without splitting the `## Tools` splice.)
- [ ] Salience tuning: starting values for **N** (≈30?), the `category_weight` table (which categories
      score high — identity/family/health), and the 0.6/0.3/0.1 weights. Start simple, tune against the
      owner's real fact set; treat as constants, not config, for v1.
- [ ] Should Fix 2's deterministic block let us *shrink* the volatile per-turn people injection
      (avoid double-emission)? (Defer; measure first — and note per Bug 3 there is little owner-DM
      people text to shrink today, since the graph is fetched but not rendered.)
