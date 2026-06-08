# Memory: Reliable Relational Recall — Design

> Status: **design proposed, awaiting user review → plan.**
> Date: 2026-06-08.
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

### Bug 2 — Per-turn fact retrieval is non-deterministic

`src/state/sqlite/facts.rs:769-902` (`get_relevant_facts_for_channel`):

- Embeds the **short user message** ("who is my wife") each turn and scores every fact (O(n) scan).
- Score = cosine + lexical (≤0.55) + freshness boost (≤0.15); must clear a tight `0.3` threshold;
  only the top `max_facts` survive. `default_max_facts = 20` (`config.rs:1019`).

Facts near the 0.3 boundary pass on one turn and fail the next → "I don't have information" one
moment, correct answer later. The most important, most stable facts (partner, children) are subject
to the same lottery as trivia.

### Bug 3 — The structured entity layer already exists, and the broken path ignores it

`route_people_fact` (`src/memory/manager.rs:823-962`) already routes `category == "people"` facts
into the `people` + `person_facts` tables — a real entity layer (Person record + PersonFacts linked
by `person_id` FK). That is *why* "who is aracely?" is clean: it resolves a Person. But:

- The guardrail dump (Bug 1) reads the **flat `facts` table**, not the people graph.
- Children are stored as flat facts, **not linked** to the owner Person.
- There are **no explicit edges** between entities (owner —partner→ Aracely; owner —child→ Bella),
  so nothing can traverse.

In owner DMs the people block IS injected deterministically (`system_prompt.rs:418-431`:
`get_all_people()` + `get_person_facts(owner.id, None)`), but it lives in the **volatile per-turn
tail**, capped/rendered ad hoc (`skills/mod.rs:1076-1081` takes 10), and the relationship guardrail
doesn't read it.

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
- Leave any **non-relationship** critical-fact guardrails untouched (scope the change narrowly to the
  relationship case; do not gut unrelated deterministic recalls).
- Net effect: "who is my wife?" now takes the same path as "who is bella's mom?" — which works.

Fix 1 alone is insufficient because it re-exposes Bug 2 (the LLM path is itself unreliable). Fix 2
closes that.

### Fix 2: A deterministic, cache-resident "core relationships" block

Guarantee the owner's core relationships are **always present and reliable**, independent of the
semantic-threshold lottery — and place them where they are **cached**, not re-paid every turn.

**What goes in it:** a small, deterministic summary derived from the people graph (source of truth
for relationships):

- The owner's partner/spouse (name + birthday if known), resolved by relationship role.
- The owner's children (name + birthday if known), resolved by relationship role / owner-linked facts.
- Synonym normalization so "wife"/"spouse"/"husband" all resolve to the `partner` role and
  "kids"/"daughter"/"son" to `children`.

**Where it goes — the caching decision:** inject it into the **session-static CORE prompt**
(`src/agent/runtime/core_prompt.rs` → `render_core_prompt`), NOT the volatile per-turn tail.

Rationale (verified against the prefix-stability architecture on this branch):

- The CORE prompt is cached per `session_id`, keyed by a component-hash aggregate
  (`core_cache_decision`, `core_prompt.rs:157`). On a HIT the bytes are reused — it is the cacheable
  prefix.
- The owner's partner/children are **session-stable** (they don't change mid-conversation), so they
  belong in the cached prefix, not the per-turn tail where the current people/facts injection lives.
- Cost: paid **once per session** as a cache write (~50-120 tokens for partner + 2-4 children), then a
  cache read on every later turn → effectively free per-turn.
- It only busts the session cache when a relationship actually changes — `core_cache_decision`
  already diffs component hashes and re-renders only on change. Rare and correct.
- This removes the most important facts from the 20-fact semantic lottery → **kills Bug 2's variance**
  for relationship recall.

**Prefill impact:** near-zero net new volume. Owner DMs already inject all people + all owner
person-facts into the tail today; Fix 2 is a *reliability + placement* change, not a bulk addition. It
may even let us trust the core block and reduce noisy tail injection later (out of scope here).

**Prefix-stability constraint:** the block must render deterministically (stable field order, stable
formatting) so identical inputs produce identical bytes — consistent with the existing
`core_prompt_renders_identically_for_identical_inputs` invariant. Adding it changes the core
component-hash aggregate; ensure the new component participates in `core_cache_decision` so a
relationship change correctly invalidates and re-renders.

### Phase 0 data flow (after)

```
user: "who is my wife?"
  → (Fix 1) no guardrail short-circuit
  → normal LLM path
  → system prompt CORE prefix already contains (cached) deterministic core-relationships block:
        Partner: Aracely Zambrano (b. 1990-06-30)
        Children: Bella (b. 2012-05-17), Cami (b. 2020-08-20)
  → LLM answers naturally: "Your wife is Aracely Zambrano."
user: "who's bella's mom?"
  → same cached block present every turn
  → LLM reasons Bella → you → Aracely → "Aracely, your partner, is Bella's mom."
```

Deterministic because the block is always there (cache-resident), not because of a hard-coded reply.

---

## Phase 1 — Durable relational layer (scoped, not fully specified here)

Once Phase 0 is in, promote relationships from string keys to **explicit, queryable edges** so
associative recall generalizes beyond the hand-picked core block:

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
  "who are my children", etc.; non-relationship critical queries (if any remain) still classify.
- **Unit (Fix 2):** core-relationships block renders deterministically (byte-identical for identical
  inputs); role synonyms (wife/spouse/husband → partner) resolve to the partner entity; empty graph
  renders an empty/omitted block, not an error.
- **Cache invariants:** the new core component participates in the aggregate hash; unchanged
  relationships → cache HIT (block byte-stable across turns); a changed relationship → exactly one
  re-render. Extends the existing `core_prompt.rs` cache-decision tests.
- **Integration (mock LLM):** in an owner DM with a seeded partner + 2 children, "who is my wife?",
  "who is my spouse?", and "who's <child>'s mom?" all surface the partner/children in the system
  prompt and let the LLM answer — across repeated turns (no variance).
- **Regression:** "who is aracely?" (direct entity lookup) still works.

## Out of scope

- `sqlite-vec` / ANN index (deferred; not on this critical path).
- The full polyglot memory build (typed numeric records, artifacts, reflection loop, decay).
- Re-extraction / backfill of historically shredded flat facts — Phase 1 entity work may revisit;
  not required for Phase 0 since the people graph is already the source of truth.

## Open questions

- [ ] Fix 2 source of truth for children: resolve from `people` records with a child relationship, or
      from owner-linked person-facts? (Leaning: people graph, consistent with partner resolution.)
- [ ] Exact placement/heading of the core-relationships block within `render_core_prompt` ordering.
- [ ] Should Fix 2's deterministic block let us *shrink* the volatile per-turn people injection
      (avoid double-emission)? (Defer; measure first.)
- [ ] Are there non-relationship `CriticalFactQuery` arms worth keeping, or is the whole deterministic
      critical-reply mechanism suspect for the same "dump beats LLM" reason?
