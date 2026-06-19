# Read-time Neighborhood Retrieval (Phase 1) — Design

**Date:** 2026-06-18
**Status:** Approved design (reconciled with the 2026-06-08 relational-recall spec) — pending implementation plan
**Topic:** Memory recall that assembles the *connected neighborhood* of the entities a query is about, so the model can derive relational answers instead of failing on lookup misses.
**Companion / predecessor:** [`2026-06-08-memory-relational-recall-design.md`](./2026-06-08-memory-relational-recall-design.md). This spec supersedes that spec's **unshipped** relational pieces (the `{intent,entities}` classifier, the full search-before-deny gate, and the 0e edge layer) and reuses their designs; see §9.

---

## 1. Problem

Recall is **lookup-shaped, not reasoning-shaped**. The pipeline retrieves the top-k facts by similarity and the model looks for a *direct hit*. When the answer is not a verbatim stored fact but must be **derived** from several facts, recall fails:

- "Who is Conchi's spouse?" — `mother=Conchi` and `father=Galo` are stored as separate facts; no `Conchi's spouse` fact exists. The model deflects to "your partner Aracely."
- "Who is my kid's mom?" — `children=Bella/Cami` and `partner=Aracely` are stored separately; no `kids' mother` fact exists.

### Evidence (measured this session)

Three distinct sub-problems, only one of which is the bottleneck:

| # | Problem | Status |
|---|---------|--------|
| 1 | Find the right fact despite wording ("spouse"≈"partner") | mostly handled by the cross-encoder reranker (v0.11.5) |
| 2 | **Assemble the connected context** (get Conchi *and* Galo together) | **the bottleneck — this spec** |
| 3 | Derive the answer from assembled facts (Conchi+Galo → spouses) | **already works on the local model** |

**#3 was proven empirically:** given the facts in context ("If my mother is Conchi and my father is Galo Loor, who is Conchi's spouse?"), `gemma-4-26b` answered *"Based on what you've said, Conchi's spouse is Galo Loor."* — correct, framed as an inference. So the reasoning step is **not** the limiter, and **neither is embedding strength** (a stronger model still wouldn't retrieve `father=Galo` for a "Conchi's spouse" query, because that fact mentions neither "Conchi" nor "spouse"). The limiter is **#2: assembling the neighborhood.**

This independently re-confirms the 2026-06-08 spec's conclusion: its salience core profile (shipped) guarantees the relevant facts are *present*, but **presence is necessary, not sufficient** — derivation still fails because the facts aren't assembled as a connected set the model is prompted to reason over.

---

## 2. Goal & non-goals

**Goal:** When a query is about an entity, assemble that entity's fact cluster plus the facts of the entities it is connected to (1-hop), inject that connected context, let the model reason over it — and, if the model still tries to *deny* knowledge of a queried entity without searching, force a search first. General across people, projects, and concepts; not relationship-type-specific.

**Non-goals (this phase):**
- No explicit graph / typed-edge store / materialized adjacency index. (That is **Phase 2** = the prior spec's **0e**.)
- No always-on passive-injection rewrite. v1 hooks the **explicit search path** plus the **completion-phase gate**.
- No provenance work (prior spec's 0f).

### Relationship to the eventual graph (Phase 2)

The agreed long-term architecture (for a scaling, open-source aidaemon) is a **hybrid**: an incrementally, LLM-built graph layer (populated by background consolidation) traversed at read time, **with read-time neighborhood assembly as its bootstrap and permanent fallback**. This spec is **Phase 1 = the fallback layer**, built first because it works day-one with no new persistent structure, it is the safety net that lets the future graph never need to be perfect, and its entity-resolution + clustering logic is the same input a graph-builder needs.

**Phase 2 already has a blueprint:** the prior spec's **0e** (minimal owner-relationship edges: `(subject_person_id, relation, object_person_id)`, `relation ∈ {partner, child, parent}`, fail-open migration, co-parent-as-labeled-inference). Phase 2 is a separate spec → plan that adopts 0e and generalizes it beyond the owner star.

---

## 3. Scope decisions (resolved during brainstorming)

- **Coverage:** unified — both people/relationships and concepts/projects.
- **Entity resolution:** **semantic / LLM-driven via a shared intent classifier, NOT deterministic token matching.** Token matching was rejected: surface overlap ≠ meaning, so it misses synonyms/paraphrase ("my mom's husband"), fails cross-lingual ("esposa" vs "spouse"; facts are mixed EN/ES), and mis-resolves homonyms (two "David" person nodes). A wrong token match assembles the *wrong* neighborhood and the model confidently derives a *wrong* answer — worse than a miss.
- **Integration points (v1):** (a) the explicit search path (`manage_memories action=search` → `search_facts_semantic`) for neighborhood assembly; (b) `completion_phase` for the search-before-deny gate. Passive injection (`core_profile`) is unchanged — it stays the foundation.
- **Search-before-deny gate is IN Phase 1** (user decision), bundled with neighborhood retrieval despite the prior spec's "ship the gate separately" caution. Accepted as a larger first change for immediate robustness; mitigated by reusing the existing `answer_grounding.rs` plumbing and the bounded-retry pattern.

---

## 4. Architecture

A new retrieval step that **wraps** the existing search, plus a completion-phase gate, both driven by **one shared classifier**:

```
                         ┌──────────────────────────────────────────┐
user query ──────────────▶  SHARED LLM CLASSIFIER (fail-open, owner-DM-gated)
                         │  → { intent: relational|recall|none,       │
                         │      entities: Vec<String> }               │
                         └───────────────┬──────────────────────────┘
            ┌────────────────────────────┴───────────────┐
            ▼ (during explicit search)                    ▼ (at completion)
  search_facts_semantic (existing: vector+lexical+rerank) │
            │                                             │
  + semantic entity resolve  (entities → stored entities) │
  + neighborhood fetch       (each entity's cluster)       │
  + merge + bound            (dedupe, salience, caps)      │
            │                                             │
  enriched results → model reasons                         │
                                                           ▼
                              SEARCH-BEFORE-DENY GATE: if the reply
                              denies/asserts a specific personal fact
                              about a classifier-named entity that was
                              NOT retrieved this turn → block, force a
                              lookup, retry (bounded). Extends the
                              existing list-fabrication gate.
```

The new retrieval logic lives in the **state layer** (needs `FactStore` + `people` access); the gate lives in `completion_phase` alongside the existing `answer_grounding` gate.

### Components (each one purpose, isolated)

1. **Shared intent classifier** — the entity-resolver, and the single piece that serves both the assembly path and the gate. Wire `src/agent/intent/llm_classifier.rs` (today shadow-only — this is its first production use) with a new function returning `{ intent: relational | recall | none, entities: Vec<String> }`. Cheap fast-model call, **fail-open** (provider error/disabled → no-op, never blocks), **owner-DM-gated**, and gated behind an optional cheap pre-filter so it doesn't run on every turn. `entities` is a **list** (compound questions name multiple entities). Reuses the prior spec's PR3 design verbatim.

2. **Semantic entity resolution** — map each classifier-extracted entity string → stored entity (person node id / fact subject / namespace) via **embeddings**, not string overlap. Wording/language-robust; disambiguates by context.

3. **Neighborhood fetch** — input: resolved entities. Three fetch rules over flat facts:
   - **Namespace rule:** entity is a namespaced subject `X` → pull all `X:*` facts (project/concept cluster).
   - **Co-mention rule:** entity name `E` → pull facts where `E` appears in key or value.
   - **Relationship-cluster rule:** entity is a person related to the owner → pull the owner's *other* relationship facts. This is what connects `Conchi`→`Galo` (they never co-occur in one fact; they're siblings in the owner's relationship set). Relationship-*general* (driven by a relationship key vocabulary / person-typed values), not spouse-specific.

4. **Merge + bound** — dedupe additions vs initial matches by id; rank by **salience** (recency + `recall_count`); hard caps (max entities expanded, max facts added). Additive — never replaces the top matches. *(Mirror the prior spec's "cap after merge over entities" rule so a selected entity carries all its attributes or none.)*

5. **Search-before-deny gate** (`completion_phase`, extends `answer_grounding.rs`) — when the classifier flags a **relational/recall** turn and the candidate reply **denies or asserts** a specific personal fact about a named entity **not retrieved this turn**, inject a forced-continuation directive ("you didn't search for `<entity>` — look it up before answering"), bounded by a monotonic retry counter (≈2) so it can never loop. Reuses the existing block-and-retry plumbing. The shipped `find_ungrounded_list_entities` (list-fabrication, keyword) stays; this adds the **single-entity denial/assertion** case it doesn't cover, now classifier-triggered.

6. **Honesty** — the model derives + states inferences (proven). Carry the prior spec's **co-parent rule**: never *assert* "partner ⇒ biological mother/co-parent"; phrase such derivations tentatively ("Aracely, your partner, is most likely Bella's mom"). Direct facts are asserted; cross-relationship inferences are labeled.

7. **Telemetry** — reuse the `memory_recall` target: log entities resolved, facts added by expansion, and every gate trigger (entity, fired/escaped), so both behaviors are tunable against real traffic.

### Precedence / composition (three interventions now exist — order them)

Per turn, at most one relational intervention fires, in this order (extending the prior spec's order; the coreference gate is already shipped and live):

1. **Coreference gate** (shipped: `looks_like_pronoun_referent_followup` → `CoreferenceGroundingRequired`) — resolves *which* entity a pronoun refers to; runs first because it determines the subject the others would act on.
2. **Neighborhood assembly** (this spec, during retrieval) — assembles the connected cluster; no forced search, just better context.
3. **Search-before-deny gate** (this spec, at completion) — only if the reply still denies/asserts an unretrieved entity. Forces a search.

This keyway prevents double-forcing a lookup in one turn.

---

## 5. Bounds (must not bloat the prompt)

- `MAX_ENTITIES_EXPANDED` per query; `MAX_NEIGHBORHOOD_FACTS` added; dedupe by id; rank additions by salience; drop the tail past the cap.
- Gate retries capped (≈2) by a dedicated monotonic counter.
- Classifier kept off most turns by the cheap pre-filter.
- (Concrete numbers chosen in the plan, tuned against prompt budget.)

---

## 6. Error handling / degradation

- **Classifier fails/disabled** → fail-open: no entity resolution (skip expansion), gate degrades to the existing keyword/list-fabrication behavior. Never blocks a reply.
- Semantic resolution finds nothing → no expansion for that mention.
- Gate: if both lookup tools are unavailable → skip the gate (nothing to force).
- Everything is additive; any failure degrades to current behavior, never worse.

---

## 7. Testing

- **Neighborhood unit** (seeded facts): (a) namespace cluster for a concept query; (b) owner-relationship cluster when a relationship fact resolves (the Conchi/Galo shape — the key case); (c) bounds (caps, dedupe); (d) no expansion when nothing resolves; (e) semantic resolution: a synonym/cross-lingual mention resolves to the same stored entity (guards against regressing to token matching).
- **Classifier**: gate/pre-filter logic deterministic; the call itself mocked (fast-model).
- **Gate**: (a) relational denial of an unretrieved entity is blocked then corrected after a forced search (cf. existing `test_ungrounded_list_reply_is_rejected_then_corrected`); (b) a *grounded* partial answer ("I don't have Juan's phone, but he's your coworker") is NOT blocked; (c) bounded escape after K retries lets an earned denial through; (d) general-knowledge denial ("president of France") does not fire.
- **Precedence**: coreference + gate both matching → coreference anchors first, no double force.

---

## 8. Open items for the implementation plan

- Exact pre-filter for "run the classifier this turn" (reuse `looks_like_personal_memory_recall_question` / `tool_result_indicates_no_evidence` as a cheap recall-biased gate, never the sole authority).
- Concrete cap values, salience formula, retry cap.
- The relationship key vocabulary for the relationship-cluster rule (kept general, derived from existing relationship categories).
- Whether the classifier output struct is shared field-for-field with the prior spec's planned `{intent, entities}` so Phase 2 (0e/kinship) and a future provenance route can consume the same call.

---

## 9. Reconciliation with the 2026-06-08 spec (what shipped, what we reuse)

Verified against the code:

| Prior-spec piece | Shipped? | Disposition here |
|---|---|---|
| Fix 2 — salience core profile | ✅ `build_core_profile` (`system_prompt.rs:589`) | **Foundation.** Unchanged; we assemble on top of it. Its "presence ≠ derivation" finding motivates this spec. |
| Fix 3 — search-before-deny gate | ⚠️ only `answer_grounding.rs` list-fabrication (keyword) shipped | **Revived & completed** as Component 5 — adds the classifier-triggered single-entity denial/assertion case, extending the existing module. |
| Shared `{intent, entities}` classifier | ❌ `llm_classifier` still shadow-only | **Built here** (Component 1) — first production wiring; the entity-resolver both this spec and Phase 2 need. |
| 0e — owner-relationship edges | ❌ not shipped | **Deferred to Phase 2**; the prior 0e (fail-open migration, co-parent-as-inference, owner-star edges) is the blueprint. |
| Coreference pronoun gate | ✅ live (`main_loop.rs:528`) | **Kept**; sits first in the precedence order (§4). |
| 0f — provenance | ⚠️ capture only (`source_excerpt`) | Out of scope. |

**Net:** this spec doesn't duplicate the prior work — it reuses the shipped foundation (core profile, coreference gate, list-grounding plumbing) and revives the prior design's two unshipped pieces we need (the classifier + the full gate), while taking the **read-time** route to #2 that the prior spec reached for via edges (0e). Edges become Phase 2.
