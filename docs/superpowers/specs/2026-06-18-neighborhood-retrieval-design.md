# Read-time Neighborhood Retrieval (Phase 1) — Design

**Date:** 2026-06-18
**Status:** Approved design — pending implementation plan
**Topic:** Memory recall that assembles the *connected neighborhood* of the entities a query is about, so the model can derive relational answers instead of failing on lookup misses.

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

**#3 was proven empirically:** given the facts in context ("If my mother is Conchi and my father is Galo Loor, who is Conchi's spouse?"), `gemma-4-26b` answered *"Based on what you've said, Conchi's spouse is Galo Loor."* — correct, and framed as an inference. So the reasoning step is **not** the limiter, and **neither is embedding strength** (a stronger model still wouldn't retrieve `father=Galo` for a "Conchi's spouse" query, because that fact mentions neither "Conchi" nor "spouse"). The limiter is **#2: assembling the neighborhood.**

---

## 2. Goal & non-goals

**Goal:** When a query is about an entity, assemble that entity's fact cluster plus the facts of the entities it is connected to (1-hop), inject that connected context, and let the model reason over it. General across people, projects, and concepts — not relationship-type-specific.

**Non-goals (this phase):**
- No explicit graph, no typed-edge extraction, no materialized adjacency index. (That is **Phase 2**, below.)
- No always-on passive-injection change. v1 hooks the **explicit search path** only.

### Relationship to the eventual graph (Phase 2)

The agreed long-term architecture (for a scaling, open-source aidaemon) is a **hybrid**: an incrementally, LLM-built graph layer (populated by the existing background consolidation job) traversed at read time, **with read-time neighborhood assembly as its bootstrap and permanent fallback**. This spec is **Phase 1 = the fallback layer**, deliberately built first because:
- It works on day one with no new persistent structure (no cold start for self-hosted users).
- It is the safety net that lets the future graph never need to be perfect (misses/stale/wrong edges are caught by read-time).
- Its entity-resolution + clustering logic is the same input a Phase-2 graph-builder needs.

Phase 2 (incremental LLM-built graph + traverse-with-fallback) is a **separate** spec → plan → build.

---

## 3. Scope decisions (resolved during brainstorming)

- **Coverage:** unified — both people/relationships and concepts/projects.
- **Entity resolution:** **semantic / LLM-driven, NOT deterministic token matching.** Token/surface matching was rejected: surface overlap ≠ meaning, so it (a) misses synonyms/paraphrase ("my mom's husband"), (b) fails cross-lingual ("esposa" vs "spouse"; facts are mixed EN/ES), (c) mis-resolves homonyms (there are two "David" person nodes). A wrong token match assembles the *wrong* neighborhood and the model then confidently derives a *wrong* answer — worse than a miss. Resolution must understand meaning.
- **Integration point (v1):** the explicit search path (`manage_memories action=search` → `search_facts_semantic`). That is exactly where the observed failures occurred (the model *did* search), it reuses the existing reranker + `memory_recall` telemetry, and it is a contained change. Passive injection can reuse the same function later.

---

## 4. Architecture

A new retrieval step that **wraps** the existing search rather than replacing it:

```
query
  │
  ├─ (existing) search_facts_semantic: vector + lexical + cross-encoder rerank → initial matches
  │
  ├─ (NEW) intent extraction        : fast-model pass → { entities[], relation_sought? }
  ├─ (NEW) semantic entity resolve  : map each extracted entity → stored entity (embeddings, not tokens)
  ├─ (NEW) neighborhood fetch       : pull each resolved entity's cluster (bounded)
  ├─ (NEW) merge + bound            : dedupe vs initial matches by id, rank by salience, cap
  │
  └─ enriched result set → model reasons (derives answer, states inferences as inferences)
```

The new logic lives in the **state layer** (it needs `FactStore` + `people` access). It is a single isolated, independently testable unit.

### Components (each one purpose, isolated)

1. **Intent extraction** — input: the query string. Output: `{ entities: Vec<String>, relation_sought: Option<String> }`. A cheap **fast-model** structured call that understands meaning/structure ("my mom's husband" → entity = the mother, relation = husband). **Gated** to recall/relational queries so it does not run on every message.

2. **Semantic entity resolution** — input: extracted entity strings. Output: resolved stored entities (person node ids and/or fact subjects/namespaces). Uses **embeddings** to map a mention to the closest stored entity, so wording/language does not break it. Disambiguates by context where multiple candidates exist.

3. **Neighborhood fetch** — input: resolved entities. Output: their fact clusters. Three fetch rules, applied to the *resolved* (not guessed) entity:
   - **Namespace rule:** entity is a namespaced subject `X` → pull all `X:*` facts (whole project/concept cluster).
   - **Co-mention rule:** entity name `E` → pull facts where `E` appears in key or value.
   - **Relationship-cluster rule:** entity is a person related to the owner → pull the owner's *other* relationship facts. This is what connects `Conchi`→`Galo` (they never co-occur in one fact; they are siblings in the owner's relationship set). It is relationship-*general* (driven by a relationship key vocabulary / person-typed values), not spouse-specific.

4. **Merge + bound** — dedupe additions against the initial matches by fact id; rank additions by **salience** (recency + `recall_count`); apply hard caps (max entities expanded, max facts added). Expansion **appends** to the top matches, never replaces them.

5. **Honesty** — keep a light instruction so the model *attempts* derivation rather than declining, while phrasing derived links as inferences. No new mechanism (proven behavior).

6. **Telemetry** — reuse the `memory_recall` tracing target: log entities resolved + facts added by expansion, so the behavior is observable in prod.

---

## 5. Bounds (must not bloat the prompt)

- `MAX_ENTITIES_EXPANDED` per query.
- `MAX_NEIGHBORHOOD_FACTS` added by expansion.
- Dedupe by fact id against initial matches.
- Rank additions by salience; drop the tail past the cap.
- Expansion is additive and capped; the original top matches always remain.

(Concrete numbers chosen during implementation, tuned against prompt budget.)

---

## 6. Error handling / degradation

- Intent extraction fails or returns nothing → **skip expansion**, return the existing search results unchanged. Never block recall on the new step.
- Semantic resolution finds no stored entity → no expansion for that mention.
- The step is purely **additive**: any failure degrades to current behavior, never worse.

---

## 7. Testing

Unit-test the neighborhood unit directly with seeded facts:
- (a) namespace cluster pulled for a concept query (`X:*`).
- (b) owner-relationship cluster pulled when a relationship fact resolves (the Conchi/Galo shape) — the key correctness case.
- (c) bounds respected (entity cap, fact cap, dedupe vs initial matches).
- (d) no expansion when nothing resolves; results unchanged.
- (e) resolution is semantic: a synonym/cross-lingual mention resolves to the same stored entity (guards against regressing to token matching).

Intent extraction is gated and model-driven; test the gate (which query shapes trigger it) deterministically, and treat the extraction call itself like other fast-model calls (mock in unit tests).

---

## 8. Open items for the implementation plan

- Exact gate for "is this a recall/relational query" (reuse existing recall heuristics where possible).
- Whether intent extraction is a dedicated fast-model call or piggybacks on an existing pass.
- Concrete cap values and salience formula.
- The relationship key vocabulary used by the relationship-cluster rule (kept general, derived from existing relationship categories).
