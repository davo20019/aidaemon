# Memory: Reliable Relational Recall — Design

> Status: **design proposed, revised after code-verified review + live re-test → ready for plan.**
> Date: 2026-06-08.
> Revision note: Bug 2 narrowed to derived recall; Bug 3 people-injection claim corrected
> (graph is fetched but not rendered in owner DMs); Fix 1 widened to close the tail `CRITICAL FACTS`
> leak (Fix 1+2 must ship together); Fix 2 given concrete `CoreInputs` wiring + cost accounting;
> open questions on children source-of-truth and the deterministic arms resolved.
> **Generalization revision:** Fix 2 reframed from a partner/children block to a **salience-selected
> core *profile*** (auto-salience + user pin override, bounded, session-frozen) so it fixes the *class*
> — any high-salience fact dropping out of recall — not just the wife/children instance.
> **Fix 3 added:** a **loop-level "search-before-deny" groundedness gate** (in `completion_phase`, not a
> prompt-only rule) so the long-tail/non-core lookups are trustworthy — the agent may not deny knowledge
> of a named entity it never searched for. No schema; sliced as **0d**, independent of 0a-0c.
> **Implementation-readiness revision (2nd review pass):** split the profile gate from the
> people-graph gate (use runtime `people_enabled`, not `config.people.enabled` — fixes a People-off
> regression); replaced the "frozen vs per-turn" prose with one explicit frozen-membership/content-hash
> algorithm; added per-source salience rules (`person_facts` has no `recall_count`); added partner
> merge/render rules (the real shredded-pairing bug); sliced Phase 0 into 0a/0b/0c; flagged the
> owner-DM integration test that must be rewritten.
> **3rd review pass:** resolved the Fix 1 `personal_facts` arm contradiction by slice-aligning the
> tail-block removal (0a removes relationships only; 0b removes `personal_facts` — closes the 0a-only
> identity regression); defined "session start" and distinguished structural edits vs `remember_fact`
> for cache busting; gave children the same name↔date pairing rigor as partner; revised the PR sequence
> so **0d ships in its own PR** (riskiest heuristic piece, not bundled with the first PR); pointed Fix 3
> at `completion_phase` + the existing `tool_result_indicates_no_evidence` / `looks_like_personal_memory_recall_question`
> helpers, narrowed its trigger to personal-recall (no general-knowledge mis-fire), specified entity
> extraction + empty/tool-unavailable fallbacks, and clarified Fix 2 (not Fix 3) fixes the transcript.
> **4th review pass (post-fix live re-test, 2026-06-08):** a fresh-context re-test (after shipping the
> coreference pronoun gate, below) showed Fix 2's premise — *facts-in-context ⇒ correct answer* — is
> **necessary but not sufficient**: with partner/children/parent facts all present in context, the model
> still (a) **misattributed** the owner's own father (`father: Galo`) as the *children's* father and
> (b) **failed to derive** "Bella's mom" — answering both **without searching** (2 lookups in ~17 turns).
> Three changes result: **(1)** Fix 2's render gains an explicit-**ownership** rule (`Your father: Galo
> Loor`, never a bare `father: Galo`) — the Galo bug is subject-ambiguity, not the name↔date pairing Fix 2
> already handles; **(2)** Fix 3's trigger moves **off the keyword lists**
> (`looks_like_personal_memory_recall_question` + denial-phrase list + lowercase proper-noun extraction)
> **onto the existing LLM intent classifier** (`agent/intent/llm_classifier.rs`), with the keyword lists
> demoted to an optional cheap pre-filter — the gate fires rarely, so a semantic decision there is cheap
> and generalizes across phrasings/languages; **(3)** a **minimal owner-relationship edge slice (0e)** is
> pulled forward from Phase 1 — kinship queries ("whose dad/mom") resolve by edge traversal,
> deterministically, with **no phrase matching**. Related shipped fix: a **coreference pronoun gate**
> (`looks_like_pronoun_referent_followup` → `CoreferenceGroundingRequired`) now anchors "...infer about
> *her*?" to the conversational subject and forces a lookup — it is itself keyword-based and is a
> candidate to fold into the same classifier (2).
> **5th review pass (consistency + 0e correctness):** fixed the co-parent contradiction (deriving a
> mother from partner∧parent is the biology the spec forbids — now stored-edge = exact, co-parent =
> labeled inference); dropped the unimplementable `∧ female` clause (`Person` has no gender field,
> verified); added 0e's missing **delivery contract** (classifier-gated in-loop traversal → system
> directive, no phrase matching) and a **precedence order** for the three relational mechanisms
> (0e → Fix 3 → coreference); narrowed PR1 to "fixes the *dump* transcript, not the re-test kinship —
> that's PR3"; called out that Fix 3/0e are the **first production wiring** of the shadow-only
> `llm_classifier` (new `needs_grounding` fn, fail-open, owner-DM-gated); moved "who's <child>'s mom?"
> out of the 0b test suite into 0e; aligned merge/render test + data-flow with rule 4's `Your partner:`;
> scoped rule 4 to owner-only facts; de-stale-d Bug 2 (presence ≠ correct derivation).
> **6th review pass (provenance gap — live forensics, 2026-06-08):** a second live probe — "when did I
> first mention Consuelo Montesdeoca?" — exposed a **fourth** failure class the prior fixes don't touch:
> **memory provenance is destroyed by consolidate-then-prune.** Verified: messages are events; the
> Consolidator distills them into facts/episodes/goals and the Pruner then physically deletes them
> (`events/store.rs:342` `DELETE FROM events WHERE consolidated_at IS NOT NULL AND created_at < ?`) — a
> term with ~12 message hits vanished to **0** within ~25 min. So "when/where did I learn X" is
> structurally unanswerable: only the distilled fact survives, with no link to the originating utterance,
> and the model also misroutes such questions to `goal_trace`. New **slice 0f**: capture provenance
> (`first_seen_at`, `source` ∈ {user_stated, derived, inferred}, `source_excerpt`/event id) **at
> consolidation time, before pruning**, + a `manage_memories(action='provenance')` recall path routed via
> the same classifier (so it stops grabbing `goal_trace`). Bonus: `source=derived` makes the
> bot-fabricated "Conchi" (web-searched, never user-stated) answerable honestly instead of masquerading as
> a user fact. Open question added on prune-timing aggressiveness.
> **7th review pass (sequencing + consistency):** resolved the **PR3↔PR4 classifier cycle** — the shared
> `{intent, entity}` classifier now ships in **PR3** (0e needs it), PR4/PR6 only consume it; deleted the
> stale "Fix 2 fixes Bella's mom / gate never fires" scope note (invalidated by the live re-test);
> aligned Phase 1 co-parent language with 0e (stored edge = exact, partner∧parent = labeled inference,
> never asserted); flagged that `facts`/`person_facts` **already have a `source` column** (verified) so
> 0f must *repurpose* it with a backfill decision, not add it; split 0f into **PR2.5 capture (early,
> time-sensitive) + PR6 recall (depends on PR3)**; added the minimum-shippable-path floor (PR1+PR2+PR3),
> a 0e migration-from-flat-facts test, and a precedence-keyway integration test.
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

### Live re-test (2026-06-08, post-coreference-fix) — presence is not enough

A fresh-context re-test confirmed the diagnosis and sharpened it. With the owner's partner, children,
and parent all present in the injected profile, on a **fresh** session:

```
who's cami's dad?        → "Galo Loor."                ✗  (Galo is the OWNER's father, not the children's)
whose Bella's mom?       → "I don't have information…"  ✗  (Aracely is partner + mother — derivable)
```

Telemetry: across ~17 turns the model issued **2** `manage_memories` calls total — both failing
relational turns answered from the **in-context profile with no search**. Two consequences for this
spec: (1) a flat `father: Galo` line is read as *the nearest subject's* father → **render ownership
explicitly** (Fix 2 revision); (2) presence alone doesn't make derivation reliable → **edges + a
groundedness gate that doesn't depend on keyword scoping** (slice 0e + revised Fix 3).

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

**But the live re-test (above) sharpened this:** retrieval variance is **not the whole story for derived
recall.** With partner + children + parent facts *all present* in context, the model still misattributed
(Galo) and failed to derive (Bella→mom). So derived kinship has **two** failure modes — (i) the fact
isn't retrieved (Bug 2), and (ii) the fact *is* present but **subject-binding / derivation** fails. Fix 2
addresses (i) by guaranteeing presence; (ii) is why 0e (edge traversal) and Fix 3 (assertion gate) exist.
Presence is necessary, not sufficient.

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

- **Phase 0 (this spec's priority):** stop the embarrassment now on data we can eyeball. **Fix 1** stops
  the bad deterministic dump, **Fix 2** guarantees the high-salience core is always present (cached),
  **0e** makes kinship derivation deterministic via owner-relationship edges, **Fix 3** makes the
  long-tail lookup trustworthy via a loop-level groundedness gate, and **0f** preserves provenance.
  **Scope honesty:** this started as "two surgical fixes" but, driven by two live re-tests, Phase 0 is now
  **six slices (0a-0f)** — effectively a relational-memory + provenance rewrite, not a surgical patch.
  That may be the right call, but it is a deliberate expansion, not the original framing. Phase 0 is
  sliced (see **Phase 0 sequencing** below) with an explicit **minimum shippable path** (PR1+PR2+PR3) so
  the surgical core can ship first if time-boxed.
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
  to Fix 2).
- **Tail-block removal is slice-aligned — never remove an arm before CORE covers it (resolves the
  arm-migration question, and the 0a regression):** the end state is that the tail `CRITICAL FACTS`
  context-injection block stops emitting owner-identity context entirely (**both** `relationships`
  *and* `personal_facts`), because Fix 2 becomes the single cache-resident home for all stable
  owner-identity context (avoids double-emission and the "exact values, do not infer" instruction
  fighting the LLM). But because Fix 2 lands in slices, the removal must track it so nothing regresses
  in between:
  - **In 0a** (CORE carries a *relationship*-only section): remove **only** the `relationships` arm
    from the tail block. **Keep** the `personal_facts` / name arms in the tail — 0a's CORE section does
    not cover employer/location/allergy yet, and removing them now would regress non-relationship
    identity recall (the 0a-only risk @cursor flagged).
  - **In 0b** (CORE carries the full salience *profile*): remove the `personal_facts` arm too, now that
    CORE covers it. After 0b the tail identity block is fully retired.

  This keeps the fix at the level of the *mechanism* (deterministic dump beats LLM) while never leaving
  a window where identity context lives in neither place.
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

**Salience varies by source — the formula above assumes `recall_count`, which only flat `facts`
have (verified).** `person_facts` has **no `recall_count`** column (`migrations.rs:642-653`: only
`confidence`, `created_at`, `updated_at`). So apply the formula **per source type**, or partner
birthdays stored only as `person_facts` rank below flat-fact trivia:

- **Flat `facts`:** full formula (has `recall_count`).
- **`Person` records with a relationship role** (partner/spouse/children): **always high salience** —
  they are the relationships this whole spec is about; do not subject them to the recall lottery.
- **`person_facts`:** `category_weight + recency` only (no recall term); or inherit the owner-person's
  recall signal if cheap. Confidence may stand in for the missing recall component.

- **Pinned wins (force-in); unpinned can be force-out** — "force-out" means **not selected into the
  top-N**, i.e. an unpinned fact competes on salience and can lose its slot; it is *not* a separate
  user "exclude" action (that would need its own API — out of scope). Direct pin control sits on top of
  automatic capture (addresses the "no control / unpredictable memory" complaint from the research
  notes).
- **Bounded** by N so the cached core stays small no matter how many facts accumulate over years.
- **Rendered deterministically**, grouped for readability (an "About you" block: identity lines, then
  relationships with linked name+birthday, then other high-salience facts). Synonym normalization so
  "wife"/"spouse"/"husband" resolve to the `partner` role and "kids"/"daughter"/"son" to `children`.

**Relationship merge/render rules (the actual hard part — don't leave it at "render
deterministically"):** the original bug *is* ambiguous name+date pairing across shredded rows
(`partner: Aracely`, `partner: June 30, 1990` are two rows). Specify the join, mirroring the resolved
children fallback:

1. **Partner:** prefer a `Person` whose `relationship ∈ {partner, spouse, wife, husband}`; render its
   name; attach birthday from *that person's* `person_facts` (`birthday`/`birth_date`). Only if no such
   Person exists, fall back to flat identity facts — and then pair name↔date by a deterministic
   heuristic (a value parseable as a date → birthday; otherwise → name), never emitting a bare
   `partner: <date>` line. If the pairing is ambiguous, render name only and drop the unattached date.
2. **Children:** same precedence (per resolved Open Question #1) — `people` records with a child-like
   role first, owner-linked / flat facts as fallback. The flat fallback needs the **same name↔date
   pairing heuristic as partner**: shredded `children`→`Bella` / `children`→`August 20, 2020` rows must
   merge into `Bella (b. 2020-08-20)`, never a bare `children: <date>` line. With multiple children the
   pairing is genuinely ambiguous across un-keyed flat rows — if names and dates can't be associated
   deterministically, render names only and drop unattached dates rather than guess a wrong pairing.
3. The render never emits a label with a lone date value — that exact output is the bug.
4. **Explicit relationship ownership (the Galo bug — verified in live re-test):** every relationship
   line names **whose** relation it is, from the owner's perspective. Render the owner's *ascending*
   relations as `Your father: Galo Loor` / `Your mother: …`, partner as `Your partner: Aracely …`,
   children as `Your children: Bella …` — never a bare `father: Galo` / `partner: Aracely` that a flat
   reader (or the LLM) can re-bind to the nearest entity in the question. This is pure rendering (no
   keywords, no schema) and is what stops "who's Cami's dad?" from grabbing the owner's *own* father.
   **Applies only to owner-scoped facts:** a `person_facts` `father:` row belonging to *another* person
   must NOT get the "Your" prefix — qualify ownership from the fact's subject, not unconditionally.
   Ownership labels are necessary but not sufficient for *derived* relations (Bella→mom) — those need
   the 0e edges.

**Cache-stability guard — frozen membership, content-hashed (resolves the "frozen vs per-turn"
tension):** salience inputs like `recall_count` change every turn; hashed live they would thrash the
cache. Pin the algorithm precisely:

0. **"Session start" defined:** the freeze runs on the **first owner-DM turn for a `session_id`** —
   i.e. when `core_profile` is first assembled for that session — **not** on daemon restart or a calendar
   day boundary. For a long-lived reused `session_id` (e.g. a persistent Telegram DM), the frozen
   membership persists across days until a structural edit busts it; that is intended (stability over
   freshness within a session).
1. **Session start:** load the union source → score (per-source salience above) → select the top-N
   ordered **fact-ID list** → **freeze that ID list for the session.**
2. **Each turn:** re-fetch *content* for the frozen IDs **plus** any facts whose **membership** changed
   structurally this session (see step 3) → render → the `core_profile` component hash =
   `canonical(sorted [(id, value, pinned)] of the rendered set)`. **`recall_count` and `recency` are
   inputs to selection only and are NOT in the hash.**
3. **What changes membership mid-session (busts the hash once):** only *structural* edits — `pin`/`unpin`
   (0c), a `manage_people` add/edit (new child, changed partner value). These recompute the frozen ID
   list and bust the hash exactly once, then restabilize.
4. **What does NOT change membership mid-session:** (a) a `recall_count` bump — changes neither the
   frozen ID list nor the hashed content; (b) a plain **`remember_fact`** mid-session — the new fact is
   available same-session via the normal/tail + Fix 3 search paths, but it does **not** enter the frozen
   core membership until the next session (this is the "Accepted limitation" below; `remember_fact` is
   *not* a structural membership edit, unlike `pin`/`manage_people`). Both → **no core re-render, no
   thrash.**

This gives both properties at once: passive recall drift is invisible to the cache (property the
freeze buys), while a genuine edit still propagates the same session (property the per-turn content
read buys). It is the single most important contract for the implementer; build to step 2's hash
definition exactly.

**Accepted limitation (state it, don't hide it):** because membership freezes at session start, a fact
the user supplies *mid-session* ("remember I'm allergic to penicillin") does **not** enter the cached
core profile until the **next** session. It is still recalled same-session via the normal/tail path,
and is promoted to core next session. This is acceptable but is exactly the "why didn't it remember
what I just told it?" complaint class — so it is a deliberate trade (cache stability over same-session
core promotion), not an oversight.

**Pin mechanism (small additive piece):** the `Fact` model has no `pinned` field today. Add a
`pinned` flag (a `pinned INTEGER DEFAULT 0` column on `facts`/`person_facts`, or a small `pinned_facts`
table) plus a `manage_memories(action='pin'|'unpin')` action. The salience formula's `pinned ? 1.0`
branch is **inert until that column lands** — v1 auto-salience never hits it.

**Phase 0 sequencing (slice it; pick the slice in the plan):**

| Slice | Delivers | Schema | Risk |
|-------|----------|--------|------|
| **0a** | Fix 1 + a minimal deterministic **relationship** section in CORE (partner/children via people-graph-first + flat fallback, with the merge/render rules above) | none | Low — fixes the transcript |
| **0b** | Generalize 0a's section to the full **salience profile** (top-N across categories, frozen snapshot, per-source salience) | none | Medium — fixes the class |
| **0c** | `pinned` column + `manage_memories(pin/unpin)` + pin override in selection | yes | Low, additive |
| **0d** | Fix 3 — classifier-triggered search-before-deny groundedness gate (+ bounded retry + soft prompt rule) | none | Medium — semantic detector, bounded |
| **0e** | Minimal owner-relationship **edges** (owner—partner/child/parent→ + inverses), pulled forward from Phase 1; kinship queries resolve by traversal — no phrase matching | yes | Medium — small schema, high payoff (kills the Galo / derived-recall bugs) |
| **0f** | Provenance capture at consolidation (`first_seen_at`/`source`/excerpt, **before** prune) + `manage_memories(action='provenance')` recall routed via the classifier | yes | Low-med — additive schema; capture is time-sensitive (land early) |

0a alone closes the reproduced *dump* failure and the leaks; 0b delivers the *class* fix this spec argues
for; **0e** makes kinship / derived recall deterministic (the live-re-test Galo / Bella's-mom bugs); 0c
adds user control; **0d** makes the long-tail (non-core) lookups trustworthy. 0d's *code* is
**independent** of 0a-0c/0e (no shared schema; lives in `completion_phase`, not `core_prompt`) — but note
one soft dependency: its no-op-path test assumes 0b's core profile exists. **Pin tests (force-include a
low-salience pinned fact, etc.) belong to 0c** — do not assert them before the column exists. 0e's edges
are also what let 0d's forced search resolve a derived denial by traversal rather than re-deriving from
flat facts. **0f** is orthogonal to all of the above — it adds *provenance* (when/how a fact was learned),
the one recall class 0a-0e don't address; its capture half should land **early** because provenance can
only be stamped going forward (the prune already destroyed the past).

**Recommended PR sequence (revised — 0d ships separately):**

| PR | Contents | What it actually closes |
|----|----------|-------------------------|
| **PR1** | 0a + Fix 1 (relationships arm removed from tail; `personal_facts` kept) + Fix 2 rule 4 (explicit-ownership render) | The **original garbled-dump transcript** + the *display* half of the Galo bug (`Your father: Galo`, never re-bound to the child). **Does NOT** make derived kinship correct — "who's Bella's mom?" still waits for PR3. |
| **PR2** | 0b salience profile (+ remove `personal_facts` tail arm) | The *class* fix for **direct** recall (wife, allergy, employer). Not kinship derivation. |
| **PR2.5** | 0f **capture** only (provenance stamping at consolidation, before prune) — *not* the recall action | Stops the bleeding: provenance can only accrue forward, so land the capture hook early. No user-facing change yet. |
| **PR3** | **Shared intent classifier** (`needs_grounding` + `kinship` + `provenance` outputs, one call) + 0e owner-relationship edges + traversal + **kinship injection path** | The live-re-test kinship failures (Cami's-dad→owner, Bella's-mom→Aracely). This is the PR that fixes the re-test, not PR1. |
| **PR4** | 0d completion-phase groundedness **gate** (consumes the PR3 classifier; adds the block-and-retry) | The long tail + assertion fallback (entities 0e's owner-star edges don't cover). |
| **PR5** | 0c pins + `manage_memories(pin/unpin)` | User control. |
| **PR6** | 0f **recall** (`manage_memories(action='provenance')`, routed via the PR3 classifier's `provenance` intent) | "When/where did I learn X" + honest `derived` sourcing (the "Conchi" fabrication). |

**Classifier-dependency resolution (critical sequencing — Option A):** 0e's kinship injection is
**classifier-gated**, and 0d and 0f also route through a classifier. Rather than ship the classifier in
PR4 (which 0e/PR3 precedes) or build three calls, **PR3 ships the one shared classifier** with a single
output struct — e.g. `{ intent: kinship | provenance | needs_grounding | none, entity: Option<String> }`
— and PR4/PR6 only **consume** it (PR4 adds the gate, PR6 adds the recall route). This removes the
PR3-needs-PR4 cycle and the "one classifier, two/three outputs" promise becomes a concrete deliverable
owned by PR3. Name that struct once in the plan.

**Honest scoping:** the *original* transcript (garbled partner dump) is closed by PR1+PR2; the **live
re-test** failures are closed by **PR3 (0e + classifier)**. Rule 4 only stops the *prompt* from misreading
`Your father: Galo` as the child's father — it does not make Bella→mom derivable. Do not claim the
re-test is fixed before 0e lands.

**Minimum shippable path (if time-boxed):** **PR1 + PR2 + PR3** (dump + direct recall + kinship) is the
must-have set — it closes both the original transcript and the live re-test. **PR2.5 (0f capture)** is
high-value-early because it is time-sensitive and tiny. **PR4 (gate), PR5 (pins), PR6 (provenance
recall)** are deferrable. The plan should offer this as the floor.

**Do NOT bundle 0d into the first PR.** It is the single *riskiest* piece — a heuristic behavioral gate
with a real false-positive surface and loop re-entry — and it is exactly the kind of change that needs
isolated bake time and trigger metrics. Co-shipping it with the already-medium-risk 0b enlarges the
blast radius of the first PR for no sequencing benefit (0d doesn't depend on 0b's *code*). Ship
**PR1+PR2 first** (these close the reproduced failure and the class), then **0d as its own PR**. 0a
alone remains the time-boxed escape hatch — with the documented caveat that 0a keeps `personal_facts`
in the tail, so non-relationship identity recall is unaffected until 0b.

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
2. **Gating — two levels, do NOT gate the whole profile on People (regression, verified):** the tail
   `CRITICAL FACTS` block that Fix 1 removes is gated on `inject_personal && user_role == Owner`
   (`system_prompt.rs:283`) — **not** on People being enabled. Gating the whole core profile on People
   would mean an owner who has disabled People still loses *all* deterministic identity recall (name,
   allergy, employer — none of which are people-graph data). Two distinct gates:
   - **Whole profile:** `user_role == Owner` **and** `should_inject_personal_memory()` **and**
     `depth == 0` (root agent only). This is the same gate as the block it replaces, so no config loses
     coverage.
   - **People-graph subsection only** (partner/children resolved from `people`/`person_facts`):
     additionally requires the **runtime** `people_enabled` flag — `get_setting("people_enabled")`
     (`system_prompt.rs:405-414`), **not** `config.people.enabled`. The rest of the people path keys off
     the runtime setting; the core profile must match it or the two will disagree (profile shows a
     partner the user just disabled, or vice-versa). When People is off, the people-graph subsection is
     empty but the **flat-identity half still renders** — "am I allergic to anything?" stays reliable.

   For any non-owner session the component hashes to **empty** — CORE is keyed by `session_id`, so an
   unguarded block would leak the owner's profile into a non-owner's cached prefix. Tests: (a) a
   non-owner session renders no core-profile block; (b) an owner with `people_enabled = false` but
   identity flat facts present **still** gets a profile (people-graph subsection empty, flat half
   present).
3. **Sub-agents:** spawned executors (`depth > 0`) should **not** receive this block (no owner-DM
   recall context); gate it off for non-root agents.
4. **Pre-fetch + freeze:** read the **union source** (owner `person_facts` ∪ owner identity/profile
   flat `facts`) into the inputs *before* `assemble_core_inputs`, mirroring the existing async pre-fetch
   for skills/specialists. The selection/freeze/hash behavior is specified exactly by the 4-step
   "frozen membership, content-hashed" algorithm in Fix 2 above — build to that, especially step 2's
   hash definition (`recall_count`/`recency` excluded from the hash). The per-turn content read must be
   the same cheap owner query already run, not a new heavy scan; the top-N scan+sort happens **once at
   session start** over the union source (O(n), bounded, off the per-turn path).

**Cost note (tighten the "paid once per session" claim):** the *render + token write* is once per
session (until a real edit); the *owner read + hash* is **per-turn** (that's how cache invalidation
detects an edit). Because salience scoring uses the **frozen** snapshot, recall-count drift does **not**
change the hash — only a genuine pin/relationship/identity edit does. Net DB cost is unchanged from
today — `system_prompt.rs:283-309` already fetches identity facts every owner turn — so the win is
**prompt tokens**, not the query. State it that way to avoid implying the DB read is eliminated.

### Fix 3: "Search-before-deny" groundedness gate (loop-level, not prompt-only)

Fix 1 + Fix 2 make the *common* case reliable (the high-salience core is always present). Fix 3 makes
the *long tail* reliable: facts **not** in the core profile must be pulled on demand — and the agent
must not be allowed to *deny knowledge* of something it never searched for. The reproduced failure
*"I don't have information about Bella's mother"* was exactly this: a denial emitted **without a
lookup**.

**Why a loop gate, not a prompt rule (the design decision):** a system-prompt instruction ("look it
up, don't guess") already exists (`skills/mod.rs:1034`) and clearly does **not** reliably fire — that
is the whole reason this transcript happened. Prompting is the soft layer; we do **not** rely on it
alone. aidaemon's loop already evaluates outputs and forces re-iteration via structural gates
(`stopping_phase`, `tool_execution/guards.rs`, completion/verification, `result_learning`). Those
gates today check **task completion / mutation success** — none checks **answer groundedness**. Fix 3
adds that missing category of evaluation as a real gate.

**Anchor it in `completion_phase.rs`, not the generic "stopping_phase":** that is where the
`verification_block_count` counter and the existing forced-continuation machinery already live
(`completion_contract.rs:78`, `completion_phase.rs:1009/1156`). Build Fix 3 as a sibling gate there so
it reuses the proven block-and-retry plumbing rather than a parallel mechanism.

**Trigger via the LLM intent classifier, not keyword lists (revised — addresses the keyword-rot
objection).** The decision "is this reply an *ungrounded* answer to a *personal-recall* question?" is a
semantic judgment; approximating it with `looks_like_personal_memory_recall_question` + a denial-phrase
list + lowercase proper-noun extraction is exactly the brittle, per-phrasing surface this codebase keeps
accreting (it misses paraphrases and other languages — e.g. "quién es el papá de Cami" matches none of
the lists). The gate fires on a small minority of turns, so it is the **cheapest possible place** to
spend one classifier call. Route it through the existing `src/agent/intent/llm_classifier.rs`:

- **Primary signal — classifier.** Add a lightweight intent: *given* `(user_text, candidate_reply,
  whether a lookup tool fired this turn)`, return `{ needs_grounding: bool, entity: Option<String> }` —
  true when the candidate reply **asserts or denies a specific personal fact** (about a person/relationship
  the owner would expect the agent to know) that was **not** retrieved this turn. The classifier returns
  the entity span too, replacing the fragile hand-rolled extractor.
- **Optional cheap pre-filter (cost guard, not the decision).** To avoid a classifier call on every turn,
  a fast word-boundary denial-phrase check (`tool_result_indicates_no_evidence`) **plus** "no
  `is_personal_memory_tool` call this turn" may gate *whether the classifier runs*. This is a
  recall-biased pre-filter (a false positive just costs one classifier call, which then says no); it must
  **never** be the sole trigger, and the **assertion** case — a confident *wrong* answer like "Galo is
  Cami's dad", which carries no denial phrase — bypasses the pre-filter and relies on the classifier when
  budget allows. Keep `looks_like_personal_memory_recall_question` only as part of this pre-filter, never
  as the scoping authority.

1. **General-knowledge guard is now the classifier's job.** "Who is the president of France?" → "I don't
   have information" must not fire the gate; the classifier distinguishes public-figure / general-knowledge
   from personal-recall directly, instead of relying on a keyword allow-list to approximate it.
2. **Entity comes from the classifier.** No separate lowercase proper-noun span heuristic. Empty/`None`
   entity → force a generic `manage_memories(search)` over the user message; if the classifier is
   unavailable (provider error / disabled), **fall back to the pre-filter-only path** (denial phrase +
   no-lookup) so the gate degrades to today's keyword behavior rather than vanishing — and if even the
   entity is unscoped there, let the denial through rather than force a blind search.
3. **Intervention** — do **not** send the denial. Inject a forced-continuation directive: *"You asserted
   you don't know about `<entity>` but did not search memory. Call `manage_people` / `manage_memories`
   for `<entity>` before answering."* If `manage_people` is disabled (runtime `people_enabled = false`),
   point only at `manage_memories`; if **both** lookup tools are unavailable, **skip the gate** (there is
   nothing to force). Mirrors the existing forced-nudge pattern (edit-stall hint, read-saturation nudge).
4. **Bounded escape** — cap forced retries with a dedicated monotonic counter (same shape as
   `verification_block_count`): after K (≈2) attempts, allow the denial through so the gate can never
   infinite-loop. A genuine "not in memory" answer survives — it just must be *earned* by a real search
   first. **Log every gate trigger** (entity, fired/escaped) so the heuristic can be tuned against real
   traffic.

**Detector honesty (the tuning surface):** "is this an ungrounded personal-fact claim?" is a judgment,
now made by the classifier rather than phrase lists — which is *more* robust to paraphrase/language but
is not free of error (a mis-classification is possible). The sharpest false-positive case is the
**nuanced partial answer** — "I don't have Juan's *phone number*, but he's your coworker" is grounded
and must NOT block; this is exactly the kind of distinction a classifier handles better than a denial
regex, so the classifier prompt must call it out with that example. The bounded escape (below) caps the
cost of any wrong trigger at K wasted iterations regardless, so the gate stays safe while the classifier
is tuned against logged triggers.

**Honest cost — denials are no longer free (not just wrong ones):** because aidaemon recalls mostly via
*automatic context injection*, not tool calls, "no lookup tool this turn" is true on the vast majority
of turns. So when the gate *correctly* fires on a genuine unknown, it **always** costs a forced
search + an extra LLM turn before the (now earned) denial. That is the intended trade — correctness over
latency on denials — but state it plainly; it is not only mis-triggers that cost iterations.

**Scope note — what each fix actually owns (corrected after the live re-test).** Fix 2 guarantees
partner/children are *present* in CORE, which fixes the **original dump** transcript and direct recall —
but the live re-test proved presence is **not** enough for *derived* kinship: "who's bella's mom?" still
failed with everything in context. **0e (edge traversal) owns kinship derivation**, not Fix 2 and not
Fix 3. Fix 3 owns a *third* class: a denial/assertion about an entity in memory but **not** resolvable by
the core profile or an owner-star edge (the "who is Juan Perez?" coworker case). So: PR1+PR2 close the
dump; **PR3/0e** closes the re-test kinship; **0d/Fix 3** closes the long tail. Do not conflate them.

**Positive synergy with Fix 2's accepted limitation:** Fix 3 partially *covers* the mid-session gap —
a fact the user stated this session that never reached the frozen core profile will still be found,
because the gate forces a search before letting a denial through. The two fixes reinforce each other.

**Layering (defense in depth):** keep the soft prompt rule *and* add the gate. Prompt nudges the model
to search proactively; the gate refuses to let a denial through unsearched. Neither alone is trusted.

**Classifier wiring — this is the first production decision-path use of `llm_classifier` (verified):**
the module is today **shadow-mode scaffolding** ("*not* wired into the agent's decision path") and its
`classify_intent` returns a coarse `LlmIntentClass` (schedule / memory-storage / recall / action /
knowledge / other) — **not** the `{ intent, entity }` shape these fixes need. So the **classifier ships in
PR3** (it's on 0e's critical path), and PR4/PR6 consume it:
- Add a **new** classifier function with a **single shared output struct** —
  `{ intent: kinship | provenance | needs_grounding | none, entity: Option<String> }` — not three calls,
  not an overload of `classify_intent`. One call serves 0e's kinship flag (PR3), Fix 3's groundedness
  (PR4), and 0f's provenance routing (PR6).
- **Fail-open**, matching the module's existing ~5s-timeout / any-error → no-op contract: a classifier
  error degrades to the pre-filter-only path (Fix 3) and to "no kinship injection" (0e), never to a hang
  or a blocked reply.
- Gate behind a config flag, **default on in owner DMs only** (where personal-recall lives), so the
  classifier never runs for non-owner / public traffic.
- Budget: the pre-filter keeps the classifier off the vast majority of turns; state the per-turn cap so
  PR4 isn't estimated as "a regex change."

**No schema.** Pure loop/policy logic + prompt text + the new classifier call. Generalizes beyond denials
later — the same gate shape can enforce "don't *assert* a specific fact you didn't retrieve," not just
"don't deny."

### Slice 0e: Minimal owner-relationship edges (pulled forward from Phase 1)

Fix 2's ownership-labeled render (rule 4) stops the **display** ambiguity ("whose father?"), but it does
not make **derived** relations answerable — "who's Bella's mom?" requires connecting *owner—child→Bella*
with *owner—partner→Aracely*. The live re-test proved presence + ownership labels are not enough for that
hop: the model must be able to *traverse*, or it guesses / denies. 0e adds the smallest edge layer that
makes the reproduced kinship questions deterministic — owner-centric only, not the full graph.

- **Schema:** a single directed-edge store keyed to the owner Person — `(subject_person_id, relation,
  object_person_id)` with `relation ∈ {partner, child, parent}` and a stored inverse (`parent`↔`child`;
  `partner` symmetric). Either a small `relationship_edges` table or reuse `person_facts` with a typed
  `relation` + `object_person_id` FK. Children become **owner-linked entities** here (the spec already
  flags they are orphan flat facts today).
- **Population:** derive edges from existing data at migration time (owner's `partner`/`children`/parent
  facts + `people` records) and keep them in sync when `manage_people` / `remember_fact` writes a
  relationship. No new user action required.
- **Retrieval — direct edges are ground truth; co-parent is an *inference*, not "deterministic"
  (resolves an internal contradiction):** a parent named by a **stored edge** is exact —
  `dad_of(X) = { p : p —parent→ X }` returns the owner when `owner —parent→ X` is stored, and that is
  what makes "who's Cami's dad? → you, not Galo" reliable. But deriving a *mother* from
  `partner(owner) ∧ owner —parent→ X` is the **partner-of-a-parent-is-the-other-parent assumption** —
  exactly the biology this spec says elsewhere to **never assert** (step-parent, ex-partner, remarriage
  all break it). Two honest options; the plan picks one:
  - **Preferred — store both parent edges.** When the data supports it, persist `Aracely —parent→ Bella`
    *as its own edge* (not derived), so `mom_of(Bella)` is a real edge lookup. This is the only way the
    answer is genuinely deterministic.
  - **Otherwise — return co-parent as a labeled *hint*, not fact.** Hand the LLM
    "owner —parent→ Bella; owner —partner→ Aracely (possible co-parent)" and let it phrase tentatively
    ("Aracely, your partner, is most likely Bella's mom") rather than asserting it as ground truth.

  **Drop the `∧ female` clause entirely (verified):** `Person` has **no gender/sex field** in the schema
  or `traits.rs`, so a gender predicate is unimplementable in 0e v1 and not needed — the owner's known
  partner edge already supplies the co-parent candidate. Gender inference is out of scope.
- **Delivery contract — when traversal runs and where the result goes (was unspecified):** 0e is useless
  if the edges exist but never reach the LLM. **Do NOT add a phrase-matching trigger.** Run the kinship
  resolver **in the loop, gated by the same Fix 3 intent classifier** (one classifier, two outputs):
  when the classifier flags a turn as a **kinship/relationship question over the owner star**, the loop
  (a) runs the deterministic traversal, (b) injects the resolved entities/edges as a **system directive**
  (same mechanism as `CoreferenceGroundingRequired`) *before* the LLM composes its answer. This keeps
  "no phrase lists" intact (the classifier decides, not keywords) and makes the kinship answer
  edge-driven rather than re-derived from flat facts. Precomputing every kinship line into CORE is
  rejected — unbounded (N children × relations) and still leaves derivation to the model.
- **Why this and not another rule:** there is **no keyword surface** — "who's Cami's dad", "quién es el
  papá de Cami", "Bella's mother?" all resolve through the same `child`/`parent`/`partner` edges via the
  classifier, not a phrase list. It is the structural answer to the class the keyword objection is about.
- **Composition / precedence (three mechanisms now touch relational recall — order them explicitly):**
  per turn, at most one intervention fires, in this priority:
  1. **0e traversal** — if the classifier flags an owner-star kinship question *and* an edge resolves it,
     inject the traversal result and let the LLM phrase it. No gate, no forced search.
  2. **Fix 3 gate** — only if 0e did **not** resolve it (no edge / non-kinship / long-tail entity) and the
     reply asserts/denies an unretrieved personal fact. Forces a search.
  3. **Coreference gate** (shipped, keyword-based `looks_like_pronoun_referent_followup`) — governs
     pronoun-referent binding ("...about *her*?"), orthogonal to kinship; if it and Fix 3 both match,
     the coreference anchor runs first (it determines *which* entity Fix 3/0e would even be about). This
     keyway prevents double-forcing a lookup in one turn. The coreference gate is a candidate to fold into
     the same classifier later (noted in Phase 1).

### Slice 0f: Provenance preservation (capture-at-consolidation) + recall

0a-0e make recall *correct*; 0f makes it *accountable* — answering "when/where did I learn this?" and
"did I actually tell you, or did you infer it?". The live forensics showed this is impossible today: the
**Pruner deletes consolidated events** (`events/store.rs:342`), so by the time the user asks, the
originating utterance is gone and only the distilled fact remains — with no backlink. The model compounds
it by misrouting "when did I…" to `goal_trace`.

**Why it must live at consolidation:** the raw text exists *only* in the event, and the event is pruned
after consolidation. Provenance must be stamped onto the fact **at extraction time, before the prune** —
it cannot be reconstructed later. This makes 0f's *capture* half time-sensitive: it only protects facts
learned **after** it ships (historical provenance is already lost — see Out of scope).

- **Capture (in the Consolidator, before prune):** when a fact is extracted from an event, stamp:
  - `first_seen_at` — the originating event's timestamp (the answer to "when").
  - `source` ∈ `{user_stated, derived, inferred, imported}` — `user_stated` only when the value came from
    a user message; `derived` for web-search/tool-derived values (the "Conchi" case); `inferred` for
    consolidation-LLM guesses. This is the truthfulness lever. **NB — `facts` and `person_facts` ALREADY
    have a `source` column (verified):** `person_facts.source` defaults to `'agent'`; `facts.source` is
    bound on every insert (facts.rs:489/495) with free-text like `'agent'`/`'user'`/tool name. 0f does
    **not** add `source` — it must **repurpose** that column to this controlled vocabulary, which forces a
    **backfill decision for existing rows** (they cannot be retroactively proven `user_stated` → backfill
    to `inferred`/`unknown`). Decide repurpose-in-place vs a new `origin` column explicitly; do not add a
    duplicate "source".
  - `first_seen_at`, `source_excerpt` (short, bounded), and/or `source_event_id` ARE net-new — the
    "when/where". `source_event_id` may dangle after prune; `source_excerpt` is the durable copy.
  - **Default must not be `user_stated`** — default to `inferred`/`unknown` and set `user_stated` only on
    a verified user-message origin, or the fabrication problem persists.
- **Recall path:** `manage_memories(action='provenance', query)` returns matching facts with
  `first_seen_at` + `source` + excerpt. Route "when/where/how did I tell you X" to it via the **same
  classifier** added in 0d (a `provenance` intent), so the model stops reaching for `goal_trace`. The
  closest-available answer for a *pre-0f* fact is its `created_at` (extraction time), labeled clearly as
  *"recorded"*, not *"you told me"*.
- **Truthfulness synergy:** with `source` populated, a `derived` value is answered honestly ("I inferred
  'Conchi' from a web search — you didn't tell me that") instead of being recalled as a user fact. Same
  groundedness principle as Fix 3, applied to *storage* rather than *reply*.
- **Capture half is independent of 0a-0e** (own schema + consolidator hook) and worth landing **early**
  (PR2.5) so provenance accrues, even though it fixes no reproduced transcript. **The recall half is NOT
  independent** — `manage_memories(action='provenance')` routes through the PR3 shared classifier's
  `provenance` intent, so it depends on PR3. State that dependency, don't claim full independence.

### Phase 0 data flow (after)

```
user: "who is my wife?"
  → (Fix 1) no guardrail short-circuit
  → normal LLM path
  → system prompt CORE prefix already contains (cached) deterministic core-PROFILE block,
    salience-selected (not hard-coded to relationships):
        ## About you
        Name: David Loor · Location: … · Employer: …      ← identity (also salience-selected)
        Your partner: Aracely Zambrano (b. 1990-06-30)    ← explicit ownership (Fix 2 rule 4)
        Your children: Bella (b. 2012-05-17), Cami (b. 2020-08-20)
        Your father: Galo Loor                            ← owner's OWN parent, never a bare `father:`
        Allergy: penicillin   [pinned]                    ← arbitrary high-salience / pinned fact
  → LLM answers naturally: "Your wife is Aracely Zambrano."
user: "who's bella's mom?"
  → same cached block present every turn (with explicit-ownership lines)
  → (Fix 3/0e classifier) flags owner-star kinship → loop traverses edges, injects result as a directive:
      exact if `Aracely —parent→ Bella` is stored; else co-parent *candidate* (owner—child→Bella ∧
      owner—partner→Aracely) — labeled as inference, not asserted as biology
  → LLM phrases from the injected edges → "Aracely, your partner, is Bella's mom." (tentative if inferred)
user: "am I allergic to anything?"
  → same block → "Yes — penicillin."   (the *class* fix: not a relationship, still reliable)
user: "who is Juan Perez?"            (a coworker — NOT in the core profile)
  → not in CORE block → LLM is about to answer "I don't have information about Juan"
  → (Fix 3 gate) intent classifier flags: personal-recall answer asserting/denying a specific fact with
    no lookup this turn → BLOCKED
  → forced retry → LLM calls manage_people(view, "Juan Perez") → fact found → answers correctly
     (or, after a real search, legitimately: "I don't have anything on Juan" — now *earned*)
```

Deterministic because (Fix 2) the core block is always present and (Fix 3) the loop refuses to let a
denial through unsearched — not because of a hard-coded reply for one question type.

---

## Phase 1 — Durable relational layer (scoped, not fully specified here)

Once Phase 0 is in, promote relationships from string keys to **explicit, queryable edges** so
associative recall generalizes beyond the bounded core-profile block (which guarantees the top-N but
can't traverse arbitrary connections).

**Note (4th pass):** the *minimal* owner-centric edges — `owner —partner→ Aracely`,
`owner —child→ Bella/Cami`, `owner —parent→ Galo` and inverses — are **pulled forward into Phase 0 as
slice 0e**, because the live re-test showed they are on the critical path for the reproduced kinship bugs
(whose-dad / whose-mom), not a later nicety. Phase 1 below is then the *generalization* beyond the
owner-centric star:

- Explicit edges **beyond the owner star** (person↔person edges not involving the owner) and richer
  relation types. Convert any remaining orphan flat facts into owner-linked entities.
- Relation-typed retrieval: a relationship question resolves a role → entity deterministically, then
  expands 1 hop for context, instead of semantic-cosine-on-flat-facts.
- Co-parent / derived relations resolved the **same way 0e settled it** — a stored parent edge is exact;
  `partner-of-owner ∧ parent-of-child` is a **labeled co-parent inference, surfaced as a hint, never
  asserted as ground truth** (and never materialized as a stored `parent` edge). This avoids the
  biological/step assumption the data doesn't
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
- **Unit (Fix 1, tail leak — slice-aware):** after **0a**, the tail `CRITICAL FACTS` block no longer
  emits the `relationships` arm (no `partner:`/`children:` line, no `partner: <date>` bullet given
  shredded facts) but **still emits `personal_facts`** (kept until 0b). After **0b**, it emits neither
  arm — owner-identity context is delegated entirely to Fix 2 (no double-emission). Assert the
  slice-appropriate state, not "both removed" unconditionally.
- **Unit (Fix 2):** core-profile block renders deterministically (byte-identical for identical
  inputs); role synonyms (wife/spouse/husband → partner) resolve to the partner entity; empty graph
  renders an empty/omitted block, not an error.
- **Unit (Fix 2 generality — proves it's not narrowed to relationships):** given a mix of facts,
  selection picks the **top-N by salience across categories** (a high-`recall_count` non-relationship
  fact — e.g. an allergy or employer — is included; a low-salience trivia fact is excluded); the block
  is capped at N. **Pin behavior (a pinned fact force-included at low salience; an unpinned
  high-salience fact losing its slot) is a 0c test** — do not assert it before the `pinned` column
  exists.
- **Unit (Fix 2 merge/render):** seed **shredded** flat facts (`partner`→`Aracely`,
  `partner`→`June 30, 1990` as two rows, mirroring the real transcript) and assert the rendered block
  pairs them into `Your partner: Aracely (b. 1990-06-30)` (ownership-qualified per rule 4) and **never**
  emits a bare `partner: <date>` line. Also seed the people-graph form (`Person{relationship: partner}` +
  a `person_facts` birthday) and assert it wins over the flat fallback. This is the test that proves the
  actual bug is fixed.
- **Unit (Fix 2 per-source salience):** a partner stored only as a `Person`/`person_facts` (no
  `recall_count`) still ranks into the profile above high-`recall_count` flat trivia.
- **Unit (Fix 2 rule 4 — explicit ownership):** seed an owner `father`/`parent` fact and assert the
  rendered line is `Your father: Galo Loor` (ownership-qualified), and that **no** bare `father: Galo`
  line is ever emitted. This is the render-level guard against the live-re-test Galo misattribution.
- **Unit (0e migration — from today's data shape):** children are **flat facts** today (Bug 3), so the
  migration must derive `owner —child→ Bella/Cami` (+ inverse `owner —parent→` ) from those flat
  `children` facts, `owner —partner→ Aracely` from the partner Person/facts, and `owner —parent→ Galo`
  from the owner's parent facts. Assert the migration produces these from realistic seed data (flat child
  rows, not pre-linked entities), and — for the **exact** `mom_of` path — optionally creates
  `Aracely —parent→ Bella/Cami` when partner+child are both owner-linked.
- **Unit (0e edges):** migration derives `owner —child→ Bella/Cami`, `owner —partner→ Aracely`,
  `owner —parent→ Galo` (+ inverses) from seeded facts/people. **Exact-edge cases:** `dad_of("Cami")`
  returns the **owner** (not the owner's father Galo) via the stored `owner —parent→ Cami` edge;
  `grandparent_of("Bella")` returns Galo via two stored hops. **Co-parent (inference) case:** with only
  `owner —parent→ Bella` + `owner —partner→ Aracely` stored, `mom_of("Bella")` returns Aracely **labeled
  as a co-parent candidate, not ground truth** (or, if the dual edge `Aracely —parent→ Bella` is stored,
  as exact). Assert the inference is **never silently materialized** as a stored `parent` edge, and that
  the `∧ female` predicate is absent (no gender field).
- **Integration (0e, mock LLM):** in an owner DM seeded via edges, "who's Cami's dad?" resolves to the
  owner and "whose Bella's mom?" resolves to Aracely **across repeated turns, no variance, no forced
  search** — the exact pair the live re-test failed. Assert the resolved entities reach the system
  prompt / tool result the LLM sees.
- **Integration (precedence keyway — 0e/Fix3/coreference):** a turn that matches **both** the coreference
  gate and a kinship question ("who's *her* mom?" after Bella was the subject) fires **exactly one**
  intervention in the defined order (coreference anchors the referent → 0e traverses → no Fix 3 forced
  search). Assert no double-forced lookup and that the precedence order (0e → Fix 3 → coreference anchor
  first) holds.
- **Unit (0f capture):** a fact extracted from a user-message event gets `source = user_stated` and
  `first_seen_at` = the event's timestamp; a fact extracted from a web_search/tool-derived value gets
  `source = derived`; a consolidation-LLM guess gets `source = inferred`. Assert the **default is not**
  `user_stated`.
- **Unit (0f survives prune):** stamp provenance, then run the Pruner (`DELETE FROM events WHERE
  consolidated_at IS NOT NULL`); assert the fact still answers `first_seen_at`/`source`/excerpt —
  provenance lives on the fact, not the deleted event.
- **Integration (0f recall routing):** "when did I tell you my mom is X?" routes via the classifier to
  `manage_memories(action='provenance')`, **not** `goal_trace`, and returns `first_seen_at` + `source`; a
  `derived` fact ("Conchi") is reported as inferred, not user-stated. This is the exact live-forensics
  failure (the bot picked `goal_trace` twice).
- **Gate regression (Fix 1+2):** owner DM with `people_enabled = false` and identity flat facts
  present still renders a core profile (people-graph subsection empty, flat-identity half present);
  non-owner session renders no profile.
- **Unit (Fix 2 snapshot/cache):** incrementing `recall_count` mid-session does **not** change the
  rendered block or its component hash (frozen snapshot); changing membership (pin/add relationship)
  **does** change it exactly once.
- **Cache invariants:** the new core component participates in the aggregate hash; unchanged
  relationships → cache HIT (block byte-stable across turns); a changed relationship → exactly one
  re-render. Extends the existing `core_prompt.rs` cache-decision tests.
- **Integration (mock LLM) — 0b is DIRECT recall only, NOT kinship derivation:** in an owner DM seeded
  with a partner + 2 children **and** a non-relationship high-salience fact (for 0b use a
  **high-`recall_count`** allergy/employer — *not* a pinned one; pinned-fact inclusion is a 0c test),
  "who is my wife?", "who is my spouse?", **and "am I allergic to anything?"** all surface the relevant
  fact in the system prompt and let the LLM answer — across repeated turns (no variance). The allergy
  case is the generality proof: a non-relationship recall is just as reliable. **Assert on the
  system-prompt bytes** (the core-profile block is present), not only on mock LLM output. **Do NOT put
  "who's <child>'s mom?" in this 0b suite** — the live re-test showed that can fail even with the full
  profile present; it belongs to the **0e integration** test above (kinship is edge-derived, not a
  presence guarantee).
- **Seed via the graph, not flat facts:** integration tests must seed with `upsert_person` +
  person-facts (and, for the fallback path, owner identity-category flat facts), so Fix 2's actual
  source of truth is exercised — not just `upsert_fact`. (Add a pinned fact to the seed only in the 0c
  test suite, once the column exists.)
- **Cache bust:** update the partner via `manage_people`; assert `core_cache_decision` reports the
  `core_profile` component changed on the next turn (and only that component).
- **Regression:** "who is aracely?" (direct entity lookup) still works. Note `text_relates_to_critical_identity`
  still matches `wife`/`children` for personal-memory tool-scoping (separate from the removed
  short-circuit) — confirm that tool-scoping behavior is unchanged (add a test asserting "who is my
  wife?" still scopes personal-memory tools after `CoreRelationships` removal).
- **Existing test to rewrite:** `test_system_prompt_pins_critical_facts_for_owner_dm`
  (`src/integration_tests/part_00.rs:784`) asserts the old `CRITICAL FACTS` block with
  `partner: Alice`-style lines. Fix 1 removes that block's identity arms, so this test must be rewritten
  to assert the **core-profile** block instead. Budget for it — it is not optional.
- **Unit (Fix 3 classifier trigger):** with a **mocked classifier**, the gate fires when it returns
  `needs_grounding=true` and is suppressed when false. Cases the *classifier prompt* must get right
  (assert via fixture, not regex): (a) "who is the president of France?" → "I don't have information" →
  **no fire** (general knowledge); (b) nuanced partial "I don't have Juan's phone number, but he's your
  coworker" → **no fire** (grounded); (c) a confident *assertion* with no lookup ("Galo is Cami's dad")
  → **fire** — the case the old denial-phrase list could never catch (no denial phrase). **Pre-filter
  test:** the cheap denial-phrase + no-lookup pre-filter only gates *whether the classifier runs*, never
  the final decision; assert a denial-phrase reply with a lookup this turn does not even call the
  classifier.
- **Unit (Fix 3 classifier-unavailable fallback):** classifier mocked to error → the gate degrades to
  the pre-filter-only (denial phrase + no-lookup) path rather than disappearing.
- **Unit (Fix 3 entity from classifier):** the entity returned by the classifier is passed to the forced
  directive; the **empty-entity fallback** is exercised (classifier returns `entity=None` → generic
  `manage_memories` search, or gate skip — assert it does not interpolate an empty `<entity>`).
- **Unit (Fix 3 tool availability):** with `people_enabled = false` the directive points only at
  `manage_memories`; with **both** lookup tools unavailable the gate is skipped (no forced retry).
- **Integration (Fix 3 gate, mock LLM):** a person exists in memory but is **not** in the core profile;
  scripted LLM denies on turn 1 **without** a lookup → the gate blocks the reply and forces a retry;
  turn 2 the LLM calls `manage_people(view)` and answers correctly. Assert the denial was **not**
  delivered to the channel and that a lookup tool was invoked before the final answer.
- **Integration (Fix 3 bounded escape):** scripted LLM denies **and** refuses to search across K+1
  attempts → after the cap the denial is delivered (no infinite loop); assert the dedicated retry
  counter stops it and a genuine "not in memory" answer survives once a search has occurred.
- **Integration (Fix 3 no-op path):** when the answer is already in the core profile (Fix 2), the gate
  never triggers (no denial, no forced lookup) — confirms Fix 3 only governs the long tail and does not
  add latency to the common case.

## Out of scope

- `sqlite-vec` / ANN index (deferred; not on this critical path).
- The full polyglot memory build (typed numeric records, artifacts, reflection loop, decay).
- Re-extraction / backfill of historically shredded flat facts — Phase 1 entity work may revisit;
  not required for Phase 0 since the people graph is already the source of truth.
- **Historical provenance backfill (0f):** facts learned **before** 0f ships have already lost their
  originating event to the Pruner; their provenance is unrecoverable (only `created_at` survives). 0f is
  capture-forward only — it cannot reconstruct when you first said something the system already pruned.

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
- [ ] Assistant name: if the core profile carries an assistant-name line, define its relationship to
      `infer_assistant_name_from_prompt` (`system_prompt.rs:611-613`) so the two sources can't diverge.
      (Leaning: keep assistant name out of the profile — it's persona-derived, not owner memory — and
      leave the existing `AssistantName` scalar short-circuit untouched.)
- [ ] Placement (Open Q above) changes CORE byte layout and will invalidate **all** existing session
      caches once on deploy — expected and harmless, but note it in the changelog.
- [ ] Should Fix 2's deterministic block let us *shrink* the volatile per-turn people injection
      (avoid double-emission)? (Defer; measure first — and note per Bug 3 there is little owner-DM
      people text to shrink today, since the graph is fetched but not rendered.)
- [ ] **Prune-timing aggressiveness (0f-adjacent):** consolidated events are pruned within minutes
      (~25 min observed in live forensics), which is what makes raw-utterance provenance unrecoverable and
      may also hurt recent-conversation recall. 0f preserves *fact* provenance regardless, but should the
      event-prune cutoff be lengthened, or an append-only utterance log kept, so the *raw* record survives
      longer? (Heavier; measure recall impact before changing the prune contract.)
