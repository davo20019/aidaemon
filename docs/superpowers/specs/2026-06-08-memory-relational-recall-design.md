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
> **Fix 3 added:** a **loop-level "search-before-deny" groundedness gate** (in `stopping_phase`, not a
> prompt-only rule) so the long-tail/non-core lookups are trustworthy — the agent may not deny knowledge
> of a named entity it never searched for. No schema; sliced as **0d**, independent of 0a-0c.
> **Implementation-readiness revision (2nd review pass):** split the profile gate from the
> people-graph gate (use runtime `people_enabled`, not `config.people.enabled` — fixes a People-off
> regression); replaced the "frozen vs per-turn" prose with one explicit frozen-membership/content-hash
> algorithm; added per-source salience rules (`person_facts` has no `recall_count`); added partner
> merge/render rules (the real shredded-pairing bug); sliced Phase 0 into 0a/0b/0c; flagged the
> owner-DM integration test that must be rewritten.
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

- **Phase 0 (this spec's priority):** stop the embarrassment now on data we can eyeball. Three
  complementary fixes: **Fix 1** stops the bad deterministic dump, **Fix 2** guarantees the
  high-salience core is always present (cached), **Fix 3** makes the long-tail lookup trustworthy via a
  loop-level groundedness gate. **Note:** the generalization of Fix 2 (salience profile + snapshot + pin
  schema) plus Fix 3 makes Phase 0 several times the original wife/children fix — it is no longer "two
  surgical fixes." Phase 0 is therefore sliced (see **Phase 0 sequencing** below) so a minimal
  relationship fix can ship first if time-boxed.
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
   role first, owner-linked / flat facts as fallback.
3. The render never emits a label with a lone date value — that exact output is the bug.

**Cache-stability guard — frozen membership, content-hashed (resolves the "frozen vs per-turn"
tension):** salience inputs like `recall_count` change every turn; hashed live they would thrash the
cache. Pin the algorithm precisely:

1. **Session start:** load the union source → score (per-source salience above) → select the top-N
   ordered **fact-ID list** → **freeze that ID list for the session.**
2. **Each turn:** re-fetch *content* for the frozen IDs **plus** any pinned/newly-added facts → render →
   the `core_profile` component hash = `canonical(sorted [(id, value, pinned)] of the rendered set)`.
   **`recall_count` and `recency` are inputs to selection only and are NOT in the hash.**
3. **Mid-session structural edit** (pin/unpin, add a child via `manage_people`, change a partner's
   value): membership or content changes → hash changes → cache busts **exactly once**, then restabilizes.
4. **Mid-session `recall_count` bump:** changes neither the frozen ID list nor the hashed content →
   **no re-render, no thrash.**

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
| **0d** | Fix 3 — search-before-deny groundedness gate (+ bounded retry + soft prompt rule) | none | Medium — heuristic detector, but bounded |

0a alone closes the reproduced failure and the leaks; 0b delivers the *class* fix this spec argues for;
0c adds user control; **0d** makes the long-tail (non-core) lookups trustworthy and is **independent**
of 0a-0c (no shared schema or code — can land in parallel or after). **Pin tests (force-include a
low-salience pinned fact, etc.) belong to 0c** — do not assert them before the column exists. The plan
author chooses 0a-only (time-boxed) vs 0a+0b vs all four; the spec recommends 0a+0b+0d in the first PR
(the trio that fixes the user-visible failure class), 0c as a fast follow.

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

**The gate (in `stopping_phase` / completion, before a reply is delivered):**

1. **Trigger detection** — the candidate reply makes a *negative-knowledge claim* ("I don't have / I
   don't know / no record of / no information about …") **about an entity the user named** (person /
   proper noun / identity term in the user message), **and** no memory-lookup tool
   (`manage_people` view, `manage_memories` search, people/facts retrieval) was invoked **this turn**.
2. **Intervention** — do **not** send the denial. Inject a forced-continuation directive: *"You
   asserted you don't know about `<entity>` but did not search memory. Call `manage_people` /
   `manage_memories` for `<entity>` before answering."* — mirroring the existing forced-nudge pattern
   (edit-stall hint, read-saturation nudge).
3. **Bounded escape** — cap forced retries with a dedicated monotonic counter (same shape as
   `verification_block_count`): after K (≈2) attempts, allow the denial through so the gate can never
   infinite-loop. A genuine "not in memory" answer survives — it just must be *earned* by a real search
   first.

**Detector honesty (the tuning surface):** "is this a negative-knowledge claim about a named entity?"
is heuristic — pattern-match on denial phrases + the user having referenced a person/proper noun. The
codebase already does this class of detection (cancel-intent, deferred-action, low-signal completion),
so it is consistent, but false positives (blocking a legitimate "I can't help with that") and false
negatives are real. Mitigations: keep patterns narrow and word-boundary matched
(`contains_keyword_as_words`); scope the trigger to identity/people questions; the bounded escape caps
the cost of a wrong trigger at K wasted iterations.

**Layering (defense in depth):** keep the soft prompt rule *and* add the gate. Prompt nudges the model
to search proactively; the gate refuses to let a denial through unsearched. Neither alone is trusted.

**No schema.** Pure loop/policy logic + prompt text. Generalizes beyond denials later — the same gate
shape can enforce "don't *assert* a specific fact you didn't retrieve," not just "don't deny."

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
user: "who is Juan Perez?"            (a coworker — NOT in the core profile)
  → not in CORE block → LLM is about to answer "I don't have information about Juan"
  → (Fix 3 gate) negative-knowledge claim about a named entity + no lookup this turn → BLOCKED
  → forced retry → LLM calls manage_people(view, "Juan Perez") → fact found → answers correctly
     (or, after a real search, legitimately: "I don't have anything on Juan" — now *earned*)
```

Deterministic because (Fix 2) the core block is always present and (Fix 3) the loop refuses to let a
denial through unsearched — not because of a hard-coded reply for one question type.

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
  is capped at N. **Pin behavior (a pinned fact force-included at low salience; an unpinned
  high-salience fact losing its slot) is a 0c test** — do not assert it before the `pinned` column
  exists.
- **Unit (Fix 2 merge/render):** seed **shredded** flat facts (`partner`→`Aracely`,
  `partner`→`June 30, 1990` as two rows, mirroring the real transcript) and assert the rendered block
  pairs them into `Partner: Aracely (b. 1990-06-30)` and **never** emits a bare `partner: <date>` line.
  Also seed the people-graph form (`Person{relationship: partner}` + a `person_facts` birthday) and
  assert it wins over the flat fallback. This is the test that proves the actual bug is fixed.
- **Unit (Fix 2 per-source salience):** a partner stored only as a `Person`/`person_facts` (no
  `recall_count`) still ranks into the profile above high-`recall_count` flat trivia.
- **Gate regression (Fix 1+2):** owner DM with `people_enabled = false` and identity flat facts
  present still renders a core profile (people-graph subsection empty, flat-identity half present);
  non-owner session renders no profile.
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
  short-circuit) — confirm that tool-scoping behavior is unchanged (add a test asserting "who is my
  wife?" still scopes personal-memory tools after `CoreRelationships` removal).
- **Existing test to rewrite:** `test_system_prompt_pins_critical_facts_for_owner_dm`
  (`src/integration_tests/part_00.rs:784`) asserts the old `CRITICAL FACTS` block with
  `partner: Alice`-style lines. Fix 1 removes that block's identity arms, so this test must be rewritten
  to assert the **core-profile** block instead. Budget for it — it is not optional.
- **Unit (Fix 3 detector):** the negative-knowledge detector fires on "I don't have information about
  Juan", "I don't know who that is", "no record of …" when the user named an entity, and does **not**
  fire on a benign answer or a refusal unrelated to memory. Word-boundary matched; covers
  false-positive guards (e.g. "I don't think you should…" must NOT trigger).
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
