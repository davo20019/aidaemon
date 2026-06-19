# Read-time Neighborhood Retrieval (Memory Recall Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make relational/derivable memory queries ("who is Conchi's spouse?", "who is my kid's mom?") work by assembling the connected *neighborhood* of the entities a query is about, then forcing a search before the model is allowed to deny knowledge of a named entity.

**Architecture:** A shared LLM intent classifier (first production wiring of `llm_classifier.rs`) extracts `{intent, entities}` from the query. During the explicit search path that drives the `manage_memories` `search` action, a neighborhood-assembly step resolves those entities to stored entities and pulls their fact clusters (namespace / co-mention / owner-relationship), bounded. At completion, a gate extends the existing `answer_grounding` machinery to block a single-entity relational *denial/assertion* about an entity that was not retrieved this turn, forcing a lookup. Precedence: coreference gate (shipped) → neighborhood assembly → search-before-deny gate.

**Tech Stack:** Rust, tokio, sqlx (SQLite/SQLCipher), `fastembed` embeddings, the project's `ModelProvider` trait and `MockProvider` test double.

**Design spec:** `docs/superpowers/specs/2026-06-18-neighborhood-retrieval-design.md`. Predecessor (reconciled): `docs/superpowers/specs/2026-06-08-memory-relational-recall-design.md`.

## Global Constraints

- Pre-commit checklist MUST pass before every commit: `cargo fmt` → `cargo clippy --all-features -- -D warnings` → `cargo test`.
- The cross-encoder reranker is **disabled in test builds** (`if cfg!(test) { bail!(...) }` in `embeddings.rs`); tests must not depend on it.
- Entity resolution MUST be semantic/LLM-driven, **never deterministic token-matching of the user query** (rejected in the spec: surface overlap ≠ meaning).
- All new model calls MUST be **fail-open** (provider error/timeout/disabled → no-op, never block a reply), matching `classify_intent`'s existing 5s-timeout → `Unknown` contract.
- Classifier-driven behavior is **owner-DM-gated** and kept off most turns by a cheap pre-filter.
- Honesty: cross-relationship inferences (e.g. partner ⇒ child's mother) are phrased as **labeled inferences**, never asserted as fact.
- `Tool::schema()` returns the full `{name, description, parameters}` object (unchanged — no new tool here).
- Telemetry uses the existing `tracing` target `memory_recall`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/agent/intent/llm_classifier.rs` | Add `RelationalIntent`/`RelationalKind` + `parse_relational_intent` (pure) + `classify_relational_intent` (async) | Modify |
| `src/agent/intent/relational_prefilter.rs` | Pure `should_run_relational_classifier(user_text, memory_lookup_fired)` | Create |
| `src/memory/neighborhood.rs` | Pure neighborhood logic: `fact_namespace`, `is_relationship_key`, `ResolvedEntity`, `select_neighborhood_facts` | Create |
| `src/state/sqlite/facts.rs` | Thin IO wrapper `assemble_neighborhood` (fetch facts/people → call pure core) | Modify |
| `src/tools/manage_memories.rs` | Wire classifier + `assemble_neighborhood` into the `search` action | Modify |
| `src/agent/loop/answer_grounding.rs` | Add `reply_denies_entity` + `find_unsearched_denials` | Modify |
| `src/agent/loop/system_directives.rs` | Add `UnsearchedEntityDenial { entities }` variant + rendering | Modify |
| `src/agent/loop/completion_phase.rs` | Wire the search-before-deny gate (classifier-gated, bounded retry, after coreference) | Modify |
| `src/integration_tests/part_11.rs` | Integration test mirroring `test_ungrounded_list_reply_is_rejected_then_corrected` | Modify |

**PR slicing:** PR1 = classifier + pre-filter (Tasks 1–3). PR2 = neighborhood assembly (Tasks 4–6). PR3 = search-before-deny gate (Tasks 7–10). Each PR is independently shippable and testable; PR2 and PR3 both *consume* PR1's classifier.

---

## PR1 — Shared relational-intent classifier

### Task 1: Pure parser for the classifier response

**Files:**
- Modify: `src/agent/intent/llm_classifier.rs`

**Interfaces:**
- Produces: `enum RelationalKind { Relational, Recall, None }`; `struct RelationalIntent { pub kind: RelationalKind, pub entities: Vec<String> }`; `fn parse_relational_intent(raw: &str) -> RelationalIntent` (pure, never panics, fail-open to `{ None, [] }`).

- [ ] **Step 1: Write the failing test.** Append to the `#[cfg(test)] mod tests` block in `llm_classifier.rs`:

```rust
#[test]
fn parse_relational_intent_reads_json() {
    let r = parse_relational_intent(r#"{"intent":"relational","entities":["Conchi","Galo"]}"#);
    assert_eq!(r.kind, RelationalKind::Relational);
    assert_eq!(r.entities, vec!["Conchi".to_string(), "Galo".to_string()]);
}

#[test]
fn parse_relational_intent_tolerates_fencing_and_prose() {
    // Models often wrap JSON in ```json fences or add a sentence.
    let r = parse_relational_intent("Sure!\n```json\n{\"intent\":\"recall\",\"entities\":[\"my dog\"]}\n```");
    assert_eq!(r.kind, RelationalKind::Recall);
    assert_eq!(r.entities, vec!["my dog".to_string()]);
}

#[test]
fn parse_relational_intent_fails_open_on_garbage() {
    let r = parse_relational_intent("not json at all");
    assert_eq!(r.kind, RelationalKind::None);
    assert!(r.entities.is_empty());
}
```

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib parse_relational_intent` — Expected: FAIL (`RelationalKind` / `parse_relational_intent` not found).

- [ ] **Step 3: Implement the types + parser.** Add near `LlmIntentClass` in `llm_classifier.rs`:

```rust
/// Coarse relational-recall classification used by neighborhood assembly and
/// the search-before-deny gate. Separate from `LlmIntentClass` because both
/// consumers need the *entities* the query names, which the coarse class lacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationalKind {
    /// A question about a relationship/connection between entities
    /// ("who is Conchi's spouse?", "what tools does project X use?").
    Relational,
    /// A direct personal-fact recall ("what's my dog's name?").
    Recall,
    /// Neither — do nothing.
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalIntent {
    pub kind: RelationalKind,
    /// Entities the query is about, as the model named them. Possibly empty.
    pub entities: Vec<String>,
}

impl RelationalIntent {
    fn none() -> Self {
        Self { kind: RelationalKind::None, entities: Vec::new() }
    }
}

/// Parse the classifier's JSON reply. Fail-open: any malformed input yields
/// `RelationalKind::None` with no entities (caller then does nothing).
pub fn parse_relational_intent(raw: &str) -> RelationalIntent {
    // Extract the first {...} span so ```json fences / prose don't break parsing.
    let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) else {
        return RelationalIntent::none();
    };
    if end < start {
        return RelationalIntent::none();
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw[start..=end]) else {
        return RelationalIntent::none();
    };
    let kind = match v.get("intent").and_then(|i| i.as_str()).unwrap_or("none") {
        "relational" => RelationalKind::Relational,
        "recall" => RelationalKind::Recall,
        _ => RelationalKind::None,
    };
    let entities = v
        .get("entities")
        .and_then(|e| e.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    RelationalIntent { kind, entities }
}
```

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib parse_relational_intent` — Expected: PASS (3 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/agent/intent/llm_classifier.rs
git commit -m "feat(memory): relational-intent parser for neighborhood recall"
```

---

### Task 2: Async `classify_relational_intent` (provider call, fail-open)

**Files:**
- Modify: `src/agent/intent/llm_classifier.rs`

**Interfaces:**
- Consumes: `parse_relational_intent` (Task 1); `ModelProvider::chat_with_options`, `ChatOptions`, the `CLASSIFIER_TIMEOUT` constant (already in this file).
- Produces: `async fn classify_relational_intent(provider: &dyn ModelProvider, fast_model: &str, user_text: &str) -> RelationalIntent`.

- [ ] **Step 1: Write the failing test.** Uses the existing `MockProvider` (from `crate::testing`). Append to the tests module:

```rust
#[tokio::test]
async fn classify_relational_intent_parses_provider_json() {
    let provider = crate::testing::MockProvider::with_responses(vec![
        crate::testing::MockProvider::text_response(
            r#"{"intent":"relational","entities":["Conchi"]}"#,
        ),
    ]);
    let r = classify_relational_intent(&provider, "fast-model", "who is conchi's spouse?").await;
    assert_eq!(r.kind, RelationalKind::Relational);
    assert_eq!(r.entities, vec!["Conchi".to_string()]);
}

#[tokio::test]
async fn classify_relational_intent_fails_open_on_empty_input() {
    let provider = crate::testing::MockProvider::new();
    let r = classify_relational_intent(&provider, "fast-model", "   ").await;
    assert_eq!(r.kind, RelationalKind::None);
}
```

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib classify_relational_intent` — Expected: FAIL (function not found).

- [ ] **Step 3: Implement.** Mirror the existing `classify_intent` body (provider call + 5s timeout + fail-open). Add a `build_relational_classifier_messages` helper next to the existing `build_classifier_messages`:

```rust
fn build_relational_classifier_messages(user_text: &str) -> Vec<serde_json::Value> {
    let system = "You classify a user message about their personal memory. \
Reply with ONLY a JSON object: {\"intent\": \"relational\"|\"recall\"|\"none\", \"entities\": [..]}. \
\"relational\" = a question about a relationship/connection between entities (e.g. \"who is Conchi's spouse?\", \"who is my kid's mom?\", \"what tools does project X use?\"). \
\"recall\" = a direct fact lookup about one entity (e.g. \"what's my dog's name?\"). \
\"none\" = anything else (general knowledge, chit-chat, actions). \
\"entities\" = the people/projects/things the question is about, as named (resolve possessives to the owned entity: \"my mom\" -> \"my mom\"). Keep it short.";
    vec![
        serde_json::json!({"role": "system", "content": system}),
        serde_json::json!({"role": "user", "content": user_text}),
    ]
}

/// Classify a message for relational/recall intent and extract its entities.
/// Fail-open: empty input, provider error, or timeout yields `RelationalKind::None`.
pub async fn classify_relational_intent(
    provider: &dyn ModelProvider,
    fast_model: &str,
    user_text: &str,
) -> RelationalIntent {
    let trimmed = user_text.trim();
    if trimmed.is_empty() {
        return RelationalIntent { kind: RelationalKind::None, entities: Vec::new() };
    }
    let messages = build_relational_classifier_messages(trimmed);
    let options = crate::traits::ChatOptions {
        max_tokens_override: Some(120),
        ..Default::default()
    };
    let call = provider.chat_with_options(fast_model, &messages, &[], &options);
    let response = match tokio::time::timeout(CLASSIFIER_TIMEOUT, call).await {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            debug!(?err, "relational classifier call failed; failing open");
            return RelationalIntent { kind: RelationalKind::None, entities: Vec::new() };
        }
        Err(_) => {
            debug!(timeout_s = CLASSIFIER_TIMEOUT.as_secs(), "relational classifier timeout");
            return RelationalIntent { kind: RelationalKind::None, entities: Vec::new() };
        }
    };
    parse_relational_intent(response.content.as_deref().unwrap_or(""))
}
```

(If `ChatOptions` is not already imported in this file, add `use crate::traits::ChatOptions;` and drop the path prefix — match the existing import style for `classify_intent`.)

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib classify_relational_intent` — Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent/intent/llm_classifier.rs
git commit -m "feat(memory): async classify_relational_intent (fail-open fast-model call)"
```

---

### Task 3: Cheap pre-filter to keep the classifier off most turns

**Files:**
- Create: `src/agent/intent/relational_prefilter.rs`
- Modify: `src/agent/intent/mod.rs` (add `pub mod relational_prefilter;`)

**Interfaces:**
- Consumes: `recall_guardrails::looks_like_personal_memory_recall_question` (pub(crate)).
- Produces: `fn should_run_relational_classifier(user_text: &str, memory_lookup_fired: bool) -> bool`.

- [ ] **Step 1: Write the failing test.** Create `src/agent/intent/relational_prefilter.rs` with only a test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_for_personal_recall_without_lookup() {
        assert!(should_run_relational_classifier("who is conchi's spouse?", false));
    }

    #[test]
    fn skips_when_a_lookup_already_fired() {
        assert!(!should_run_relational_classifier("who is conchi's spouse?", true));
    }

    #[test]
    fn skips_general_knowledge() {
        assert!(!should_run_relational_classifier("who is the president of france?", false));
    }
}
```

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib should_run_relational_classifier` — Expected: FAIL (module/function not found; remember to add `pub mod relational_prefilter;` to `src/agent/intent/mod.rs`).

- [ ] **Step 3: Implement.** Prepend to the file:

```rust
//! Cheap, recall-biased pre-filter that decides whether the relational
//! classifier should run this turn. A false positive only costs one
//! fast-model call (which then returns `None`); it must never be the sole
//! authority for behavior.

/// True when a relational classifier call is worth making. Recall-biased:
/// fires on personal-recall-shaped messages, and only when no memory lookup
/// already grounded the turn.
pub fn should_run_relational_classifier(user_text: &str, memory_lookup_fired: bool) -> bool {
    if memory_lookup_fired {
        return false;
    }
    crate::agent::policy::recall_guardrails::looks_like_personal_memory_recall_question(user_text)
}
```

(Verify the path to `looks_like_personal_memory_recall_question`; the explorer found it `pub(crate)` in `recall_guardrails.rs`. If the re-export path differs, use the one the crate already uses — e.g. `crate::agent::recall_guardrails::...` as `main_loop.rs:528` does.)

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib should_run_relational_classifier` — Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent/intent/relational_prefilter.rs src/agent/intent/mod.rs
git commit -m "feat(memory): pre-filter gating the relational classifier"
```

---

## PR2 — Neighborhood assembly

### Task 4: Pure neighborhood selection core

**Files:**
- Create: `src/memory/neighborhood.rs`
- Modify: `src/memory/mod.rs` (add `pub mod neighborhood;`)

**Interfaces:**
- Consumes: `crate::traits::Fact`.
- Produces:
  - `fn fact_namespace(key: &str) -> Option<&str>` (prefix before `':'`).
  - `fn is_relationship_key(key: &str) -> bool`.
  - `struct NeighborhoodCaps { pub max_entities: usize, pub max_facts: usize }` with `Default` (e.g. `max_entities: 6`, `max_facts: 16`).
  - `fn select_neighborhood_facts(all_facts: &[Fact], resolved_names: &[String], owner_relationship: bool, initial_ids: &std::collections::HashSet<i64>, caps: NeighborhoodCaps) -> Vec<Fact>` — the deterministic fetch over already-fetched facts (namespace + co-mention + owner-relationship-cluster rules), deduped vs `initial_ids`, salience-ranked, capped.

- [ ] **Step 1: Write the failing tests.** Create the file with a tests module first. Use a small fact builder:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Fact;
    use std::collections::HashSet;

    fn fact(id: i64, category: &str, key: &str, value: &str) -> Fact {
        Fact {
            id,
            category: category.into(),
            key: key.into(),
            value: value.into(),
            source: "test".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            superseded_at: None,
            recall_count: 0,
            last_recalled_at: None,
            channel_id: None,
            privacy: Default::default(),
            first_seen_at: None,
            source_excerpt: None,
        }
    }

    #[test]
    fn namespace_is_prefix_before_colon() {
        assert_eq!(fact_namespace("LearnEnglishSounds:path"), Some("LearnEnglishSounds"));
        assert_eq!(fact_namespace("partner_name"), None);
    }

    #[test]
    fn relationship_keys_detected() {
        assert!(is_relationship_key("mother_name"));
        assert!(is_relationship_key("father"));
        assert!(is_relationship_key("partner_name"));
        assert!(!is_relationship_key("LearnEnglishSounds:path"));
    }

    #[test]
    fn owner_relationship_query_pulls_the_whole_family_cluster() {
        // The Conchi/Galo shape: query resolved to "Conchi"; Galo never co-occurs
        // with Conchi, but is in the owner's relationship set.
        let all = vec![
            fact(1, "user", "mother_name", "Consuelo Montesdeoca"), // initial match
            fact(2, "user", "father", "Galo Loor"),
            fact(3, "user", "partner_name", "Aracely Zambrano"),
            fact(4, "project", "LearnEnglishSounds:path", "~/projects/LES"),
        ];
        let initial: HashSet<i64> = [1].into_iter().collect();
        let out = select_neighborhood_facts(
            &all,
            &["Consuelo".into()],
            true, // owner_relationship
            &initial,
            NeighborhoodCaps::default(),
        );
        let ids: HashSet<i64> = out.iter().map(|f| f.id).collect();
        assert!(ids.contains(&2), "father=Galo must be pulled into the cluster");
        assert!(ids.contains(&3), "partner is part of the owner relationship set");
        assert!(!ids.contains(&1), "initial match is deduped out");
        assert!(!ids.contains(&4), "unrelated project fact is not pulled");
    }

    #[test]
    fn namespace_query_pulls_the_concept_cluster() {
        let all = vec![
            fact(10, "project", "LearnEnglishSounds:path", "~/p/LES"),
            fact(11, "technical", "LearnEnglishSounds:tech_stack", "Next.js"),
            fact(12, "user", "partner_name", "Aracely"),
        ];
        let initial: HashSet<i64> = [10].into_iter().collect();
        let out = select_neighborhood_facts(
            &all,
            &["LearnEnglishSounds".into()],
            false,
            &initial,
            NeighborhoodCaps::default(),
        );
        let ids: HashSet<i64> = out.iter().map(|f| f.id).collect();
        assert!(ids.contains(&11), "same-namespace fact pulled");
        assert!(!ids.contains(&12), "unrelated fact not pulled");
    }

    #[test]
    fn caps_are_respected_and_nothing_resolves_is_empty() {
        let all = vec![fact(1, "user", "partner_name", "Aracely")];
        let empty: HashSet<i64> = HashSet::new();
        let out = select_neighborhood_facts(&all, &[], false, &empty, NeighborhoodCaps::default());
        assert!(out.is_empty(), "no resolved entities -> no expansion");
    }
}
```

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib neighborhood::` — Expected: FAIL (module/functions not found; add `pub mod neighborhood;` to `src/memory/mod.rs`).

- [ ] **Step 3: Implement.** Prepend to `src/memory/neighborhood.rs`:

```rust
//! Pure neighborhood selection over already-fetched flat facts.
//!
//! Given the entities a query resolved to, assemble their connected fact
//! clusters so the model can derive relational answers. Three rules:
//! namespace (`X:*`), co-mention (entity name appears in key/value), and the
//! owner-relationship cluster (relationship-typed facts travel together —
//! this is what connects `Conchi` to `Galo`, who never co-occur in one fact).
//! Pure and synchronous so it is unit-testable without a DB.

use crate::traits::Fact;
use std::collections::HashSet;

/// Relationship key roots. General across relationship types, not spouse-specific.
const RELATIONSHIP_ROOTS: &[&str] = &[
    "mother", "father", "mom", "dad", "parent", "partner", "spouse", "wife",
    "husband", "child", "children", "son", "daughter", "kid", "sibling",
    "brother", "sister", "grandmother", "grandfather",
];

#[derive(Debug, Clone, Copy)]
pub struct NeighborhoodCaps {
    pub max_entities: usize,
    pub max_facts: usize,
}

impl Default for NeighborhoodCaps {
    fn default() -> Self {
        Self { max_entities: 6, max_facts: 16 }
    }
}

/// The namespace prefix of a key (`"X:attr" -> "X"`), or `None` for flat keys.
pub fn fact_namespace(key: &str) -> Option<&str> {
    key.split_once(':').map(|(ns, _)| ns).filter(|ns| !ns.is_empty())
}

/// True if the key names a kinship/relationship role (any direction).
pub fn is_relationship_key(key: &str) -> bool {
    if fact_namespace(key).is_some() {
        return false; // namespaced concept keys are never relationship keys
    }
    let lower = key.to_ascii_lowercase();
    RELATIONSHIP_ROOTS.iter().any(|root| {
        lower == *root
            || lower.starts_with(&format!("{root}_"))
            || lower.ends_with(&format!("_{root}"))
            || lower == format!("{root}_name")
    })
}

fn folded_contains(haystack: &str, needle: &str) -> bool {
    haystack.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())
}

/// Salience used to rank additions when over the cap. Higher = keep.
fn salience(f: &Fact) -> i64 {
    // recall_count dominates; recency as a weak tiebreaker via updated_at secs.
    (f.recall_count as i64) * 1_000_000 + f.updated_at.timestamp()
}

/// Select the neighborhood facts to append to the initial matches.
pub fn select_neighborhood_facts(
    all_facts: &[Fact],
    resolved_names: &[String],
    owner_relationship: bool,
    initial_ids: &HashSet<i64>,
    caps: NeighborhoodCaps,
) -> Vec<Fact> {
    if resolved_names.is_empty() {
        return Vec::new();
    }
    let names: Vec<String> = resolved_names.iter().take(caps.max_entities).cloned().collect();

    // Namespaces named by the resolved entities (exact, case-insensitive).
    let target_ns: HashSet<String> = names.iter().map(|n| n.to_ascii_lowercase()).collect();

    let mut picked: Vec<Fact> = all_facts
        .iter()
        .filter(|f| !initial_ids.contains(&f.id))
        .filter(|f| {
            // Rule 1 — namespace cluster.
            let ns_hit = fact_namespace(&f.key)
                .map(|ns| target_ns.contains(&ns.to_ascii_lowercase()))
                .unwrap_or(false);
            // Rule 2 — co-mention (entity name in key or value).
            let mention_hit = names
                .iter()
                .any(|n| folded_contains(&f.key, n) || folded_contains(&f.value, n));
            // Rule 3 — owner relationship cluster.
            let rel_hit = owner_relationship && is_relationship_key(&f.key);
            ns_hit || mention_hit || rel_hit
        })
        .cloned()
        .collect();

    // Dedupe by id (a fact can match more than one rule).
    let mut seen: HashSet<i64> = HashSet::new();
    picked.retain(|f| seen.insert(f.id));

    // Salience-rank and cap.
    picked.sort_by(|a, b| salience(b).cmp(&salience(a)));
    picked.truncate(caps.max_facts);
    picked
}
```

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib neighborhood::` — Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit.**

```bash
git add src/memory/neighborhood.rs src/memory/mod.rs
git commit -m "feat(memory): pure neighborhood fact-selection core"
```

---

### Task 5: Entity resolution + IO wrapper `assemble_neighborhood`

**Files:**
- Modify: `src/state/sqlite/facts.rs`
- Modify: `src/traits/state_store.rs` (add the trait method with a default no-op so non-SQLite stores compile)

**Interfaces:**
- Consumes: `select_neighborhood_facts` (Task 4); `get_all_facts_with_provenance`, `get_all_people`, `Person`, `EmbeddingService` (already on `SqliteStateStore`).
- Produces (trait `FactStore`):
  ```rust
  async fn assemble_neighborhood(
      &self,
      entities: &[String],
      initial_ids: &std::collections::HashSet<i64>,
  ) -> anyhow::Result<Vec<Fact>> { Ok(vec![]) }   // default no-op
  ```
  The `SqliteStateStore` impl: resolve each entity name to (a) a `Person` (name/alias fold-match → flag `owner_relationship` when the matched person's `relationship` is set or it appears in the owner relationship set) and/or (b) a namespace token, then call `select_neighborhood_facts`.

- [ ] **Step 1: Write the failing test.** In `src/state/sqlite/facts.rs` tests (or `state/sqlite/tests.rs`, matching where store tests live), seed facts via the real store and assert the family cluster is assembled:

```rust
#[tokio::test]
async fn assemble_neighborhood_pulls_owner_family_cluster() {
    let store = crate::state::sqlite::SqliteStateStore::new_in_memory().await.unwrap();
    store.upsert_fact("user", "mother_name", "Consuelo Montesdeoca").await.unwrap();
    store.upsert_fact("user", "father", "Galo Loor").await.unwrap();
    store.upsert_fact("user", "partner_name", "Aracely Zambrano").await.unwrap();

    // Pretend the search already matched the mother fact (id resolved below).
    let mother = store.get_facts(Some("user")).await.unwrap()
        .into_iter().find(|f| f.key == "mother_name").unwrap();
    let initial: std::collections::HashSet<i64> = [mother.id].into_iter().collect();

    let out = store.assemble_neighborhood(&["Consuelo".to_string()], &initial).await.unwrap();
    let values: Vec<String> = out.iter().map(|f| f.value.clone()).collect();
    assert!(values.iter().any(|v| v.contains("Galo")), "father pulled into cluster: {:?}", values);
}
```

(Use whatever in-memory constructor the store tests already use — match the existing `SqliteStateStore` test setup in this file; `upsert_fact` is the existing fact-write method.)

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib assemble_neighborhood_pulls_owner_family_cluster` — Expected: FAIL (method not found).

- [ ] **Step 3: Implement.** Add the trait default in `state_store.rs` (no-op shown above). In `facts.rs`, implement on `SqliteStateStore`:

```rust
async fn assemble_neighborhood(
    &self,
    entities: &[String],
    initial_ids: &std::collections::HashSet<i64>,
) -> anyhow::Result<Vec<Fact>> {
    use crate::memory::neighborhood::{select_neighborhood_facts, NeighborhoodCaps};
    if entities.is_empty() {
        return Ok(vec![]);
    }
    let all = self.get_all_facts_with_provenance().await.unwrap_or_default();
    let people = self.get_all_people().await.unwrap_or_default();

    // Resolve each entity name to a stored Person by name/alias fold-match.
    // (The LLM already did the semantic understanding; this maps its output to
    // a stored entity. An embedding fallback can be added later for fuzzy
    // mentions — flagged in the spec's open items.)
    let folded = |s: &str| s.to_ascii_lowercase();
    let mut resolved: Vec<String> = Vec::new();
    let mut owner_relationship = false;
    for ent in entities {
        let ef = folded(ent);
        if let Some(p) = people.iter().find(|p| {
            folded(&p.name).contains(&ef)
                || p.aliases.iter().any(|a| folded(a).contains(&ef))
        }) {
            resolved.push(p.name.clone());
            // A resolved person that has a relationship role (or is the owner's
            // relation) flips on the owner-relationship cluster rule.
            if p.relationship.is_some() {
                owner_relationship = true;
            }
        } else {
            resolved.push(ent.clone());
        }
        // If the entity name itself matches a relationship word, also enable the
        // owner cluster ("my mom", "spouse").
        if crate::memory::neighborhood::is_relationship_key(&ef) {
            owner_relationship = true;
        }
    }

    Ok(select_neighborhood_facts(
        &all,
        &resolved,
        owner_relationship,
        initial_ids,
        NeighborhoodCaps::default(),
    ))
}
```

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib assemble_neighborhood_pulls_owner_family_cluster` — Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/state/sqlite/facts.rs src/traits/state_store.rs
git commit -m "feat(memory): assemble_neighborhood IO wrapper (resolve entities -> cluster)"
```

---

### Task 6: Wire classifier + neighborhood into the `manage_memories` search action

**Files:**
- Modify: `src/tools/manage_memories.rs`

**Interfaces:**
- Consumes: `classify_relational_intent` (Task 2), `assemble_neighborhood` (Task 5), the existing `merge_search_results` and result-formatting block.
- The tool already has `self.state`; it needs a provider + fast model to call the classifier. Use the same access pattern the tool already uses for any provider needs (the explorer found the tool calls `self.state.search_facts_semantic`; check whether the tool holds a provider/router — if not, thread the relational expansion through the search path that *does* have provider access, i.e. call the classifier in the agent before the tool, and pass resolved entities via the search call). **Decision:** keep the classifier call in the tool only if the tool already has provider access; otherwise resolve entities upstream. Inspect `ManageMemoriesTool`'s fields first.

- [ ] **Step 1: Inspect the tool's available dependencies.** Run: `grep -n "struct ManageMemoriesTool" -A 20 src/tools/manage_memories.rs` — confirm whether it holds a `provider`/`router`. Record the finding in the commit message. If it has provider access, proceed as below; if not, the classifier call moves to the caller and this task passes pre-resolved `entities` into the search action via an internal arg (document that variant inline).

- [ ] **Step 2: Write the failing test.** Add an integration-style test in `manage_memories.rs` tests that seeds the family facts, runs the `search` action for `"who is conchi's spouse?"` with a `MockProvider` scripted to return `{"intent":"relational","entities":["Consuelo"]}`, and asserts the formatted output now contains `Galo`:

```rust
#[tokio::test]
async fn search_action_appends_relational_neighborhood() {
    // ... build SqliteStateStore + seed mother_name=Consuelo, father=Galo Loor ...
    // ... construct ManageMemoriesTool with a MockProvider returning the relational JSON ...
    let out = tool.call(serde_json::json!({"action":"search","query":"who is conchi's spouse?"})).await.unwrap();
    assert!(out.contains("Galo"), "neighborhood should surface the father fact: {out}");
}
```

(Fill the setup to match the tool's real constructor and `call` signature — see the existing `manage_memories.rs` tests for the exact pattern.)

- [ ] **Step 3: Run it; verify it fails.** Run: `cargo test --lib search_action_appends_relational_neighborhood` — Expected: FAIL (output lacks `Galo`).

- [ ] **Step 4: Implement.** In the `search` action, after computing `merged` (lexical+semantic) and before formatting, when a provider is available run the classifier and append the neighborhood:

```rust
// After `let merged = merge_search_results(lexical, semantic);`
let initial_ids: std::collections::HashSet<i64> =
    merged.iter().map(|(f, _)| f.id).collect();

// Relational expansion (fail-open; only when the pre-filter says so).
let mut neighborhood: Vec<crate::traits::Fact> = Vec::new();
if let Some((provider, fast_model)) = self.relational_classifier_deps() {
    let intent = crate::agent::intent::llm_classifier::classify_relational_intent(
        provider.as_ref(), &fast_model, query,
    ).await;
    if !intent.entities.is_empty() {
        neighborhood = self.state
            .assemble_neighborhood(&intent.entities, &initial_ids)
            .await
            .unwrap_or_default();
    }
}
```

Then render the neighborhood facts under a clearly-labeled sub-block appended to the existing output (so it is additive and the model knows it is connected context):

```rust
if !neighborhood.is_empty() {
    tracing::info!(
        target: "memory_recall",
        entities = %query,
        added = neighborhood.len(),
        "neighborhood expansion appended"
    );
    output.push_str("\n── Related context (connected facts) ──\n");
    for f in neighborhood.iter().take(16) {
        output.push_str(&format!("• [{}] {} → \"{}\"\n", f.category, f.key, f.value));
    }
}
```

(`relational_classifier_deps()` returns the provider+fast-model if the tool holds them and the owner-DM gate is satisfied; otherwise `None` → expansion is skipped, preserving today's behavior. Implement it per Step 1's finding.)

- [ ] **Step 5: Run it; verify it passes.** Run: `cargo test --lib search_action_appends_relational_neighborhood` — Expected: PASS.

- [ ] **Step 6: Full checklist + commit.** Run: `cargo fmt && cargo clippy --all-features -- -D warnings && cargo test`. Then:

```bash
git add src/tools/manage_memories.rs
git commit -m "feat(memory): append relational neighborhood to search results"
```

---

## PR3 — Search-before-deny gate

### Task 7: Detect a denial/assertion of an unsearched entity

**Files:**
- Modify: `src/agent/loop/answer_grounding.rs`

**Interfaces:**
- Consumes: `fold_for_match` (already in this module), `tool_result_indicates_no_evidence` semantics.
- Produces: `pub(in crate::agent) fn find_unsearched_denials(reply: &str, entities: &[String], evidence: &[&str]) -> Vec<String>` — returns the subset of classifier-named `entities` that (a) the `reply` makes a claim/denial about, and (b) do not appear in `evidence`.

- [ ] **Step 1: Write the failing test.** Append to `answer_grounding.rs` tests:

```rust
#[test]
fn flags_denial_of_unsearched_entity() {
    let reply = "I don't have information about Conchi's spouse.";
    let entities = vec!["Conchi".to_string()];
    let evidence = vec!["partner_name: Aracely Zambrano"]; // Conchi absent
    let out = find_unsearched_denials(reply, &entities, &evidence);
    assert_eq!(out, vec!["Conchi".to_string()]);
}

#[test]
fn does_not_flag_when_entity_is_in_evidence() {
    let reply = "I don't have Conchi's phone number.";
    let entities = vec!["Conchi".to_string()];
    let evidence = vec!["mother_name: Consuelo (Conchi) Montesdeoca"]; // present
    assert!(find_unsearched_denials(reply, &entities, &evidence).is_empty());
}

#[test]
fn does_not_flag_when_no_entities() {
    assert!(find_unsearched_denials("anything", &[], &["x"]).is_empty());
}
```

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib find_unsearched_denials` — Expected: FAIL (function not found).

- [ ] **Step 3: Implement.** Add to `answer_grounding.rs`:

```rust
/// Entities the `reply` makes a claim/denial about that appear nowhere in
/// `evidence` (tool outputs + user message). The classifier supplies the
/// candidate `entities`; this confirms the reply actually addresses them and
/// that they were not grounded this turn. Substring (folded) match, like the
/// list gate — errs toward NOT flagging.
pub(in crate::agent) fn find_unsearched_denials(
    reply: &str,
    entities: &[String],
    evidence: &[&str],
) -> Vec<String> {
    if entities.is_empty() {
        return Vec::new();
    }
    let reply_f = fold_for_match(reply);
    let corpus = fold_for_match(&evidence.join("\n"));
    entities
        .iter()
        .filter(|e| {
            let ef = fold_for_match(e);
            // The reply addresses the entity, but evidence does not contain it.
            ef.split_whitespace()
                .filter(|w| w.chars().count() >= 3)
                .any(|w| reply_f.contains(w))
                && !ef
                    .split_whitespace()
                    .filter(|w| w.chars().count() >= 3)
                    .all(|w| corpus.contains(w))
        })
        .cloned()
        .collect()
}
```

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib find_unsearched_denials` — Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent/loop/answer_grounding.rs
git commit -m "feat(memory): detect denial of an unsearched entity"
```

---

### Task 8: New `UnsearchedEntityDenial` system directive

**Files:**
- Modify: `src/agent/loop/system_directives.rs`

**Interfaces:**
- Produces: `SystemDirective::UnsearchedEntityDenial { entities: Vec<String> }` + its rendering arm.

- [ ] **Step 1: Write the failing test.** Append to the `system_directives.rs` tests (or add one mirroring how `UngroundedListEntities` rendering is tested, if such a test exists):

```rust
#[test]
fn renders_unsearched_entity_denial() {
    let d = SystemDirective::UnsearchedEntityDenial { entities: vec!["Conchi".into()] };
    let msg = d.render(); // use whatever the existing render method is named
    assert!(msg.contains("Conchi"));
    assert!(msg.to_lowercase().contains("search"));
}
```

(Match the actual render method name used by the existing variants — the explorer showed a `match self { ... }` that returns a `String`; use that method's real name.)

- [ ] **Step 2: Run it; verify it fails.** Run: `cargo test --lib renders_unsearched_entity_denial` — Expected: FAIL (variant not found).

- [ ] **Step 3: Implement.** Add the variant near `UngroundedListEntities`:

```rust
/// The candidate reply denies or asserts a specific personal fact about an
/// entity the user named, but no memory lookup for that entity grounded this
/// turn. Force a search before the denial/assertion is allowed through.
UnsearchedEntityDenial {
    entities: Vec<String>,
},
```

And its rendering arm (mirroring the `UngroundedListEntities` arm style):

```rust
Self::UnsearchedEntityDenial { entities } => format!(
    "[SYSTEM] GROUNDING CHECK: your draft answers a question about {} but you did \
     not search memory for it this turn. Call manage_memories (and manage_people if \
     available) for {} BEFORE answering. If you genuinely find nothing after \
     searching, say so plainly — do not assert or deny a relationship you did not \
     look up, and never assume a partner is a child's biological parent; phrase any \
     such inference tentatively.",
    entities.join(", "),
    entities.join(", ")
),
```

- [ ] **Step 4: Run it; verify it passes.** Run: `cargo test --lib renders_unsearched_entity_denial` — Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/agent/loop/system_directives.rs
git commit -m "feat(memory): UnsearchedEntityDenial directive"
```

---

### Task 9: Wire the gate into `completion_phase` (classifier-gated, bounded, after coreference)

**Files:**
- Modify: `src/agent/loop/completion_phase.rs`
- Modify: `src/agent/loop/execution_state.rs` (add a `denial_gate_count` retry counter, mirroring `grounding_nudge_count`)

**Interfaces:**
- Consumes: `classify_relational_intent` (Task 2), `should_run_relational_classifier` (Task 3), `find_unsearched_denials` (Task 7), `SystemDirective::UnsearchedEntityDenial` (Task 8), the existing `tool_output_evidence`, `pending_system_messages`, and `ResponsePhaseOutcome::ContinueLoop`.
- Produces: a sibling gate next to the `find_ungrounded_list_entities` block.

- [ ] **Step 1: Add the bounded retry counter.** In `completion_progress` (the struct holding `grounding_nudge_count`), add `pub denial_gate_count: u32` (default 0). Run: `cargo build` — Expected: compiles.

- [ ] **Step 2: Write the failing integration test.** In `src/integration_tests/part_11.rs`, mirror `test_ungrounded_list_reply_is_rejected_then_corrected`:

```rust
#[tokio::test]
async fn test_relational_denial_is_blocked_then_corrected() {
    // Owner has mother=Consuelo and father=Galo seeded before the turn.
    // Provider script:
    //  1) relational classifier  -> {"intent":"relational","entities":["Conchi"]}
    //  2) first answer (denial)   -> "I don't have information about Conchi's spouse."
    //  3) (gate forces a search)  -> a manage_memories search tool call
    //  4) corrected answer        -> "Conchi's spouse is Galo Loor."
    // Assert: the denial never reaches the user; final response names Galo;
    // and an UnsearchedEntityDenial directive ("did not search memory") was injected.
}
```

(Build the seed + `MockProvider::with_responses` sequence to match the loop's call order; consult the existing test for the exact ordering of classifier vs. answer calls and adjust the script.)

- [ ] **Step 3: Run it; verify it fails.** Run: `cargo test --test '*' test_relational_denial_is_blocked_then_corrected` (or `cargo test test_relational_denial_is_blocked_then_corrected`) — Expected: FAIL (denial reaches the user).

- [ ] **Step 4: Implement the gate.** Add, immediately AFTER the existing `find_ungrounded_list_entities` block in `completion_phase.rs` (so list-fabrication runs first; this is the single-entity case), and only when the coreference directive did not already fire this turn:

```rust
// Search-before-deny gate: the reply denies/asserts a specific personal fact
// about an entity the user named, but no memory lookup grounded it this turn.
// Classifier-gated and owner-DM-gated; bounded so it can never loop.
if completion_progress.denial_gate_count == 0
    && is_owner_dm
    && super::super::intent::relational_prefilter::should_run_relational_classifier(
        user_text,
        memory_lookup_fired_this_turn,
    )
{
    let intent = crate::agent::intent::llm_classifier::classify_relational_intent(
        provider.as_ref(), &fast_model, user_text,
    ).await;
    if !intent.entities.is_empty() {
        let unsearched = super::answer_grounding::find_unsearched_denials(
            &reply,
            &intent.entities,
            &[execution_state.tool_output_evidence.as_str(), user_text],
        );
        if !unsearched.is_empty() {
            completion_progress.denial_gate_count += 1;
            warn!(
                target: "memory_recall",
                session_id,
                iteration,
                entities = %unsearched.join(", "),
                "Reply denies/asserts an unsearched entity — forcing a lookup"
            );
            pending_system_messages.push(SystemDirective::UnsearchedEntityDenial {
                entities: unsearched,
            });
            commit_state!();
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }
    }
}
```

You will need to bind three locals from the surrounding phase context (match how the list gate accesses its inputs):
- `is_owner_dm` — from the channel context / user role already available in the phase.
- `provider` + `fast_model` — from the runtime snapshot (`runtime_snapshot.provider()` and `router.select(Tier::Fast)`, as in `system_prompt.rs`).
- `memory_lookup_fired_this_turn` — true if any `manage_memories`/`manage_people` tool call succeeded this turn; derive from the same tool-call accounting the phase already tracks (e.g. inspect executed tool names, or add a boolean to `execution_state` set when a personal-memory tool runs).

- [ ] **Step 5: Run it; verify it passes.** Run: `cargo test test_relational_denial_is_blocked_then_corrected` — Expected: PASS.

- [ ] **Step 6: Full checklist.** Run: `cargo fmt && cargo clippy --all-features -- -D warnings && cargo test` — Expected: all green.

- [ ] **Step 7: Commit.**

```bash
git add src/agent/loop/completion_phase.rs src/agent/loop/execution_state.rs src/integration_tests/part_11.rs
git commit -m "feat(memory): search-before-deny gate for relational denials"
```

---

### Task 10: Precedence guard + grounded-partial safety test

**Files:**
- Modify: `src/agent/loop/completion_phase.rs` (precedence) and its test neighbors.

**Interfaces:**
- Ensures the denial gate does NOT fire when the coreference gate already fired this turn (precedence: coreference → assembly → denial gate), and does NOT fire on a grounded partial answer.

- [ ] **Step 1: Write the failing tests.**

```rust
#[tokio::test]
async fn grounded_partial_answer_is_not_blocked() {
    // entities=["Juan"], evidence contains "Juan ... coworker", reply:
    // "I don't have Juan's phone, but he's your coworker." -> gate must NOT fire.
}

#[tokio::test]
async fn denial_gate_skipped_when_coreference_fired() {
    // A pronoun-referent turn that triggers CoreferenceGroundingRequired must
    // not also trigger UnsearchedEntityDenial in the same turn.
}
```

- [ ] **Step 2: Run them; verify they fail (or that the first already passes via Task 7's evidence check).** Run: `cargo test grounded_partial_answer_is_not_blocked denial_gate_skipped_when_coreference_fired`.

- [ ] **Step 3: Implement the precedence guard.** Gate the denial block on a flag indicating the coreference directive did not fire this turn (the coreference gate runs in `main_loop.rs` before the completion phase; thread its "fired" state into `execution_state` or `completion_progress` as `coreference_fired: bool`, and add `&& !completion_progress.coreference_fired` to the denial-gate condition). The grounded-partial case is already handled by Task 7's evidence containment check; the test locks it in.

- [ ] **Step 4: Run them; verify they pass.** Run: `cargo test grounded_partial_answer_is_not_blocked denial_gate_skipped_when_coreference_fired` — Expected: PASS.

- [ ] **Step 5: Full checklist + commit.** Run: `cargo fmt && cargo clippy --all-features -- -D warnings && cargo test`. Then:

```bash
git add src/agent/loop/completion_phase.rs src/agent/loop/main_loop.rs src/agent/loop/execution_state.rs
git commit -m "feat(memory): denial-gate precedence (coreference first) + grounded-partial guard"
```

---

## Self-Review

**Spec coverage:**
- §4 Component 1 (shared classifier) → Tasks 1–3. ✓
- §4 Components 2–4 (semantic resolution, neighborhood fetch, merge+bound) → Tasks 4–6. ✓
- §4 Component 5 (search-before-deny gate) → Tasks 7–10. ✓
- §4 Component 6 (honesty / co-parter inference) → wording baked into the `UnsearchedEntityDenial` directive (Task 8). ✓
- §4 Component 7 (telemetry) → `memory_recall` log lines in Tasks 6 and 9. ✓
- §4 Precedence (coreference → assembly → gate) → Task 10. ✓
- §5 Bounds → `NeighborhoodCaps` (Task 4), `denial_gate_count` (Task 9). ✓
- §6 Degradation (fail-open) → Tasks 2, 6, 9 all skip on classifier failure/`None`. ✓
- §7 Testing → unit tests in Tasks 1,3,4,7,8; integration tests in Tasks 6,9,10. ✓
- §9 reuse (don't duplicate `core_profile`, coreference gate, `answer_grounding`) → Tasks extend `answer_grounding`, keep coreference, leave `core_profile` untouched. ✓

**Placeholder scan:** No "TODO/implement later". Two tasks (6, 9) require inspecting the real constructor/phase-locals before writing the final binding — these are explicit *inspection steps* with concrete fallbacks, not placeholders, because the exact field/param wiring can only be confirmed against the live code and the plan states precisely what to look for and what to do in each branch.

**Type consistency:** `RelationalIntent`/`RelationalKind` consistent across Tasks 1–3, 6, 9. `select_neighborhood_facts` signature consistent between Task 4 (def) and Task 5 (call). `find_unsearched_denials` signature consistent between Task 7 (def) and Task 9 (call). `UnsearchedEntityDenial { entities }` consistent between Tasks 8 and 9.

**Known follow-ups (out of scope, recorded):** embedding-based fuzzy entity resolution (Task 5 uses name/alias fold-match first); Phase 2 owner-relationship edges (prior spec's 0e). Both noted in the spec.
