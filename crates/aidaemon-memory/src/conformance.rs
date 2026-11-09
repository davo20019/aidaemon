//! Memory store conformance kit (feature `conformance`).
//!
//! Exercises the documented [`MemoryStore`] contract against fresh stores
//! made by a factory, so a third-party backend (pgvector, Qdrant, a document
//! DB) can prove it behaves like the bundled stores:
//!
//! ```
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! use std::sync::Arc;
//! use aidaemon_memory::conformance::check_store;
//! use aidaemon_memory::{InMemoryMemoryStore, MemoryStore};
//!
//! check_store(|| async { Arc::new(InMemoryMemoryStore::new()) as Arc<dyn MemoryStore> })
//!     .await
//!     .assert_all_passed();
//! # }
//! ```
//!
//! Every check gets its own store from the factory. Facts arrive already
//! normalized (as [`crate::Memory`] sends them) and all timestamps are whole
//! seconds, so stores that keep microseconds compare exactly. Check names
//! are the constants in [`store_checks`], each documenting its rule.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use aidaemon_core::conformance::ConformanceReport;
use chrono::{DateTime, Duration, TimeZone, Utc};

use crate::store::MemoryStore;
use crate::types::{
    DecayReport, Episode, Fact, FactAccess, NewEpisode, NewFact, RetentionConfig, StoredEmbedding,
    UpsertOutcome,
};

/// Stable names of the checks run by [`check_store`].
pub mod store_checks {
    /// `upsert_fact` of a new key returns `Inserted` with a non-empty id;
    /// the fact is active, keeps its fields, and starts with zero recalls.
    pub const UPSERT_INSERTS_NEW_FACT: &str = "store.upsert_inserts_new_fact";
    /// Upserting the same value (modulo case and whitespace, see
    /// [`crate::same_value`]) returns `Reinforced` with the same id, adds no
    /// version, refreshes `updated_at` and keeps the higher confidence.
    pub const SAME_VALUE_REINFORCES: &str = "store.same_value_reinforces";
    /// Upserting a different value returns `Superseded` (new id, previous id
    /// and value); the old value is only in `fact_history` (oldest first,
    /// with `superseded_at` set) and never in `active_facts` or searches.
    pub const DIFFERENT_VALUE_SUPERSEDES: &str = "store.different_value_supersedes";
    /// Scopes are hard boundaries: facts, history, lexical scores, searches,
    /// episodes and supersession never cross scopes.
    pub const SCOPE_ISOLATION: &str = "store.scope_isolation";
    /// `supersede_fact` (forget) hides the active fact, keeps it in history,
    /// returns `true` once and `false` when nothing is active.
    pub const FORGET_SUPERSEDES: &str = "store.forget_supersedes";
    /// `insert_episode` returns the stored episode (non-empty id, fields
    /// kept, zero recalls); `episodes` lists it; `episode_lexical_scores`
    /// scores matching episodes in `(0, 1]` and omits non-matching ones.
    pub const EPISODES_ADD_LIST_SEARCH: &str = "store.episodes_add_list_search";
    /// `fact_lexical_scores` returns scores in `(0, 1]` only for active facts
    /// that match the query; non-matching facts are absent (or 0).
    pub const LEXICAL_SCORES_ONLY_MATCHING: &str = "store.lexical_scores_only_matching";
    /// When `search_facts` / `search_episodes` return `Some`, results are in
    /// scope, active (facts), and at most `limit` long. `None` passes.
    pub const NATIVE_SEARCH_CONTRACT: &str = "store.native_search_contract";
    /// `record_fact_recalls` bumps `recall_count` by one per call and sets
    /// `last_recalled_at`; unknown ids are ignored; other facts unchanged.
    pub const FACT_RECALL_BOOKKEEPING: &str = "store.fact_recall_bookkeeping";
    /// `record_episode_recalls` bumps `recall_count` and sets
    /// `last_recalled_at`, like facts (episode retention depends on it).
    pub const EPISODE_RECALL_BOOKKEEPING: &str = "store.episode_recall_bookkeeping";
    /// Fact and episode embeddings come back exactly as stored: same model
    /// string, same length, every `f32` equal (including subnormals and
    /// extremes); `None` stays `None`.
    pub const EMBEDDINGS_ROUND_TRIP: &str = "store.embeddings_round_trip";
    /// `stats` counts active facts, superseded facts, episodes and distinct
    /// scopes.
    pub const STATS_COUNTS: &str = "store.stats_counts";
    /// `decay` with every [`crate::RetentionConfig`] window `None` changes
    /// nothing and reports nothing.
    pub const DECAY_DISABLED_IS_NOOP: &str = "store.decay_disabled_is_noop";
    /// `decay` with the default config either leaves everything alone (the
    /// trait default, for backends that expire data themselves; noted) or
    /// applies each documented step exactly: superseded versions past the
    /// retention window deleted, never-recalled old episodes deleted, stale
    /// fact recalls lowered by a tenth (at least one), stale episode recalls
    /// lowered by one. The report matches what changed, and active facts or
    /// recently recalled rows are never touched.
    pub const DECAY_FOLLOWS_RETENTION: &str = "store.decay_follows_retention";
    /// `fact_access` / `episode_access` are either empty (the trait default:
    /// records stay private; noted) or return exactly what was written, for
    /// the asked scope only. Restating a fact never widens its access, and a
    /// new version takes the access it was written with.
    pub const ACCESS_LEVELS_ROUND_TRIP: &str = "store.access_levels_round_trip";
}

/// Run every store check (see [`store_checks`]); each check uses a fresh
/// store from `make`.
pub async fn check_store<F, Fut>(make: F) -> ConformanceReport
where
    F: Fn() -> Fut,
    Fut: Future<Output = Arc<dyn MemoryStore>>,
{
    use store_checks::*;

    let mut report = ConformanceReport::new("memory store");
    report.record(UPSERT_INSERTS_NEW_FACT, upsert_inserts(make().await).await);
    report.record(SAME_VALUE_REINFORCES, reinforces(make().await).await);
    report.record(DIFFERENT_VALUE_SUPERSEDES, supersedes(make().await).await);
    report.record(SCOPE_ISOLATION, scope_isolation(make().await).await);
    report.record(FORGET_SUPERSEDES, forget(make().await).await);
    report.record(EPISODES_ADD_LIST_SEARCH, episodes(make().await).await);
    report.record(LEXICAL_SCORES_ONLY_MATCHING, lexical(make().await).await);
    match native_search(make().await).await {
        Ok(notes) => {
            report.pass(NATIVE_SEARCH_CONTRACT);
            for note in notes {
                report.note(format!("{NATIVE_SEARCH_CONTRACT}: {note}"));
            }
        }
        Err(e) => report.fail(NATIVE_SEARCH_CONTRACT, e),
    }
    report.record(FACT_RECALL_BOOKKEEPING, fact_recalls(make().await).await);
    report.record(
        EPISODE_RECALL_BOOKKEEPING,
        episode_recalls(make().await).await,
    );
    report.record(EMBEDDINGS_ROUND_TRIP, embeddings(make().await).await);
    report.record(STATS_COUNTS, stats(make().await).await);
    report.record(DECAY_DISABLED_IS_NOOP, decay_disabled(make().await).await);
    match decay_default(make().await).await {
        Ok(true) => report.pass(DECAY_FOLLOWS_RETENTION),
        Ok(false) => {
            report.pass(DECAY_FOLLOWS_RETENTION);
            report.note(format!(
                "{DECAY_FOLLOWS_RETENTION}: decay changed nothing (the trait default); \
                 make sure the backend expires data itself"
            ));
        }
        Err(e) => report.fail(DECAY_FOLLOWS_RETENTION, e),
    }
    match access_levels(make().await).await {
        Ok(true) => report.pass(ACCESS_LEVELS_ROUND_TRIP),
        Ok(false) => {
            report.pass(ACCESS_LEVELS_ROUND_TRIP);
            report.note(format!(
                "{ACCESS_LEVELS_ROUND_TRIP}: the store keeps no access levels, so its records \
                 are hidden from shared channels"
            ));
        }
        Err(e) => report.fail(ACCESS_LEVELS_ROUND_TRIP, e),
    }
    report
}

const SCOPE: &str = "conformance:synthetic-user-1";
const OTHER_SCOPE: &str = "conformance:synthetic-user-2";

type Check<T = ()> = Result<T, String>;

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
        .single()
        .expect("valid timestamp")
}

fn at(seconds: i64) -> DateTime<Utc> {
    t0() + Duration::seconds(seconds)
}

fn fact(scope: &str, category: &str, key: &str, value: &str) -> NewFact {
    NewFact::new(scope, category, key, value)
        .normalized()
        .expect("kit facts are valid")
}

fn episode(scope: &str, summary: &str, topics: &[&str]) -> NewEpisode {
    NewEpisode {
        scope: scope.to_string(),
        session_id: format!("{scope}:session"),
        summary: summary.to_string(),
        topics: topics.iter().map(|t| t.to_string()).collect(),
        importance: 0.5,
    }
}

/// `?` for store calls, naming the call that failed.
macro_rules! call {
    ($what:expr, $fut:expr) => {
        $fut.await.map_err(|e| format!("{} failed: {e}", $what))?
    };
}

fn ensure(cond: bool, message: impl FnOnce() -> String) -> Check {
    if cond {
        Ok(())
    } else {
        Err(message())
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

async fn active(store: &dyn MemoryStore, scope: &str) -> Check<Vec<Fact>> {
    Ok(call!("active_facts", store.active_facts(scope))
        .into_iter()
        .map(|r| r.fact)
        .collect())
}

async fn upsert_inserts(store: Arc<dyn MemoryStore>) -> Check {
    let new = fact(SCOPE, "person", "daughters", "Mia and Zoe").confidence(0.8);
    let outcome = call!("upsert_fact", store.upsert_fact(new, None, at(0)));
    let UpsertOutcome::Inserted { id } = outcome else {
        return Err(format!("expected Inserted, got {outcome:?}"));
    };
    ensure(!id.is_empty(), || "Inserted id is empty".into())?;
    let facts = active(&*store, SCOPE).await?;
    ensure(facts.len() == 1, || {
        format!("expected 1 active fact, got {}", facts.len())
    })?;
    let f = &facts[0];
    ensure(f.id == id, || {
        format!("active fact id {:?} != {id:?}", f.id)
    })?;
    ensure(
        (
            f.scope.as_str(),
            f.category.as_str(),
            f.key.as_str(),
            f.value.as_str(),
        ) == (SCOPE, "person", "daughters", "Mia and Zoe"),
        || format!("fields not preserved: {f:?}"),
    )?;
    ensure(close(f.confidence, 0.8), || {
        format!("confidence {} != 0.8", f.confidence)
    })?;
    ensure(
        f.is_active() && f.recall_count == 0 && f.last_recalled_at.is_none(),
        || format!("a new fact must be active with no recalls: {f:?}"),
    )
}

async fn reinforces(store: Arc<dyn MemoryStore>) -> Check {
    let key = ("preference", "theme");
    let first = call!(
        "upsert_fact",
        store.upsert_fact(
            fact(SCOPE, key.0, key.1, "Dark mode").confidence(0.6),
            None,
            at(0)
        )
    );
    let id = first.id().to_string();
    let again = call!(
        "upsert_fact (same value)",
        store.upsert_fact(
            fact(SCOPE, key.0, key.1, "dark  MODE").confidence(0.9),
            None,
            at(60)
        )
    );
    ensure(
        again == UpsertOutcome::Reinforced { id: id.clone() },
        || format!("expected Reinforced {{ id: {id:?} }}, got {again:?}"),
    )?;
    let lower = call!(
        "upsert_fact (same value, lower confidence)",
        store.upsert_fact(
            fact(SCOPE, key.0, key.1, "Dark mode").confidence(0.3),
            None,
            at(120)
        )
    );
    ensure(
        lower == UpsertOutcome::Reinforced { id: id.clone() },
        || format!("expected Reinforced again, got {lower:?}"),
    )?;
    let facts = active(&*store, SCOPE).await?;
    ensure(facts.len() == 1, || {
        format!("reinforcing created versions: {} active facts", facts.len())
    })?;
    let history = call!("fact_history", store.fact_history(SCOPE, key.0, key.1));
    ensure(history.len() == 1, || {
        format!(
            "reinforcing must not add versions; history has {}",
            history.len()
        )
    })?;
    let f = &facts[0];
    ensure(f.updated_at == at(120), || {
        format!(
            "updated_at {} was not refreshed to {}",
            f.updated_at,
            at(120)
        )
    })?;
    ensure(close(f.confidence, 0.9), || {
        format!(
            "confidence {} should be the highest seen (0.9)",
            f.confidence
        )
    })
}

async fn supersedes(store: Arc<dyn MemoryStore>) -> Check {
    let first = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "work", "employer", "Acme Corp"), None, at(0))
    );
    let old_id = first.id().to_string();
    let second = call!(
        "upsert_fact (new value)",
        store.upsert_fact(fact(SCOPE, "work", "employer", "Globex"), None, at(60))
    );
    let UpsertOutcome::Superseded {
        id,
        previous_id,
        previous_value,
    } = &second
    else {
        return Err(format!("expected Superseded, got {second:?}"));
    };
    ensure(*id != old_id && !id.is_empty(), || {
        format!("the new version needs a new id, got {id:?} (old {old_id:?})")
    })?;
    ensure(
        *previous_id == old_id && previous_value == "Acme Corp",
        || {
            format!("previous should be ({old_id:?}, \"Acme Corp\"), got ({previous_id:?}, {previous_value:?})")
        },
    )?;
    let facts = active(&*store, SCOPE).await?;
    let values: Vec<&str> = facts.iter().map(|f| f.value.as_str()).collect();
    ensure(values == ["Globex"], || {
        format!("only the new value may be active, got {values:?}")
    })?;
    let history = call!(
        "fact_history",
        store.fact_history(SCOPE, "work", "employer")
    );
    let summary: Vec<(&str, bool)> = history
        .iter()
        .map(|f| (f.value.as_str(), f.superseded_at.is_some()))
        .collect();
    ensure(summary == [("Acme Corp", true), ("Globex", false)], || {
        format!("history must be oldest first with the old value superseded, got {summary:?}")
    })?;
    let scores = call!(
        "fact_lexical_scores",
        store.fact_lexical_scores(SCOPE, "Acme Corp")
    );
    ensure(scores.get(&old_id).is_none_or(|s| *s <= 0.0), || {
        "a superseded fact received a lexical score".into()
    })?;
    if let Some(hits) = call!(
        "search_facts",
        store.search_facts(SCOPE, "Acme Corp employer", None, 10)
    ) {
        ensure(hits.iter().all(|h| h.fact.id != old_id), || {
            "search_facts returned a superseded fact".into()
        })?;
    }
    Ok(())
}

async fn scope_isolation(store: Arc<dyn MemoryStore>) -> Check {
    let mine = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "pet", "dog_name", "Rex"), None, at(0))
    );
    let theirs = call!(
        "upsert_fact (other scope)",
        store.upsert_fact(
            fact(OTHER_SCOPE, "pet", "dog_name", "Biscuit"),
            None,
            at(10)
        )
    );
    ensure(matches!(theirs, UpsertOutcome::Inserted { .. }), || {
        format!("the same key in another scope must be Inserted, got {theirs:?}")
    })?;
    let facts = active(&*store, SCOPE).await?;
    ensure(
        facts.len() == 1 && facts[0].id == mine.id() && facts[0].scope == SCOPE,
        || format!("active_facts leaked across scopes: {facts:?}"),
    )?;
    let history = call!("fact_history", store.fact_history(SCOPE, "pet", "dog_name"));
    ensure(
        history.iter().all(|f| f.scope == SCOPE) && history.len() == 1,
        || format!("fact_history leaked across scopes: {history:?}"),
    )?;
    let scores = call!(
        "fact_lexical_scores",
        store.fact_lexical_scores(SCOPE, "Biscuit dog")
    );
    ensure(!scores.contains_key(theirs.id()), || {
        "fact_lexical_scores scored another scope's fact".into()
    })?;
    if let Some(hits) = call!(
        "search_facts",
        store.search_facts(SCOPE, "Biscuit dog", None, 10)
    ) {
        ensure(hits.iter().all(|h| h.fact.scope == SCOPE), || {
            "search_facts returned another scope's fact".into()
        })?;
    }
    call!(
        "insert_episode",
        store.insert_episode(
            episode(OTHER_SCOPE, "Planned the dog's vet visit", &[]),
            None,
            at(20)
        )
    );
    let eps = call!("episodes", store.episodes(SCOPE));
    ensure(eps.is_empty(), || "episodes leaked across scopes".into())?;
    let ep_scores = call!(
        "episode_lexical_scores",
        store.episode_lexical_scores(SCOPE, "vet visit")
    );
    ensure(ep_scores.is_empty(), || {
        "episode_lexical_scores scored another scope's episode".into()
    })?;
    let forgot = call!(
        "supersede_fact",
        store.supersede_fact(SCOPE, "pet", "dog_name", at(30))
    );
    ensure(forgot, || {
        "supersede_fact returned false for an active fact".into()
    })?;
    let other = active(&*store, OTHER_SCOPE).await?;
    ensure(other.len() == 1 && other[0].value == "Biscuit", || {
        format!("forgetting in one scope affected another: {other:?}")
    })
}

async fn forget(store: Arc<dyn MemoryStore>) -> Check {
    call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "user", "city", "Springfield"), None, at(0))
    );
    let first = call!(
        "supersede_fact",
        store.supersede_fact(SCOPE, "user", "city", at(60))
    );
    ensure(first, || {
        "supersede_fact returned false for an active fact".into()
    })?;
    ensure(active(&*store, SCOPE).await?.is_empty(), || {
        "a forgotten fact is still active".into()
    })?;
    let history = call!("fact_history", store.fact_history(SCOPE, "user", "city"));
    ensure(
        history.len() == 1 && history[0].superseded_at.is_some(),
        || format!("a forgotten fact must stay in history as superseded: {history:?}"),
    )?;
    let again = call!(
        "supersede_fact (again)",
        store.supersede_fact(SCOPE, "user", "city", at(120))
    );
    ensure(!again, || {
        "supersede_fact returned true with nothing active".into()
    })?;
    let unknown = call!(
        "supersede_fact (unknown key)",
        store.supersede_fact(SCOPE, "user", "never_set", at(120))
    );
    ensure(!unknown, || {
        "supersede_fact returned true for an unknown key".into()
    })
}

async fn episodes(store: Arc<dyn MemoryStore>) -> Check {
    let new = episode(
        SCOPE,
        "Fixed the deployment pipeline for Acme Corp",
        &["deploy", "ci"],
    );
    let stored = call!(
        "insert_episode",
        store.insert_episode(new.clone(), None, at(0))
    );
    ensure(!stored.id.is_empty(), || "episode id is empty".into())?;
    ensure(
        stored.scope == new.scope
            && stored.session_id == new.session_id
            && stored.summary == new.summary
            && stored.topics == new.topics
            && close(stored.importance, new.importance)
            && stored.recall_count == 0,
        || format!("insert_episode did not keep the fields: {stored:?}"),
    )?;
    let other = call!(
        "insert_episode",
        store.insert_episode(
            episode(SCOPE, "Chose a birthday cake for Mia", &[]),
            None,
            at(10)
        )
    );
    let listed = call!("episodes", store.episodes(SCOPE));
    let ids: Vec<&str> = listed.iter().map(|r| r.episode.id.as_str()).collect();
    ensure(
        listed.len() == 2 && ids.contains(&stored.id.as_str()),
        || format!("episodes should list both episodes, got {ids:?}"),
    )?;
    let listed_stored = listed
        .iter()
        .find(|r| r.episode.id == stored.id)
        .map(|r| &r.episode);
    ensure(listed_stored == Some(&stored), || {
        format!("episodes returned {listed_stored:?}, insert_episode returned {stored:?}")
    })?;
    let scores = call!(
        "episode_lexical_scores",
        store.episode_lexical_scores(SCOPE, "deployment pipeline")
    );
    check_scores(&scores, &[stored.id.as_str()], &[other.id.as_str()])
}

fn check_scores(scores: &HashMap<String, f32>, matching: &[&str], not: &[&str]) -> Check {
    for id in matching {
        match scores.get(*id) {
            Some(s) if *s > 0.0 && *s <= 1.0 => {}
            other => return Err(format!("matching row {id} scored {other:?}, want (0, 1]")),
        }
    }
    for id in not {
        if scores.get(*id).is_some_and(|s| *s > 0.0) {
            return Err(format!("non-matching row {id} scored {:?}", scores[*id]));
        }
    }
    if let Some((id, s)) = scores.iter().find(|(_, s)| !(0.0..=1.0).contains(*s)) {
        return Err(format!("score {s} for {id} is outside 0..=1"));
    }
    Ok(())
}

async fn lexical(store: Arc<dyn MemoryStore>) -> Check {
    let fruit = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "food", "favorite_fruit", "mango"), None, at(0))
    );
    let dog = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "pet", "dog_name", "Rex"), None, at(0))
    );
    let scores = call!(
        "fact_lexical_scores",
        store.fact_lexical_scores(SCOPE, "mango")
    );
    check_scores(&scores, &[fruit.id()], &[dog.id()])?;
    let none = call!(
        "fact_lexical_scores",
        store.fact_lexical_scores(SCOPE, "zeppelin")
    );
    ensure(none.values().all(|s| *s <= 0.0), || {
        format!("a query matching nothing produced scores: {none:?}")
    })
}

async fn native_search(store: Arc<dyn MemoryStore>) -> Check<Vec<String>> {
    let mut notes = Vec::new();
    for (i, value) in ["mango smoothie", "mango salad", "mango lassi"]
        .iter()
        .enumerate()
    {
        call!(
            "upsert_fact",
            store.upsert_fact(
                fact(SCOPE, "food", &format!("dish_{i}"), value),
                None,
                at(0)
            )
        );
    }
    call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "food", "dish_0", "papaya"), None, at(10))
    );
    call!(
        "upsert_fact",
        store.upsert_fact(
            fact(OTHER_SCOPE, "food", "dish_9", "mango pie"),
            None,
            at(0)
        )
    );
    match call!("search_facts", store.search_facts(SCOPE, "mango", None, 1)) {
        None => notes.push("search_facts uses the scorer (None)".to_string()),
        Some(hits) => {
            ensure(hits.len() <= 1, || {
                format!("search_facts returned {} hits for limit 1", hits.len())
            })?;
            ensure(
                hits.iter()
                    .all(|h| h.fact.scope == SCOPE && h.fact.is_active()),
                || format!("search_facts returned out-of-scope or superseded facts: {hits:?}"),
            )?;
        }
    }
    for summary in ["Mango tasting", "Mango market trip"] {
        call!(
            "insert_episode",
            store.insert_episode(episode(SCOPE, summary, &[]), None, at(0))
        );
    }
    call!(
        "insert_episode",
        store.insert_episode(episode(OTHER_SCOPE, "Mango farm", &[]), None, at(0))
    );
    match call!(
        "search_episodes",
        store.search_episodes(SCOPE, "mango", None, 1)
    ) {
        None => notes.push("search_episodes uses the scorer (None)".to_string()),
        Some(hits) => {
            ensure(hits.len() <= 1, || {
                format!("search_episodes returned {} hits for limit 1", hits.len())
            })?;
            ensure(hits.iter().all(|h| h.episode.scope == SCOPE), || {
                "search_episodes returned another scope's episode".into()
            })?;
        }
    }
    Ok(notes)
}

async fn fact_recalls(store: Arc<dyn MemoryStore>) -> Check {
    let a = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "user", "city", "Springfield"), None, at(0))
    );
    let b = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "user", "team", "Platform"), None, at(0))
    );
    let ids = vec![a.id().to_string(), "conformance-unknown-id".to_string()];
    call!(
        "record_fact_recalls",
        store.record_fact_recalls(&ids, at(100))
    );
    call!(
        "record_fact_recalls",
        store.record_fact_recalls(&ids[..1], at(200))
    );
    let facts = active(&*store, SCOPE).await?;
    let get = |id: &str| facts.iter().find(|f| f.id == id).cloned();
    let fa = get(a.id()).ok_or("recalled fact disappeared")?;
    ensure(
        fa.recall_count == 2 && fa.last_recalled_at == Some(at(200)),
        || {
            format!(
                "after two recalls expected count 2 at {}, got {} at {:?}",
                at(200),
                fa.recall_count,
                fa.last_recalled_at
            )
        },
    )?;
    let fb = get(b.id()).ok_or("untouched fact disappeared")?;
    ensure(
        fb.recall_count == 0 && fb.last_recalled_at.is_none(),
        || format!("a fact that was not recalled changed: {fb:?}"),
    )
}

async fn episode_recalls(store: Arc<dyn MemoryStore>) -> Check {
    let a = call!(
        "insert_episode",
        store.insert_episode(
            episode(SCOPE, "Reviewed the quarterly plan", &[]),
            None,
            at(0)
        )
    );
    let b = call!(
        "insert_episode",
        store.insert_episode(episode(SCOPE, "Booked flights", &[]), None, at(0))
    );
    let ids = vec![a.id.clone(), "conformance-unknown-id".to_string()];
    call!(
        "record_episode_recalls",
        store.record_episode_recalls(&ids, at(100))
    );
    call!(
        "record_episode_recalls",
        store.record_episode_recalls(&ids[..1], at(200))
    );
    let eps: Vec<Episode> = call!("episodes", store.episodes(SCOPE))
        .into_iter()
        .map(|r| r.episode)
        .collect();
    let get = |id: &str| eps.iter().find(|e| e.id == id).cloned();
    let ea = get(&a.id).ok_or("recalled episode disappeared")?;
    ensure(
        ea.recall_count == 2 && ea.last_recalled_at == Some(at(200)),
        || {
            format!(
                "after two recalls expected count 2 at {}, got {} at {:?}",
                at(200),
                ea.recall_count,
                ea.last_recalled_at
            )
        },
    )?;
    let eb = get(&b.id).ok_or("untouched episode disappeared")?;
    ensure(
        eb.recall_count == 0 && eb.last_recalled_at.is_none(),
        || format!("an episode that was not recalled changed: {eb:?}"),
    )
}

fn tricky_vector() -> Vec<f32> {
    vec![
        0.1,
        -0.0,
        1.0 / 3.0,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        1.0e-45, // smallest subnormal
        -123.456,
        f32::EPSILON,
        7.0,
    ]
}

fn same_embedding(got: &Option<StoredEmbedding>, want: &Option<StoredEmbedding>) -> Check {
    match (got, want) {
        (None, None) => Ok(()),
        (Some(g), Some(w)) => {
            ensure(g.model == w.model, || {
                format!("model {:?} != {:?}", g.model, w.model)
            })?;
            ensure(g.vector.len() == w.vector.len(), || {
                format!("length {} != {}", g.vector.len(), w.vector.len())
            })?;
            match g.vector.iter().zip(&w.vector).position(|(a, b)| a != b) {
                None => Ok(()),
                Some(i) => Err(format!(
                    "component {i}: got {:e}, stored {:e}",
                    g.vector[i], w.vector[i]
                )),
            }
        }
        _ => Err(format!(
            "embedding presence changed: stored {}, got {}",
            want.is_some(),
            got.is_some()
        )),
    }
}

async fn embeddings(store: Arc<dyn MemoryStore>) -> Check {
    let embedding = Some(StoredEmbedding {
        model: "conformance-embedder-v1".to_string(),
        vector: tricky_vector(),
    });
    let with = call!(
        "upsert_fact",
        store.upsert_fact(
            fact(SCOPE, "user", "city", "Springfield"),
            embedding.clone(),
            at(0)
        )
    );
    let without = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "user", "team", "Platform"), None, at(0))
    );
    let records = call!("active_facts", store.active_facts(SCOPE));
    for (id, want) in [(with.id(), &embedding), (without.id(), &None)] {
        let got = records
            .iter()
            .find(|r| r.fact.id == id)
            .map(|r| r.embedding.clone())
            .ok_or_else(|| format!("fact {id} missing"))?;
        same_embedding(&got, want).map_err(|e| format!("fact embedding: {e}"))?;
    }
    let ep = call!(
        "insert_episode",
        store.insert_episode(
            episode(SCOPE, "Embedded episode", &[]),
            embedding.clone(),
            at(0)
        )
    );
    let eps = call!("episodes", store.episodes(SCOPE));
    let got = eps
        .iter()
        .find(|r| r.episode.id == ep.id)
        .map(|r| r.embedding.clone())
        .ok_or("episode missing")?;
    same_embedding(&got, &embedding).map_err(|e| format!("episode embedding: {e}"))
}

async fn stats(store: Arc<dyn MemoryStore>) -> Check {
    call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "work", "employer", "Acme Corp"), None, at(0))
    );
    call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "work", "employer", "Globex"), None, at(10))
    );
    call!(
        "upsert_fact",
        store.upsert_fact(fact(OTHER_SCOPE, "pet", "dog_name", "Rex"), None, at(0))
    );
    call!(
        "insert_episode",
        store.insert_episode(episode(SCOPE, "Changed jobs", &[]), None, at(20))
    );
    let s = call!("stats", store.stats());
    let got = (s.active_facts, s.superseded_facts, s.episodes, s.scopes);
    ensure(got == (2, 1, 1, 2), || {
        format!(
            "expected (active 2, superseded 1, episodes 1, scopes 2), got (active {}, \
             superseded {}, episodes {}, scopes {})",
            got.0, got.1, got.2, got.3
        )
    })
}

/// Rows the decay checks set up, and a snapshot of their observable state.
struct DecayFixture {
    recalled_old: String,
    recalled_recent: String,
    superseded: String,
    replacement: String,
    episode_never: String,
    episode_old: String,
    episode_recent: String,
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    active: Vec<(String, u32)>,
    history: Vec<String>,
    episodes: Vec<(String, u32)>,
}

fn now() -> DateTime<Utc> {
    t0() + Duration::days(400)
}

async fn decay_setup(store: &dyn MemoryStore) -> Check<DecayFixture> {
    let recalled_old = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "user", "city", "Springfield"), None, t0())
    )
    .id()
    .to_string();
    let ids = [recalled_old.clone()];
    for _ in 0..10 {
        call!("record_fact_recalls", store.record_fact_recalls(&ids, t0()));
    }
    let recalled_recent = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "user", "team", "Platform"), None, t0())
    )
    .id()
    .to_string();
    let ids = [recalled_recent.clone()];
    for _ in 0..5 {
        call!(
            "record_fact_recalls",
            store.record_fact_recalls(&ids, now() - Duration::days(1))
        );
    }
    let superseded = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "work", "employer", "Acme Corp"), None, t0())
    )
    .id()
    .to_string();
    let replacement = call!(
        "upsert_fact",
        store.upsert_fact(fact(SCOPE, "work", "employer", "Globex"), None, t0())
    )
    .id()
    .to_string();
    let episode_never = call!(
        "insert_episode",
        store.insert_episode(episode(SCOPE, "Never recalled", &[]), None, t0())
    )
    .id;
    let episode_old = call!(
        "insert_episode",
        store.insert_episode(episode(SCOPE, "Recalled long ago", &[]), None, t0())
    )
    .id;
    let ids = [episode_old.clone()];
    for _ in 0..3 {
        call!(
            "record_episode_recalls",
            store.record_episode_recalls(&ids, t0())
        );
    }
    let episode_recent = call!(
        "insert_episode",
        store.insert_episode(episode(SCOPE, "Recalled yesterday", &[]), None, t0())
    )
    .id;
    let ids = [episode_recent.clone()];
    call!(
        "record_episode_recalls",
        store.record_episode_recalls(&ids, now() - Duration::days(1))
    );
    Ok(DecayFixture {
        recalled_old,
        recalled_recent,
        superseded,
        replacement,
        episode_never,
        episode_old,
        episode_recent,
    })
}

async fn snapshot(store: &dyn MemoryStore) -> Check<Snapshot> {
    let mut active: Vec<(String, u32)> = call!("active_facts", store.active_facts(SCOPE))
        .into_iter()
        .map(|r| (r.fact.id, r.fact.recall_count))
        .collect();
    active.sort();
    let mut history: Vec<String> = call!(
        "fact_history",
        store.fact_history(SCOPE, "work", "employer")
    )
    .into_iter()
    .map(|f| f.id)
    .collect();
    history.sort();
    let mut episodes: Vec<(String, u32)> = call!("episodes", store.episodes(SCOPE))
        .into_iter()
        .map(|r| (r.episode.id, r.episode.recall_count))
        .collect();
    episodes.sort();
    Ok(Snapshot {
        active,
        history,
        episodes,
    })
}

async fn decay_disabled(store: Arc<dyn MemoryStore>) -> Check {
    decay_setup(&*store).await?;
    let before = snapshot(&*store).await?;
    let disabled = RetentionConfig {
        fact_recall_decay_after: None,
        episode_recall_decay_after: None,
        superseded_retention: None,
        episode_retention: None,
    };
    let report = call!("decay", store.decay(now(), &disabled));
    ensure(report == DecayReport::default(), || {
        format!("disabled decay reported changes: {report:?}")
    })?;
    let after = snapshot(&*store).await?;
    ensure(before == after, || {
        format!("disabled decay changed state: before {before:?}, after {after:?}")
    })
}

/// `Ok(true)` when decay applied the documented steps, `Ok(false)` when it
/// was a complete no-op.
async fn decay_default(store: Arc<dyn MemoryStore>) -> Check<bool> {
    let f = decay_setup(&*store).await?;
    let report = call!("decay", store.decay(now(), &RetentionConfig::default()));
    let s = snapshot(&*store).await?;
    let fact_count = |id: &str| s.active.iter().find(|(i, _)| i == id).map(|(_, c)| *c);
    let episode_count = |id: &str| s.episodes.iter().find(|(i, _)| i == id).map(|(_, c)| *c);
    let mut problems = Vec::new();

    // Always: active and recently recalled rows are untouched.
    if fact_count(&f.recalled_recent) != Some(5) {
        problems.push(format!(
            "recently recalled fact changed: {:?}",
            fact_count(&f.recalled_recent)
        ));
    }
    if fact_count(&f.replacement) != Some(0) {
        problems.push("the active replacement fact was deleted or changed".to_string());
    }
    if episode_count(&f.episode_recent) != Some(1) {
        problems.push(format!(
            "recently recalled episode changed: {:?}",
            episode_count(&f.episode_recent)
        ));
    }

    // Superseded retention: untouched, or exactly the old version deleted.
    let old_present = s.history.contains(&f.superseded);
    match (report.superseded_deleted, old_present) {
        (0, true) | (1, false) => {}
        (n, present) => problems.push(format!(
            "superseded retention: reported {n} deleted but the 400-day-old superseded \
             version is {}",
            if present { "still present" } else { "gone" }
        )),
    }
    // Episode retention: untouched, or exactly the never-recalled one gone.
    let never_present = episode_count(&f.episode_never).is_some();
    match (report.episodes_deleted, never_present) {
        (0, true) | (1, false) => {}
        (n, present) => problems.push(format!(
            "episode retention: reported {n} deleted but the never-recalled old episode is {}",
            if present { "still present" } else { "gone" }
        )),
    }
    if episode_count(&f.episode_old).is_none() {
        problems.push("a recalled episode was deleted by retention".to_string());
    }
    // Fact recall decay: 10 -> 9 (a tenth, at least one), or untouched.
    match (report.facts_decayed, fact_count(&f.recalled_old)) {
        (0, Some(10)) | (1, Some(9)) => {}
        (n, count) => problems.push(format!(
            "fact recall decay: reported {n} decayed, stale fact count is {count:?} \
             (expected 10 untouched or 9 decayed)"
        )),
    }
    // Episode recall decay: 3 -> 2, or untouched.
    match (report.episodes_decayed, episode_count(&f.episode_old)) {
        (0, Some(3)) | (1, Some(2)) => {}
        (n, count) => problems.push(format!(
            "episode recall decay: reported {n} decayed, stale episode count is {count:?} \
             (expected 3 untouched or 2 decayed)"
        )),
    }
    if problems.is_empty() {
        Ok(report != DecayReport::default())
    } else {
        Err(problems.join("; "))
    }
}

/// `Ok(true)` when the store keeps access levels, `Ok(false)` for the trait
/// default.
async fn access_levels(store: Arc<dyn MemoryStore>) -> Check<bool> {
    let shared = FactAccess::channel("conformance-channel");
    let first = call!(
        "upsert_fact_with_access",
        store.upsert_fact_with_access(
            fact(SCOPE, "work", "employer", "Acme Corp"),
            None,
            &FactAccess::global(),
            at(0)
        )
    );
    let id = first.id().to_string();
    let ids = [id.clone()];
    let access = call!("fact_access", store.fact_access(SCOPE, &ids));
    if access.is_empty() {
        return Ok(false);
    }
    ensure(access.get(&id) == Some(&FactAccess::global()), || {
        format!("expected global access for {id}, got {access:?}")
    })?;
    let other = call!("fact_access", store.fact_access(OTHER_SCOPE, &ids));
    ensure(other.is_empty(), || {
        format!("fact_access answered for another scope: {other:?}")
    })?;

    call!(
        "upsert_fact_with_access (restated, narrower)",
        store.upsert_fact_with_access(
            fact(SCOPE, "work", "employer", "Acme Corp"),
            None,
            &shared,
            at(10)
        )
    );
    call!(
        "upsert_fact_with_access (restated, wider)",
        store.upsert_fact_with_access(
            fact(SCOPE, "work", "employer", "Acme Corp"),
            None,
            &FactAccess::global(),
            at(20)
        )
    );
    let access = call!("fact_access", store.fact_access(SCOPE, &ids));
    ensure(access.get(&id) == Some(&shared), || {
        format!("restating must keep the narrowest access, got {access:?}")
    })?;

    let next = call!(
        "upsert_fact_with_access (new value)",
        store.upsert_fact_with_access(
            fact(SCOPE, "work", "employer", "Globex"),
            None,
            &FactAccess::global(),
            at(30)
        )
    );
    let ids = [next.id().to_string()];
    let access = call!("fact_access", store.fact_access(SCOPE, &ids));
    ensure(access.get(next.id()) == Some(&FactAccess::global()), || {
        format!("a new version takes the access it was written with, got {access:?}")
    })?;

    let stored = call!(
        "insert_episode_with_access",
        store.insert_episode_with_access(
            episode(SCOPE, "Reviewed the roadmap", &[]),
            None,
            &shared,
            at(40)
        )
    );
    let ids = [stored.id.clone()];
    let access = call!("episode_access", store.episode_access(SCOPE, &ids));
    ensure(access.get(&stored.id) == Some(&shared), || {
        format!("expected the channel access back for the episode, got {access:?}")
    })?;
    let other = call!("episode_access", store.episode_access(OTHER_SCOPE, &ids));
    ensure(other.is_empty(), || {
        format!("episode_access answered for another scope: {other:?}")
    })?;
    Ok(true)
}
