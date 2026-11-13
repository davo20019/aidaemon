//! Recall quality on realistic memory: does the right fact come back, and
//! does a corrected or forgotten fact stay gone?
//!
//! Ignored by default (it loads the daemon's local embedding model). Run:
//!
//! ```text
//! cargo test -p aidaemon-memory --features fastembed --release \
//!     --test recall_quality_eval -- --ignored --nocapture
//! ```
//!
//! `AIDAEMON_EVAL_RERANK=1` adds the daemon's cross-encoder reranker.
//! `AIDAEMON_EVAL_NOISE` sets the number of unrelated facts (default 400).
//!
//! The report has four parts:
//! - quality: recall@1, recall@5 and MRR over paraphrased questions whose
//!   wording shares little with the stored fact (`retrieval_eval`);
//! - stale leaks: after a correction or a delete, how often the old value
//!   is still recalled for the question it answers;
//! - canonical leaks: the same, read from the canonical personal-memory
//!   projection the owner's prompt receives as a block (profile keys such as
//!   residence, birthplace and birth_date never come back by question);
//! - conflicts: a changed fact stored under a different key, where recall
//!   returns both the old and new value with nothing marking which is current.
//!
//! All data is synthetic (Alice Rivera, Mia and Zoe, Acme Corp).
#![cfg(feature = "fastembed")]

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::ChannelVisibility;
use aidaemon_memory::records::{
    PersonalEntityCandidate, PersonalFactCandidate, PersonalMemoryWrite,
};
use aidaemon_memory::retrieval_eval::evaluate_ranked;
use aidaemon_memory::sqlite::{Reranker, SqliteMemory, migrate};
use aidaemon_memory::stores::FactStore;
use aidaemon_memory::{FactPrivacy, FastEmbedder};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};

const K: usize = 12;

/// (label, category, key, value). The label is a phrase that appears only
/// in that fact's value, so a recalled row is graded by its text, whichever
/// store (legacy facts or canonical personal memory) returned it.
const FACTS: &[(&str, &str, &str, &str)] = &[
    (
        "cortado",
        "preference",
        "coffee_order",
        "Prefers an oat-milk cortado with no sugar",
    ),
    (
        "lakeside",
        "family",
        "daughter_mia_school",
        "Mia attends Lakeside Elementary, third grade",
    ),
    (
        "epipen",
        "family",
        "daughter_zoe_allergy",
        "Zoe is allergic to peanuts; always carry her EpiPen",
    ),
    (
        "payments team",
        "work",
        "employer",
        "Works as a staff engineer at Acme Corp on the payments team",
    ),
    (
        "thursdays",
        "work",
        "manager",
        "Manager is Jordan Lee; weekly one-on-one on Thursdays",
    ),
    (
        "half marathon",
        "health",
        "running_goal",
        "Training for the Portland half marathon in May",
    ),
    (
        "heirloom",
        "project",
        "garden",
        "Growing heirloom tomatoes and basil in raised beds",
    ),
    (
        "jazz piano",
        "preference",
        "focus_music",
        "Listens to jazz piano while coding",
    ),
    (
        "subaru",
        "vehicle",
        "car",
        "Drives a 2019 blue Subaru Outback",
    ),
    (
        "biscuit",
        "pet",
        "dog",
        "Has a golden retriever named Biscuit",
    ),
    (
        "grocery",
        "finance",
        "grocery_budget",
        "Monthly grocery budget is 600 dollars",
    ),
    (
        "lisbon",
        "travel",
        "upcoming_trip",
        "Flying to Lisbon on June 12 for a conference",
    ),
    (
        "vegetarian",
        "preference",
        "diet",
        "Vegetarian, but eats fish occasionally",
    ),
    (
        "6am",
        "schedule",
        "gym",
        "Goes to the gym Monday, Wednesday and Friday at 6am",
    ),
    (
        "neovim",
        "tech",
        "editor",
        "Uses Neovim with a custom Lua config",
    ),
    (
        "tp-link",
        "home",
        "router",
        "Home network is a TP-Link mesh with the base on the second floor",
    ),
    (
        "pediatric nurse",
        "family",
        "partner",
        "Partner Sam is a pediatric nurse who works night shifts",
    ),
    (
        "le guin",
        "preference",
        "reading",
        "Favorite author is Ursula K. Le Guin; reads mostly science fiction",
    ),
    (
        "vitamin d",
        "health",
        "supplements",
        "Takes vitamin D and magnesium every morning",
    ),
    (
        "household chores",
        "project",
        "side_project",
        "Building a Rust CLI that tracks household chores",
    ),
];

/// (question, labels that answer it). Worded to share few words with the fact.
const QUESTIONS: &[(&str, &[&str])] = &[
    ("How do I take my coffee?", &["cortado"]),
    ("Where does Mia go to school?", &["lakeside"]),
    (
        "Is there any food my youngest daughter must avoid?",
        &["epipen"],
    ),
    ("Where do I work?", &["payments team"]),
    ("When is my regular meeting with my boss?", &["thursdays"]),
    ("What race am I preparing for?", &["half marathon"]),
    ("What am I planting this season?", &["heirloom"]),
    ("What should I put on to concentrate?", &["jazz piano"]),
    ("What car do I have?", &["subaru"]),
    ("What's my dog called?", &["biscuit"]),
    (
        "How much can I spend on food shopping each month?",
        &["grocery"],
    ),
    ("Do I have any trips coming up?", &["lisbon"]),
    ("Do I eat meat?", &["vegetarian"]),
    ("Which days do I work out?", &["6am"]),
    ("Which text editor do I use?", &["neovim"]),
    (
        "My internet is slow, what networking gear do I have?",
        &["tp-link"],
    ),
    (
        "What does my partner do for a living?",
        &["pediatric nurse"],
    ),
    ("Recommend a novel I'd enjoy", &["le guin"]),
    ("What pills do I take in the morning?", &["vitamin d"]),
    ("What am I coding in my spare time?", &["household chores"]),
];

const NOISE_TOPICS: &[&str] = &[
    "a podcast episode about urban planning",
    "a recipe for lentil soup",
    "the weather forecast for the weekend",
    "a documentary about deep sea creatures",
    "an article on solar panel efficiency",
    "a museum exhibit on ancient Rome",
    "a news story about a city council vote",
    "a tutorial on spreadsheet pivot tables",
    "a review of a mechanical keyboard",
    "a thread about houseplant watering",
    "a video on bicycle maintenance",
    "a blog post about minimalism",
];

struct JinaReranker(Mutex<fastembed::TextRerank>);

#[async_trait::async_trait]
impl Reranker for JinaReranker {
    async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let docs: Vec<&String> = documents.iter().collect();
        let ranked = self.0.lock().unwrap().rerank(&query, docs, false, None)?;
        Ok(ranked.into_iter().map(|r| (r.index, r.score)).collect())
    }
}

fn model_cache() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.fastembed_cache")
}

async fn memory() -> (tempfile::TempDir, SqliteMemory) {
    let dir = tempfile::tempdir().unwrap();
    let options = SqliteConnectOptions::new()
        .filename(dir.path().join("eval.db"))
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .busy_timeout(Duration::from_secs(30));
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await
        .unwrap();
    migrate(&pool).await.unwrap();
    let embedder = FastEmbedder::new().cache_dir(model_cache());
    let mut memory = SqliteMemory::new(pool, Arc::new(embedder));
    if std::env::var("AIDAEMON_EVAL_RERANK").is_ok_and(|v| v == "1") {
        let options = fastembed::RerankInitOptions::new(
            fastembed::RerankerModel::JINARerankerV2BaseMultilingual,
        )
        .with_cache_dir(model_cache());
        let model = fastembed::TextRerank::try_new(options).unwrap();
        memory = memory.with_reranker(Arc::new(JinaReranker(Mutex::new(model))));
    }
    (dir, memory)
}

async fn put(memory: &SqliteMemory, category: &str, key: &str, value: &str) {
    memory
        .upsert_fact(
            category,
            key,
            value,
            "user_stated",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
}

/// The owner's DM recall path, the one the daemon uses in a private chat.
async fn recall(memory: &SqliteMemory, question: &str) -> Vec<String> {
    memory
        .get_relevant_facts_for_channel(question, K, None, ChannelVisibility::Private, true)
        .await
        .unwrap()
        .into_iter()
        .map(|fact| format!("{} {}", fact.key, fact.value).to_lowercase())
        .collect()
}

fn labels(recalled: &[String]) -> Vec<String> {
    recalled
        .iter()
        .map(|text| {
            FACTS
                .iter()
                .find(|(label, ..)| text.contains(label))
                .map_or_else(|| "other".to_string(), |(label, ..)| label.to_string())
        })
        .collect()
}

fn rank_of(recalled: &[String], needle: &str) -> Option<usize> {
    recalled.iter().position(|text| text.contains(needle))
}

/// A leak probe means something only if the value was recallable before the
/// change; print that control.
async fn control(memory: &SqliteMemory, name: &str, question: &str, value: &str) {
    let recalled = recall(memory, question).await;
    println!(
        "  control {name}: {value:?} before change {} (recalled {})",
        rank_of(&recalled, value)
            .map_or("NOT recalled".to_string(), |r| format!("at rank {}", r + 1)),
        recalled.len()
    );
}

fn owner_write(predicate: &str, value: &str, correction: bool) -> PersonalMemoryWrite {
    PersonalMemoryWrite {
        entities: vec![PersonalEntityCandidate {
            local_id: "owner".into(),
            entity_type: "person".into(),
            canonical_name: "Owner".into(),
            is_reference: false,
            canonical_name_confirmed: false,
        }],
        facts: vec![PersonalFactCandidate {
            subject_local_id: "owner".into(),
            predicate: predicate.into(),
            value: value.into(),
            display_value: None,
            valid_from: None,
            valid_to: None,
        }],
        direct_user_statement: true,
        correction,
        ..Default::default()
    }
}

async fn reconcile(memory: &SqliteMemory, write: &PersonalMemoryWrite) -> anyhow::Result<()> {
    memory
        .reconcile_personal_memory(write, "agent", None, None, FactPrivacy::Private)
        .await
        .map(|_| ())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "eval: loads the local embedding model"]
async fn recall_quality_eval() {
    let noise: usize = std::env::var("AIDAEMON_EVAL_NOISE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let (_dir, memory) = memory().await;

    // An older employer and an older trip, both replaced below by an upsert
    // of the same key: the realistic "that changed" path.
    put(
        &memory,
        "work",
        "employer",
        "Works as a data analyst at Globex",
    )
    .await;
    put(
        &memory,
        "travel",
        "upcoming_trip",
        "Flying to Kyoto on April 2 for vacation",
    )
    .await;
    for (_, category, key, value) in FACTS {
        put(&memory, category, key, value).await;
    }
    for i in 0..noise {
        let topic = NOISE_TOPICS[i % NOISE_TOPICS.len()];
        put(
            &memory,
            "notes",
            &format!("note_{i}"),
            &format!("Alice Rivera saved {topic} (item {i})"),
        )
        .await;
    }
    while memory.backfill_fact_embeddings().await.unwrap() > 0 {}

    // ---- quality ----
    let mut r1 = 0.0;
    let mut r5 = 0.0;
    let mut mrr = 0.0;
    let mut misses = Vec::new();
    for (question, gold) in QUESTIONS {
        let ranked = labels(&recall(&memory, question).await);
        let relevant: HashSet<String> = gold.iter().map(|g| g.to_string()).collect();
        let at1 = evaluate_ranked(&ranked, &relevant, 1);
        let at5 = evaluate_ranked(&ranked, &relevant, 5);
        let atk = evaluate_ranked(&ranked, &relevant, K);
        r1 += at1.recall_at_k;
        r5 += at5.recall_at_k;
        mrr += atk.reciprocal_rank;
        if at5.recall_at_k < 1.0 {
            misses.push(format!(
                "{question} -> top3 {:?}",
                &ranked[..ranked.len().min(3)]
            ));
        }
    }
    let n = QUESTIONS.len() as f32;
    println!(
        "quality facts={} noise={noise} questions={} recall@1={:.2} recall@5={:.2} mrr@{K}={:.2}",
        FACTS.len(),
        QUESTIONS.len(),
        r1 / n,
        r5 / n,
        mrr / n
    );
    for miss in &misses {
        println!("  miss: {miss}");
    }

    // ---- stale leaks (each probe: question, value that must be gone) ----
    let mut probes: Vec<(String, &str, &str)> = Vec::new();

    // 1. Same-key correction through upsert.
    probes.push((
        "same-key correction (employer)".into(),
        "Where do I work?",
        "globex",
    ));
    probes.push((
        "same-key correction (trip)".into(),
        "Do I have any trips coming up?",
        "kyoto",
    ));

    // 2. remember_fact delete marker on an ordinary key.
    control(&memory, "dog", "What's my dog called?", "biscuit").await;
    memory.delete_fact_by_key("pet", "dog").await.unwrap();
    probes.push((
        "delete_fact_by_key (dog)".into(),
        "What's my dog called?",
        "biscuit",
    ));

    // 3. manage_memories forget (delete_fact by id).
    control(
        &memory,
        "supplements",
        "What pills do I take in the morning?",
        "vitamin d",
    )
    .await;
    let supplements = memory
        .get_facts(Some("health"))
        .await
        .unwrap()
        .into_iter()
        .find(|f| f.key == "supplements")
        .unwrap();
    memory.delete_fact(supplements.id).await.unwrap();
    probes.push((
        "forget by id (supplements)".into(),
        "What pills do I take in the morning?",
        "vitamin d",
    ));

    // 4. Canonical profile key, then "forget it" the way remember_fact does
    //    it for every other key (delete_fact_by_key on user/residence).
    reconcile(&memory, &owner_write("residence", "Denver", false))
        .await
        .unwrap();
    control(&memory, "residence", "Where do I live?", "denver").await;
    let removed = memory
        .delete_fact_by_key("user", "residence")
        .await
        .unwrap();
    println!("canonical residence: delete_fact_by_key reported removed={removed}");
    probes.push((
        "forget canonical residence".into(),
        "Where do I live?",
        "denver",
    ));

    // 5. The same forget sent as remember_fact value "delete": profile keys
    //    route the marker into the canonical store as a value.
    let marker = reconcile(&memory, &owner_write("birthplace", "Springfield", false)).await;
    control(&memory, "birthplace", "Where was I born?", "springfield").await;
    let marker_write = reconcile(&memory, &owner_write("birthplace", "delete", false)).await;
    println!(
        "canonical birthplace: seed ok={} 'delete' marker write ok={}",
        marker.is_ok(),
        marker_write.is_ok()
    );
    probes.push((
        "'delete' marker on birthplace".into(),
        "Where was I born?",
        "springfield",
    ));

    // 6. A correction the owner states without the correction flag, against
    //    an earlier high-confidence owner statement.
    reconcile(&memory, &owner_write("birth_date", "1990-04-12", false))
        .await
        .unwrap();
    control(&memory, "birth_date", "When is my birthday?", "1990-04-12").await;
    reconcile(&memory, &owner_write("birth_date", "1990-05-12", false))
        .await
        .unwrap();
    probes.push((
        "unflagged correction (birth_date)".into(),
        "When is my birthday?",
        "1990-04-12",
    ));

    // 7. The same with the correction flag set.
    reconcile(&memory, &owner_write("current_employer", "Initech", false))
        .await
        .unwrap();
    control(
        &memory,
        "current_employer",
        "Who is my employer?",
        "initech",
    )
    .await;
    reconcile(&memory, &owner_write("current_employer", "Hooli", true))
        .await
        .unwrap();
    probes.push((
        "flagged correction (current_employer)".into(),
        "Who is my employer?",
        "initech",
    ));

    // The canonical store reaches the prompt as a block, not by question:
    // read the projection the daemon injects for the owner.
    let canonical: Vec<String> = memory
        .get_canonical_memory_facts()
        .await
        .unwrap()
        .into_iter()
        .map(|f| format!("{}={}", f.key, f.value).to_lowercase())
        .collect();
    println!("canonical projection after deletes/corrections: {canonical:?}");
    let mut canonical_leaks = 0;
    for (name, stale) in [
        ("forgotten residence", "denver"),
        ("'delete' marker birthplace", "springfield"),
        ("'delete' stored as a value", "=delete"),
        ("unflagged correction birth_date", "1990-04-12"),
        ("flagged correction employer", "initech"),
    ] {
        let present = canonical.iter().any(|row| row.contains(stale));
        if present {
            canonical_leaks += 1;
        }
        println!(
            "  canonical {name}: {stale:?} {}",
            if present { "STILL ACTIVE" } else { "gone" }
        );
    }
    println!("canonical_leaks={canonical_leaks}/5");

    let mut leaks = 0;
    for (name, question, stale) in &probes {
        let recalled = recall(&memory, question).await;
        let rank = rank_of(&recalled, stale);
        if rank.is_some() {
            leaks += 1;
        }
        let delete_value = rank_of(&recalled, "delete").map(|r| recalled[r].clone());
        if std::env::var("AIDAEMON_EVAL_VERBOSE").is_ok() {
            println!("    recalled: {:?}", &recalled[..recalled.len().min(4)]);
        }
        println!(
            "  leak-probe {name}: stale {:?} {}{}",
            stale,
            rank.map_or("gone".to_string(), |r| format!(
                "RECALLED at rank {}",
                r + 1
            )),
            delete_value.map_or(String::new(), |v| format!(" | recalled literal: {v:?}"))
        );
    }
    println!("stale_leaks={leaks}/{}", probes.len());

    // ---- conflicts: the change is stored under a new key ----
    put(
        &memory,
        "vehicle",
        "current_vehicle",
        "Sold the old car; now drives a white Tesla Model 3",
    )
    .await;
    while memory.backfill_fact_embeddings().await.unwrap() > 0 {}
    let recalled = recall(&memory, "What car do I have?").await;
    println!(
        "conflict (car under a new key): new at {:?}, old at {:?}",
        rank_of(&recalled, "tesla").map(|r| r + 1),
        rank_of(&recalled, "subaru").map(|r| r + 1)
    );
}
