//! Calibration of the explicit-search relevance floor for the daemon's
//! cross-encoder (`jina-reranker-v2-base-multilingual`).
//!
//! Ignored by default (it loads the local reranker model). Run:
//!
//! ```text
//! cargo test -p aidaemon-memory --features fastembed --release \
//!     --test reranker_floor_calibration -- --ignored --nocapture
//! ```
//!
//! The queries are shaped like the ones the model actually sends to
//! `manage_memories search`: keyword lists of near-synonyms, names, and
//! namespace words, not questions. Each query labels which facts answer it.
//! The report prints, per query, the best relevant and best irrelevant
//! score, then evaluates candidate floors: how many relevant facts each
//! would drop and how many irrelevant facts it would let through.
//!
//! All data is synthetic (Alice Rivera, Mia and Zoe, Acme Corp).
#![cfg(feature = "fastembed")]

use std::collections::HashSet;
use std::path::PathBuf;

use aidaemon_memory::records::Fact;
use aidaemon_memory::search::LexicalIndex;
use aidaemon_memory::sqlite::build_fact_embedding_text;
use aidaemon_memory::FactPrivacy;
use chrono::{TimeZone, Utc};

/// `(category, key, value)`.
const CORPUS: &[(&str, &str, &str)] = &[
    ("user", "spouse_name", "Alice Rivera"),
    ("user", "daughter_names", "Mia and Zoe"),
    ("user", "mother_name", "Elena Rivera"),
    ("user", "father_name", "Tomas Rivera"),
    ("user", "birth_date", "1984-03-12"),
    ("user", "home_city", "Springfield"),
    ("user", "employer", "Acme Corp"),
    ("user", "job_title", "staff engineer"),
    ("user", "work_email", "owner@acme.example"),
    ("user", "email_personal", "owner@mail.example"),
    ("user", "hostname", "owner-laptop"),
    ("user", "daily_ritual", "walks the dog before breakfast"),
    ("user", "morning_drink", "green tea"),
    ("user", "favorite_language", "Rust"),
    ("user", "editor", "Neovim"),
    ("user", "operating_system", "macOS"),
    ("preference", "reply_style", "short and direct, no preamble"),
    ("preference", "verbosity", "terse"),
    ("preference", "time_zone", "US Eastern"),
    ("people", "Jordan Lee · favorite_coffee", "cortado"),
    ("people", "Jordan Lee · relationship_to_owner", "colleague"),
    ("people", "Sam Patel · relationship_to_owner", "neighbor"),
    ("people", "Sam Patel · dog_name", "Pixel"),
    ("people", "Mia · birthday", "2015-06-02"),
    ("people", "Zoe · birthday", "2018-09-21"),
    ("project", "orion_database", "Postgres 16"),
    (
        "project",
        "orion_deploy_schedule",
        "deploys to staging every Tuesday",
    ),
    ("project", "orion_repo", "github.com/acme/orion"),
    ("project", "aidemo:version", "0.4.2"),
    ("project", "aidemo:tech_stack", "Rust, SQLite, Tokio"),
    ("project", "blog_repo", "github.com/owner/blog"),
    ("project", "owner_website_domain", "owner.example"),
    (
        "workflow",
        "workflow:blog_deployment",
        "push to main, the worker redeploys",
    ),
    (
        "workflow",
        "workflow:release_checklist",
        "bump version, changelog, tag",
    ),
    (
        "workflow",
        "workflow:scheduling",
        "recurring jobs run at 9am",
    ),
    ("health", "allergy", "penicillin"),
    ("hobby", "weekend_hobby", "trail running"),
    ("vehicle", "car", "2019 Subaru Outback"),
];

/// `(query, keys that answer it)`. An empty list means nothing answers it.
const QUERIES: &[(&str, &[&str])] = &[
    (
        "coworker teammate colleague coffee",
        &["Jordan Lee · favorite_coffee"],
    ),
    ("Jordan Lee coffee", &["Jordan Lee · favorite_coffee"]),
    ("Sam Patel dog_name", &["Sam Patel · dog_name"]),
    ("neighbor dog pet name", &["Sam Patel · dog_name"]),
    (
        "orion",
        &["orion_database", "orion_deploy_schedule", "orion_repo"],
    ),
    ("orion deploy schedule", &["orion_deploy_schedule"]),
    ("orion database postgres", &["orion_database"]),
    (
        "mother father parents names",
        &["mother_name", "father_name"],
    ),
    ("spouse partner wife", &["spouse_name"]),
    ("daughter names kids children", &["daughter_names"]),
    ("daughters birthdays", &["Mia · birthday", "Zoe · birthday"]),
    ("birth date birthday", &["birth_date"]),
    ("reply style concise direct", &["reply_style", "verbosity"]),
    ("employer company work", &["employer", "job_title"]),
    ("email address", &["work_email", "email_personal"]),
    ("programming language", &["favorite_language"]),
    ("allergies medication", &["allergy"]),
    ("car vehicle", &["car"]),
    ("blog deploy", &["workflow:blog_deployment", "blog_repo"]),
    ("boat own sailboat motorboat vessel", &[]),
    ("boat", &[]),
    ("zzqx nonexistent topic 4417", &[]),
    ("favorite movie film", &[]),
    ("gym membership", &[]),
];

fn fact(id: i64, category: &str, key: &str, value: &str) -> Fact {
    let at = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    Fact {
        id,
        category: category.into(),
        key: key.into(),
        value: value.into(),
        source: "synthetic".into(),
        created_at: at,
        updated_at: at,
        superseded_at: None,
        recall_count: 0,
        last_recalled_at: None,
        channel_id: None,
        privacy: FactPrivacy::Global,
        first_seen_at: None,
        source_excerpt: None,
    }
}

fn model_cache() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.fastembed_cache")
}

#[test]
#[ignore = "loads the local reranker model"]
fn reranker_floor_calibration() {
    let options =
        fastembed::RerankInitOptions::new(fastembed::RerankerModel::JINARerankerV2BaseMultilingual)
            .with_cache_dir(model_cache());
    let mut model = fastembed::TextRerank::try_new(options).unwrap();
    let docs: Vec<String> = CORPUS
        .iter()
        .map(|(c, k, v)| build_fact_embedding_text(c, k, v))
        .collect();

    let facts: Vec<Fact> = CORPUS
        .iter()
        .enumerate()
        .map(|(i, (c, k, v))| fact(i as i64 + 1, c, k, v))
        .collect();
    let lexical_index = LexicalIndex::build(&facts);

    // (query index, score, relevant, lexical match)
    let mut scored: Vec<(usize, f32, bool, bool)> = Vec::new();
    for (qi, (query, answers)) in QUERIES.iter().enumerate() {
        let lexical: HashSet<usize> = lexical_index
            .score(query)
            .into_iter()
            .map(|hit| hit.index)
            .collect();
        let ranked = model
            .rerank(
                *query,
                docs.iter().map(String::as_str).collect::<Vec<_>>(),
                false,
                None,
            )
            .unwrap();
        let mut best_rel = f32::NEG_INFINITY;
        let mut best_irr = (f32::NEG_INFINITY, "", false);
        for r in &ranked {
            let key = CORPUS[r.index].1;
            let relevant = answers.contains(&key);
            let lex = lexical.contains(&r.index);
            scored.push((qi, r.score, relevant, lex));
            if relevant {
                best_rel = best_rel.max(r.score);
            } else if r.score > best_irr.0 {
                best_irr = (r.score, key, lex);
            }
        }
        let worst_rel = scored
            .iter()
            .filter(|s| s.0 == qi && s.2)
            .map(|s| (s.1, s.3))
            .fold((f32::INFINITY, false), |a, b| if b.0 < a.0 { b } else { a });
        println!(
            "{query:<40} rel best={best_rel:>7.3} worst={:>7.3}{} | irr best={:>7.3}{} ({})",
            worst_rel.0,
            if worst_rel.1 { "L" } else { " " },
            best_irr.0,
            if best_irr.2 { "L" } else { " " },
            best_irr.1
        );
    }

    println!("\nfloor for semantic-only candidates (S) and lexically matched ones (L):");
    for lexical_floor in [-3.0_f32, -2.75, -2.5] {
        for semantic_floor in [-2.5_f32, -2.25, -2.0, -1.75, -1.5] {
            let mut dropped_rel = 0;
            let mut kept_irr = 0;
            let mut no_answer_rows = 0;
            for (qi, (_, answers)) in QUERIES.iter().enumerate() {
                for s in scored.iter().filter(|s| s.0 == qi) {
                    let floor = if s.3 { lexical_floor } else { semantic_floor };
                    match (s.2, s.1 >= floor) {
                        (true, false) => dropped_rel += 1,
                        (false, true) => {
                            kept_irr += 1;
                            if answers.is_empty() {
                                no_answer_rows += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
            println!(
                "L={lexical_floor:>5.2} S={semantic_floor:>5.2}: dropped relevant={dropped_rel:>2}  kept irrelevant={kept_irr:>3}  rows on no-answer queries={no_answer_rows}"
            );
        }
    }
}
