//! Synthetic corpus and labelled queries for explicit fact search.
//!
//! Reproduces the failure shapes seen on real stores without any real data:
//! a high-recall cluster that used to win every expansion, namespaces named
//! after common words, a tie flood on two common words, namespace spelling
//! variants, a family cluster, a synonym-only answer, and a canonical
//! projection beside its raw row.

use chrono::{DateTime, Duration, TimeZone, Utc};

use crate::records::{Fact, FactPrivacy};

/// Id of the canonical projection that duplicates [`HOME_CITY`].
pub(crate) const HOME_CITY_PROJECTION: i64 = -1_000_000_000 - HOME_CITY;
pub(crate) const HOME_CITY: i64 = 7000;
pub(crate) const PARTNER_NAME: i64 = 5000;
pub(crate) const MOTHER_NAME: i64 = 5001;
pub(crate) const FATHER_NAME: i64 = 5002;
pub(crate) const DAUGHTER_NAMES: i64 = 5003;
pub(crate) const OWNER_NAME: i64 = 5004;
pub(crate) const BIRTH_DATE: i64 = 5010;
pub(crate) const REPLY_STYLE: i64 = 5011;
pub(crate) const CODE_REVIEW: i64 = 5012;
pub(crate) const STAGING_DEPLOY_PORT: i64 = 6100;
pub(crate) const WORKFLOW_RELEASE_CHECKLIST: i64 = 2000;

/// Hot cluster: `acme_newsletter_*` keys and the `newsletter.acme.test:` namespace.
pub(crate) const HOT_CLUSTER: std::ops::Range<i64> = 1000..1040;
/// `workflow:` namespace facts.
pub(crate) const WORKFLOW_NS: std::ops::Range<i64> = 2000..2030;
/// `project:` namespace facts.
pub(crate) const PROJECT_NS: std::ops::Range<i64> = 2100..2120;
/// Flat facts whose values mention "workflow" or "project".
pub(crate) const GENERIC_MENTIONS: std::ops::Range<i64> = 3000..3120;
/// The 60 facts that each mention "deploy" and "staging" once.
pub(crate) const TIE_FLOOD: std::ops::Range<i64> = 6000..6060;

const HOT_CLUSTER_IDS: [i64; 40] = {
    let mut ids = [0i64; 40];
    let mut i = 0;
    while i < 40 {
        ids[i] = 1000 + i as i64;
        i += 1;
    }
    ids
};

/// Distinct words so each namespaced key is individually addressable.
const WORDS: [&str; 30] = [
    "release_checklist",
    "lint",
    "format",
    "bundle",
    "sign",
    "notarize",
    "upload",
    "mirror",
    "tag",
    "announce",
    "rollback",
    "smoke",
    "canary",
    "migrate",
    "seed",
    "vacuum",
    "rotate",
    "archive",
    "prune",
    "index",
    "warm",
    "profile",
    "audit",
    "triage",
    "label",
    "close",
    "merge",
    "rebase",
    "squash",
    "bisect",
];

const FILLER_TOPICS: [&str; 10] = [
    "gardening tomatoes",
    "bicycle tires",
    "jazz records",
    "sourdough starter",
    "chess openings",
    "hiking boots",
    "tea kettle",
    "piano scales",
    "camera lenses",
    "pottery glaze",
];

fn base_time() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
}

pub(crate) fn fact(id: i64, category: &str, key: &str, value: &str, recall_count: i32) -> Fact {
    // Deterministic, distinct timestamps: larger ids are newer.
    let at = base_time() + Duration::seconds(id.rem_euclid(1_000_000));
    Fact {
        id,
        category: category.into(),
        key: key.into(),
        value: value.into(),
        source: "synthetic".into(),
        created_at: at,
        updated_at: at,
        superseded_at: None,
        recall_count,
        last_recalled_at: None,
        channel_id: None,
        privacy: FactPrivacy::Global,
        first_seen_at: None,
        source_excerpt: None,
    }
}

/// About 400 synthetic facts with stable ids.
pub(crate) fn corpus() -> Vec<Fact> {
    let mut facts = Vec::new();

    // Hot cluster: recalled constantly by a recurring job.
    for (i, id) in HOT_CLUSTER.enumerate() {
        let recall = 80 + (i as i32 * 3);
        if i < 20 {
            facts.push(fact(
                id,
                "project",
                &format!("acme_newsletter_{}", WORDS[i % 30].replace('_', "")),
                &format!("Acme newsletter issue {i} goes out Monday to subscribers"),
                recall,
            ));
        } else {
            facts.push(fact(
                id,
                "project",
                &format!("newsletter.acme.test:{}", WORDS[i % 30].replace('_', "")),
                &format!("Newsletter section {i} lists subscriber growth numbers"),
                recall,
            ));
        }
    }

    // Generic-word namespaces.
    for (i, id) in WORKFLOW_NS.enumerate() {
        facts.push(fact(
            id,
            "technical",
            &format!("workflow:{}", WORDS[i]),
            &format!("step {i} runs on the build machine"),
            (i % 10) as i32,
        ));
    }
    for (i, id) in PROJECT_NS.enumerate() {
        facts.push(fact(
            id,
            "project",
            &format!("project:{}", WORDS[i]),
            &format!("milestone {i} is tracked on the board"),
            (i % 10) as i32,
        ));
    }
    for (i, id) in GENERIC_MENTIONS.enumerate() {
        let topic = FILLER_TOPICS[i % FILLER_TOPICS.len()];
        let value = if i % 2 == 0 {
            format!("the workflow for {topic} batch {i}")
        } else {
            format!("the project about {topic} round {i}")
        };
        facts.push(fact(
            id,
            "general",
            &format!("note_{i}"),
            &value,
            (i % 10) as i32,
        ));
    }

    // Namespace spelling variants: one namespace, three spellings.
    for (offset, ns) in [(4000, "Acme.Blog"), (4010, "acme_blog"), (4020, "acmeblog")] {
        for i in 0..5 {
            facts.push(fact(
                offset + i,
                "project",
                &format!("{ns}:{}", WORDS[(offset as usize / 10 + i as usize) % 30]),
                &format!("blog item {i} for the {ns} site"),
                (i % 10) as i32,
            ));
        }
    }

    // Family cluster for the owner.
    facts.push(fact(PARTNER_NAME, "user", "partner_name", "Jordan Lee", 3));
    facts.push(fact(MOTHER_NAME, "user", "mother_name", "Maria Rivera", 2));
    facts.push(fact(FATHER_NAME, "user", "father_name", "Frank Rivera", 2));
    facts.push(fact(
        DAUGHTER_NAMES,
        "user",
        "daughter_names",
        "Mia and Zoe",
        4,
    ));
    facts.push(fact(OWNER_NAME, "user", "name", "Alice Rivera", 5));

    // Plain preferences the hot cluster used to crowd out.
    facts.push(fact(BIRTH_DATE, "user", "birth_date", "1990-04-12", 1));
    facts.push(fact(
        REPLY_STYLE,
        "preference",
        "reply_style",
        "short and direct answers, no preamble",
        1,
    ));
    facts.push(fact(
        CODE_REVIEW,
        "preference",
        "code_review_preference",
        "look at tests before implementation",
        1,
    ));

    // Tie flood: every value has exactly "deploy" and "staging".
    for (i, id) in TIE_FLOOD.enumerate() {
        facts.push(fact(
            id,
            "technical",
            &format!("release_note_{i}"),
            &format!("deploy to staging after check {i}"),
            (i % 10) as i32,
        ));
    }
    facts.push(fact(
        STAGING_DEPLOY_PORT,
        "technical",
        "staging_deploy_port",
        "8443",
        0,
    ));

    // Duplicate pair: a canonical projection beside its raw row.
    facts.push(fact(HOME_CITY, "user", "home_city", "Springfield", 1));
    facts.push(fact(
        HOME_CITY_PROJECTION,
        "user",
        "home_city",
        "Springfield",
        0,
    ));

    // Filler so idf has a realistic denominator.
    for i in 0..100 {
        let topic = FILLER_TOPICS[i % FILLER_TOPICS.len()];
        facts.push(fact(
            8000 + i as i64,
            "general",
            &format!("trivia_{i}"),
            &format!("{topic} detail number {i}"),
            (i % 10) as i32,
        ));
    }

    facts
}

pub(crate) struct LabelledQuery {
    pub query: &'static str,
    pub limit: usize,
    pub relevant: &'static [i64],
    pub forbidden_related: &'static [i64],
}

pub(crate) fn queries() -> Vec<LabelledQuery> {
    vec![
        LabelledQuery {
            query: "birth date",
            limit: 5,
            relevant: &[BIRTH_DATE],
            forbidden_related: &HOT_CLUSTER_IDS,
        },
        LabelledQuery {
            query: "reply style",
            limit: 5,
            relevant: &[REPLY_STYLE],
            forbidden_related: &HOT_CLUSTER_IDS,
        },
        LabelledQuery {
            query: "code review preference",
            limit: 5,
            relevant: &[CODE_REVIEW],
            forbidden_related: &HOT_CLUSTER_IDS,
        },
        LabelledQuery {
            query: "staging deploy port",
            limit: 10,
            relevant: &[STAGING_DEPLOY_PORT],
            forbidden_related: &[],
        },
        LabelledQuery {
            query: "spouse partner wife",
            limit: 5,
            relevant: &[PARTNER_NAME],
            forbidden_related: &[],
        },
        LabelledQuery {
            query: "home city town residence location where live hometown address place springfield",
            limit: 5,
            relevant: &[HOME_CITY],
            forbidden_related: &[],
        },
        LabelledQuery {
            query: "workflow release checklist",
            limit: 5,
            relevant: &[WORKFLOW_RELEASE_CHECKLIST],
            forbidden_related: &[],
        },
        LabelledQuery {
            query: "Alice mother",
            limit: 5,
            relevant: &[MOTHER_NAME],
            forbidden_related: &[],
        },
    ]
}

fn words(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Deterministic token-overlap stand-in for an embedding cosine, so pure
/// tests need no model: `|q ∩ f| / sqrt(|q| * |f|)`.
pub(crate) fn fake_cosine(query: &str, fact: &Fact) -> f32 {
    let q = words(query);
    let f = words(&format!("{} {} {}", fact.category, fact.key, fact.value));
    if q.is_empty() || f.is_empty() {
        return 0.0;
    }
    let shared = q.intersection(&f).count() as f32;
    shared / ((q.len() * f.len()) as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn fixture_is_self_consistent() {
        let facts = corpus();
        assert!(
            (380..=420).contains(&facts.len()),
            "about 400 facts, got {}",
            facts.len()
        );
        let mut seen: HashMap<i64, usize> = HashMap::new();
        for f in &facts {
            *seen.entry(f.id).or_default() += 1;
        }
        assert!(seen.values().all(|n| *n == 1), "ids are unique");
        let ids: HashSet<i64> = seen.keys().copied().collect();
        for q in queries() {
            for id in q.relevant.iter().chain(q.forbidden_related) {
                assert!(ids.contains(id), "{}: id {id} missing", q.query);
            }
        }
        assert!(queries().len() >= 8);
        assert!(
            queries()
                .iter()
                .any(|q| q.query.split_whitespace().count() == 11)
        );

        // The only same-key-same-value pair is the deliberate duplicate.
        let mut by_content: HashMap<(String, String, String), Vec<i64>> = HashMap::new();
        for f in &facts {
            by_content
                .entry((f.category.clone(), f.key.clone(), f.value.clone()))
                .or_default()
                .push(f.id);
        }
        let dupes: Vec<_> = by_content.values().filter(|v| v.len() > 1).collect();
        assert_eq!(dupes.len(), 1);
        let mut pair = dupes[0].clone();
        pair.sort();
        assert_eq!(pair, vec![HOME_CITY_PROJECTION, HOME_CITY]);

        // The hot cluster is the only high-recall group.
        for f in &facts {
            assert_eq!(
                HOT_CLUSTER.contains(&f.id),
                f.recall_count >= 80,
                "fact {}",
                f.id
            );
        }
        // The synonym answer shares no query synonym but "partner".
        let partner = facts.iter().find(|f| f.id == PARTNER_NAME).unwrap();
        assert!(!partner.value.to_lowercase().contains("wife"));
        assert!(!partner.value.to_lowercase().contains("spouse"));
    }

    /// Every labelled query through the pure pipeline (fake cosine, no
    /// reranker): the answer is returned within its limit, and no forbidden
    /// fact appears in the hits or the related list.
    #[test]
    fn ranking_metrics_meet_bar() {
        use crate::neighborhood::{eligible_neighbors, is_relationship_key, seeds_from_hits};
        use crate::retrieval_eval::evaluate_ranked;
        use crate::search::{FactSearchConfig, LexicalIndex, candidates, rank, rank_related};

        let facts = corpus();
        let cfg = FactSearchConfig::default();
        let index = LexicalIndex::build(&facts);
        for q in queries() {
            let lexical = index.score(q.query);
            let cosine: Vec<(usize, f32)> = facts
                .iter()
                .enumerate()
                .map(|(i, f)| (i, fake_cosine(q.query, f)))
                .collect();
            let pool = candidates(&facts, &lexical, &cosine, &cfg);
            let report = rank(&facts, &pool, None, None, q.limit, &cfg);
            let hits: Vec<Fact> = report.hits.iter().map(|h| h.fact.clone()).collect();

            let ranked: Vec<String> = hits.iter().map(|f| f.id.to_string()).collect();
            let relevant: HashSet<String> = q.relevant.iter().map(i64::to_string).collect();
            let metrics = evaluate_ranked(&ranked, &relevant, q.limit);
            assert_eq!(metrics.recall_at_k, 1.0, "{}: {ranked:?}", q.query);

            let seeds = seeds_from_hits(&hits, &[]);
            let cluster = hits.iter().any(|f| is_relationship_key(&f.key));
            let exclude: HashSet<i64> = hits.iter().map(|f| f.id).collect();
            let neighbors = eligible_neighbors(&facts, &seeds, cluster, &exclude);
            let scores: Vec<(usize, f32)> = neighbors
                .iter()
                .enumerate()
                .map(|(i, f)| (i, fake_cosine(q.query, f)))
                .collect();
            let related = rank_related(&neighbors, &scores, None, None, &cfg);
            assert!(related.len() <= cfg.related_max);
            for hit in report.hits.iter().chain(&related) {
                assert!(
                    !q.forbidden_related.contains(&hit.fact.id),
                    "{}: forbidden fact {}",
                    q.query,
                    hit.fact.id
                );
            }
        }
    }
}
