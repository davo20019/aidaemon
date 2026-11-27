//! Which active facts carry the same knowledge, and which copy keeps it.
//!
//! Two rules, both pure so the maintenance pass and `db_probe`'s dry run
//! agree on every pair:
//!
//! - A restatement ([`facts_are_restatements`]): one category, near-identical
//!   text that makes one claim.
//! - One value ([`facts_hold_one_value`]): the same normalized value under
//!   reworded keys, in any category. Extractors file one fact under
//!   `project` and `technical` alike, and reword its key every time.
//!
//! Similarity alone never decides: above 0.93 there are pairs with different
//! values (two expiry dates), and between 0.85 and 0.93 duplicates mix with
//! distinct facts. Both rules need the values to agree first.

use super::facts::{
    canonicalize_fact_key, fact_category_is_personal, facts_are_restatements,
    keys_name_distinct_concepts,
};

/// Full-fact similarity at which a value of two or more words, identical
/// under another key, is the same fact. Measured on a live store of 4,747
/// facts: from here up pairs are reworded keys for one value; below it
/// different things start to share a value (two sites on one platform).
pub const ONE_VALUE_MIN_SIMILARITY: f32 = 0.80;

/// The same for a one-word value (`ddev`, `Rust`), which also needs a key
/// word in common: one word is shared by many unrelated attributes.
pub const ONE_WORD_VALUE_MIN_SIMILARITY: f32 = 0.85;

/// A value that says nothing on its own: a boolean, a bare number, a null.
fn is_bare_scalar(normalized: &str) -> bool {
    matches!(
        normalized,
        "true" | "false" | "yes" | "no" | "none" | "null" | "n a" | "unknown"
    ) || normalized
        .split(' ')
        .all(|word| word.chars().all(|c| c.is_ascii_digit()))
}

/// Whether two active facts hold one piece of knowledge under different
/// keys: the same normalized value, similar enough as whole facts, and, when
/// either is about a person, keys naming the same attribute (`pet_name` and
/// `daughter_name` are both `Mia`, `birth_place` and `hometown` one city).
pub fn facts_hold_one_value(
    (category_a, key_a, value_a): (&str, &str, &str),
    (category_b, key_b, value_b): (&str, &str, &str),
    similarity: f32,
) -> bool {
    let value = crate::text::normalize_fact_value(value_a);
    if value.is_empty() || value != crate::text::normalize_fact_value(value_b) {
        return false;
    }
    if is_bare_scalar(&value) {
        return false;
    }
    let canonical_a = canonicalize_fact_key(key_a);
    let canonical_b = canonicalize_fact_key(key_b);
    let tokens_a: std::collections::HashSet<&str> =
        canonical_a.split('_').filter(|t| !t.is_empty()).collect();
    let tokens_b: std::collections::HashSet<&str> =
        canonical_b.split('_').filter(|t| !t.is_empty()).collect();
    let share_a_word = tokens_a.intersection(&tokens_b).next().is_some();
    let threshold = if value.contains(' ') {
        ONE_VALUE_MIN_SIMILARITY
    } else {
        if !share_a_word {
            return false;
        }
        ONE_WORD_VALUE_MIN_SIMILARITY
    };
    if similarity < threshold {
        return false;
    }
    if fact_category_is_personal(category_a) || fact_category_is_personal(category_b) {
        return canonical_a == canonical_b
            || share_a_word && !keys_name_distinct_concepts(&tokens_a, &tokens_b);
    }
    true
}

/// An active fact as the fold planner sees it.
#[derive(Debug, Clone)]
pub struct FoldCandidate {
    pub id: i64,
    pub category: String,
    pub key: String,
    pub value: String,
    /// [`super::facts::fact_source_rank`]: stated by the owner (2) is never
    /// folded away; a closer source is kept over a farther one.
    pub source_rank: u8,
    /// Pinned in owner DMs (`user`, `preference`...): kept over a copy in an
    /// unpinned category, so folding never unpins the knowledge.
    pub pinned: bool,
    pub recall_count: i64,
    pub privacy: String,
    pub channel_id: Option<String>,
    pub vector: Vec<f32>,
}

/// Which rule found a fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldRule {
    Restatement,
    OneValue,
}

/// One fact to retire into the fact that keeps its knowledge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannedFold {
    pub folded: i64,
    pub kept: i64,
    pub rule: FoldRule,
    pub similarity: f32,
}

/// Plan every fold in `candidates`.
///
/// Star clustering in authority order: closest source (owner-stated, then
/// quoted, then inferred), then pinned, then most recalled, then oldest (its key is the one in use). Each fact folds into
/// the first kept fact it matches; no chaining, so a cluster never drifts
/// from its representative. Facts fold only within one scope (privacy and
/// channel): folding a channel fact into a private one hides it, the
/// reverse widens it. Vectors of different lengths (another embedding model)
/// never compare.
pub fn plan_folds(candidates: &mut [FoldCandidate]) -> Vec<PlannedFold> {
    candidates.sort_by(|a, b| {
        b.source_rank
            .cmp(&a.source_rank)
            .then(b.pinned.cmp(&a.pinned))
            .then(b.recall_count.cmp(&a.recall_count))
            .then(a.id.cmp(&b.id))
    });
    // Candidates are compared only within a bucket that can match: a
    // restatement needs the category, one value needs the value.
    let mut kept_by_category: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    let mut kept_by_value: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    let mut folds = Vec::new();
    for index in 0..candidates.len() {
        let candidate = &candidates[index];
        let category = candidate.category.trim().to_ascii_lowercase();
        let value = crate::text::normalize_fact_value(&candidate.value);
        let same_scope = |other: &FoldCandidate| {
            other.privacy == candidate.privacy && other.channel_id == candidate.channel_id
        };
        let restated = kept_by_category
            .get(&category)
            .into_iter()
            .flatten()
            .find_map(|&kept| {
                let rep = &candidates[kept];
                if !same_scope(rep) {
                    return None;
                }
                let similarity = crate::cosine(&candidate.vector, &rep.vector);
                facts_are_restatements(
                    &candidate.category,
                    (&candidate.key, &candidate.value),
                    (&rep.key, &rep.value),
                    similarity,
                )
                .then_some((kept, FoldRule::Restatement, similarity))
            });
        let matched = restated.or_else(|| {
            kept_by_value
                .get(&value)
                .into_iter()
                .flatten()
                .find_map(|&kept| {
                    let rep = &candidates[kept];
                    if !same_scope(rep) {
                        return None;
                    }
                    let similarity = crate::cosine(&candidate.vector, &rep.vector);
                    facts_hold_one_value(
                        (&candidate.category, &candidate.key, &candidate.value),
                        (&rep.category, &rep.key, &rep.value),
                        similarity,
                    )
                    .then_some((kept, FoldRule::OneValue, similarity))
                })
        });
        match matched {
            Some((kept, rule, similarity)) if candidate.source_rank < 2 => {
                folds.push(PlannedFold {
                    folded: candidate.id,
                    kept: candidates[kept].id,
                    rule,
                    similarity,
                });
            }
            _ => {
                kept_by_category.entry(category).or_default().push(index);
                if !value.is_empty() {
                    kept_by_value.entry(value).or_default().push(index);
                }
            }
        }
    }
    folds
}

/// Minimum full-fact similarity to flag two active facts as a probable
/// conceptual duplicate/overlap for UI review when they did not auto-fold
/// (e.g. different value shapes, different ranks, or cross-scope owner DMs).
pub const OVERLAP_REVIEW_MIN_SIMILARITY: f32 = 0.78;

/// Identify active facts that look like conceptual duplicates or overlaps of
/// another active fact so the Memory UI can highlight them for review without
/// mutating or losing distinct constraints automatically.
///
/// Returns `lower_priority_id -> representative_id` (authority order: highest
/// `source_rank`, then pinned, then most recalled, then oldest id).
pub fn detect_overlaps(candidates: &mut [FoldCandidate]) -> std::collections::HashMap<i64, i64> {
    use std::collections::{HashMap, HashSet};

    candidates.sort_by(|a, b| {
        b.source_rank
            .cmp(&a.source_rank)
            .then(b.pinned.cmp(&a.pinned))
            .then(b.recall_count.cmp(&a.recall_count))
            .then(a.id.cmp(&b.id))
    });

    // Every fact's normalized forms are computed once: the Memory list runs
    // this over thousands of facts, and normalizing both sides on every
    // comparison made it quadratic in string work (tens of seconds).
    let n = candidates.len();
    let cats: Vec<String> = candidates
        .iter()
        .map(|c| c.category.trim().to_ascii_lowercase())
        .collect();
    let keys: Vec<String> = candidates
        .iter()
        .map(|c| canonicalize_fact_key(&c.key))
        .collect();
    let vals: Vec<String> = candidates
        .iter()
        .map(|c| crate::text::normalize_fact_value(&c.value))
        .collect();
    let words: Vec<HashSet<&str>> = vals
        .iter()
        .map(|v| v.split_whitespace().filter(|w| w.len() >= 3).collect())
        .collect();
    // Positive owner naming facet per fact (`None` when not an owner-name fact).
    let naming: Vec<Option<String>> = candidates
        .iter()
        .zip(&cats)
        .map(|(c, cat)| {
            (crate::text::is_owner_name_key(&c.key)
                && matches!(cat.as_str(), "user" | "preference" | "personal" | "profile"))
            .then(|| crate::text::canonical_personal_key(&c.key))
        })
        .collect();
    let norms: Vec<f32> = candidates
        .iter()
        .map(|c| c.vector.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    // `crate::cosine` with both norms precomputed.
    let similarity = |a: usize, b: usize| -> f32 {
        let (va, vb) = (&candidates[a].vector, &candidates[b].vector);
        if va.len() != vb.len() || va.is_empty() {
            return 0.0;
        }
        if norms[a] * norms[a] <= f32::EPSILON || norms[b] * norms[b] <= f32::EPSILON {
            return 0.0;
        }
        let dot: f32 = va.iter().zip(vb).map(|(x, y)| x * y).sum();
        dot / (norms[a] * norms[b])
    };
    let is_positive_naming = |facet: &str| {
        matches!(
            facet,
            "name" | "full_name" | "preferred_name" | "nickname" | "addressing"
        )
    };

    // Representatives, indexed by what each rule matches on. Every list is
    // in push order (ascending candidate index), so the first match found in
    // a list is that rule's earliest representative.
    let mut naming_reps: Vec<usize> = Vec::new();
    let mut reps_by_key: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut reps_by_val: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut reps_by_cat: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut overlaps = HashMap::new();

    for index in 0..n {
        // The earliest representative any rule matches, as the original
        // single scan over all representatives in order found it.
        let mut best: Option<usize> = None;
        let offer = |rep: usize, best: &mut Option<usize>| {
            if best.is_none_or(|b| rep < b) {
                *best = Some(rep);
            }
        };

        // 1. Two owner-name / addressing facts (e.g. `user/preferred_name = Alice`
        // and `preference/address_as = Call the owner Alice, not Al.`) are typed
        // overlaps when they map to the same canonical naming facet or both
        // express positive owner naming/addressing (keeping `rejected_alias`
        // distinct as a separate constraint).
        if let Some(cand_facet) = &naming[index]
            && let Some(&rep) = naming_reps.iter().find(|&&rep| {
                let rep_facet = naming[rep].as_deref().unwrap_or_default();
                cand_facet == rep_facet
                    || (is_positive_naming(cand_facet) && is_positive_naming(rep_facet))
            })
        {
            offer(rep, &mut best);
        }

        // 2. Same canonical key across different categories (e.g.
        // `user/preferred_name` beside `preference/preferred_name`).
        if !keys[index].is_empty()
            && let Some(&rep) = reps_by_key
                .get(keys[index].as_str())
                .and_then(|list| list.iter().find(|&&rep| cats[rep] != cats[index]))
        {
            offer(rep, &mut best);
        }

        // 3. Same non-scalar normalized value across different keys or
        // scopes (cross-writer owner-DM duplicates that could not auto-fold).
        if !vals[index].is_empty()
            && !is_bare_scalar(&vals[index])
            && let Some(&rep) = reps_by_val.get(vals[index].as_str()).and_then(|list| {
                list.iter().find(|&&rep| {
                    candidates[index].vector.is_empty()
                        || candidates[rep].vector.is_empty()
                        || similarity(index, rep) >= ONE_VALUE_MIN_SIMILARITY
                })
            })
        {
            offer(rep, &mut best);
        }

        // 4. High semantic similarity within the same category where
        // values share meaningful vocabulary (catches paraphrased or
        // partially overlapping rules/preferences). Only representatives
        // earlier than a match already found can change the answer.
        if !candidates[index].vector.is_empty()
            && !words[index].is_empty()
            && let Some(list) = reps_by_cat.get(cats[index].as_str())
        {
            let personal = fact_category_is_personal(&cats[index]);
            for &rep in list {
                if best.is_some_and(|b| rep >= b) {
                    break;
                }
                if candidates[rep].vector.is_empty()
                    || words[index].is_disjoint(&words[rep])
                    || similarity(index, rep) < OVERLAP_REVIEW_MIN_SIMILARITY
                {
                    continue;
                }
                if personal {
                    let cand_tokens: HashSet<&str> =
                        keys[index].split('_').filter(|t| !t.is_empty()).collect();
                    let rep_tokens: HashSet<&str> =
                        keys[rep].split('_').filter(|t| !t.is_empty()).collect();
                    if keys_name_distinct_concepts(&cand_tokens, &rep_tokens) {
                        continue;
                    }
                }
                offer(rep, &mut best);
                break;
            }
        }

        if let Some(rep) = best {
            overlaps.insert(candidates[index].id, candidates[rep].id);
        } else {
            if naming[index].is_some() {
                naming_reps.push(index);
            }
            reps_by_key
                .entry(keys[index].as_str())
                .or_default()
                .push(index);
            reps_by_val
                .entry(vals[index].as_str())
                .or_default()
                .push(index);
            reps_by_cat
                .entry(cats[index].as_str())
                .or_default()
                .push(index);
        }
    }
    overlaps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_value_under_reworded_keys_is_one_fact() {
        let a = ("project", "local_sites_directory", "~/projects/");
        assert!(facts_hold_one_value(
            a,
            ("technical", "projects_directory", "~/projects"),
            0.86
        ));
        assert!(facts_hold_one_value(
            ("technical", "local_dev_tool", "ddev"),
            ("project", "local_dev_tooling", "DDEV"),
            0.878
        ));
    }

    #[test]
    fn a_shared_value_alone_is_not_one_fact() {
        // Too dissimilar as whole facts.
        assert!(!facts_hold_one_value(
            ("project", "cars_site_platform", "Cloudflare Workers"),
            ("project", "store_deployment_target", "Cloudflare Workers"),
            0.75
        ));
        // One word, no key word in common.
        assert!(!facts_hold_one_value(
            ("project", "development_platform", "Drupal"),
            ("project", "framework", "Drupal"),
            0.9
        ));
        // Booleans and numbers say nothing on their own.
        assert!(!facts_hold_one_value(
            ("project", "tests_enabled", "true"),
            ("project", "tests_passing", "true"),
            0.99
        ));
        // Values differ.
        assert!(!facts_hold_one_value(
            ("project", "version", "0.9.10"),
            ("project", "current_version", "0.9.17"),
            0.95
        ));
    }

    #[test]
    fn people_keys_must_name_the_same_attribute() {
        assert!(!facts_hold_one_value(
            ("user", "pet_name", "Mia"),
            ("user", "daughter_name", "Mia"),
            0.9
        ));
        assert!(!facts_hold_one_value(
            ("user", "birth_place", "Quito, Ecuador"),
            ("user", "hometown", "Quito, Ecuador"),
            0.9
        ));
        assert!(facts_hold_one_value(
            ("user", "favorite_food", "sushi"),
            ("user", "food", "sushi"),
            0.9
        ));
    }

    fn candidate(id: i64, category: &str, key: &str, value: &str) -> FoldCandidate {
        FoldCandidate {
            id,
            category: category.into(),
            key: key.into(),
            value: value.into(),
            source_rank: 0,
            pinned: false,
            recall_count: 0,
            privacy: "channel".into(),
            channel_id: None,
            vector: vec![1.0, 0.0],
        }
    }

    #[test]
    fn the_owner_copy_then_the_pinned_one_survives() {
        let mut candidates = vec![
            candidate(1, "project", "editor_choice", "Helix editor"),
            FoldCandidate {
                pinned: true,
                ..candidate(2, "preference", "preferred_editor", "Helix editor")
            },
            FoldCandidate {
                source_rank: 2,
                ..candidate(3, "technical", "editor", "helix editor")
            },
        ];
        let folds = plan_folds(&mut candidates);
        assert_eq!(folds.len(), 2);
        assert!(folds.iter().all(|fold| fold.kept == 3));

        let mut candidates = vec![
            FoldCandidate {
                recall_count: 50,
                ..candidate(1, "project", "editor_choice", "Helix editor")
            },
            FoldCandidate {
                pinned: true,
                ..candidate(2, "preference", "preferred_editor", "Helix editor")
            },
        ];
        assert_eq!(plan_folds(&mut candidates)[0].kept, 2);
    }

    #[test]
    fn scope_and_owner_statements_are_never_folded_away() {
        let mut candidates = vec![
            candidate(1, "project", "sites_dir", "~/projects"),
            FoldCandidate {
                privacy: "private".into(),
                ..candidate(2, "technical", "sites_directory", "~/projects")
            },
            FoldCandidate {
                channel_id: Some("slack:synthetic-team".into()),
                ..candidate(3, "technical", "sites_directory", "~/projects")
            },
        ];
        assert!(plan_folds(&mut candidates).is_empty());

        let mut candidates = vec![
            FoldCandidate {
                source_rank: 2,
                ..candidate(1, "project", "sites_dir", "~/projects")
            },
            FoldCandidate {
                source_rank: 2,
                ..candidate(2, "technical", "sites_directory", "~/projects")
            },
        ];
        assert!(plan_folds(&mut candidates).is_empty());
    }

    #[test]
    fn other_embedding_models_never_compare() {
        let mut candidates = vec![
            candidate(1, "project", "sites_dir", "~/projects"),
            FoldCandidate {
                vector: vec![1.0, 0.0, 0.0],
                ..candidate(2, "technical", "sites_directory", "~/projects")
            },
        ];
        assert!(plan_folds(&mut candidates).is_empty());
    }

    #[test]
    fn detect_overlaps_flags_owner_name_and_addressing_duplicates() {
        let mut candidates = vec![
            FoldCandidate {
                source_rank: 2,
                pinned: true,
                ..candidate(-2_000_000_001, "user", "preferred_name", "Alice")
            },
            FoldCandidate {
                source_rank: 0,
                pinned: false,
                ..candidate(
                    42,
                    "preference",
                    "address_as",
                    "Call the owner Alice. Don't call her Ally or Al.",
                )
            },
            FoldCandidate {
                source_rank: 2,
                pinned: false,
                vector: vec![0.0, 1.0],
                ..candidate(43, "preference", "favorite_dish", "ramen")
            },
        ];
        let overlaps = detect_overlaps(&mut candidates);
        assert_eq!(overlaps.get(&42), Some(&-2_000_000_001));
        assert!(!overlaps.contains_key(&43));
    }
}
