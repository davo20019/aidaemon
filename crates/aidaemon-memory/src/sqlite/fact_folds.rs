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
}
