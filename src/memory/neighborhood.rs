//! Pure neighborhood selection over already-fetched flat facts.
//!
//! Given the entities a query resolved to, assemble their connected fact
//! clusters so the model can derive relational answers. Three rules:
//! namespace (`X:*`), co-mention (entity name appears in key/value), and the
//! owner-relationship cluster (relationship-typed facts travel together —
//! this is what connects `Conchi` to `Galo`, who never co-occur in one fact).
//! Pure and synchronous so it is unit-testable without a DB.

// Not yet called by production code — wired in Task 5.
#![allow(dead_code)]

use crate::traits::Fact;
use std::collections::HashSet;

/// Relationship key roots. General across relationship types, not spouse-specific.
const RELATIONSHIP_ROOTS: &[&str] = &[
    "mother",
    "father",
    "mom",
    "dad",
    "parent",
    "partner",
    "spouse",
    "wife",
    "husband",
    "child",
    "children",
    "son",
    "daughter",
    "kid",
    "sibling",
    "brother",
    "sister",
    "grandmother",
    "grandfather",
];

#[derive(Debug, Clone, Copy)]
pub struct NeighborhoodCaps {
    pub max_entities: usize,
    pub max_facts: usize,
}

impl Default for NeighborhoodCaps {
    fn default() -> Self {
        Self {
            max_entities: 6,
            max_facts: 16,
        }
    }
}

/// The namespace prefix of a key (`"X:attr" -> "X"`), or `None` for flat keys.
pub fn fact_namespace(key: &str) -> Option<&str> {
    key.split_once(':')
        .map(|(ns, _)| ns)
        .filter(|ns| !ns.is_empty())
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
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
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
    let names: Vec<String> = resolved_names
        .iter()
        .take(caps.max_entities)
        .cloned()
        .collect();

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
    picked.sort_by_key(|f| std::cmp::Reverse(salience(f)));
    picked.truncate(caps.max_facts);
    picked
}

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
            privacy: crate::types::FactPrivacy::Global,
            first_seen_at: None,
            source_excerpt: None,
        }
    }

    #[test]
    fn namespace_is_prefix_before_colon() {
        assert_eq!(
            fact_namespace("LearnEnglishSounds:path"),
            Some("LearnEnglishSounds")
        );
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
        assert!(
            ids.contains(&2),
            "father=Galo must be pulled into the cluster"
        );
        assert!(
            ids.contains(&3),
            "partner is part of the owner relationship set"
        );
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
