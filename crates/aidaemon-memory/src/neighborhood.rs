//! Pure expansion over already-loaded flat facts.
//!
//! Given the hits a search actually returned, decide which other facts may
//! travel with them. Rules are typed by seed kind:
//!
//! - a namespace seed (`X:attr` keys) makes eligible only facts in the same
//!   normalized namespace, never facts that merely mention the word;
//! - a person seed (the value of a relationship-keyed hit, or a known
//!   person's name in a hit) makes eligible facts that mention that name as
//!   whole words;
//! - a returned relationship-keyed hit makes the owner's relationship
//!   cluster eligible, which is what connects a mother to a father who never
//!   co-occur in one fact.
//!
//! Eligibility only: [`crate::search::rank_related`] scores the neighbors
//! against the query and caps them.

use crate::records::Fact;
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
            || lower.starts_with(&format!("{root}_")) // e.g. "mother_name", "partner_name"
            || lower.ends_with(&format!("_{root}")) // e.g. "step_mother", "half_brother"
    })
}

/// Tokenise a string for word-boundary matching.
///
/// Underscores and hyphens in fact KEYS (e.g. `"partner_name"`) act as word
/// separators just like spaces, so `"partner"` is a whole word inside
/// `"partner_name"`.  All tokens are lowercased.
fn word_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// True if every significant word of `entity` appears as a whole word in
/// `haystack` (case-insensitive, `_`/`-` treated as word boundaries).
///
/// "Significant" words are those with at least 2 characters, so short
/// connectives don't produce spurious matches.
pub fn entity_mentioned_as_words(haystack: &str, entity: &str) -> bool {
    let haystack_tokens: HashSet<String> = word_tokens(haystack).into_iter().collect();
    let entity_words: Vec<String> = word_tokens(entity);
    // Require at least one meaningful word to match.
    let significant: Vec<&String> = entity_words.iter().filter(|w| w.len() >= 2).collect();
    if significant.is_empty() {
        return false;
    }
    significant.iter().all(|w| haystack_tokens.contains(*w))
}

/// One namespace however it is spelled: lowercase with every
/// non-alphanumeric character removed (`Acme.Blog`, `acme_blog` and
/// `acmeblog` are one namespace).
pub fn normalize_namespace(ns: &str) -> String {
    ns.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Where an expansion starts from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seed {
    /// A normalized namespace.
    Namespace(String),
    /// A person's name.
    Person(String),
}

/// Most seeds one search expands from.
const MAX_SEEDS: usize = 6;

/// Seeds from the hits actually returned, in hit order, at most six.
/// `known_people` are names and aliases from the people store; pass an
/// empty slice when there is none.
pub fn seeds_from_hits(hits: &[Fact], known_people: &[String]) -> Vec<Seed> {
    let mut seeds: Vec<Seed> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut push = |seed: Seed, seeds: &mut Vec<Seed>| {
        let identity = match &seed {
            Seed::Namespace(ns) => format!("ns:{ns}"),
            Seed::Person(name) => format!("person:{}", name.to_lowercase()),
        };
        if seen.insert(identity) {
            seeds.push(seed);
        }
    };
    for hit in hits {
        if let Some(ns) = fact_namespace(&hit.key) {
            let normalized = normalize_namespace(ns);
            if !normalized.is_empty() {
                push(Seed::Namespace(normalized), &mut seeds);
            }
        }
        if is_relationship_key(&hit.key) {
            let name = hit.value.trim();
            if !name.is_empty() {
                push(Seed::Person(name.to_string()), &mut seeds);
            }
        }
        for person in known_people {
            if entity_mentioned_as_words(&hit.key, person)
                || entity_mentioned_as_words(&hit.value, person)
            {
                push(Seed::Person(person.clone()), &mut seeds);
            }
        }
    }
    seeds.truncate(MAX_SEEDS);
    seeds
}

/// Facts eligible for expansion, excluding `exclude_ids`; unranked and
/// uncapped. `relationship_cluster` makes relationship-keyed facts eligible
/// and should be true only when a returned hit has a relationship key.
pub fn eligible_neighbors<'a>(
    all: &'a [Fact],
    seeds: &[Seed],
    relationship_cluster: bool,
    exclude_ids: &HashSet<i64>,
) -> Vec<&'a Fact> {
    let namespaces: HashSet<&str> = seeds
        .iter()
        .filter_map(|seed| match seed {
            Seed::Namespace(ns) => Some(ns.as_str()),
            Seed::Person(_) => None,
        })
        .collect();
    let people: Vec<&str> = seeds
        .iter()
        .filter_map(|seed| match seed {
            Seed::Person(name) => Some(name.as_str()),
            Seed::Namespace(_) => None,
        })
        .collect();
    let mut seen: HashSet<i64> = HashSet::new();
    all.iter()
        .filter(|f| !exclude_ids.contains(&f.id))
        .filter(|f| {
            let in_namespace = fact_namespace(&f.key)
                .is_some_and(|ns| namespaces.contains(normalize_namespace(ns).as_str()));
            let mentions_person = people.iter().any(|name| {
                entity_mentioned_as_words(&f.key, name) || entity_mentioned_as_words(&f.value, name)
            });
            let in_cluster = relationship_cluster && is_relationship_key(&f.key);
            in_namespace || mentions_person || in_cluster
        })
        .filter(|f| seen.insert(f.id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_fixture::{self as fx, corpus};

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
            privacy: crate::records::FactPrivacy::Global,
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

    fn ids(facts: &[&Fact]) -> HashSet<i64> {
        facts.iter().map(|f| f.id).collect()
    }

    fn by_id(all: &[Fact], id: i64) -> Fact {
        all.iter().find(|f| f.id == id).unwrap().clone()
    }

    #[test]
    fn generic_namespace_does_not_co_mention() {
        let all = corpus();
        let hit = by_id(&all, fx::WORKFLOW_RELEASE_CHECKLIST);
        let seeds = seeds_from_hits(std::slice::from_ref(&hit), &[]);
        assert_eq!(seeds, vec![Seed::Namespace("workflow".into())]);
        let exclude: HashSet<i64> = [hit.id].into_iter().collect();
        let out = ids(&eligible_neighbors(&all, &seeds, false, &exclude));
        assert_eq!(out.len(), 29);
        assert!(out.iter().all(|id| fx::WORKFLOW_NS.contains(id)));
        assert!(!out.iter().any(|id| fx::GENERIC_MENTIONS.contains(id)));
    }

    #[test]
    fn namespace_variants_are_one_namespace() {
        let all = corpus();
        let hit = by_id(&all, 4000); // Acme.Blog:
        let seeds = seeds_from_hits(std::slice::from_ref(&hit), &[]);
        let exclude: HashSet<i64> = [hit.id].into_iter().collect();
        let out = ids(&eligible_neighbors(&all, &seeds, false, &exclude));
        for id in (4001..4005).chain(4010..4015).chain(4020..4025) {
            assert!(out.contains(&id), "{id} missing: {out:?}");
        }
        assert_eq!(out.len(), 14);
    }

    #[test]
    fn seeds_only_from_returned_hits() {
        let hits = vec![
            fact(1, "user", "birth_date", "1990-04-12"),
            fact(2, "preference", "reply_style", "short"),
        ];
        assert!(seeds_from_hits(&hits, &[]).is_empty());
    }

    #[test]
    fn family_cluster_still_travels() {
        let all = corpus();
        let hit = by_id(&all, fx::MOTHER_NAME);
        let seeds = seeds_from_hits(std::slice::from_ref(&hit), &[]);
        assert_eq!(seeds, vec![Seed::Person("Maria Rivera".into())]);
        let exclude: HashSet<i64> = [hit.id].into_iter().collect();
        let out = ids(&eligible_neighbors(&all, &seeds, true, &exclude));
        assert!(out.contains(&fx::FATHER_NAME));
        assert!(out.contains(&fx::PARTNER_NAME));
        assert!(!out.iter().any(|id| fx::HOT_CLUSTER.contains(id)));
    }

    #[test]
    fn known_person_in_a_hit_is_a_seed() {
        let hits = vec![fact(1, "general", "gift_idea", "a kettle for Jordan Lee")];
        let seeds = seeds_from_hits(&hits, &["Jordan Lee".into(), "Ana".into()]);
        assert_eq!(seeds, vec![Seed::Person("Jordan Lee".into())]);
    }

    #[test]
    fn seeds_are_capped() {
        let hits: Vec<Fact> = (0..10)
            .map(|i| fact(i, "project", &format!("ns{i}:attr"), "x"))
            .collect();
        assert_eq!(seeds_from_hits(&hits, &[]).len(), 6);
    }

    #[test]
    fn co_mention_is_word_boundary_not_substring() {
        let all = vec![
            fact(1, "food", "preference", "I like banana bread"),
            fact(2, "user", "father", "Frank Mendez"),
        ];
        let none: HashSet<i64> = HashSet::new();
        let ana = eligible_neighbors(&all, &[Seed::Person("Ana".into())], false, &none);
        assert!(ana.is_empty(), "'Ana' must not match 'banana'");
        let frank = eligible_neighbors(&all, &[Seed::Person("Frank".into())], false, &none);
        assert_eq!(ids(&frank), [2].into_iter().collect());
    }

    #[test]
    fn known_person_match_is_word_boundary() {
        let hits = vec![fact(1, "user", "best_friend_note", "Banana Doe likes tea")];
        assert!(seeds_from_hits(&hits, &["Ana".into()]).is_empty());
    }
}
