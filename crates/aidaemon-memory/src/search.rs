//! Explicit fact search ranking, pure and database-free.
//!
//! A store gathers facts and scores (cosine, rerank) and hands them here as
//! plain data; every ranking decision lives in this module so any backend
//! gets the same order:
//!
//! - [`LexicalIndex`]: term-weighted (BM25-style idf) lexical scoring, with
//!   exact-key and full-phrase hits pinned first.

use std::collections::{HashMap, HashSet};

use crate::records::Fact;

/// Weight of a query term found in a fact's key, relative to one found in
/// its value or category.
const KEY_FIELD_WEIGHT: f32 = 3.0;

/// Lowercased word tokens of at least two characters, split on every
/// non-alphanumeric character, so `local_dev_db_port` yields
/// `["local", "dev", "db", "port"]` and `port` never matches `report`.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Lowercase, with `_`, `-` and runs of whitespace folded to one space.
fn fold_phrase(text: &str) -> String {
    text.to_lowercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

struct IndexedFact {
    key_tokens: HashSet<String>,
    other_tokens: HashSet<String>,
    folded_key: String,
    folded_value: String,
    updated_at: chrono::DateTime<chrono::Utc>,
    id: i64,
}

/// One fact's lexical match. `index` is the fact's position in the slice
/// given to [`LexicalIndex::build`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalHit {
    pub index: usize,
    pub score: f32,
    pub pinned: bool,
}

/// Term-weighted lexical index over one search's loaded facts.
pub struct LexicalIndex {
    facts: Vec<IndexedFact>,
    document_frequency: HashMap<String, usize>,
}

impl LexicalIndex {
    pub fn build(facts: &[Fact]) -> Self {
        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        let facts: Vec<IndexedFact> = facts
            .iter()
            .map(|fact| {
                let key_tokens: HashSet<String> = tokenize(&fact.key).into_iter().collect();
                let other_tokens: HashSet<String> = tokenize(&fact.value)
                    .into_iter()
                    .chain(tokenize(&fact.category))
                    .collect();
                for token in key_tokens.union(&other_tokens) {
                    *document_frequency.entry(token.clone()).or_default() += 1;
                }
                IndexedFact {
                    key_tokens,
                    other_tokens,
                    folded_key: fold_phrase(&fact.key),
                    folded_value: fold_phrase(&fact.value),
                    updated_at: fact.updated_at,
                    id: fact.id,
                }
            })
            .collect();
        Self {
            facts,
            document_frequency,
        }
    }

    fn idf(&self, token: &str) -> f32 {
        let n = self.facts.len() as f32;
        let df = self.document_frequency.get(token).copied().unwrap_or(0) as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Matching facts, best first: pinned hits, then score descending, then
    /// `updated_at` descending, then id ascending. Facts that share no query
    /// term and are not pinned are left out.
    pub fn score(&self, query: &str) -> Vec<LexicalHit> {
        let mut seen = HashSet::new();
        let terms: Vec<(String, f32)> = tokenize(query)
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .map(|t| {
                let idf = self.idf(&t);
                (t, idf)
            })
            .collect();
        let phrase = fold_phrase(query);
        let multi_word = phrase.split(' ').count() >= 2;

        let mut hits: Vec<LexicalHit> = self
            .facts
            .iter()
            .enumerate()
            .filter_map(|(index, fact)| {
                let score: f32 = terms
                    .iter()
                    .map(|(term, idf)| {
                        if fact.key_tokens.contains(term) {
                            idf * KEY_FIELD_WEIGHT
                        } else if fact.other_tokens.contains(term) {
                            *idf
                        } else {
                            0.0
                        }
                    })
                    .sum();
                // The one substring match allowed: the whole multi-word query.
                let pinned = !phrase.is_empty()
                    && (fact.folded_key == phrase
                        || multi_word
                            && (fact.folded_key.contains(&phrase)
                                || fact.folded_value.contains(&phrase)));
                (score > 0.0 || pinned).then_some(LexicalHit {
                    index,
                    score,
                    pinned,
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.pinned
                .cmp(&a.pinned)
                .then(b.score.total_cmp(&a.score))
                .then_with(|| self.tie_order(a.index, b.index))
        });
        hits
    }

    /// Tie order between two facts: newer first, then lower id.
    fn tie_order(&self, a: usize, b: usize) -> std::cmp::Ordering {
        let (a, b) = (&self.facts[a], &self.facts[b]);
        b.updated_at.cmp(&a.updated_at).then(a.id.cmp(&b.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_fixture::{self as fx, corpus};

    fn ids(facts: &[Fact], hits: &[LexicalHit]) -> Vec<i64> {
        hits.iter().map(|h| facts[h.index].id).collect()
    }

    #[test]
    fn rare_term_outranks_common_terms() {
        let facts = corpus();
        let hits = LexicalIndex::build(&facts).score("staging deploy port");
        assert!(hits.len() >= 61, "{} hits", hits.len());
        assert_eq!(facts[hits[0].index].id, fx::STAGING_DEPLOY_PORT);
    }

    #[test]
    fn ties_do_not_fall_back_to_category_order() {
        let facts = corpus();
        let hits = LexicalIndex::build(&facts).score("deploy staging");
        let flood: Vec<&LexicalHit> = hits
            .iter()
            .filter(|h| fx::TIE_FLOOD.contains(&facts[h.index].id))
            .collect();
        assert_eq!(flood.len(), 60);
        assert!(flood.windows(2).all(|w| w[0].score == w[1].score));
        assert!(flood
            .windows(2)
            .all(|w| facts[w[0].index].updated_at > facts[w[1].index].updated_at));
    }

    #[test]
    fn synonym_list_keeps_single_word_answer() {
        let facts = corpus();
        let hits = LexicalIndex::build(&facts).score("spouse partner wife");
        assert_eq!(ids(&facts, &hits).first(), Some(&fx::PARTNER_NAME));
    }

    #[test]
    fn word_boundary_not_substring() {
        let facts = vec![
            fx::fact(1, "general", "monthly_report", "sent on Fridays", 0),
            fx::fact(2, "general", "important_note", "support is closed", 0),
        ];
        assert!(LexicalIndex::build(&facts).score("port").is_empty());
    }

    #[test]
    fn exact_key_is_pinned() {
        let facts = corpus();
        let index = LexicalIndex::build(&facts);
        let hits = index.score("staging deploy port");
        assert!(hits[0].pinned);
        assert_eq!(facts[hits[0].index].id, fx::STAGING_DEPLOY_PORT);
        assert!(hits[1..].iter().all(|h| !h.pinned));

        let hits = index.score("partner name");
        assert!(hits[0].pinned);
        assert_eq!(facts[hits[0].index].id, fx::PARTNER_NAME);
    }

    #[test]
    fn single_word_is_never_a_phrase_match() {
        let facts = vec![fx::fact(1, "general", "notes", "the report is late", 0)];
        let hits = LexicalIndex::build(&facts).score("report");
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].pinned);
    }
}
