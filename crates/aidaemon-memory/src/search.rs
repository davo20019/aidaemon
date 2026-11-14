//! Explicit fact search ranking, pure and database-free.
//!
//! A store gathers facts and cosine scores and hands them here as
//! plain data; every ranking decision lives in this module so any backend
//! gets the same order:
//!
//! - [`LexicalIndex`]: term-weighted (BM25-style idf) lexical scoring, with
//!   exact-key and full-phrase hits pinned first.
//! - [`candidates`] and [`rank`]: lexical and semantic nominations become one
//!   pool, ordered by reciprocal rank fusion, which needs no score
//!   calibration and so works with any embedder.
//! - [`rank_related`]: expansion candidates scored against the same query,
//!   capped and returned apart from the main hits.

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

/// Most facts pinned by containing the whole query phrase. A phrase that
/// many facts mention would otherwise pin them all ahead of every better
/// answer. Exact-key pins are not capped.
const MAX_PHRASE_PINS: usize = 5;

/// Every lowercased alphanumeric word, in order: the unit a phrase match
/// compares, so `port 80` never matches inside `report 8080`.
fn word_sequence(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `phrase` occurs in `words` as consecutive whole words.
fn contains_word_run(words: &[String], phrase: &[String]) -> bool {
    !phrase.is_empty() && words.windows(phrase.len()).any(|window| window == phrase)
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
    key_words: Vec<String>,
    value_words: Vec<String>,
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
                    key_words: word_sequence(&fact.key),
                    value_words: word_sequence(&fact.value),
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
        let phrase_words = word_sequence(query);
        let multi_word = phrase_words.len() >= 2;

        let mut matches: Vec<(LexicalHit, bool)> = self
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
                let exact_key = !phrase.is_empty() && fact.folded_key == phrase;
                // The whole multi-word query as consecutive words.
                let phrase_pin = !exact_key
                    && multi_word
                    && (contains_word_run(&fact.key_words, &phrase_words)
                        || contains_word_run(&fact.value_words, &phrase_words));
                (score > 0.0 || exact_key || phrase_pin).then_some((
                    LexicalHit {
                        index,
                        score,
                        pinned: exact_key || phrase_pin,
                    },
                    exact_key,
                ))
            })
            .collect();
        // Keep the best phrase pins; the rest compete on score like any hit.
        matches.sort_by(|(a, _), (b, _)| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| self.tie_order(a.index, b.index))
        });
        let mut phrase_pins = 0usize;
        for (hit, exact_key) in &mut matches {
            if hit.pinned && !*exact_key {
                phrase_pins += 1;
                if phrase_pins > MAX_PHRASE_PINS {
                    hit.pinned = false;
                }
            }
        }
        let mut hits: Vec<LexicalHit> = matches
            .into_iter()
            .map(|(hit, _)| hit)
            .filter(|hit| hit.pinned || hit.score > 0.0)
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

/// Tunables for [`crate::stores::FactStore::search_facts`].
#[derive(Debug, Clone, PartialEq)]
pub struct FactSearchConfig {
    /// Unpinned lexical hits nominated to the pool.
    pub lexical_pool: usize,
    /// Semantic hits nominated to the pool.
    pub semantic_pool: usize,
    /// Cosine at or below which a semantic hit is not nominated.
    pub semantic_min_cosine: f32,
    /// Reciprocal rank fusion constant.
    pub rrf_k: f32,
    /// Most related facts returned beside the hits.
    pub related_max: usize,
    /// Most expansion-eligible facts scored against the query.
    pub related_pool: usize,
}

impl Default for FactSearchConfig {
    fn default() -> Self {
        Self {
            lexical_pool: 25,
            semantic_pool: 25,
            semantic_min_cosine: 0.22,
            rrf_k: 60.0,
            related_max: 5,
            related_pool: 30,
        }
    }
}

/// Why a fact is in the results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    /// Exact key or full-phrase match; always returned, always first.
    Exact,
    Lexical,
    Semantic,
    Both,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub fact: Fact,
    /// Reciprocal rank fusion score.
    pub score: f32,
    pub kind: MatchKind,
}

/// One explicit search: the ranked hits, related facts, and honest counts.
#[derive(Debug, Clone, Default)]
pub struct FactSearchReport {
    /// At most `limit`.
    pub hits: Vec<SearchHit>,
    /// At most `related_max`, none of them in `hits`.
    pub related: Vec<SearchHit>,
    /// Size of the candidate union.
    pub considered: usize,
}

/// One pooled candidate. Ranks start at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Position in the searched facts.
    pub index: usize,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub pinned: bool,
}

/// The candidate union in a deterministic order (lexical order, then
/// semantic-only in semantic order).
///
/// `cosine` holds `(fact index, cosine)`. Pinned lexical hits are always
/// kept. Duplicate facts (same id, or same category, key and value, as a
/// canonical projection beside its raw row) collapse into one candidate that
/// keeps the positive id.
pub fn candidates(
    facts: &[Fact],
    lexical: &[LexicalHit],
    cosine: &[(usize, f32)],
    cfg: &FactSearchConfig,
) -> Vec<Candidate> {
    let mut nominated: Vec<Candidate> = Vec::new();
    let mut unpinned = 0usize;
    for (rank, hit) in lexical.iter().enumerate() {
        if !hit.pinned {
            if unpinned >= cfg.lexical_pool {
                continue;
            }
            unpinned += 1;
        }
        nominated.push(Candidate {
            index: hit.index,
            lexical_rank: Some(rank + 1),
            semantic_rank: None,
            pinned: hit.pinned,
        });
    }

    let mut semantic: Vec<(usize, f32)> = cosine
        .iter()
        .copied()
        .filter(|(_, score)| *score > cfg.semantic_min_cosine)
        .collect();
    semantic.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    semantic.truncate(cfg.semantic_pool);
    for (rank, (index, _)) in semantic.into_iter().enumerate() {
        nominated.push(Candidate {
            index,
            lexical_rank: None,
            semantic_rank: Some(rank + 1),
            pinned: false,
        });
    }

    // Collapse duplicates, keeping each identity's first position.
    type Identity<'a> = (&'a str, &'a str, &'a str);
    let mut by_id: HashMap<i64, usize> = HashMap::new();
    let mut by_content: HashMap<Identity, usize> = HashMap::new();
    let mut pool: Vec<Candidate> = Vec::new();
    for candidate in nominated {
        let fact = &facts[candidate.index];
        let content = (
            fact.category.as_str(),
            fact.key.as_str(),
            fact.value.as_str(),
        );
        let existing = by_id
            .get(&fact.id)
            .or_else(|| by_content.get(&content))
            .copied();
        let slot = match existing {
            Some(slot) => {
                let kept = &mut pool[slot];
                kept.lexical_rank = min_rank(kept.lexical_rank, candidate.lexical_rank);
                kept.semantic_rank = min_rank(kept.semantic_rank, candidate.semantic_rank);
                kept.pinned |= candidate.pinned;
                if facts[kept.index].id <= 0 && fact.id > 0 {
                    kept.index = candidate.index;
                }
                slot
            }
            None => {
                pool.push(candidate);
                pool.len() - 1
            }
        };
        by_id.insert(fact.id, slot);
        by_content.insert(content, slot);
    }
    pool
}

fn min_rank(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// The final order, by reciprocal rank fusion: pinned candidates first,
/// then by score descending, then newest, then lowest id.
pub fn rank(
    facts: &[Fact],
    candidates: &[Candidate],
    limit: usize,
    cfg: &FactSearchConfig,
) -> FactSearchReport {
    let mut scored: Vec<(&Candidate, f32)> = candidates
        .iter()
        .map(|candidate| (candidate, rrf_score(candidate, cfg.rrf_k)))
        .collect();
    scored.sort_by(|(a, a_score), (b, b_score)| {
        b.pinned
            .cmp(&a.pinned)
            .then(b_score.total_cmp(a_score))
            .then_with(|| newest_first(&facts[a.index], &facts[b.index]))
    });
    let hits = scored
        .into_iter()
        .take(limit)
        .map(|(candidate, score)| SearchHit {
            fact: facts[candidate.index].clone(),
            score,
            kind: match (
                candidate.pinned,
                candidate.lexical_rank,
                candidate.semantic_rank,
            ) {
                (true, _, _) => MatchKind::Exact,
                (false, Some(_), Some(_)) => MatchKind::Both,
                (false, None, Some(_)) => MatchKind::Semantic,
                (false, _, None) => MatchKind::Lexical,
            },
        })
        .collect();
    FactSearchReport {
        hits,
        related: Vec::new(),
        considered: candidates.len(),
    }
}

/// Rank expansion-eligible facts and cap them at `cfg.related_max`.
/// `scores` holds `(position in neighbors, score)`; a neighbor with no score
/// is left out. There is no score floor: a relationship cluster (a parent of
/// the person searched) shares no words with the query, so a cutoff would
/// drop exactly the designed case. Ties break by `recall_count`, then lowest id: recall frequency is never a
/// relevance signal, only a tie breaker.
pub fn rank_related(
    neighbors: &[&Fact],
    scores: &[(usize, f32)],
    cfg: &FactSearchConfig,
) -> Vec<SearchHit> {
    let mut seen = HashSet::new();
    let mut scored: Vec<(&Fact, f32)> = scores
        .iter()
        .filter(|(position, _)| seen.insert(*position))
        .filter_map(|(position, score)| neighbors.get(*position).map(|f| (*f, *score)))
        .collect();
    scored.sort_by(|(a, a_score), (b, b_score)| {
        b_score
            .total_cmp(a_score)
            .then(b.recall_count.cmp(&a.recall_count))
            .then(a.id.cmp(&b.id))
    });
    scored
        .into_iter()
        .take(cfg.related_max)
        .map(|(fact, score)| SearchHit {
            fact: fact.clone(),
            score,
            kind: MatchKind::Semantic,
        })
        .collect()
}

fn rrf_score(candidate: &Candidate, k: f32) -> f32 {
    [candidate.lexical_rank, candidate.semantic_rank]
        .into_iter()
        .flatten()
        .map(|rank| 1.0 / (k + rank as f32))
        .sum()
}

fn newest_first(a: &Fact, b: &Fact) -> std::cmp::Ordering {
    b.updated_at.cmp(&a.updated_at).then(a.id.cmp(&b.id))
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
        assert!(
            flood
                .windows(2)
                .all(|w| facts[w[0].index].updated_at > facts[w[1].index].updated_at)
        );
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

    fn lexical(index: usize, pinned: bool) -> LexicalHit {
        LexicalHit {
            index,
            score: 1.0,
            pinned,
        }
    }

    fn filler(n: usize) -> Vec<Fact> {
        (0..n as i64)
            .map(|i| {
                fx::fact(
                    i + 1,
                    "general",
                    &format!("item_{i}"),
                    &format!("value {i}"),
                    0,
                )
            })
            .collect()
    }

    fn hit_ids(report: &FactSearchReport) -> Vec<i64> {
        report.hits.iter().map(|h| h.fact.id).collect()
    }

    #[test]
    fn lexical_flood_does_not_hide_semantic() {
        let facts = filler(65);
        let cfg = FactSearchConfig::default();
        let lex: Vec<LexicalHit> = (0..60).map(|i| lexical(i, false)).collect();
        let cosine: Vec<(usize, f32)> = (60..65).map(|i| (i, 0.5)).collect();
        let pool = candidates(&facts, &lex, &cosine, &cfg);
        assert_eq!(pool.len(), 30, "lexical capped at 25, plus 5 semantic");
        let report = rank(&facts, &pool, 10, &cfg);
        let ids = hit_ids(&report);
        for i in 61..=65 {
            assert!(ids.contains(&i), "semantic-only fact {i} hidden: {ids:?}");
        }
        assert_eq!(report.considered, 30);
    }

    #[test]
    fn pinned_leads_whatever_its_fusion_score() {
        let facts = filler(5);
        let cfg = FactSearchConfig::default();
        let lex = vec![lexical(0, false), lexical(1, false), lexical(3, true)];
        let cosine = vec![(0, 0.9), (1, 0.8)];
        let pool = candidates(&facts, &lex, &cosine, &cfg);
        let report = rank(&facts, &pool, 10, &cfg);
        assert_eq!(hit_ids(&report), vec![4, 1, 2]);
        assert_eq!(report.hits[0].kind, MatchKind::Exact);
    }

    #[test]
    fn fusion_orders_by_both_ranks() {
        let facts = filler(3);
        let cfg = FactSearchConfig::default();
        // Fact 0 is lexical rank 1 only; fact 1 is lexical rank 2 and
        // semantic rank 1; fact 2 is semantic rank 2 only.
        let lex = vec![lexical(0, false), lexical(1, false)];
        let cosine = vec![(1, 0.9), (2, 0.8)];
        let pool = candidates(&facts, &lex, &cosine, &cfg);
        let report = rank(&facts, &pool, 10, &cfg);
        assert_eq!(hit_ids(&report), vec![2, 1, 3]);
        assert_eq!(report.hits[0].kind, MatchKind::Both);
    }

    #[test]
    fn no_vectors_is_lexical_only() {
        let facts = corpus();
        let cfg = FactSearchConfig::default();
        let lex = LexicalIndex::build(&facts).score("deploy staging");
        let pool = candidates(&facts, &lex, &[], &cfg);
        let report = rank(&facts, &pool, 20, &cfg);
        let expected: Vec<i64> = lex.iter().take(20).map(|h| facts[h.index].id).collect();
        assert_eq!(hit_ids(&report), expected);
    }

    #[test]
    fn duplicate_projection_collapses() {
        let facts = corpus();
        let cfg = FactSearchConfig::default();
        let lex = LexicalIndex::build(&facts).score("home city");
        let cosine: Vec<(usize, f32)> = facts
            .iter()
            .enumerate()
            .map(|(i, f)| (i, fx::fake_cosine("home city", f)))
            .collect();
        let pool = candidates(&facts, &lex, &cosine, &cfg);
        let report = rank(&facts, &pool, 10, &cfg);
        let ids = hit_ids(&report);
        assert!(ids.contains(&fx::HOME_CITY), "{ids:?}");
        assert!(!ids.contains(&fx::HOME_CITY_PROJECTION), "{ids:?}");
    }

    #[test]
    fn hot_cluster_does_not_win_on_recall_count() {
        let facts = corpus();
        let cfg = FactSearchConfig::default();
        let neighbors: Vec<&Fact> = facts
            .iter()
            .filter(|f| fx::HOT_CLUSTER.contains(&f.id) || f.id == fx::FATHER_NAME)
            .collect();
        let scores: Vec<(usize, f32)> = neighbors
            .iter()
            .enumerate()
            .map(|(p, f)| (p, if f.id == fx::FATHER_NAME { 0.4 } else { -1.0 }))
            .collect();
        let related = rank_related(&neighbors, &scores, &cfg);
        assert_eq!(related[0].fact.id, fx::FATHER_NAME);
    }

    #[test]
    fn related_is_capped() {
        let facts = filler(30);
        let cfg = FactSearchConfig::default();
        let neighbors: Vec<&Fact> = facts.iter().collect();
        let scores: Vec<(usize, f32)> = (0..30).map(|p| (p, 0.5 + p as f32 / 100.0)).collect();
        let related = rank_related(&neighbors, &scores, &cfg);
        assert_eq!(related.len(), 5);
        assert_eq!(related[0].fact.id, 30, "best score first");
    }

    #[test]
    fn related_ties_break_on_recall_count() {
        let a = fx::fact(1, "user", "a", "x", 2);
        let b = fx::fact(2, "user", "b", "x", 9);
        let neighbors = vec![&a, &b];
        let related = rank_related(
            &neighbors,
            &[(0, 0.5), (1, 0.5)],
            &FactSearchConfig::default(),
        );
        assert_eq!(related[0].fact.id, 2);
    }

    #[test]
    fn single_word_is_never_a_phrase_match() {
        let facts = vec![fx::fact(1, "general", "notes", "the report is late", 0)];
        let hits = LexicalIndex::build(&facts).score("report");
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].pinned);
    }

    #[test]
    fn phrase_pin_matches_whole_words_only() {
        let facts = vec![
            fx::fact(1, "notes", "schedule", "report 8080 daily", 0),
            fx::fact(2, "project", "web_port", "port 80 is open", 0),
        ];
        let hits = LexicalIndex::build(&facts).score("port 80");
        let pinned: Vec<usize> = hits.iter().filter(|h| h.pinned).map(|h| h.index).collect();
        assert_eq!(pinned, vec![1]);
        assert!(
            hits.iter().all(|h| h.index != 0),
            "`report 8080` shares no word with `port 80`"
        );
    }

    #[test]
    fn phrase_pins_are_capped_but_exact_keys_are_not() {
        let mut facts: Vec<Fact> = (0..40)
            .map(|i| {
                fx::fact(
                    i,
                    "notes",
                    &format!("note_{i}"),
                    "a new blog post went out",
                    0,
                )
            })
            .collect();
        facts.push(fx::fact(100, "project", "blog_post", "weekly", 0));
        let hits = LexicalIndex::build(&facts).score("blog post");
        let pinned: Vec<&LexicalHit> = hits.iter().filter(|h| h.pinned).collect();
        assert_eq!(pinned.len(), MAX_PHRASE_PINS + 1);
        assert!(pinned.iter().any(|h| facts[h.index].id == 100));
        assert_eq!(hits.len(), 41, "unpinned phrase matches still score");
    }
}
