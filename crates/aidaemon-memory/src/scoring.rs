//! Retrieval scoring: the replaceable [`Scorer`] and its default,
//! [`HybridScorer`] (port of the daemon's `memory/hybrid.rs`, the fact
//! thresholds in `state/sqlite/facts.rs`, and `memory/scoring.rs`).

use std::collections::HashSet;

use chrono::{DateTime, Utc};

use crate::types::{Episode, Fact, Score, Signals};

/// Weights of the fused score. Defaults are the daemon's calibrated values
/// and sum to 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridWeights {
    pub semantic: f32,
    pub lexical: f32,
    /// Entity-graph match, from [`crate::MemoryStore::fact_graph_scores`].
    /// Stays 0 for a store with no graph.
    pub graph: f32,
    pub freshness: f32,
    pub confidence: f32,
    pub provenance: f32,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            semantic: 0.45,
            lexical: 0.30,
            graph: 0.10,
            freshness: 0.10,
            confidence: 0.03,
            provenance: 0.02,
        }
    }
}

impl HybridWeights {
    /// Fuse independently bounded signals; inputs are clamped to 0..=1.
    pub fn fuse(&self, s: &Signals) -> f32 {
        self.semantic * s.semantic.clamp(0.0, 1.0)
            + self.lexical * s.lexical.clamp(0.0, 1.0)
            + self.graph * s.graph.clamp(0.0, 1.0)
            + self.freshness * s.freshness.clamp(0.0, 1.0)
            + self.confidence * s.confidence.clamp(0.0, 1.0)
            + self.provenance * s.provenance.clamp(0.0, 1.0)
    }
}

/// Tunables for [`HybridScorer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoringConfig {
    pub weights: HybridWeights,
    /// A candidate is kept when its semantic similarity is above this...
    pub semantic_min: f32,
    /// ...or its lexical score is above this. Both 0.3 in the daemon. The
    /// freshness/confidence/provenance signals only order candidates; they
    /// never let an irrelevant one through.
    pub lexical_min: f32,
    /// Freshness falls linearly from 1 to 0 over this window (7 days).
    pub freshness_window: chrono::Duration,
    /// Provenance signal for a fact with / without a supporting excerpt.
    pub provenance_with_excerpt: f32,
    pub provenance_without_excerpt: f32,
    /// Drop candidates whose fused score is at or below this (default 0).
    pub min_score: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            weights: HybridWeights::default(),
            semantic_min: 0.3,
            lexical_min: 0.3,
            freshness_window: chrono::Duration::days(7),
            provenance_with_excerpt: 1.0,
            provenance_without_excerpt: 0.3,
            min_score: 0.0,
        }
    }
}

/// Turns retrieval signals into a ranking score, or drops the candidate.
///
/// [`crate::Memory`] gathers the raw signals (cosine similarity from the
/// embedder, the store's lexical score) and asks the scorer. Replace it to
/// change ranking without touching storage; a store that ranks on its own
/// can bypass scoring entirely through [`crate::MemoryStore::search_facts`].
pub trait Scorer: Send + Sync {
    /// `semantic` is `None` when either side has no usable embedding.
    /// Return `None` to drop the fact.
    fn score_fact(
        &self,
        fact: &Fact,
        semantic: Option<f32>,
        lexical: f32,
        now: DateTime<Utc>,
    ) -> Option<Score>;

    /// [`Scorer::score_fact`] with the fact's graph relation to the query's
    /// other hits (0.0-1.0). The default ignores it, so a scorer written
    /// before this signal existed keeps ranking as it did.
    fn score_fact_with_graph(
        &self,
        fact: &Fact,
        semantic: Option<f32>,
        lexical: f32,
        _graph: f32,
        now: DateTime<Utc>,
    ) -> Option<Score> {
        self.score_fact(fact, semantic, lexical, now)
    }

    fn score_episode(
        &self,
        episode: &Episode,
        semantic: Option<f32>,
        lexical: f32,
        now: DateTime<Utc>,
    ) -> Option<Score>;
}

/// The daemon's hybrid ranking: weighted semantic + lexical + graph +
/// freshness + confidence + provenance, gated on relevance thresholds.
#[derive(Debug, Clone, Default)]
pub struct HybridScorer {
    pub config: ScoringConfig,
}

impl HybridScorer {
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    /// 1.0 when just updated, 0.0 at or beyond the freshness window.
    pub fn freshness(&self, now: DateTime<Utc>, updated_at: DateTime<Utc>) -> f32 {
        let window = self.config.freshness_window.num_seconds().max(1) as f32;
        let age = (now - updated_at).num_seconds().max(0) as f32;
        (1.0 - age / window).clamp(0.0, 1.0)
    }

    fn relevant(&self, semantic: Option<f32>, lexical: f32) -> bool {
        semantic.is_some_and(|s| s > self.config.semantic_min) || lexical > self.config.lexical_min
    }
}

impl Scorer for HybridScorer {
    fn score_fact(
        &self,
        fact: &Fact,
        semantic: Option<f32>,
        lexical: f32,
        now: DateTime<Utc>,
    ) -> Option<Score> {
        self.score_fact_with_graph(fact, semantic, lexical, 0.0, now)
    }

    /// The graph signal orders relevant facts; like freshness, it never lets
    /// an irrelevant one through.
    fn score_fact_with_graph(
        &self,
        fact: &Fact,
        semantic: Option<f32>,
        lexical: f32,
        graph: f32,
        now: DateTime<Utc>,
    ) -> Option<Score> {
        if !self.relevant(semantic, lexical) {
            return None;
        }
        let signals = Signals {
            semantic: semantic.unwrap_or(0.0).clamp(0.0, 1.0),
            lexical: lexical.clamp(0.0, 1.0),
            graph: graph.clamp(0.0, 1.0),
            freshness: self.freshness(now, fact.updated_at),
            confidence: fact.confidence.clamp(0.0, 1.0),
            provenance: if fact.source.excerpt.is_some() {
                self.config.provenance_with_excerpt
            } else {
                self.config.provenance_without_excerpt
            },
        };
        let value = self.config.weights.fuse(&signals);
        (value > self.config.min_score).then_some(Score { value, signals })
    }

    /// Episodes rank on relevance plus a small importance term, boosted by
    /// recall count. Age is deliberately absent (daemon `scoring.rs`): an
    /// old conversation must stay discoverable.
    fn score_episode(
        &self,
        episode: &Episode,
        semantic: Option<f32>,
        lexical: f32,
        _now: DateTime<Utc>,
    ) -> Option<Score> {
        if !self.relevant(semantic, lexical) {
            return None;
        }
        let signals = Signals {
            semantic: semantic.unwrap_or(0.0).clamp(0.0, 1.0),
            lexical: lexical.clamp(0.0, 1.0),
            confidence: episode.importance.clamp(0.0, 1.0),
            provenance: 1.0,
            ..Signals::default()
        };
        let value = self.config.weights.fuse(&signals) * episode_recall_boost(episode.recall_count);
        (value > self.config.min_score).then_some(Score { value, signals })
    }
}

/// Bounded reinforcement for often-recalled episodes: at most +16%.
pub fn episode_recall_boost(recall_count: u32) -> f32 {
    1.0 + ((recall_count as f32 + 1.0).ln() * 0.04).min(0.16)
}

/// Episode importance from typed session features (port of the daemon's
/// `calculate_episode_importance`). Base 0.5, clamped to 0.1..=1.0.
pub fn episode_importance(
    message_count: usize,
    has_errors: bool,
    has_decisions: bool,
    has_goals: bool,
    emotional_intensity: f32,
) -> f32 {
    let mut importance = 0.5;
    if message_count > 20 {
        importance += 0.2;
    } else if message_count > 10 {
        importance += 0.1;
    }
    if has_errors {
        importance += 0.15;
    }
    if has_decisions {
        importance += 0.1;
    }
    if has_goals {
        importance += 0.1;
    }
    let intensity = if emotional_intensity.is_finite() {
        emotional_intensity.clamp(0.0, 1.0)
    } else {
        0.0
    };
    importance += intensity * 0.2;
    importance.clamp(0.1, 1.0)
}

/// Function words that carry no retrieval signal. Used only to tokenize
/// for lexical overlap and hashing, never to decide what is remembered.
fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "me"
            | "my"
            | "of"
            | "on"
            | "or"
            | "please"
            | "tell"
            | "that"
            | "the"
            | "to"
            | "was"
            | "we"
            | "were"
            | "what"
            | "when"
            | "where"
            | "who"
            | "why"
            | "you"
            | "your"
    )
}

/// Lowercase alphanumeric words, stopwords removed. Splits on `_` too, so
/// `favorite_color` yields `favorite` and `color`.
pub fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && !is_stopword(t))
        .map(str::to_owned)
        .collect()
}

/// Fraction of distinct query tokens that occur as words in `text`
/// (0.0-1.0). The in-memory store's lexical signal.
pub fn term_overlap(query: &str, text: &str) -> f32 {
    let query: HashSet<String> = tokens(query).into_iter().collect();
    if query.is_empty() {
        return 0.0;
    }
    let text: HashSet<String> = tokens(text).into_iter().collect();
    query.intersection(&text).count() as f32 / query.len() as f32
}

/// A conversation turn's importance (0.1-1.0) from structure only: its role
/// (system prompts are not memories; tool output counts less than what the
/// user says) and its length. Nothing is inferred from the words used;
/// deciding what is durable is the extraction model's job.
pub fn turn_importance(role: &str, content: Option<&str>) -> f32 {
    let base: f32 = match role {
        "system" => return 0.0,
        "tool" => 0.3,
        "assistant" => 0.5,
        "user" => 0.6,
        _ => 0.5,
    };
    let length = content.map_or(0, str::len);
    let adjust = if length > 200 {
        0.2
    } else if length < 20 {
        -0.2
    } else {
        0.0
    };
    (base + adjust).clamp(0.1, 1.0)
}

/// Fraction of distinct query words that occur as words in `text`
/// (0.0-1.0), counting every word, stopwords included. Unlike
/// [`term_overlap`], a short query such as "who is she" keeps all its words:
/// the signal for ranking when no embedder is available.
pub fn word_overlap(query: &str, text: &str) -> f32 {
    let words = |value: &str| {
        value
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_owned)
            .collect::<HashSet<_>>()
    };
    let query = words(query);
    if query.is_empty() {
        return 0.0;
    }
    query.intersection(&words(text)).count() as f32 / query.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_importance_reads_role_and_length_only() {
        assert_eq!(turn_importance("system", Some("anything")), 0.0);
        assert!(
            turn_importance("user", Some(&"a".repeat(300)))
                > turn_importance("user", Some("a medium length message"))
        );
        assert!(
            turn_importance("user", Some("ok"))
                < turn_importance("user", Some("a medium length message"))
        );
        assert!(
            turn_importance("tool", Some("output of some length"))
                < turn_importance("user", Some("output of some length"))
        );
        // Wording does not matter, only structure.
        assert_eq!(
            turn_importance("user", Some("please remember this one")),
            turn_importance("user", Some("the weather is mild today"))
        );
    }

    #[test]
    fn word_overlap_counts_every_word() {
        assert_eq!(word_overlap("who is she", "she is here"), 2.0 / 3.0);
        assert_eq!(word_overlap("", "anything"), 0.0);
        assert_eq!(word_overlap("Mia", "mia and zoe"), 1.0);
    }

    #[test]
    fn corroborated_result_beats_single_weak_signal() {
        let w = HybridWeights::default();
        let corroborated = w.fuse(&Signals {
            semantic: 0.55,
            lexical: 0.8,
            graph: 1.0,
            confidence: 1.0,
            provenance: 1.0,
            ..Default::default()
        });
        let vector_only = w.fuse(&Signals {
            semantic: 0.7,
            ..Default::default()
        });
        assert!(corroborated > vector_only);
        let max = w.fuse(&Signals {
            semantic: 2.0,
            lexical: 2.0,
            graph: 2.0,
            freshness: 2.0,
            confidence: 2.0,
            provenance: 2.0,
        });
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn importance_and_boost_are_bounded() {
        assert_eq!(episode_importance(3, false, false, false, 0.0), 0.5);
        assert_eq!(episode_importance(30, true, true, true, 1.0), 1.0);
        assert!(episode_importance(12, true, false, false, 0.0) > 0.7);
        assert_eq!(episode_recall_boost(0), 1.0);
        assert!(episode_recall_boost(1_000_000) <= 1.16 + 1e-6);
    }

    #[test]
    fn overlap_counts_words_not_substrings() {
        assert_eq!(term_overlap("dog", "dodger stadium"), 0.0);
        assert_eq!(
            term_overlap("favorite color", "preference favorite_color: blue"),
            1.0
        );
        assert_eq!(term_overlap("the", "the"), 0.0);
    }
}
