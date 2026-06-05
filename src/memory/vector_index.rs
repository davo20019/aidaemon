//! In-memory vector index for fast top-K semantic retrieval.
//!
//! Holds decoded f32 embeddings keyed by row id so similarity queries avoid
//! re-fetching every row from SQLite and re-decoding every embedding blob on
//! each call (the previous O(n)-per-query cost in `facts.rs`).
//!
//! The v1 backend is an exact brute-force cosine scan over the cached vectors.
//! This eliminates the dominant per-query costs (DB I/O + blob decode) while
//! keeping results identical to the previous scoring. The `search` interface is
//! deliberately backend-agnostic so an approximate HNSW backend can be swapped
//! in later, when single-user fact volume actually grows large enough to need
//! sub-linear search.

use std::collections::HashMap;

use crate::memory::math::cosine_similarity;

/// In-memory map of `id -> embedding` supporting top-K cosine search.
#[derive(Debug, Default)]
pub struct VectorIndex {
    entries: HashMap<i64, Vec<f32>>,
}

impl VectorIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Number of vectors currently indexed.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `id` is currently indexed.
    pub fn contains(&self, id: i64) -> bool {
        self.entries.contains_key(&id)
    }

    /// Insert or replace the embedding for `id`.
    pub fn insert(&mut self, id: i64, embedding: Vec<f32>) {
        self.entries.insert(id, embedding);
    }

    /// Remove the embedding for `id`, if present.
    pub fn remove(&mut self, id: i64) {
        self.entries.remove(&id);
    }

    /// Return up to `k` `(id, cosine_similarity)` pairs, ordered by descending
    /// similarity. Returns an empty vec if the index or `query` is empty, or if
    /// `k == 0`. Ties are broken by ascending id for deterministic ordering.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if k == 0 || query.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(i64, f32)> = self
            .entries
            .iter()
            .map(|(&id, vec)| (id, cosine_similarity(query, vec)))
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_search_returns_empty() {
        let index = VectorIndex::new();
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn finds_inserted_vector_with_high_self_similarity() {
        let mut index = VectorIndex::new();
        index.insert(42, vec![1.0, 0.0, 0.0]);
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
        assert!((results[0].1 - 1.0).abs() < 1e-6, "self-sim should be ~1.0");
    }

    #[test]
    fn orders_results_by_descending_similarity() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0, 0.0]); // identical to query
        index.insert(2, vec![0.0, 1.0, 0.0]); // orthogonal to query
        index.insert(3, vec![0.7, 0.7, 0.0]); // 45° from query
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        let ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        // similarities are non-increasing
        for pair in results.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
    }

    #[test]
    fn respects_k_limit() {
        let mut index = VectorIndex::new();
        for i in 0..10 {
            index.insert(i, vec![i as f32, 1.0, 0.0]);
        }
        let results = index.search(&[1.0, 1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn k_larger_than_count_returns_all() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0]);
        index.insert(2, vec![0.0, 1.0]);
        let results = index.search(&[1.0, 1.0], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0]);
        assert!(index.search(&[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn insert_with_existing_id_upserts() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0, 0.0]);
        index.insert(1, vec![0.0, 1.0, 0.0]); // replace
        assert_eq!(index.len(), 1);
        let results = index.search(&[0.0, 1.0, 0.0], 5);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn remove_deletes_entry() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0]);
        index.insert(2, vec![0.0, 1.0]);
        assert!(index.contains(1));
        index.remove(1);
        assert!(!index.contains(1));
        assert_eq!(index.len(), 1);
        let ids: Vec<i64> = index
            .search(&[1.0, 1.0], 5)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn remove_missing_id_is_noop() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0]);
        index.remove(999);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut index = VectorIndex::new();
        index.insert(1, vec![1.0, 0.0]);
        assert!(index.search(&[], 5).is_empty());
    }
}
