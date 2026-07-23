//! Small deterministic metrics used by memory retrieval benchmarks.

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RetrievalMetrics {
    pub recall_at_k: f32,
    pub reciprocal_rank: f32,
}

/// Evaluate an ordered result list against one or more relevant owner IDs.
/// Duplicate results do not inflate recall.
pub fn evaluate_ranked(
    ranked_owner_ids: &[String],
    relevant_owner_ids: &HashSet<String>,
    k: usize,
) -> RetrievalMetrics {
    if relevant_owner_ids.is_empty() || k == 0 {
        return RetrievalMetrics::default();
    }
    let mut seen = HashSet::new();
    let mut hits = 0usize;
    let mut reciprocal_rank = 0.0;
    for (rank, owner_id) in ranked_owner_ids.iter().take(k).enumerate() {
        if !seen.insert(owner_id) || !relevant_owner_ids.contains(owner_id) {
            continue;
        }
        hits += 1;
        if reciprocal_rank == 0.0 {
            reciprocal_rank = 1.0 / (rank + 1) as f32;
        }
    }
    RetrievalMetrics {
        recall_at_k: hits as f32 / relevant_owner_ids.len() as f32,
        reciprocal_rank,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_recall_and_mrr_without_duplicate_inflation() {
        let ranked = ["noise", "target-a", "target-a", "target-b"]
            .map(str::to_string)
            .to_vec();
        let relevant = ["target-a", "target-b"]
            .map(str::to_string)
            .into_iter()
            .collect();
        let metrics = evaluate_ranked(&ranked, &relevant, 4);
        assert_eq!(metrics.recall_at_k, 1.0);
        assert_eq!(metrics.reciprocal_rank, 0.5);
    }

    #[test]
    fn empty_judgment_is_well_defined() {
        assert_eq!(
            evaluate_ranked(&["x".to_string()], &HashSet::new(), 10),
            RetrievalMetrics::default()
        );
    }
}
