//! Calibrated fusion for heterogeneous memory-retrieval signals.

#[derive(Debug, Clone, Copy, Default)]
pub struct HybridSignals {
    pub semantic: f32,
    pub lexical: f32,
    pub graph: f32,
    pub freshness: f32,
    pub confidence: f32,
    pub provenance: f32,
}

/// Fuse independently bounded signals into one stable score. Inputs are
/// clamped because individual backends may use different score ranges.
pub fn fused_score(signals: HybridSignals) -> f32 {
    0.45 * signals.semantic.clamp(0.0, 1.0)
        + 0.30 * signals.lexical.clamp(0.0, 1.0)
        + 0.10 * signals.graph.clamp(0.0, 1.0)
        + 0.10 * signals.freshness.clamp(0.0, 1.0)
        + 0.03 * signals.confidence.clamp(0.0, 1.0)
        + 0.02 * signals.provenance.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corroborated_result_beats_single_weak_signal() {
        let corroborated = fused_score(HybridSignals {
            semantic: 0.55,
            lexical: 0.8,
            graph: 1.0,
            confidence: 1.0,
            provenance: 1.0,
            ..Default::default()
        });
        let vector_only = fused_score(HybridSignals {
            semantic: 0.7,
            ..Default::default()
        });
        assert!(corroborated > vector_only);
    }

    #[test]
    fn score_is_bounded() {
        assert!((fused_score(HybridSignals::default()) - 0.0).abs() < f32::EPSILON);
        assert!(
            (fused_score(HybridSignals {
                semantic: 2.0,
                lexical: 2.0,
                graph: 2.0,
                freshness: 2.0,
                confidence: 2.0,
                provenance: 2.0,
            }) - 1.0)
                .abs()
                < f32::EPSILON
        );
    }
}
