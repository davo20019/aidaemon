//! Per-task harness effectiveness evaluation (routing, progress, quality, cost).

mod accumulator;
mod scoring;

pub use accumulator::{HarnessEvalAccumulator, HarnessEvalSeed, StopReason};
pub use scoring::HarnessEvalConfig;

pub(in crate::agent) type HarnessEvalHandle =
    std::sync::Arc<tokio::sync::RwLock<Option<HarnessEvalAccumulator>>>;

impl From<&crate::config::DiagnosticsHarnessEvalConfig> for HarnessEvalConfig {
    fn from(cfg: &crate::config::DiagnosticsHarnessEvalConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            weight_routing: cfg.weight_routing,
            weight_progress: cfg.weight_progress,
            weight_quality: cfg.weight_quality,
            weight_cost: cfg.weight_cost,
            cost_tier_cheap: cfg.cost_tier_cheap,
            cost_tier_balanced: cfg.cost_tier_balanced,
            cost_tier_strong: cfg.cost_tier_strong,
            cost_tier_unknown: cfg.cost_tier_unknown,
        }
    }
}
