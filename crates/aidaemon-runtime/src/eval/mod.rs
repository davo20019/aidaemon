//! Per-task harness effectiveness evaluation (routing, progress, quality, cost).

mod accumulator;
mod scoring;

pub use accumulator::{HarnessEvalAccumulator, HarnessEvalSeed};
pub use scoring::HarnessEvalConfig;
