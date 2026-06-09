//! Harness effectiveness evaluation: offline fixtures and online reporting helpers.

pub mod fixture;
pub mod report;

#[cfg(test)]
pub mod runner;

pub use fixture::{ExpectBlock, HarnessEvalFixture, MockResponseSpec};
pub use report::{format_eval_summary_row, format_eval_task_report, EvalSummaryStats, EvalTaskRow};
