//! Harness effectiveness evaluation: offline scenarios on `aidaemon-eval` and
//! online reporting helpers.

pub mod recorder;
pub mod report;

#[cfg(test)]
pub mod subject;

pub use report::{EvalSummaryStats, EvalTaskRow, format_eval_summary_row, format_eval_task_report};
