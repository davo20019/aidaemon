//! Parts of one agent turn: the system directives and tool-result notices
//! the loop injects (their model-facing text), how a task's outcome is read
//! from its completion evidence, tool-failure classification, the ledger
//! closeout's admissibility, evidence routing, the replies built from tool
//! output, structural reply checks and the post-run learning context.

pub mod closeout;
pub mod completion_checks;
pub mod inquiry;
pub mod loop_utils;
pub mod post_task;
pub mod reply_checks;
pub mod system_directives;
pub mod task_outcome;
pub mod tool_facts;
pub mod tool_result_notices;
