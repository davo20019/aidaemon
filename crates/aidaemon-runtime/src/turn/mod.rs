//! Parts of one agent turn: the system directives and tool-result notices
//! the loop injects (their model-facing text), and how a task's outcome is
//! read from its completion evidence.

pub mod system_directives;
pub mod task_outcome;
pub mod tool_result_notices;
