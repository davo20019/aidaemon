//! Parts of one agent turn: the system directives and tool-result notices
//! the loop injects (their model-facing text), how a task's outcome is read
//! from its completion evidence, tool-failure classification, the ledger
//! closeout's admissibility, evidence routing, the replies built from tool
//! output, structural reply checks and the post-run learning context.

pub mod closeout;
pub mod completion_checks;
pub mod completion_contract;
pub mod dialogue_state;
pub mod execution_state;
pub mod followup;
pub mod inquiry;
pub mod loop_utils;
pub mod post_task;
pub mod project_scope;
pub mod reply_checks;
pub mod sliding_window;
pub mod system_directives;
pub mod task_outcome;
pub mod tool_facts;
pub mod tool_result_notices;
pub mod turn_eviction;
pub mod turn_render_cache;
pub mod validation_state;
