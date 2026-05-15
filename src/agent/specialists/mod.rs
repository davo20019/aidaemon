//! Specialist registry: loads per-kind expert profiles (system prompt, tools,
//! model, budgets) from bundled `.md` files plus optional user overrides.

use crate::traits::SpecialistKind;
use std::path::PathBuf;
use std::sync::Arc;

mod parse;
mod registry;
mod render;
pub mod validation;

#[cfg(test)]
mod equivalence_tests;
#[cfg(test)]
mod override_tests;

#[allow(unused_imports)]
// re-exported for downstream consumers; registry uses super::parse path
pub use parse::parse_specialist;
#[allow(unused_imports)]
// re-exported for downstream consumers; registry uses super::render path
pub use render::render_template;

#[derive(Debug, Default, Clone)]
#[allow(dead_code)] // consumed by registry.render and Tasks 5-8
pub struct SpecialistRenderContext {
    pub mission: String,
    pub task: String,
    pub depth: usize,
    pub max_depth: usize,
    pub max_iterations: usize,
    pub goal_id: String,
    pub working_dir: String,
    pub is_scheduled: bool,
    pub parent_session_id: String,
    /// Pre-rendered execution-mode paragraph used by `task_lead.md`. The
    /// task-lead prompt has two variants depending on `is_scheduled`; the
    /// caller picks the right paragraph and passes it here, so the template
    /// stays a flat string substitution.
    pub execution_mode: String,
}

#[allow(dead_code)] // model/tools/budget/timeout fields consumed by Tasks 5-8
#[derive(Debug, Clone)]
pub struct SpecialistDef {
    pub kind: SpecialistKind,
    pub description: String,
    pub system_prompt_template: String,
    pub model: Option<String>,
    pub tools: Option<Vec<String>>,
    pub max_iterations: Option<usize>,
    pub tool_budget: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub source: SpecialistSource,
}

#[allow(dead_code)] // variants constructed by registry loader; consumed by Tasks 5-8
#[derive(Debug, Clone)]
pub enum SpecialistSource {
    Bundled,
    UserOverride(PathBuf),
}

#[allow(dead_code)] // production consumers wired in Tasks 5-8; tests cover load/get/kinds now
#[derive(Debug, Default)]
pub struct SpecialistRegistry {
    by_kind: std::collections::HashMap<SpecialistKind, Arc<SpecialistDef>>,
}
