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

// Re-exported for downstream consumers and tests. Internal callers (registry)
// use the `super::parse` / `super::render` paths directly, so these re-exports
// are currently unused outside the crate's test code.
#[allow(unused_imports)]
pub use parse::parse_specialist;
#[allow(unused_imports)]
pub use render::render_template;

#[derive(Debug, Default, Clone)]
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

// `kind`, `description`, and `tool_budget` are parsed from frontmatter and
// surfaced for completeness / future consumers (logging, schema generation),
// but the spawn flow currently does not read them. Keep them on the struct
// so the parsed representation stays faithful to the on-disk schema.
#[allow(dead_code)]
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

#[derive(Debug, Clone)]
pub enum SpecialistSource {
    Bundled,
    // `PathBuf` payload is recorded for diagnostics / future logging; current
    // production callers only discriminate on the variant tag.
    UserOverride(#[allow(dead_code)] PathBuf),
}

#[derive(Debug, Default)]
pub struct SpecialistRegistry {
    by_kind: std::collections::HashMap<SpecialistKind, Arc<SpecialistDef>>,
}
