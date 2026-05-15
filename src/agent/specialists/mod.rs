//! Specialist registry: loads per-kind expert profiles (system prompt, tools,
//! model, budgets) from bundled `.md` files plus optional user overrides.

use crate::traits::SpecialistKind;
use std::path::PathBuf;
use std::sync::Arc;

mod parse;

#[allow(unused_imports)] // consumed by registry loader in upcoming tasks
pub use parse::parse_specialist;

#[allow(dead_code)] // fields populated by loader in upcoming tasks
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

#[allow(dead_code)] // variants used by loader in upcoming tasks
#[derive(Debug, Clone)]
pub enum SpecialistSource {
    Bundled,
    UserOverride(PathBuf),
}

#[allow(dead_code)] // populated by registry loader in upcoming tasks
#[derive(Debug, Default)]
pub struct SpecialistRegistry {
    by_kind: std::collections::HashMap<SpecialistKind, Arc<SpecialistDef>>,
}
