//! Prompt-equivalence migration safety net.
//!
//! Asserts that rendering `task_lead.md` and `executor.md` from the registry
//! produces byte-identical output to the legacy `build_*_prompt` helpers in
//! `src/agent/runtime/spawn.rs` over a fixture grid. When this passes, the
//! legacy production callers can be deleted (Task 12).

#![cfg(test)]

use super::{SpecialistRegistry, SpecialistRenderContext};
use crate::agent::Agent;
use crate::traits::SpecialistKind;

fn registry() -> SpecialistRegistry {
    SpecialistRegistry::load(None)
}

/// Construct `execution_mode` exactly as the legacy task-lead builder
/// branches on `is_scheduled` (see `src/agent/runtime/spawn.rs:1124-1134`).
fn execution_mode_for(is_scheduled: bool) -> String {
    if is_scheduled {
        "You have full tool access including `terminal`. For simple steps (single shell commands, \
         file writes), execute them directly. For complex multi-step work, you may still delegate \
         to executors via the workflow below.".to_string()
    } else {
        "Your primary job is to plan and delegate work via executors or cli_agent. \
         However, you also have direct access to essential tools (read_file, write_file, \
         edit_file, terminal, search_files). Use delegation first, but if delegation fails \
         (cli_agent errors, spawn_agent blocked, executor failures), switch to direct \
         execution with your own tools rather than retrying broken delegation paths.".to_string()
    }
}

fn fixtures() -> Vec<SpecialistRenderContext> {
    let mut out = Vec::new();
    for depth in [1usize, 2, 3] {
        for is_scheduled in [false, true] {
            out.push(SpecialistRenderContext {
                mission: "Audit disk usage in ~/projects".to_string(),
                task: "List the top ten largest directories under ~/projects".to_string(),
                depth,
                max_depth: 4,
                max_iterations: 24,
                goal_id: format!("goal_{}_{}", depth, is_scheduled),
                working_dir: "/Users/test/projects".to_string(),
                is_scheduled,
                parent_session_id: "telegram:bot:42".to_string(),
                execution_mode: execution_mode_for(is_scheduled),
            });
        }
    }
    out
}

#[test]
fn task_lead_md_renders_byte_equal_to_legacy_builder() {
    let registry = registry();
    for ctx in fixtures() {
        let rendered = registry.render(SpecialistKind::TaskLead, &ctx);
        let legacy = Agent::build_task_lead_prompt(
            &ctx.goal_id,
            &ctx.mission,
            None, // goal_context — keep test grid simple
            ctx.depth,
            ctx.max_depth,
            false, // has_cli_agent — match the no-cli-agent variant
            ctx.is_scheduled,
        );
        assert_eq!(
            rendered, legacy,
            "task_lead.md drift at ctx={:?}\n--- rendered ---\n{}\n--- legacy ---\n{}",
            ctx, rendered, legacy
        );
    }
}

#[test]
fn executor_md_renders_byte_equal_to_legacy_builder() {
    let registry = registry();
    for ctx in fixtures() {
        let rendered = registry.render(SpecialistKind::Executor, &ctx);
        let legacy = Agent::build_executor_prompt(
            &ctx.task,
            &ctx.mission,
            ctx.depth,
            ctx.max_depth,
            false, // has_cli_agent
            None,  // task_id — keep test grid simple
            None,  // project_scope
        );
        assert_eq!(
            rendered, legacy,
            "executor.md drift at ctx={:?}\n--- rendered ---\n{}\n--- legacy ---\n{}",
            ctx, rendered, legacy
        );
    }
}
