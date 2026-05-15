//! Prompt-equivalence migration safety net.
//!
//! Asserts that the registry-driven composition helpers
//! (`compose_task_lead_prompt_from_registry`,
//! `compose_executor_prompt_from_registry`) produce byte-identical output to
//! the legacy `build_*_prompt` helpers in `src/agent/runtime/spawn.rs` over a
//! fixture grid that covers depth × is_scheduled × has_cli_agent. When this
//! passes, the legacy production callers can be deleted (Task 12).

#![cfg(test)]

use super::SpecialistRenderContext;
use crate::agent::spawn::task_lead_execution_mode;
use crate::agent::specialists::SpecialistRegistry;
use crate::agent::Agent;

fn registry() -> SpecialistRegistry {
    SpecialistRegistry::load(None)
}

/// Test fixture context. Carries the inputs both the legacy builders and the
/// new composition helpers need.
#[derive(Debug, Clone)]
struct FixtureCtx {
    render: SpecialistRenderContext,
    has_cli_agent: bool,
}

fn fixtures() -> Vec<FixtureCtx> {
    let mut out = Vec::new();
    for depth in [1usize, 2, 3] {
        for is_scheduled in [false, true] {
            for has_cli_agent in [false, true] {
                out.push(FixtureCtx {
                    render: SpecialistRenderContext {
                        mission: "Audit disk usage in ~/projects".to_string(),
                        task: "List the top ten largest directories under ~/projects".to_string(),
                        depth,
                        max_depth: 4,
                        max_iterations: 24,
                        goal_id: format!("goal_{}_{}_{}", depth, is_scheduled, has_cli_agent),
                        working_dir: "/Users/test/projects".to_string(),
                        is_scheduled,
                        parent_session_id: "telegram:bot:42".to_string(),
                        execution_mode: task_lead_execution_mode(is_scheduled).to_string(),
                    },
                    has_cli_agent,
                });
            }
        }
    }
    out
}

#[test]
fn task_lead_md_renders_byte_equal_to_legacy_builder() {
    let registry = registry();
    let fxs = fixtures();
    assert_eq!(
        fxs.len(),
        12,
        "expected 3 depths × 2 schedule × 2 cli_agent = 12 fixtures"
    );
    for fx in fxs {
        let ctx = &fx.render;
        let composed = Agent::compose_task_lead_prompt_from_registry(
            &registry,
            &ctx.goal_id,
            &ctx.mission,
            None, // goal_context — keep test grid simple
            ctx.depth,
            ctx.max_depth,
            fx.has_cli_agent,
            ctx.is_scheduled,
        );
        let legacy = Agent::build_task_lead_prompt(
            &ctx.goal_id,
            &ctx.mission,
            None, // goal_context — keep test grid simple
            ctx.depth,
            ctx.max_depth,
            fx.has_cli_agent,
            ctx.is_scheduled,
        );
        assert_eq!(
            composed, legacy,
            "task_lead.md drift at fx={:?}\n--- composed ---\n{}\n--- legacy ---\n{}",
            fx, composed, legacy
        );
    }
}

#[test]
fn executor_md_renders_byte_equal_to_legacy_builder() {
    let registry = registry();
    let fxs = fixtures();
    assert_eq!(
        fxs.len(),
        12,
        "expected 3 depths × 2 schedule × 2 cli_agent = 12 fixtures"
    );
    for fx in fxs {
        let ctx = &fx.render;
        let composed = Agent::compose_executor_prompt_from_registry(
            &registry,
            &ctx.task,
            &ctx.mission,
            ctx.depth,
            ctx.max_depth,
            fx.has_cli_agent,
            None, // task_id — keep test grid simple
            None, // project_scope
        );
        let legacy = Agent::build_executor_prompt(
            &ctx.task,
            &ctx.mission,
            ctx.depth,
            ctx.max_depth,
            fx.has_cli_agent,
            None, // task_id — keep test grid simple
            None, // project_scope
        );
        assert_eq!(
            composed, legacy,
            "executor.md drift at fx={:?}\n--- composed ---\n{}\n--- legacy ---\n{}",
            fx, composed, legacy
        );
    }
}
