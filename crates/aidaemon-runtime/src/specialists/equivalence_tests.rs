//! Prompt-equivalence migration safety net.
//!
//! Asserts that the registry-driven composition helpers
//! (`compose_task_lead_prompt_from_registry`,
//! `compose_executor_prompt_from_registry`) produce byte-identical output to
//! the legacy `build_*_prompt` helpers in `src/agent/runtime/spawn.rs` over a
//! fixture grid that covers depth × is_scheduled × has_cli_agent. When this
//! passes, the legacy production callers can be deleted (Task 12).

#![cfg(test)]

use super::SpecialistRegistry;
use super::SpecialistRenderContext;
use crate::spawn::task_lead_execution_mode;
use aidaemon_agents::specialist::SpecialistKind;

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
                        can_spawn: false,
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
        let composed = super::compose::compose_task_lead_prompt_from_registry(
            &registry,
            &ctx.goal_id,
            &ctx.mission,
            None, // goal_context — keep test grid simple
            ctx.depth,
            ctx.max_depth,
            fx.has_cli_agent,
            ctx.is_scheduled,
        );
        let legacy = super::compose::legacy::build_task_lead_prompt(
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
        let composed = super::compose::compose_executor_prompt_from_registry(
            &registry,
            SpecialistKind::Executor,
            &ctx.task,
            &ctx.mission,
            ctx.depth,
            ctx.max_depth,
            fx.has_cli_agent,
            None, // task_id — keep test grid simple
            None, // project_scope
            false,
        );
        let legacy = super::compose::legacy::build_executor_prompt(
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

/// Per-kind goldens for the executor-flavored specialists (Code, Research,
/// Review, etc.). Verifies two contracts:
///
/// 1. Each kind's rendered prompt opens with its role-specific tagline
///    (the part that differentiates a Code specialist from a Research
///    specialist at the prompt level).
/// 2. The shared `{{executor_base}}` partial is spliced in — the Rules
///    section and the mission/task headings must appear after the tagline,
///    so every executor-flavored specialist inherits the same discipline
///    frame.
///
/// These are *new* contracts the refactor introduces; before the refactor,
/// every kind silently rendered the Executor body regardless of role. There
/// is no legacy byte-equivalent oracle for the non-Executor kinds — these
/// goldens are the spec.
#[test]
fn executor_flavored_kinds_open_with_their_tagline_and_include_shared_base() {
    let registry = SpecialistRegistry::load(None);

    // Each entry: (kind, expected tagline prefix). `Executor` and `Generic`
    // have no role-specific tagline — their body is the shared base, so the
    // rendered prompt opens with "You are an Executor." directly.
    let cases: &[(SpecialistKind, &str)] = &[
        (SpecialistKind::Code, "You are a Code specialist."),
        (SpecialistKind::Research, "You are a Research specialist."),
        (SpecialistKind::Review, "You are a Review specialist."),
        (
            SpecialistKind::CommsDraft,
            "You are a Comms Draft specialist.",
        ),
        (
            SpecialistKind::BrowserVerifier,
            "You are a Browser Verifier specialist.",
        ),
        (
            SpecialistKind::ArtifactWriter,
            "You are an Artifact Writer specialist.",
        ),
        (
            SpecialistKind::Executor,
            "You are an Executor. Complete this single task",
        ),
        (
            SpecialistKind::Generic,
            "You are an Executor. Complete this single task",
        ),
    ];

    for (kind, tagline_prefix) in cases {
        let prompt = super::compose::compose_executor_prompt_from_registry(
            &registry,
            *kind,
            "list the largest files",
            "audit disk usage",
            2,
            4,
            false, // has_cli_agent
            None,  // task_id
            None,  // project_scope
            false, // can_spawn
        );
        assert!(
            prompt.starts_with(tagline_prefix),
            "{:?} prompt does not open with expected tagline.\nexpected prefix: {:?}\nactual head: {:?}",
            kind,
            tagline_prefix,
            prompt.chars().take(120).collect::<String>(),
        );
        // Shared base spliced in: discipline rules + headings must be present.
        for marker in [
            "## Original User Request",
            "audit disk usage", // mission substituted
            "## Your Specific Task",
            "list the largest files", // task substituted
            "Rules:",
            "- Do NOT spawn sub-agents.",
        ] {
            assert!(
                prompt.contains(marker),
                "{:?} prompt is missing expected base marker {:?}\n---\n{}\n---",
                kind,
                marker,
                prompt,
            );
        }
        // No unresolved placeholders for the variables the registry supports.
        for placeholder in [
            "{{executor_base}}",
            "{{mission}}",
            "{{task}}",
            "{{depth}}",
            "{{max_depth}}",
        ] {
            assert!(
                !prompt.contains(placeholder),
                "{:?} left {} unresolved",
                kind,
                placeholder
            );
        }
    }
}

/// A specialist the caller made a coordinator is told it may delegate: its
/// prompt follows the spawn decision, not a rule in the shared template.
#[test]
fn a_specialist_that_can_spawn_is_not_told_it_cannot() {
    let registry = SpecialistRegistry::load(None);
    for (can_spawn, says) in [
        (true, super::render::delegation_rule(true)),
        (false, "- Do NOT spawn sub-agents."),
    ] {
        let prompt = super::compose::compose_executor_prompt_from_registry(
            &registry,
            SpecialistKind::Research,
            "t",
            "m",
            1,
            3,
            false,
            None,
            None,
            can_spawn,
        );
        assert!(prompt.contains(says), "{prompt}");
        assert_eq!(prompt.contains("Do NOT spawn"), !can_spawn, "{prompt}");
        assert!(!prompt.contains("{{delegation_rule}}"));
    }
}
