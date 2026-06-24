//! Correction-bridge decision logic (Plan 3c, Task P3b.1).
//!
//! This is the PURE decision/safety core of the idle-reap → autonomous-correction
//! bridge. Given the self-correction config, a reconstructed subject context, the
//! failed command, and how long it ran idle before being reaped, it decides
//! whether (and how) a remediation task should be triggered.
//!
//! It performs NO I/O, async, or DB work. The actual remediation task spawn —
//! and the consumers of the `Dispatch`/`Shadowed` variants — arrive in later
//! tasks (P3b.2 / P3b.3), so the remediation-prompt fields are tolerated as
//! dead code until then.

use crate::config::SelfCorrectionConfig;

/// Outcome of evaluating whether a reaped command should trigger autonomous
/// remediation.
#[allow(dead_code)]
#[derive(Debug)]
pub enum CorrectionBridgeAction {
    /// `self_correction.enabled == false` — the bridge is off entirely.
    Disabled,
    /// The subject's `working_dir` is `/`, `$HOME`, or otherwise unbounded —
    /// refuse to remediate regardless of any bypass setting. This is the
    /// critical safety gate.
    UnsafeScope,
    /// Shadow mode is on: log the would-be remediation but do NOT dispatch.
    Shadowed { remediation_prompt: String },
    /// Live mode: proceed to spawn a remediation task with this prompt.
    Dispatch { remediation_prompt: String },
}

/// Decide whether/how a reaped command triggers autonomous remediation.
///
/// The order of checks is load-bearing:
/// 1. config disabled → [`CorrectionBridgeAction::Disabled`]
/// 2. unsafe scope (whole-home/whole-disk/unbounded) → [`CorrectionBridgeAction::UnsafeScope`]
///    (refused even when bypass is enabled — safety wins)
/// 3. shadow mode → [`CorrectionBridgeAction::Shadowed`]
/// 4. otherwise → [`CorrectionBridgeAction::Dispatch`]
#[allow(dead_code)]
pub fn decide_correction_bridge_action(
    config: &SelfCorrectionConfig,
    subject: &crate::agent::correction_sandbox::CorrectionSubjectContext,
    failed_command: &str,
    idle_secs: u64,
) -> CorrectionBridgeAction {
    // 1. Master switch.
    if !config.enabled {
        return CorrectionBridgeAction::Disabled;
    }

    // 2. Safety gate: unbounded scope is refused unconditionally.
    if crate::agent::correction_intent::is_unsafe_correction_working_dir(&subject.working_dir) {
        return CorrectionBridgeAction::UnsafeScope;
    }

    // 3. Build the remediation prompt once for both shadow + live paths.
    let remediation_prompt =
        build_remediation_prompt(&subject.original_request, failed_command, idle_secs);

    // 4. Shadow-first: log the would-be remediation, do not dispatch.
    if config.shadow_mode {
        return CorrectionBridgeAction::Shadowed { remediation_prompt };
    }

    // 5. Live dispatch.
    CorrectionBridgeAction::Dispatch { remediation_prompt }
}

/// Maximum chars retained from the failed command / original request when
/// embedding them in the remediation prompt. Keeps the prompt bounded without
/// risking a UTF-8 byte-boundary panic.
const MAX_EMBED_CHARS: usize = 1_000;

/// Build the instruction handed to the remediation agent.
///
/// The prompt names the failed command, the idle duration, the original goal,
/// and explicitly instructs a materially different, tightly-scoped retry — never
/// a repeat of the same command.
#[allow(dead_code)]
fn build_remediation_prompt(
    original_request: &str,
    failed_command: &str,
    idle_secs: u64,
) -> String {
    let command = crate::utils::truncate_str(failed_command, MAX_EMBED_CHARS);
    let goal = crate::utils::truncate_str(original_request, MAX_EMBED_CHARS);

    format!(
        "A previous command was stopped after {idle_secs}s without completing:\n\
         `{command}`\n\n\
         The goal it was pursuing: {goal}\n\n\
         Re-attempt this goal with a MATERIALLY DIFFERENT, faster, tightly-scoped \
         approach — do not repeat the same command. Prefer narrow, bounded \
         operations (size filters, depth limits, specific directories). Deliver the answer."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::correction_sandbox::CorrectionSubjectContext;
    use crate::traits::SelfCorrectionSubjectKind;
    use std::path::PathBuf;

    /// Build a subject context with the given working_dir and original_request
    /// for the bridge tests. Other fields are inert for this decision logic.
    fn subject_with(working_dir: &str, original_request: &str) -> CorrectionSubjectContext {
        CorrectionSubjectContext {
            subject_id: "subject-1".to_string(),
            subject_kind: SelfCorrectionSubjectKind::BackgroundCommand,
            session_id: "session-1".to_string(),
            original_request: original_request.to_string(),
            completion_contract_summary: String::new(),
            intended_accounts: Vec::new(),
            allowed_external_targets: Vec::new(),
            working_dir: PathBuf::from(working_dir),
        }
    }

    fn cfg(enabled: bool, shadow_mode: bool, bypass: bool) -> SelfCorrectionConfig {
        SelfCorrectionConfig {
            enabled,
            correction_bypass_enabled: bypass,
            max_attempts: 3,
            shadow_mode,
        }
    }

    #[test]
    fn test_bridge_disabled_when_config_off() {
        let config = cfg(false, true, false);
        let subject = subject_with("/tmp/proj", "what's the biggest file?");
        let action = decide_correction_bridge_action(&config, &subject, "find / -type f", 120);
        assert!(
            matches!(action, CorrectionBridgeAction::Disabled),
            "disabled config must short-circuit to Disabled, got {action:?}"
        );
    }

    #[test]
    fn test_bridge_unsafe_scope_refused() {
        // Unsafe scope must be refused even with bypass enabled and shadow off.
        let config = cfg(true, false, true);
        let subject = subject_with("/", "what's the biggest file?");
        let action = decide_correction_bridge_action(&config, &subject, "find / -type f", 120);
        assert!(
            matches!(action, CorrectionBridgeAction::UnsafeScope),
            "whole-disk working_dir must be refused, got {action:?}"
        );

        // Also refuse $HOME.
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/test".to_string());
        let subject_home = subject_with(&home, "what's the biggest file?");
        let action_home =
            decide_correction_bridge_action(&config, &subject_home, "find ~ -type f", 120);
        assert!(
            matches!(action_home, CorrectionBridgeAction::UnsafeScope),
            "whole-home working_dir must be refused, got {action_home:?}"
        );
    }

    #[test]
    fn test_bridge_shadowed_when_shadow_mode() {
        let config = cfg(true, true, false);
        let subject = subject_with("/tmp/proj", "what's the biggest file?");
        let action = decide_correction_bridge_action(&config, &subject, "find / -type f", 90);
        match action {
            CorrectionBridgeAction::Shadowed { remediation_prompt } => {
                assert!(
                    !remediation_prompt.is_empty(),
                    "shadowed prompt must be non-empty"
                );
            }
            other => panic!("expected Shadowed, got {other:?}"),
        }
    }

    #[test]
    fn test_bridge_dispatch_when_live() {
        let config = cfg(true, false, false);
        let subject = subject_with("/tmp/proj", "what's the biggest file?");
        let action = decide_correction_bridge_action(&config, &subject, "find / -type f", 90);
        match action {
            CorrectionBridgeAction::Dispatch { remediation_prompt } => {
                assert!(
                    !remediation_prompt.is_empty(),
                    "dispatch prompt must be non-empty"
                );
            }
            other => panic!("expected Dispatch, got {other:?}"),
        }
    }

    #[test]
    fn test_remediation_prompt_contains_goal_and_diff_instruction() {
        let prompt = build_remediation_prompt(
            "what's the biggest file in my downloads?",
            "find / -type f -size +100M",
            300,
        );
        // Contains the original goal.
        assert!(
            prompt.contains("what's the biggest file in my downloads?"),
            "prompt must contain the original request: {prompt}"
        );
        // Contains the failed command.
        assert!(
            prompt.contains("find / -type f -size +100M"),
            "prompt must contain the failed command: {prompt}"
        );
        // Instructs a different / scoped approach.
        assert!(
            prompt.contains("MATERIALLY DIFFERENT"),
            "prompt must instruct a different approach: {prompt}"
        );
        assert!(
            prompt.contains("do not repeat the same command"),
            "prompt must instruct not to repeat the command: {prompt}"
        );
        // Mentions the idle duration.
        assert!(
            prompt.contains("300s"),
            "prompt must mention the idle duration: {prompt}"
        );
    }
}
