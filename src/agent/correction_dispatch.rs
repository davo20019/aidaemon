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

use std::sync::{Arc, Weak};

use crate::agent::correction_execution::{
    build_correction_execution_context, CorrectionDispatchMode,
};
use crate::agent::correction_sandbox::CorrectionSubjectContext;
use crate::agent::{spawn_background_task_lead, Agent};
use crate::channels::ChannelHub;
use crate::config::SelfCorrectionConfig;
use crate::traits::{Goal, StateStore};
use crate::types::{ChannelContext, UserRole};

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
    let remediation_prompt = build_remediation_prompt(
        &subject.original_request,
        failed_command,
        idle_secs,
        &subject.working_dir,
    );

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
    working_dir: &std::path::Path,
) -> String {
    let command = crate::utils::truncate_str(failed_command, MAX_EMBED_CHARS);
    let goal = crate::utils::truncate_str(original_request, MAX_EMBED_CHARS);
    // Cap the scope dir the same way (UTF-8-safe) so a pathological path can't
    // blow the prompt size or panic on a byte boundary.
    let scope = crate::utils::truncate_str(&working_dir.display().to_string(), MAX_EMBED_CHARS);

    format!(
        "A previous command was stopped after {idle_secs}s without completing:\n\
         `{command}`\n\n\
         The goal it was pursuing: {goal}\n\n\
         Re-attempt this goal with a MATERIALLY DIFFERENT, faster, tightly-scoped \
         approach — do not repeat the same command. Prefer narrow, bounded \
         operations (size filters, depth limits, specific directories). \
         IMPORTANT: use explicit ABSOLUTE paths (e.g. /Users/<you>/...) — do NOT use \
         `~`, `$HOME`, or `~/*`; shorthand home references and unbounded root scans \
         are rejected by the safety sandbox and will fail. Deliver the answer.\n\
         Example of an allowed command for this goal: \
         `find {scope} -type f -size +1G -printf '%s\\t%p\\n'` — this lists \
         \"<bytes>\\t<path>\" for each file over the threshold; read the output \
         and pick the largest. Adapt the size threshold and directory as needed. \
         (Pipes and `-exec` are blocked in correction mode, so do not use them.)"
    )
}

/// Session id used for synthetic remediation goals. Remediation runs are
/// daemon-internal and never belong to a user channel session.
#[allow(dead_code)]
const CORRECTION_REMEDIATION_SESSION: &str = "internal:self-correction";

/// LIVE dispatch of an autonomous remediation task with a per-call correction
/// gate (Plan 3c, Task P3b.2).
///
/// This is the security-sensitive bridge that actually *fires* the 3b per-call
/// sandbox gate. It:
///
/// 1. Builds the correction-execution context via
///    [`build_correction_execution_context`] in [`CorrectionDispatchMode::Deferred`].
///    If the factory refuses (kill-switch: `enabled=false`, or Deferred with
///    `correction_bypass_enabled=false` — which would hang waiting on interactive
///    approval), this returns `Ok(None)` and dispatches **nothing**.
/// 2. Mints a synthetic finite [`Goal`] whose `description`/user_text is the
///    `remediation_prompt`, registers the correction context against that goal's
///    unique id on the agent, then spawns it through the existing
///    [`spawn_background_task_lead`] path with [`ChannelContext::internal`] and
///    [`UserRole::Owner`].
///
/// The registered context is keyed by the synthetic goal's globally-unique UUID.
/// Because the remediation task-lead and every executor it spawns inherit that
/// same `goal_id`, each of their agent loops reads `Some(ctx)` into
/// `ToolExecutionCtx.correction` (peek, not consume), while every other turn —
/// which carries a different (or no) `goal_id` — reads `None`. This is the
/// invariant: ONLY the deliberately-dispatched remediation task gets
/// `Some(correction)`.
///
/// Returns `Ok(Some(goal_id))` on dispatch, `Ok(None)` when the factory refused.
///
/// Tolerated as dead in plain (non-test) lib builds until the live reaper caller
/// (a later 3c task) invokes it; exercised by this module's tests.
#[allow(dead_code)]
pub async fn dispatch_correction_remediation(
    agent: Arc<Agent>,
    state: Arc<dyn StateStore>,
    hub: Option<Weak<ChannelHub>>,
    config: &SelfCorrectionConfig,
    subject: CorrectionSubjectContext,
    remediation_prompt: String,
) -> anyhow::Result<Option<String>> {
    // 1. Build the correction context. Deferred mode is mandatory here: this is
    //    an unattended, background remediation. The factory enforces the
    //    kill-switch rules (disabled, or Deferred-without-bypass → None).
    let Some(correction_ctx) = build_correction_execution_context(
        config,
        state.clone(),
        subject,
        CorrectionDispatchMode::Deferred,
    ) else {
        // Kill-switch tripped: do not dispatch anything.
        return Ok(None);
    };

    // 2. Mint a synthetic finite goal carrying the remediation prompt. Its fresh
    //    UUID id is the unique registry key for the correction context.
    let goal = Goal::new_finite(&remediation_prompt, CORRECTION_REMEDIATION_SESSION);
    let goal_id = goal.id.clone();

    // Register BEFORE spawning so the spawned hierarchy's loops observe the
    // context the moment they construct their `ToolExecutionCtx`.
    agent
        .register_correction_context(&goal_id, correction_ctx)
        .await;

    // 3. Spawn the remediation task lead in the background. It runs with an
    //    internal channel context and Owner role (remediation acts on the
    //    owner's behalf within the sandbox). `spawn_background_task_lead` is
    //    fire-and-forget (it `tokio::spawn`s internally), so we do NOT await it.
    spawn_background_task_lead(
        agent,
        goal,
        remediation_prompt,
        CORRECTION_REMEDIATION_SESSION.to_string(),
        ChannelContext::internal(),
        UserRole::Owner,
        state,
        hub,
        None, // no goal-token registry for synthetic remediation goals
        None, // no heartbeat dispatch-trigger task
        None, // no pre-posted surface to reuse
    );

    Ok(Some(goal_id))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        // After the scope relaxation, only genuinely-invalid scopes are refused:
        // empty or non-absolute (relative) working_dir. Broad scopes like `/` and
        // home are now ALLOWED (the read-only sandbox is the safety), so they are
        // NOT refused here.
        let config = cfg(true, false, true);

        // Relative (non-absolute) working_dir => still UnsafeScope.
        let subject_rel = subject_with("relative/dir", "what's the biggest file?");
        let action_rel =
            decide_correction_bridge_action(&config, &subject_rel, "find . -type f", 120);
        assert!(
            matches!(action_rel, CorrectionBridgeAction::UnsafeScope),
            "relative working_dir must be refused, got {action_rel:?}"
        );

        // `/` and home are now ALLOWED scopes (not refused). With shadow_mode=true
        // they reach Shadowed, never UnsafeScope.
        let subject_root = subject_with("/", "what's the biggest file?");
        let action_root =
            decide_correction_bridge_action(&config, &subject_root, "find / -type f", 120);
        assert!(
            !matches!(action_root, CorrectionBridgeAction::UnsafeScope),
            "whole-disk working_dir is now an allowed broad scope, got {action_root:?}"
        );

        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/test".to_string());
        let subject_home = subject_with(&home, "what's the biggest file?");
        let action_home =
            decide_correction_bridge_action(&config, &subject_home, "find ~ -type f", 120);
        assert!(
            !matches!(action_home, CorrectionBridgeAction::UnsafeScope),
            "home working_dir is now an allowed broad scope, got {action_home:?}"
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
            std::path::Path::new("/Users/synthetic/Documents"),
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

    #[test]
    fn test_remediation_prompt_contains_scoped_example_command() {
        // 3c robustness: the prompt must embed a CONCRETE worked example using
        // the goal's actual scope directory + a `find … -size …` form, so the
        // weak local model has something to adapt instead of abstract guidance.
        let scope = "/Users/synthetic/Documents";
        let prompt = build_remediation_prompt(
            "what's the biggest file?",
            "du -sh ~",
            120,
            std::path::Path::new(scope),
        );
        assert!(
            prompt.contains(scope),
            "prompt must contain the scope dir: {prompt}"
        );
        assert!(
            prompt.contains("find") && prompt.contains("-size"),
            "prompt must contain an example `find … -size` command: {prompt}"
        );
        // The example must use the scope dir as the find root (not a placeholder).
        assert!(
            prompt.contains(&format!("find {scope} -type f -size")),
            "example command must use the scope dir as the find root: {prompt}"
        );
    }

    #[test]
    fn test_remediation_prompt_example_uses_printf_not_exec() {
        // The worked example must use the Allowed `-printf` form, NOT the
        // (blocked) `-exec ls -lh {} +` form, so a model copying it is not
        // blocked again by the correction sandbox.
        let scope = "/Users/synthetic/Documents";
        let prompt = build_remediation_prompt(
            "what's the biggest file?",
            "du -sh ~",
            120,
            std::path::Path::new(scope),
        );
        assert!(
            prompt.contains("-printf"),
            "prompt example must use -printf: {prompt}"
        );
        assert!(
            !prompt.contains("-exec ls"),
            "prompt example must not use the blocked -exec form: {prompt}"
        );
        assert!(
            prompt.contains(&format!("find {scope} -type f -size +1G -printf")),
            "example must be the scoped -printf form: {prompt}"
        );
    }

    // ── P3b.2 TDD: live dispatch + correction-context threading ─────────────

    /// Build a fully-wired test agent (real SqliteStateStore, EventStore, etc.).
    async fn make_test_agent() -> crate::testing::TestHarness {
        crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .expect("test agent setup")
    }

    /// P3b.2: the factory kill-switch must short-circuit the LIVE dispatch.
    /// `enabled=false` → `Ok(None)` and NOTHING is registered/spawned.
    #[tokio::test]
    async fn test_dispatch_returns_none_when_factory_refuses() {
        let harness = make_test_agent().await;
        let agent = Arc::new(harness.agent);
        let state: Arc<dyn StateStore> = harness.state.clone();

        // Case 1: master kill-switch off.
        let disabled = cfg(false, true, true);
        let result = dispatch_correction_remediation(
            agent.clone(),
            state.clone(),
            None,
            &disabled,
            subject_with("/tmp/proj", "what's the biggest file?"),
            "remediate this".to_string(),
        )
        .await
        .expect("dispatch must not error");
        assert!(
            result.is_none(),
            "disabled config must dispatch nothing (Ok(None)), got {result:?}"
        );
        assert_eq!(
            agent.correction_context_count().await,
            0,
            "disabled config must register no correction context"
        );

        // Case 2: enabled but Deferred + bypass OFF → factory refuses (would hang
        // on interactive approval), so the live dispatch must also refuse.
        let no_bypass = cfg(true, false, false);
        let result2 = dispatch_correction_remediation(
            agent.clone(),
            state.clone(),
            None,
            &no_bypass,
            subject_with("/tmp/proj", "what's the biggest file?"),
            "remediate this".to_string(),
        )
        .await
        .expect("dispatch must not error");
        assert!(
            result2.is_none(),
            "Deferred + bypass-off must dispatch nothing (Ok(None)), got {result2:?}"
        );
        assert_eq!(
            agent.correction_context_count().await,
            0,
            "Deferred + bypass-off must register no correction context"
        );
    }

    /// P3b.2: when the factory accepts (enabled + bypass on), the live dispatch
    /// returns the synthetic remediation goal id (proving it dispatched rather
    /// than refusing). Registration is verified separately/deterministically in
    /// [`test_correction_context_threading_invariants`] — here we only assert the
    /// dispatch produced a goal id, because the fire-and-forget remediation task
    /// it spawns may *tear the context down* (P3b.3 teardown) before we can
    /// observe the registration, so counting it would race.
    #[tokio::test]
    async fn test_dispatch_returns_goal_id_when_factory_accepts() {
        let harness = make_test_agent().await;
        let agent = Arc::new(harness.agent);
        let state: Arc<dyn StateStore> = harness.state.clone();

        let enabled_bypass = cfg(true, false, true);
        let result = dispatch_correction_remediation(
            agent.clone(),
            state.clone(),
            None,
            &enabled_bypass,
            subject_with("/tmp/proj", "what's the biggest file?"),
            "Re-attempt with a bounded find".to_string(),
        )
        .await
        .expect("dispatch must not error");

        let goal_id = result.expect("enabled + bypass → must dispatch and return a goal id");
        assert!(
            !goal_id.trim().is_empty(),
            "dispatch must return a non-empty synthetic goal id"
        );
    }

    /// P3b.2 (deterministic, teardown-independent): the threading invariant —
    /// an agent whose current goal id matches a registered correction context
    /// peeks `Some` (and a second peek still sees it, i.e. peek does not
    /// consume), while any other / no goal id reads `None`. This is the heart of
    /// the non-stickiness guarantee: ONLY the deliberately-registered remediation
    /// goal id gets `Some(correction)`.
    ///
    /// Registration is done directly (not via the full dispatch+spawn path) so
    /// the fire-and-forget remediation task's P3b.3 teardown cannot race the
    /// assertions.
    #[tokio::test]
    async fn test_correction_context_threading_invariants() {
        use crate::agent::correction_execution::{
            CorrectionDispatchMode, CorrectionExecutionContext,
        };
        use crate::agent::self_correction::SelfCorrectionController;

        let harness = make_test_agent().await;
        let agent = Arc::new(harness.agent);
        let state: Arc<dyn StateStore> = harness.state.clone();

        let goal_id = "remediation-goal-threading".to_string();
        let ctx = Arc::new(CorrectionExecutionContext {
            subject: subject_with("/tmp/proj", "what's the biggest file?"),
            controller: Arc::new(SelfCorrectionController::new(state.clone(), 3)),
            dispatch_mode: CorrectionDispatchMode::Deferred,
            bypass_approvals: true,
        });
        agent.register_correction_context(&goal_id, ctx).await;
        assert_eq!(
            agent.correction_context_count().await,
            1,
            "exactly one correction context must be registered"
        );

        // An agent whose current goal id matches reads `Some` (peek, not consume).
        let mut peeker = make_test_agent().await.agent;
        // Share the same registry as the registering agent so the peek sees the
        // registered entry (this is what `create_child_agent` does in prod via
        // `self.correction_contexts.clone()`).
        peeker.correction_contexts = agent.correction_contexts.clone();
        peeker.set_test_goal_id(Some(goal_id.clone()));
        assert!(
            peeker.correction_context_for_current_goal().await.is_some(),
            "an agent running under the remediation goal id must peek Some(correction)"
        );

        // Peek must NOT consume — a second peek still sees it (so executors
        // sharing the goal id also get Some).
        assert!(
            peeker.correction_context_for_current_goal().await.is_some(),
            "peek must not consume; executors sharing the goal id must also see Some"
        );

        // An agent with a DIFFERENT goal id (a normal, unrelated turn) reads None.
        peeker.set_test_goal_id(Some("some-other-goal".to_string()));
        assert!(
            peeker.correction_context_for_current_goal().await.is_none(),
            "an unrelated goal id must read None — the invariant: only the \
             remediation task gets Some(correction)"
        );

        // An agent with no goal id (a plain user turn) reads None.
        peeker.set_test_goal_id(None);
        assert!(
            peeker.correction_context_for_current_goal().await.is_none(),
            "an agent with no goal id must read None"
        );
    }

    /// P3b.3 teardown: a dispatched remediation's correction context is cleared
    /// after the spawned task lead completes (success OR error). The MockProvider
    /// test agent has no `self_ref`, so the spawned task lead fails fast — which
    /// still drives the teardown path — and the registered context is removed
    /// within a bounded poll window. This proves contexts don't leak until FIFO
    /// eviction.
    #[tokio::test]
    async fn test_dispatched_remediation_context_is_cleared_after_completion() {
        let harness = make_test_agent().await;
        let agent = Arc::new(harness.agent);
        let state: Arc<dyn StateStore> = harness.state.clone();

        let enabled_bypass = cfg(true, false, true);
        let goal_id = dispatch_correction_remediation(
            agent.clone(),
            state.clone(),
            None,
            &enabled_bypass,
            subject_with("/tmp/proj", "what's the biggest file?"),
            "Re-attempt with a bounded find".to_string(),
        )
        .await
        .expect("dispatch must not error")
        .expect("enabled + bypass → must dispatch and return a goal id");

        // Poll until the background task lead completes and the teardown clears
        // the context. Bounded so a regression (no teardown) fails the test.
        let mut cleared = false;
        for _ in 0..100 {
            if agent.correction_context_count().await == 0 {
                cleared = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            cleared,
            "correction context for goal {goal_id} must be cleared after the \
             remediation task lead completes (P3b.3 teardown)"
        );
    }

    /// P3b.2: the bounded map evicts the oldest entry past the cap, but never
    /// drops the entry being inserted — so a fire-and-forget spawn that never
    /// tears down cannot leak unboundedly, and the just-dispatched remediation
    /// always retains its own context.
    #[tokio::test]
    async fn test_correction_context_registry_is_bounded() {
        use crate::agent::correction_execution::{
            CorrectionDispatchMode, CorrectionExecutionContext,
        };
        use crate::agent::self_correction::SelfCorrectionController;

        let harness = make_test_agent().await;
        let agent = Arc::new(harness.agent);
        let state: Arc<dyn StateStore> = harness.state.clone();

        let make_ctx = || {
            Arc::new(CorrectionExecutionContext {
                subject: super::CorrectionSubjectContext {
                    subject_id: "s".to_string(),
                    subject_kind: SelfCorrectionSubjectKind::BackgroundCommand,
                    session_id: "sess".to_string(),
                    original_request: "req".to_string(),
                    completion_contract_summary: String::new(),
                    intended_accounts: Vec::new(),
                    allowed_external_targets: Vec::new(),
                    working_dir: PathBuf::from("/tmp/proj"),
                },
                controller: Arc::new(SelfCorrectionController::new(state.clone(), 3)),
                dispatch_mode: CorrectionDispatchMode::Deferred,
                bypass_approvals: true,
            })
        };

        // Insert well past the cap.
        let total = super::super::MAX_CORRECTION_CONTEXTS + 10;
        let last_goal_id = format!("goal-{}", total - 1);
        for i in 0..total {
            agent
                .register_correction_context(&format!("goal-{i}"), make_ctx())
                .await;
        }

        assert!(
            agent.correction_context_count().await <= super::super::MAX_CORRECTION_CONTEXTS,
            "registry must stay bounded by MAX_CORRECTION_CONTEXTS"
        );

        // The most-recently inserted entry must still be present (insert never
        // evicts the key it is inserting). Use a peeker that shares the registry.
        let mut peeker = make_test_agent().await.agent;
        peeker.correction_contexts = agent.correction_contexts.clone();
        peeker.set_test_goal_id(Some(last_goal_id.clone()));
        assert!(
            peeker.correction_context_for_current_goal().await.is_some(),
            "the just-inserted context must survive bounded eviction"
        );

        // clear_correction_context removes a specific entry.
        agent.clear_correction_context(&last_goal_id).await;
        peeker.set_test_goal_id(Some(last_goal_id));
        assert!(
            peeker.correction_context_for_current_goal().await.is_none(),
            "clear_correction_context must remove the entry"
        );
    }
}
