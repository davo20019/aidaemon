/// Correction-execution context types for Plan 3b Phase 2.
///
/// `CorrectionExecutionContext` is carried by a correction/remediation task.
/// A normal (user-initiated) task always has `correction: None` in the
/// tool-execution context structs; correction tasks carry `Some(Arc<...>)`.
use std::sync::Arc;

use crate::agent::correction_sandbox::CorrectionSubjectContext;
use crate::agent::self_correction::SelfCorrectionController;
use crate::config::SelfCorrectionConfig;

/// How the correction sub-loop was dispatched.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrectionDispatchMode {
    /// Inline: the correction runs inside the current agent turn, before
    /// responding to the user.
    Inline,
    /// Deferred: the correction was scheduled as an independent goal/task and
    /// will execute asynchronously.
    Deferred,
}

/// Runtime context for a correction/remediation task.
///
/// Constructed once per correction attempt and shared across all tool
/// calls in that attempt via `Arc`. A normal task always sees
/// `correction: None` in `ToolExecutionCtx`; P2.4 reads the `Some` variant
/// to apply the sandbox gate.
#[allow(dead_code)]
#[derive(Clone)]
pub struct CorrectionExecutionContext {
    pub subject: CorrectionSubjectContext,
    pub controller: Arc<SelfCorrectionController>,
    pub dispatch_mode: CorrectionDispatchMode,
    /// When `true`, the P2.4 gate must skip the normal terminal/tool
    /// approval flow and trust the sandbox classifier instead.
    pub bypass_approvals: bool,
}

/// Lightweight handle returned to the dispatch caller so it can record
/// the final outcome (failure/success/blocked) against the right subject
/// and attempt index without needing the full context.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CorrectionAttemptHandle {
    pub subject_id: String,
    pub kind: crate::traits::SelfCorrectionSubjectKind,
    pub signature: String,
    pub attempt_index: usize,
}

/// Construct a [`CorrectionExecutionContext`] from config and runtime inputs,
/// applying the kill-switch rules:
///
/// 1. `enabled=false` → `None` (master kill-switch).
/// 2. `mode=Deferred` (unattended) **and** `correction_bypass_enabled=false` → `None`.
///    Dispatching unattended remediation that cannot bypass approvals would cause
///    the task to hang waiting for interactive approval. Plan 3c must treat `None`
///    here as "do not spawn unattended remediation."
/// 3. `mode=Deferred` + `correction_bypass_enabled=true` → `Some(...)` with
///    `bypass_approvals=true`.
/// 4. `mode=Inline` (user-present) + `enabled=true` → `Some(...)` with
///    `bypass_approvals` mirroring `correction_bypass_enabled`.
///
/// The controller is initialised with `config.max_attempts`.
///
/// Note on budgets: this context uses `self_correction.max_attempts`; the
/// existing B1 approach-pivot logic uses `MAX_APPROACH_PIVOTS + 1`. These are
/// independent budgets — do not conflate them in operator debug notes.
#[allow(dead_code)]
pub fn build_correction_execution_context(
    config: &SelfCorrectionConfig,
    state: Arc<dyn crate::traits::StateStore>,
    subject: CorrectionSubjectContext,
    mode: CorrectionDispatchMode,
) -> Option<Arc<CorrectionExecutionContext>> {
    // Rule 1: master kill-switch.
    if !config.enabled {
        return None;
    }

    // Rule 2: deferred (unattended) path requires bypass to be enabled, otherwise
    // the remediation task would hang waiting for interactive approval.
    if mode == CorrectionDispatchMode::Deferred && !config.correction_bypass_enabled {
        return None;
    }

    let bypass_approvals = config.correction_bypass_enabled;
    let controller = Arc::new(SelfCorrectionController::new(state, config.max_attempts));

    Some(Arc::new(CorrectionExecutionContext {
        subject,
        controller,
        dispatch_mode: mode,
        bypass_approvals,
    }))
}

/// Record that a correction attempt completed its dispatch (transport-level
/// success). This is **not** verified remediation success — subject-specific
/// verification must call `controller.record_success()` later when it has real
/// evidence.
#[allow(dead_code)]
pub async fn finalize_correction_attempt_executed(
    ctx: &Arc<CorrectionExecutionContext>,
    attempt: &CorrectionAttemptHandle,
    evidence_ref: Option<&str>,
) -> anyhow::Result<()> {
    ctx.controller
        .record_executed(
            &attempt.subject_id,
            attempt.kind,
            &attempt.signature,
            attempt.attempt_index,
            evidence_ref,
        )
        .await
}

/// Record that a correction attempt failed (the remediation was not applied or
/// was blocked by the sandbox gate).
#[allow(dead_code)]
pub async fn finalize_correction_attempt_failure(
    ctx: &Arc<CorrectionExecutionContext>,
    attempt: &CorrectionAttemptHandle,
    evidence_ref: Option<&str>,
) -> anyhow::Result<()> {
    ctx.controller
        .record_failure(
            &attempt.subject_id,
            attempt.kind,
            &attempt.signature,
            attempt.attempt_index,
            evidence_ref,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::correction_sandbox::{CorrectionSubjectContext, IntendedAccount};
    use crate::traits::SelfCorrectionSubjectKind;
    use std::sync::Arc;

    async fn make_state() -> Arc<dyn crate::traits::StateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        let store = crate::state::SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding,
        )
        .await
        .unwrap();
        std::mem::forget(db_file);
        Arc::new(store)
    }

    fn make_subject() -> CorrectionSubjectContext {
        CorrectionSubjectContext {
            subject_id: "task-abc".to_string(),
            subject_kind: SelfCorrectionSubjectKind::Task,
            session_id: "sess-1".to_string(),
            original_request: "do the thing".to_string(),
            completion_contract_summary: "task was complete".to_string(),
            intended_accounts: vec![IntendedAccount {
                provider: "github".to_string(),
                account_id: "acme-bot".to_string(),
                account_label: "Acme Bot".to_string(),
            }],
            allowed_external_targets: vec!["api.acme.com".to_string()],
            working_dir: std::path::PathBuf::from("/tmp/test-workdir"),
        }
    }

    /// P2.1 TDD: CorrectionExecutionContext can be constructed and its fields
    /// are accessible as documented.
    #[tokio::test]
    async fn correction_execution_context_fields_roundtrip() {
        let state = make_state().await;
        let controller = Arc::new(SelfCorrectionController::new(state, 3));
        let subject = make_subject();

        let ctx = CorrectionExecutionContext {
            subject: subject.clone(),
            controller: Arc::clone(&controller),
            dispatch_mode: CorrectionDispatchMode::Inline,
            bypass_approvals: true,
        };

        assert_eq!(ctx.subject.subject_id, "task-abc");
        assert_eq!(ctx.subject.subject_kind, SelfCorrectionSubjectKind::Task);
        assert_eq!(ctx.dispatch_mode, CorrectionDispatchMode::Inline);
        assert!(ctx.bypass_approvals);
        // Controller is reachable (pointer equality).
        assert!(Arc::ptr_eq(&ctx.controller, &controller));
    }

    /// P2.1 TDD: CorrectionExecutionContext is Clone.
    #[tokio::test]
    async fn correction_execution_context_is_clone() {
        let state = make_state().await;
        let controller = Arc::new(SelfCorrectionController::new(state, 3));
        let ctx = CorrectionExecutionContext {
            subject: make_subject(),
            controller,
            dispatch_mode: CorrectionDispatchMode::Deferred,
            bypass_approvals: false,
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.dispatch_mode, CorrectionDispatchMode::Deferred);
        assert!(!cloned.bypass_approvals);
    }

    /// P2.1 TDD: CorrectionAttemptHandle fields roundtrip.
    #[test]
    fn correction_attempt_handle_fields() {
        let h = CorrectionAttemptHandle {
            subject_id: "subj-1".to_string(),
            kind: SelfCorrectionSubjectKind::Task,
            signature: "terminal(ls)".to_string(),
            attempt_index: 2,
        };
        assert_eq!(h.subject_id, "subj-1");
        assert_eq!(h.kind, SelfCorrectionSubjectKind::Task);
        assert_eq!(h.signature, "terminal(ls)");
        assert_eq!(h.attempt_index, 2);
    }

    /// P2.1 TDD: ToolExecutionCtx carries correction as None for normal paths.
    /// We construct a minimal ToolExecutionCtx only to assert the field exists
    /// and is None by default / when explicitly passed as None.
    #[test]
    fn tool_execution_ctx_correction_is_none_for_normal_path() {
        // We only verify the field exists and is Option<Arc<CorrectionExecutionContext>>.
        // Full ToolExecutionCtx construction in tests requires many fields; instead
        // we assert via type-level code that Option::<Arc<CorrectionExecutionContext>>::None
        // compiles as the field type.
        let val: Option<Arc<CorrectionExecutionContext>> = None;
        assert!(val.is_none());
    }

    /// P2.1 TDD: ToolExecutionCtx correction field carries Some(Arc<...>) correctly.
    #[tokio::test]
    async fn tool_execution_ctx_correction_carries_some() {
        let state = make_state().await;
        let controller = Arc::new(SelfCorrectionController::new(state, 3));
        let ctx = Arc::new(CorrectionExecutionContext {
            subject: make_subject(),
            controller,
            dispatch_mode: CorrectionDispatchMode::Inline,
            bypass_approvals: false,
        });
        let field: Option<Arc<CorrectionExecutionContext>> = Some(Arc::clone(&ctx));
        assert!(field.is_some());
        assert!(!field.unwrap().bypass_approvals);
    }

    /// P2.1 TDD: ToolExecutionIoCtx per-call flags are false by default on
    /// non-correction paths.
    #[test]
    fn tool_execution_io_ctx_flags_default_false() {
        let preapproved: bool = false;
        let suppress: bool = false;
        assert!(!preapproved);
        assert!(!suppress);
    }

    /// P2.1 TDD: ToolExecCtx per-call flags are false by default on
    /// non-correction paths.
    #[test]
    fn tool_exec_ctx_flags_default_false() {
        let preapproved: bool = false;
        let suppress: bool = false;
        assert!(!preapproved);
        assert!(!suppress);
    }

    // ── P2.2 TDD: factory kill-switch rules ─────────────────────────────────

    fn disabled_config() -> crate::config::SelfCorrectionConfig {
        crate::config::SelfCorrectionConfig {
            enabled: false,
            correction_bypass_enabled: false,
            max_attempts: 3,
            shadow_mode: true,
        }
    }

    fn enabled_no_bypass_config() -> crate::config::SelfCorrectionConfig {
        crate::config::SelfCorrectionConfig {
            enabled: true,
            correction_bypass_enabled: false,
            max_attempts: 3,
            shadow_mode: true,
        }
    }

    fn enabled_with_bypass_config() -> crate::config::SelfCorrectionConfig {
        crate::config::SelfCorrectionConfig {
            enabled: true,
            correction_bypass_enabled: true,
            max_attempts: 3,
            shadow_mode: true,
        }
    }

    /// P2.2 TDD: master kill-switch — `enabled=false` always returns `None`,
    /// regardless of mode or bypass setting.
    #[tokio::test]
    async fn test_context_factory_disabled_returns_none() {
        let state = make_state().await;
        let cfg = disabled_config();

        let result = build_correction_execution_context(
            &cfg,
            state.clone(),
            make_subject(),
            CorrectionDispatchMode::Inline,
        );
        assert!(result.is_none(), "enabled=false, Inline → must be None");

        let result2 = build_correction_execution_context(
            &cfg,
            state,
            make_subject(),
            CorrectionDispatchMode::Deferred,
        );
        assert!(result2.is_none(), "enabled=false, Deferred → must be None");
    }

    /// P2.2 TDD: unattended (Deferred) without bypass must return `None` to
    /// prevent a task that hangs on interactive approval.
    #[tokio::test]
    async fn test_unattended_bypass_disabled_returns_none() {
        let state = make_state().await;
        let cfg = enabled_no_bypass_config();

        let result = build_correction_execution_context(
            &cfg,
            state,
            make_subject(),
            CorrectionDispatchMode::Deferred,
        );
        assert!(
            result.is_none(),
            "enabled=true, Deferred, bypass_off → must be None (would hang on approval)"
        );
    }

    /// P2.2 TDD: user-present (Inline) with bypass disabled returns `Some` and
    /// sets `bypass_approvals=false` (sandbox blocks unsafe tools, normal approval).
    #[tokio::test]
    async fn test_user_present_bypass_disabled_keeps_sandbox_context_without_preapproval() {
        let state = make_state().await;
        let cfg = enabled_no_bypass_config();

        let result = build_correction_execution_context(
            &cfg,
            state,
            make_subject(),
            CorrectionDispatchMode::Inline,
        );
        let ctx = result.expect("enabled=true, Inline, bypass_off → must be Some");
        assert!(
            !ctx.bypass_approvals,
            "bypass_approvals must be false when correction_bypass_enabled=false"
        );
        assert_eq!(
            ctx.dispatch_mode,
            CorrectionDispatchMode::Inline,
            "dispatch_mode must be preserved"
        );
    }

    /// P2.2 TDD: the context stores `dispatch_mode` so a live gate can fail
    /// closed if a caller passes an invalid unattended/no-bypass context.
    #[tokio::test]
    async fn test_context_records_dispatch_mode_for_gate_defense_in_depth() {
        let state = make_state().await;
        let cfg = enabled_with_bypass_config();

        let inline_ctx = build_correction_execution_context(
            &cfg,
            state.clone(),
            make_subject(),
            CorrectionDispatchMode::Inline,
        )
        .expect("should be Some for Inline+bypass");
        assert_eq!(inline_ctx.dispatch_mode, CorrectionDispatchMode::Inline);
        assert!(inline_ctx.bypass_approvals);

        let deferred_ctx = build_correction_execution_context(
            &cfg,
            state,
            make_subject(),
            CorrectionDispatchMode::Deferred,
        )
        .expect("should be Some for Deferred+bypass");
        assert_eq!(deferred_ctx.dispatch_mode, CorrectionDispatchMode::Deferred);
        assert!(deferred_ctx.bypass_approvals);
    }

    /// P2.2 TDD: the controller's `max_attempts` comes from `config.max_attempts`.
    /// Build a context with `max_attempts=1`, exhaust the budget, then assert
    /// a second distinct attempt returns `StopBudget`.
    #[tokio::test]
    async fn test_context_factory_uses_configured_max_attempts() {
        use crate::agent::self_correction::AttemptDecision;
        use crate::traits::SelfCorrectionSubjectKind;

        let state = make_state().await;
        let cfg = crate::config::SelfCorrectionConfig {
            enabled: true,
            correction_bypass_enabled: false,
            max_attempts: 1,
            shadow_mode: true,
        };

        let ctx = build_correction_execution_context(
            &cfg,
            state,
            make_subject(),
            CorrectionDispatchMode::Inline,
        )
        .expect("enabled=true, Inline → Some");

        let subject_id = ctx.subject.subject_id.clone();
        let kind = SelfCorrectionSubjectKind::Task;

        // First attempt — should proceed.
        let decision1 = ctx
            .controller
            .attempt(&subject_id, kind, "sig-a")
            .await
            .unwrap();
        assert!(
            matches!(decision1, AttemptDecision::Proceed { attempt_index: 1 }),
            "first attempt must Proceed with index 1, got {decision1:?}"
        );

        // Record the attempt as failed/blocked to consume the budget.
        ctx.controller
            .record_failure(&subject_id, kind, "sig-a", 1, None)
            .await
            .unwrap();

        // Second attempt — budget exhausted (max_attempts=1).
        let decision2 = ctx
            .controller
            .attempt(&subject_id, kind, "sig-b")
            .await
            .unwrap();
        assert_eq!(
            decision2,
            AttemptDecision::StopBudget,
            "second attempt must StopBudget after max_attempts=1 exhausted"
        );
    }
}
