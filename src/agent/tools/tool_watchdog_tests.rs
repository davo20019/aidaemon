use super::*;
use crate::testing::{setup_test_agent, MockProvider};
use crate::traits::store_prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

struct SlowTool;

#[async_trait::async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &str {
        "slow_tool"
    }

    fn description(&self) -> &str {
        "Sleeps before returning"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "slow_tool",
            "description": "Sleeps before returning",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        tokio::time::sleep(Duration::from_millis(150)).await;
        Ok("done".to_string())
    }
}

struct SlowCliAgentTool;

#[async_trait::async_trait]
impl Tool for SlowCliAgentTool {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "Simulates a long-running cli_agent tool call"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "cli_agent",
            "description": "Simulates a long-running cli_agent tool call",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        })
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        tokio::time::sleep(Duration::from_millis(150)).await;
        Ok("ok".to_string())
    }
}

struct EchoSpawnAgentTool;

struct MandateMutationSpy {
    calls: Arc<AtomicUsize>,
    saw_mandate_preapproval: Arc<AtomicBool>,
}

struct MandateTestFence {
    root_task_id: String,
    root_attempt: crate::traits::TaskAttempt,
    worker_task_id: String,
    worker_attempt: crate::traits::TaskAttempt,
}

#[async_trait::async_trait]
impl Tool for MandateMutationSpy {
    fn name(&self) -> &str {
        "mandate_mutation_spy"
    }

    fn description(&self) -> &str {
        "Records an authorized synthetic remote mutation"
    }

    fn schema(&self) -> Value {
        json!({
            "name": self.name(),
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
                "additionalProperties": false
            }
        })
    }

    fn call_semantics(&self, _arguments: &str) -> crate::traits::ToolCallSemantics {
        crate::traits::ToolCallSemantics::mutation_with(
            crate::traits::ToolMutationEffects::REMOTE_MUTATION,
        )
        .with_target_hint(
            crate::traits::ToolTargetHintKind::Url,
            "https://api.x.com/2/tweets",
        )
    }

    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        self.calls.fetch_add(1, AtomicOrdering::SeqCst);
        Ok("mutated".to_string())
    }

    async fn call_with_execution_context(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
        exec_ctx: crate::traits::ToolExecutionContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        self.saw_mandate_preapproval
            .store(exec_ctx.mandate_preapproved, AtomicOrdering::SeqCst);
        self.call(arguments)
            .await
            .map(crate::traits::ToolCallOutcome::from_output)
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }
}

async fn install_active_test_mandate(
    harness: &mut crate::testing::TestHarness,
) -> (
    crate::traits::Mandate,
    crate::traits::MandateDecisionCycle,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
    crate::traits::GoalRun,
    MandateTestFence,
) {
    use crate::traits::{
        Goal, Intention, Mandate, MandateAuthority, MandateDecisionCycle, MandateDecisionOutcome,
        Task,
    };

    let goal = Goal::new_continuous("Synthetic mandate", "owner-session", None, None);
    let authority = MandateAuthority {
        allow_observations: true,
        allowed_tools: vec!["mandate_mutation_spy".to_string()],
        allowed_mutation_effects: vec!["remote_mutation".to_string()],
        allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
        operation_scopes: Vec::new(),
        max_mutating_actions_per_cycle: 1,
        max_mutating_actions_per_rolling_24h: 8,
        min_seconds_between_mutations: 900,
    };
    let mut mandate = Mandate::new(
        &goal.id,
        None,
        "Perform one synthetic bounded mutation",
        "owner-session",
        authority,
        60,
        3_600,
        300,
    );
    mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
    harness
        .state
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    assert_eq!(
        harness
            .state
            .claim_due_mandates(1, "test-heartbeat", 300)
            .await
            .unwrap()
            .len(),
        1
    );
    let root_task_id = uuid::Uuid::new_v4().to_string();
    let run = harness
        .state
        .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
        .await
        .unwrap();
    let now = chrono::Utc::now().to_rfc3339();
    harness
        .state
        .create_task(&Task {
            id: root_task_id.clone(),
            goal_id: goal.id.clone(),
            description: "Run one bounded mandate cycle".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now,
            started_at: None,
            completed_at: None,
        })
        .await
        .unwrap();
    let root_attempt = harness
        .state
        .claim_task_with_lease(
            &root_task_id,
            "mandate-test-task-lead",
            Some("profile-task-lead"),
            180,
        )
        .await
        .unwrap()
        .expect("mandate root should be claimable");
    let decision = MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        MandateDecisionOutcome::Act,
        "The synthetic action is within the test envelope",
        mandate.version,
    );
    let intention = Intention::new(
        &mandate.id,
        &decision.id,
        &run.id,
        "Perform the synthetic action once",
        "It verifies complete mediation",
    );
    harness
        .state
        .record_mandate_decision(&decision, Some(&intention), None)
        .await
        .unwrap();
    let worker_task_id = uuid::Uuid::new_v4().to_string();
    harness
        .state
        .create_mandate_task_from_attempt(
            &Task {
                id: worker_task_id.clone(),
                goal_id: goal.id.clone(),
                description: "Perform the synthetic governed mutation".to_string(),
                status: "pending".to_string(),
                priority: "high".to_string(),
                task_order: 1,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: None,
            },
            &mandate.id,
            mandate.version,
            &run.id,
            &root_attempt.id,
            8,
        )
        .await
        .unwrap();
    let worker_attempt = harness
        .state
        .claim_mandate_task_from_attempt(
            &worker_task_id,
            "mandate-test-executor",
            &mandate.id,
            mandate.version,
            &run.id,
            &root_attempt.id,
            180,
        )
        .await
        .unwrap()
        .expect("mandate worker should be claimable");
    harness.agent.set_test_mandate_execution(
        &mandate.id,
        mandate.version,
        mandate.authority.clone(),
        &goal.id,
        &root_task_id,
        &root_attempt.id,
        &worker_attempt,
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let saw_mandate_preapproval = Arc::new(AtomicBool::new(false));
    harness.agent.tools.push(Arc::new(MandateMutationSpy {
        calls: Arc::clone(&calls),
        saw_mandate_preapproval: Arc::clone(&saw_mandate_preapproval),
    }));
    (
        mandate,
        decision,
        calls,
        saw_mandate_preapproval,
        run,
        MandateTestFence {
            root_task_id,
            root_attempt,
            worker_task_id,
            worker_attempt,
        },
    )
}

fn test_mandate_grant(
    mandate: &crate::traits::Mandate,
    decision: &crate::traits::MandateDecisionCycle,
    arguments: &str,
) -> crate::traits::MandateAuthorityGrant {
    let semantics = crate::traits::ToolCallSemantics::mutation_with(
        crate::traits::ToolMutationEffects::REMOTE_MUTATION,
    )
    .with_target_hint(
        crate::traits::ToolTargetHintKind::Url,
        "https://api.x.com/2/tweets",
    );
    match crate::mandates::authority::authorize_mandate_action(
        mandate,
        decision,
        "mandate_mutation_spy",
        arguments,
        &semantics,
        &chrono::Utc::now(),
    ) {
        crate::mandates::authority::MandateAuthorityDecision::Allow(grant) => grant,
        crate::mandates::authority::MandateAuthorityDecision::Deny(reason) => {
            panic!("test action should be authorized: {}", reason.as_str())
        }
    }
}

fn test_mandate_reservation(
    grant: &crate::traits::MandateAuthorityGrant,
    run: &crate::traits::GoalRun,
    fence: &MandateTestFence,
    sequence: u64,
) -> crate::traits::MandateMutationReservation {
    crate::traits::MandateMutationReservation {
        grant: grant.clone(),
        goal_run_id: run.id.clone(),
        root_task_id: fence.root_task_id.clone(),
        root_task_attempt_id: fence.root_attempt.id.clone(),
        task_id: fence.worker_task_id.clone(),
        task_attempt_id: fence.worker_attempt.id.clone(),
        tool_call_id: format!("watchdog-mandate-call-{sequence}"),
        tool_name: "mandate_mutation_spy".to_string(),
        mutation_effects: vec!["remote_mutation".to_string()],
        targets: vec![crate::traits::MandateMutationTarget {
            kind: "url".to_string(),
            identifier: "https://api.x.com/2/tweets".to_string(),
        }],
        account_identifiers: Vec::new(),
        reserved_at: chrono::Utc::now().to_rfc3339(),
    }
}

fn mandate_tool_ctx<'a>(
    grant: Option<&'a crate::traits::MandateAuthorityGrant>,
) -> ToolExecCtx<'a> {
    ToolExecCtx {
        session_id: "owner-session",
        task_id: None,
        status_tx: None,
        channel_visibility: ChannelVisibility::Internal,
        channel_id: None,
        project_scope: None,
        trusted: false,
        user_role: UserRole::Owner,
        workspace_grant: None,
        correction_preapproved: false,
        suppress_trusted_session: false,
        mandate_authority: grant,
        tool_call_id: grant.and_then(|value| value.tool_call_id.as_deref()),
        mutation_forbidden: false,
    }
}

#[async_trait::async_trait]
impl Tool for EchoSpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Echoes enriched spawn_agent arguments for regression tests"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "spawn_agent",
            "description": "Echoes enriched spawn_agent arguments for regression tests",
            "parameters": {
                "type": "object",
                "properties": {
                    "mission": { "type": "string" },
                    "task": { "type": "string" }
                },
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(arguments.to_string())
    }
}

#[tokio::test]
async fn execute_tool_watchdog_times_out_slow_tool() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    harness.agent.tools.push(Arc::new(SlowTool));
    harness.agent.limits.llm_call_timeout = Some(Duration::from_millis(30));

    let result = harness
        .agent
        .execute_tool_with_watchdog_outcome(
            "slow_tool",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-1"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                workspace_grant: None,
                correction_preapproved: false,
                suppress_trusted_session: false,
                mandate_authority: None,
                tool_call_id: None,
                mutation_forbidden: false,
            },
        )
        .await
        .expect("watchdog timeout should be returned as a typed tool outcome");

    assert!(
        result.output.contains("timed out"),
        "timeout result expected, got: {}",
        result.output
    );
    assert!(result.metadata.timed_out);
    assert_eq!(
        result.metadata.outcome_status,
        Some(crate::traits::ToolOutcomeStatus::FailedRetryable)
    );
}

#[tokio::test]
async fn execute_tool_watchdog_skips_cli_agent() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    harness.agent.tools.push(Arc::new(SlowCliAgentTool));
    harness.agent.limits.llm_call_timeout = Some(Duration::from_millis(30));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "cli_agent",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-3"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                workspace_grant: None,
                correction_preapproved: false,
                suppress_trusted_session: false,
                mandate_authority: None,
                tool_call_id: None,
                mutation_forbidden: false,
            },
        )
        .await
        .expect("cli_agent should bypass watchdog");

    assert_eq!(result, "ok");
}

#[tokio::test]
async fn execute_tool_watchdog_allows_fast_tool() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    // system_info runs multiple subprocesses; allow a bit of slack to avoid
    // flakiness on slower/loaded machines.
    harness.agent.limits.llm_call_timeout = Some(Duration::from_secs(5));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "system_info",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-2"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                workspace_grant: None,
                correction_preapproved: false,
                suppress_trusted_session: false,
                mandate_authority: None,
                tool_call_id: None,
                mutation_forbidden: false,
            },
        )
        .await
        .expect("fast tool should succeed");

    assert!(
        !result.is_empty(),
        "system_info should return a non-empty payload"
    );
}

#[tokio::test]
async fn execute_tool_watchdog_injects_project_scope_and_causal_tool_call_id() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    harness.agent.tools.push(Arc::new(EchoSpawnAgentTool));

    let result = harness
        .agent
        .execute_tool_with_watchdog(
            "spawn_agent",
            r#"{"mission":"delegate log analysis","task":"inspect the logs","_project_scope":"/tmp/spoofed"}"#,
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-4"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: Some("/Users/davidloor/Library/Logs/aidaemon"),
                trusted: false,
                user_role: UserRole::Owner,
                workspace_grant: None,
                correction_preapproved: false,
                suppress_trusted_session: false,
                mandate_authority: None,
                tool_call_id: Some("call-parent-synthetic"),
                mutation_forbidden: false,
            },
        )
        .await
        .expect("spawn_agent call should succeed");

    let payload: Value = serde_json::from_str(&result).expect("spawn_agent args should be JSON");
    assert_eq!(
        payload.get("_project_scope").and_then(Value::as_str),
        Some("/Users/davidloor/Library/Logs/aidaemon")
    );
    assert_eq!(
        payload.get("mission").and_then(Value::as_str),
        Some("delegate log analysis")
    );
    assert_eq!(
        payload.get("_session_id").and_then(Value::as_str),
        Some("test-session")
    );
    assert_eq!(
        payload.get("_tool_call_id").and_then(Value::as_str),
        Some("call-parent-synthetic")
    );
}

#[tokio::test]
async fn mandate_dispatch_requires_an_exact_fenced_grant_and_rechecks_policy() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (mandate, decision, calls, saw_mandate_preapproval, _run, _fence) =
        install_active_test_mandate(&mut harness).await;
    let arguments = r#"{"text":"useful update"}"#;
    let semantics = harness
        .agent
        .tools
        .iter()
        .find(|tool| tool.name() == "mandate_mutation_spy")
        .unwrap()
        .call_semantics(arguments);

    let base_ctx = |grant| ToolExecCtx {
        session_id: "owner-session",
        task_id: None,
        status_tx: None,
        channel_visibility: ChannelVisibility::Internal,
        channel_id: None,
        project_scope: None,
        trusted: false,
        user_role: UserRole::Owner,
        workspace_grant: None,
        correction_preapproved: false,
        suppress_trusted_session: false,
        mandate_authority: grant,
        tool_call_id: grant.and_then(|value| value.tool_call_id.as_deref()),
        mutation_forbidden: false,
    };

    let executor_read = harness
        .agent
        .execute_tool_with_watchdog("system_info", "{}", &base_ctx(None))
        .await
        .unwrap_err();
    assert!(
        executor_read
            .to_string()
            .contains("not permitted for this mandate execution role"),
        "a fenced executor must not bypass the task-lead observation boundary: {executor_read}"
    );

    let missing = harness
        .agent
        .execute_tool_with_watchdog("mandate_mutation_spy", arguments, &base_ctx(None))
        .await
        .unwrap_err();
    assert!(missing
        .to_string()
        .contains("no action-bound mandate grant"));
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
    assert!(!saw_mandate_preapproval.load(AtomicOrdering::SeqCst));

    let mut grant = match crate::mandates::authority::authorize_mandate_action(
        &mandate,
        &decision,
        "mandate_mutation_spy",
        arguments,
        &semantics,
        &chrono::Utc::now(),
    ) {
        crate::mandates::authority::MandateAuthorityDecision::Allow(grant) => grant,
        crate::mandates::authority::MandateAuthorityDecision::Deny(reason) => {
            panic!("test action should be authorized: {}", reason.as_str())
        }
    };
    grant.tool_call_id = Some("watchdog-mandate-call-1".to_string());
    assert_eq!(grant.reserved_action_attempt, 1);

    let changed = harness
        .agent
        .execute_tool_with_watchdog(
            "mandate_mutation_spy",
            r#"{"text":"different update"}"#,
            &base_ctx(Some(&grant)),
        )
        .await
        .unwrap_err();
    assert!(changed.to_string().contains("grant_mismatch"));
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
    assert!(!saw_mandate_preapproval.load(AtomicOrdering::SeqCst));

    let result = harness
        .agent
        .execute_tool_with_watchdog("mandate_mutation_spy", arguments, &base_ctx(Some(&grant)))
        .await
        .unwrap();
    assert_eq!(result, "mutated");
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    assert!(
        saw_mandate_preapproval.load(AtomicOrdering::SeqCst),
        "the adapter must see mandate preapproval only after final exact-grant validation"
    );

    let replay = harness
        .agent
        .execute_tool_with_watchdog("mandate_mutation_spy", arguments, &base_ctx(Some(&grant)))
        .await
        .unwrap_err();
    assert!(
        replay.to_string().contains("candidate is invalid")
            || replay.to_string().contains("already dispatched")
    );
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);

    let mut revised = harness
        .state
        .get_mandate(&mandate.id)
        .await
        .unwrap()
        .unwrap();
    revised.version += 1;
    revised.constraints.push("new owner fence".to_string());
    harness.state.update_mandate(&revised).await.unwrap();
    let revoked = harness
        .agent
        .execute_tool_with_watchdog("mandate_mutation_spy", arguments, &base_ctx(Some(&grant)))
        .await
        .unwrap_err();
    assert!(
        revoked.to_string().contains("Mandate is no longer active"),
        "unexpected revocation error: {revoked}"
    );
    assert_eq!(
        harness
            .state
            .get_mandate(&mandate.id)
            .await
            .unwrap()
            .unwrap()
            .status,
        crate::traits::MandateStatus::AwaitingInput,
        "policy revision after a dispatch claim without a strict receipt must pause for reconciliation"
    );
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn mandate_task_lead_mutation_is_denied_without_burning_quota() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (mandate, decision, calls, _saw_mandate_preapproval, run, fence) =
        install_active_test_mandate(&mut harness).await;
    harness.agent.set_test_mandate_execution(
        &mandate.id,
        mandate.version,
        mandate.authority.clone(),
        &mandate.goal_id,
        &fence.root_task_id,
        &fence.root_attempt.id,
        &fence.root_attempt,
    );
    let arguments = r#"{"text":"task lead must delegate this"}"#;
    let mut grant = test_mandate_grant(&mandate, &decision, arguments);
    grant.tool_call_id = Some("watchdog-task-lead-mutation".to_string());

    let error = harness
        .agent
        .execute_tool_with_watchdog(
            "mandate_mutation_spy",
            arguments,
            &mandate_tool_ctx(Some(&grant)),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("task lead cannot perform"),
        "unexpected role-boundary error: {error}"
    );
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
    assert!(harness
        .state
        .list_mandate_mutation_attempts_for_run(&run.id)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        harness
            .state
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .unwrap()
            .action_attempts,
        0,
        "a task-lead mutation denial must happen before quota reservation"
    );
}

#[tokio::test]
async fn mandate_executor_cannot_use_task_lead_control_plane() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (_mandate, _decision, calls, _saw_mandate_preapproval, run, _fence) =
        install_active_test_mandate(&mut harness).await;

    for (tool, arguments) in [
        (
            "manage_mandates",
            r#"{"action":"record_decision","outcome":"wait","rationale":"escape"}"#,
        ),
        (
            "manage_goal_tasks",
            r#"{"action":"create_task","description":"escape"}"#,
        ),
        (
            "spawn_agent",
            r#"{"mission":"escape","task":"escape","background":false}"#,
        ),
    ] {
        let error = harness
            .agent
            .execute_tool_with_watchdog(tool, arguments, &mandate_tool_ctx(None))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not permitted for this mandate execution role"),
            "unexpected executor role-boundary error for {tool}: {error}"
        );
    }

    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
    assert!(harness
        .state
        .list_mandate_mutation_attempts_for_run(&run.id)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn mandate_pause_resume_cannot_resurrect_a_pre_pause_grant() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (mandate, decision, calls, _saw_mandate_preapproval, run, fence) =
        install_active_test_mandate(&mut harness).await;
    let arguments = r#"{"text":"must require fresh deliberation"}"#;
    let semantics = harness
        .agent
        .tools
        .iter()
        .find(|tool| tool.name() == "mandate_mutation_spy")
        .unwrap()
        .call_semantics(arguments);
    let mut grant = match crate::mandates::authority::authorize_mandate_action(
        &mandate,
        &decision,
        "mandate_mutation_spy",
        arguments,
        &semantics,
        &chrono::Utc::now(),
    ) {
        crate::mandates::authority::MandateAuthorityDecision::Allow(grant) => grant,
        crate::mandates::authority::MandateAuthorityDecision::Deny(reason) => {
            panic!("test action should be authorized: {}", reason.as_str())
        }
    };
    grant.tool_call_id = Some("watchdog-mandate-call-2".to_string());
    assert!(harness
        .state
        .reserve_mandate_action_attempt(&test_mandate_reservation(&grant, &run, &fence, 2))
        .await
        .unwrap()
        .is_some());

    assert!(harness
        .state
        .transition_mandate_status(
            &mandate.id,
            crate::traits::MandateStatus::Active,
            crate::traits::MandateStatus::Paused,
        )
        .await
        .unwrap());
    assert!(harness
        .state
        .transition_mandate_status(
            &mandate.id,
            crate::traits::MandateStatus::Paused,
            crate::traits::MandateStatus::Active,
        )
        .await
        .unwrap());

    let blocked = harness
        .agent
        .execute_tool_with_watchdog(
            "mandate_mutation_spy",
            arguments,
            &ToolExecCtx {
                session_id: "owner-session",
                task_id: None,
                status_tx: None,
                channel_visibility: ChannelVisibility::Internal,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                workspace_grant: None,
                correction_preapproved: false,
                suppress_trusted_session: false,
                mandate_authority: Some(&grant),
                tool_call_id: grant.tool_call_id.as_deref(),
                mutation_forbidden: false,
            },
        )
        .await
        .unwrap_err();
    assert!(blocked.to_string().to_ascii_lowercase().contains("mandate"));
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
}

#[tokio::test]
async fn mandate_dispatch_rejects_a_blocked_run_even_with_a_live_attempt_and_grant() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (mandate, decision, calls, _saw_mandate_preapproval, run, fence) =
        install_active_test_mandate(&mut harness).await;
    let arguments = r#"{"text":"must not run while blocked"}"#;
    let mut grant = test_mandate_grant(&mandate, &decision, arguments);
    grant.tool_call_id = Some("watchdog-mandate-call-3".to_string());
    assert!(harness
        .state
        .reserve_mandate_action_attempt(&test_mandate_reservation(&grant, &run, &fence, 3))
        .await
        .unwrap()
        .is_some());
    assert!(harness
        .state
        .finish_goal_run(&run.id, "blocked", Some("owner input required"))
        .await
        .unwrap());

    let error = harness
        .agent
        .execute_tool_with_watchdog(
            "mandate_mutation_spy",
            arguments,
            &mandate_tool_ctx(Some(&grant)),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("stale, blocked"));
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
}

#[tokio::test]
async fn stale_mandate_child_cannot_rebind_to_the_next_goal_run() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (mandate, decision, calls, _saw_mandate_preapproval, old_run, fence) =
        install_active_test_mandate(&mut harness).await;
    let arguments = r#"{"text":"belongs only to cycle n"}"#;
    let mut grant = test_mandate_grant(&mandate, &decision, arguments);
    grant.tool_call_id = Some("watchdog-mandate-call-4".to_string());
    assert!(harness
        .state
        .reserve_mandate_action_attempt(&test_mandate_reservation(&grant, &old_run, &fence, 4,))
        .await
        .unwrap()
        .is_some());
    assert!(harness
        .state
        .finish_goal_run(&old_run.id, "completed", Some("cycle n closed"))
        .await
        .unwrap());
    let next_root = uuid::Uuid::new_v4().to_string();
    let next_run = harness
        .state
        .start_goal_run(&mandate.goal_id, "mandate", None, Some(&next_root))
        .await
        .unwrap();
    assert_ne!(next_run.id, old_run.id);

    let error = harness
        .agent
        .execute_tool_with_watchdog(
            "mandate_mutation_spy",
            arguments,
            &mandate_tool_ctx(Some(&grant)),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("stale, blocked"));
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
}

#[tokio::test]
async fn mandate_dispatch_rejects_a_lost_root_task_attempt() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");
    let (mandate, decision, calls, _saw_mandate_preapproval, run, fence) =
        install_active_test_mandate(&mut harness).await;
    let arguments = r#"{"text":"must not outlive its lease"}"#;
    let mut grant = test_mandate_grant(&mandate, &decision, arguments);
    grant.tool_call_id = Some("watchdog-mandate-call-5".to_string());
    assert!(harness
        .state
        .reserve_mandate_action_attempt(&test_mandate_reservation(&grant, &run, &fence, 5))
        .await
        .unwrap()
        .is_some());
    let patch = crate::traits::TaskAttemptPatch {
        status: "blocked".to_string(),
        blocker: Some("lease owner stopped".to_string()),
        ..Default::default()
    };
    assert!(harness
        .state
        .patch_task_from_attempt(
            &fence.root_attempt.id,
            &fence.root_attempt.lease_token,
            &patch,
        )
        .await
        .unwrap());

    let error = harness
        .agent
        .execute_tool_with_watchdog(
            "mandate_mutation_spy",
            arguments,
            &mandate_tool_ctx(Some(&grant)),
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("stale, blocked")
            || error.to_string().contains("reservation is stale"),
        "unexpected root-attempt revocation error: {error}"
    );
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
}
