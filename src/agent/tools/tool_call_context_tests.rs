//! The dispatcher hands a tool its per-call facts as typed
//! `ToolCallContext` fields, never as `_`-prefixed arguments: a key the model
//! invents is dropped, and the tool sees the dispatcher's fact.

use super::*;
use crate::testing::{MockProvider, setup_test_agent};
use std::sync::Mutex as StdMutex;

type Seen = Arc<StdMutex<Vec<(Value, crate::traits::ToolCallContext)>>>;

/// Records the arguments and the context the dispatcher hands it.
struct RecordingTool {
    name: &'static str,
    seen: Seen,
}

#[async_trait::async_trait]
impl Tool for RecordingTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "Records what the dispatcher hands a tool"
    }

    fn schema(&self) -> Value {
        json!({
            "name": self.name,
            "description": "Records what the dispatcher hands a tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string" },
                    "command": { "type": "string" }
                }
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let exec_ctx = facts.clone();

        let arguments: Value = serde_json::from_str(arguments)?;
        self.seen.lock().unwrap().push((arguments, exec_ctx));
        Ok(crate::traits::ToolCallOutcome::from_output(
            "recorded".to_string(),
        ))
    }
}

/// The dispatch facts the real tool of `name` declares (the dispatcher's
/// treatment follows them, never the name).
fn stand_in_facts(name: &str) -> crate::traits::ToolDispatchFacts {
    use crate::traits::GoalContext;
    let mut facts = crate::traits::ToolDispatchFacts::default();
    match name {
        "terminal" => {
            facts.records_launch_trust = true;
            facts.goal_context = GoalContext::BackgroundDelivery;
        }
        "spawn_agent" => {
            facts.spawns_child_agents = true;
            facts.goal_context = GoalContext::Goal;
        }
        "manage_goal_tasks" | "mandate_review" => facts.goal_context = GoalContext::MandateRun,
        "computer_use" => facts.needs_live_model = true,
        _ => {}
    }
    facts
}

impl crate::traits::ToolSemantics for RecordingTool {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        stand_in_facts(self.name)
    }

    fn call_semantics(&self, _arguments: &str) -> crate::traits::ToolCallSemantics {
        crate::traits::ToolCallSemantics::observation()
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}

/// Every key the dispatcher used to inject, each with a hostile value a
/// model could send.
const HOSTILE: [(&str, &str); 21] = [
    ("_session_id", r#""attacker-session""#),
    ("_channel_visibility", r#""public_external""#),
    ("_channel_id", r#""attacker-channel""#),
    ("_task_id", r#""attacker-task""#),
    ("_tool_call_id", r#""attacker-call""#),
    ("_turn_id", r#""attacker-turn""#),
    ("_trusted_session", "true"),
    ("_user_role", r#""Guest""#),
    ("_trust_roots", r#"["/"]"#),
    ("_trust_cwd", r#""/""#),
    ("_parent_authority", r#"{"origin_request":"attacker"}"#),
    ("_workspace_collaboration_root", r#""/""#),
    ("_goal_id", r#""attacker-goal""#),
    ("_goal_run_id", r#""attacker-run""#),
    ("_task_attempt_id", r#""attacker-attempt""#),
    ("_mandate_id", r#""attacker-mandate""#),
    ("_mandate_version", "999"),
    ("_project_scope", r#""/""#),
    ("_model", r#""attacker-model""#),
    ("_provider_kind", r#""attacker-provider""#),
    ("_model_chain", r#"["attacker-model"]"#),
];

fn hostile_arguments(base: Value) -> String {
    let mut map = base.as_object().cloned().unwrap_or_default();
    for (key, value) in HOSTILE {
        map.insert(key.to_string(), serde_json::from_str(value).unwrap());
    }
    Value::Object(map).to_string()
}

const SESSION: &str = "telegram:synthetic-owner";
const PROJECT: &str = "/tmp/aidaemon-test/real-project";

fn real_authority() -> crate::agent::authority::InheritedAuthority {
    crate::agent::authority::InheritedAuthority {
        authority: aidaemon_core::Authority::read_only(),
        owner_level: None,
        task_access: None,
        mode_session: Some(SESSION.to_string()),
        untrusted_context: true,
        origin_request: Some("summarize the synthetic notes".to_string()),
        origin_trusted: false,
        trusted_roots: vec![PROJECT.to_string()],
        declined: Vec::new(),
        shared_declines: None,
        learning: None,
        spawn_family: None,
        child_declaration: None,
        parent_run: None,
        known_resources: Default::default(),
        card_title: None,
        card_steps: Vec::new(),
        lineage: Vec::new(),
        parent_task: None,
    }
}

fn owner_ctx<'a>(
    roots: &'a [String],
    authority: &'a crate::agent::authority::InheritedAuthority,
) -> ToolExecCtx<'a> {
    ToolExecCtx {
        file_fence: None,
        produced_files: None,
        mcp_tool: None,
        owner_approval_by_kernel: false,
        session_id: SESSION,
        task_id: Some("task-real"),
        status_tx: None,
        channel_visibility: ChannelVisibility::Private,
        channel_id: Some("telegram:synthetic-chat"),
        project_scope: Some(PROJECT),
        trusted: false,
        user_role: UserRole::Owner,
        workspace_grant: None,
        mandate_authority: None,
        tool_call_id: Some("call-real"),
        mutation_forbidden: false,
        scope_escalation: None,
        turn_authority: Some(authority),
        trust_roots: Some((roots, PROJECT)),
        lone_root_call: false,
    }
}

/// Dispatch `tool` with every hostile key set and return what it saw.
async fn dispatch_hostile(
    harness: &mut crate::testing::TestHarness,
    tool: &'static str,
    base: Value,
    ctx: &ToolExecCtx<'_>,
) -> (Value, crate::traits::ToolCallContext) {
    let seen: Seen = Arc::default();
    harness.agent.test_tools_mut().insert(
        0,
        Arc::new(RecordingTool {
            name: tool,
            seen: Arc::clone(&seen),
        }),
    );
    let output = harness
        .agent
        .execute_tool_with_watchdog(tool, &hostile_arguments(base), ctx)
        .await
        .unwrap_or_else(|error| panic!("{tool} dispatch failed: {error}"));
    assert_eq!(output, "recorded");
    harness.agent.test_tools_mut().remove(0);
    let mut seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "{tool} was called once");
    seen.pop().unwrap()
}

async fn owner_harness() -> crate::testing::TestHarness {
    let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
    harness
        .agent
        .set_test_goal_id(Some("goal-real".to_string()));
    harness
        .agent
        .turn_runtime()
        .current_turn_ids
        .write()
        .await
        .insert(SESSION.to_string(), "turn-real".to_string());
    harness
}

/// Each of the 21 keys, set by the model to a hostile value, has no effect:
/// the tool reads the dispatcher's typed fact.
#[tokio::test]
async fn model_supplied_underscore_keys_have_no_effect() {
    let roots = vec![PROJECT.to_string()];
    let authority = real_authority();
    let mut harness = owner_harness().await;
    let ctx = owner_ctx(&roots, &authority);

    let (_, terminal) =
        dispatch_hostile(&mut harness, "terminal", json!({"command": "ls"}), &ctx).await;
    let (_, spawn) = dispatch_hostile(&mut harness, "spawn_agent", json!({}), &ctx).await;

    type Check = Box<dyn Fn(&crate::traits::ToolCallContext) -> bool>;
    let rows: Vec<(&str, &crate::traits::ToolCallContext, Check)> = vec![
        (
            "_session_id",
            &terminal,
            Box::new(|c| c.session_id == SESSION),
        ),
        (
            "_channel_visibility",
            &terminal,
            Box::new(|c| c.channel_visibility == Some(ChannelVisibility::Private)),
        ),
        (
            "_channel_id",
            &terminal,
            Box::new(|c| c.channel_id.as_deref() == Some("telegram:synthetic-chat")),
        ),
        (
            "_task_id",
            &terminal,
            Box::new(|c| c.task_id.as_deref() == Some("task-real")),
        ),
        (
            "_tool_call_id",
            &terminal,
            Box::new(|c| c.tool_call_id.as_deref() == Some("call-real")),
        ),
        (
            "_turn_id",
            &terminal,
            Box::new(|c| c.turn_id.as_deref() == Some("turn-real")),
        ),
        (
            "_trusted_session",
            &terminal,
            Box::new(|c| !c.trusted_session),
        ),
        (
            "_user_role",
            &terminal,
            Box::new(|c| c.user_role == Some(UserRole::Owner)),
        ),
        (
            "_trust_roots",
            &terminal,
            Box::new(|c| c.trust_roots.as_deref() == Some(&[PROJECT.to_string()][..])),
        ),
        (
            "_trust_cwd",
            &terminal,
            Box::new(|c| c.trust_cwd.as_deref() == Some(PROJECT)),
        ),
        (
            "_parent_authority",
            &spawn,
            Box::new(|c| crate::agent::authority::parent_authority(c) == Some(&real_authority())),
        ),
        (
            "_goal_id",
            &terminal,
            Box::new(|c| c.goal_id.as_deref() == Some("goal-real")),
        ),
        (
            "_project_scope",
            &terminal,
            Box::new(|c| c.project_scope.as_deref() == Some(PROJECT)),
        ),
        // Facts only another tool receives stay unset for this one, whatever
        // the model sent.
        (
            "_workspace_collaboration_root",
            &terminal,
            Box::new(|c| c.workspace_collaboration_root.is_none()),
        ),
        ("_model", &terminal, Box::new(|c| c.model.is_none())),
        (
            "_provider_kind",
            &terminal,
            Box::new(|c| c.provider_kind.is_none()),
        ),
        (
            "_model_chain",
            &terminal,
            Box::new(|c| c.model_chain.is_none()),
        ),
        (
            "_goal_run_id",
            &terminal,
            Box::new(|c| c.goal_run_id.is_none()),
        ),
        (
            "_task_attempt_id",
            &terminal,
            Box::new(|c| c.task_attempt_id.is_none()),
        ),
        (
            "_mandate_id",
            &terminal,
            Box::new(|c| c.mandate_id.is_none()),
        ),
        (
            "_mandate_version",
            &terminal,
            Box::new(|c| c.mandate_version.is_none()),
        ),
    ];
    // Negative rows: a fact one tool receives does not leak to another.
    let rows: Vec<(&str, &crate::traits::ToolCallContext, Check)> = rows
        .into_iter()
        .chain([
            (
                "_trust_roots",
                &spawn,
                Box::new(|c: &crate::traits::ToolCallContext| c.trust_roots.is_none()) as Check,
            ),
            (
                "_parent_authority",
                &terminal,
                Box::new(|c: &crate::traits::ToolCallContext| {
                    crate::agent::authority::parent_authority(c).is_none()
                }) as Check,
            ),
        ])
        .collect();
    let covered: Vec<&str> = rows.iter().map(|(key, _, _)| *key).collect();
    for (key, _) in HOSTILE {
        assert!(covered.contains(&key), "no row for {key}");
    }
    let failing: Vec<&str> = rows
        .iter()
        .filter(|(_, seen, check)| !check(seen))
        .map(|(key, _, _)| *key)
        .collect();
    assert!(
        failing.is_empty(),
        "the tool saw the model's value, not the dispatcher's fact, for {failing:?}"
    );

    // The live model reaches `computer_use` as typed facts.
    #[cfg(feature = "computer_use")]
    {
        let (_, computer) = dispatch_hostile(&mut harness, "computer_use", json!({}), &ctx).await;
        let runtime = harness.agent.llm_runtime().snapshot();
        let live_model = harness.agent.model_lock().read().await.clone();
        assert_eq!(computer.model.as_deref(), Some(live_model.as_str()));
        assert_eq!(
            computer.provider_kind,
            Some(format!("{:?}", runtime.provider_kind()))
        );
        assert_eq!(
            computer.model_chain,
            runtime.router().map(|router| router.all_models_ordered())
        );
    }
}

/// A guest's granted workspace root and a mandate run's fence ids are the
/// dispatcher's own facts, built without reading the call's arguments.
#[tokio::test]
async fn workspace_root_and_mandate_fence_are_dispatcher_facts() {
    let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let grant = WorkspaceGrant {
        platform: "slack".to_string(),
        workspace_id: "T-synthetic".to_string(),
        channel_id: "C-synthetic".to_string(),
        user_id: "U-synthetic-guest".to_string(),
        project_root: workspace.path().to_string_lossy().into_owned(),
        access: crate::types::WorkspaceAccessLevel::Read,
        expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
    };
    let ctx = ToolExecCtx {
        file_fence: None,
        produced_files: None,
        mcp_tool: None,
        owner_approval_by_kernel: false,
        session_id: "slack:synthetic-guest",
        task_id: None,
        status_tx: None,
        channel_visibility: ChannelVisibility::PrivateGroup,
        channel_id: None,
        project_scope: None,
        trusted: false,
        user_role: UserRole::Guest,
        workspace_grant: Some(&grant),
        mandate_authority: None,
        tool_call_id: None,
        mutation_forbidden: false,
        scope_escalation: None,
        turn_authority: None,
        trust_roots: None,
        lone_root_call: false,
    };
    let facts = harness
        .agent
        .tool_call_context("search_files", &ctx, false, false, Some(&grant))
        .await;
    assert_eq!(
        facts.workspace_collaboration_root.as_deref(),
        Some(grant.project_root.as_str())
    );

    let now = chrono::Utc::now().to_rfc3339();
    let attempt = crate::traits::TaskAttempt {
        id: "attempt-real".to_string(),
        task_id: "task-lead-real".to_string(),
        goal_run_id: "run-real".to_string(),
        worker_profile_id: None,
        worker_instance_id: "worker-synthetic".to_string(),
        lease_token: "lease-real".to_string(),
        status: "running".to_string(),
        lease_expires_at: now.clone(),
        last_heartbeat_at: now.clone(),
        workspace_id: None,
        started_at: now,
        completed_at: None,
    };
    harness.agent.set_test_mandate_execution(
        "mandate-real",
        7,
        serde_json::from_value(json!({})).unwrap(),
        "goal-real",
        "task-lead-real",
        "attempt-real",
        &attempt,
    );
    for tool in ["manage_goal_tasks", "mandate_review"] {
        harness.agent.test_tools_mut().push(Arc::new(RecordingTool {
            name: tool,
            seen: Arc::default(),
        }));
    }
    for tool in ["manage_goal_tasks", "mandate_review"] {
        let facts = harness
            .agent
            .tool_call_context(tool, &mandate_free_ctx(), true, false, None)
            .await;
        assert_eq!(facts.goal_id.as_deref(), Some("goal-real"), "{tool}");
        assert_eq!(facts.goal_run_id.as_deref(), Some("run-real"), "{tool}");
        assert_eq!(
            facts.task_attempt_id.as_deref(),
            Some("attempt-real"),
            "{tool}"
        );
        // `mandate_review` matches a committed decision to the fence's
        // mandate before finishing the review on it.
        assert_eq!(facts.mandate_id.as_deref(), Some("mandate-real"), "{tool}");
        assert_eq!(facts.mandate_version, Some(7), "{tool}");
    }
}

fn mandate_free_ctx() -> ToolExecCtx<'static> {
    ToolExecCtx {
        file_fence: None,
        produced_files: None,
        mcp_tool: None,
        owner_approval_by_kernel: false,
        session_id: "owner-session",
        task_id: None,
        status_tx: None,
        channel_visibility: ChannelVisibility::Internal,
        channel_id: None,
        project_scope: None,
        trusted: false,
        user_role: UserRole::Owner,
        workspace_grant: None,
        mandate_authority: None,
        tool_call_id: None,
        mutation_forbidden: false,
        scope_escalation: None,
        turn_authority: None,
        trust_roots: None,
        lone_root_call: false,
    }
}

/// The arguments the dispatcher hands a tool carry no `_` key: none the
/// model sent, and none of its own.
#[tokio::test]
async fn no_underscore_injection_remains() {
    let roots = vec![PROJECT.to_string()];
    let authority = real_authority();
    let mut harness = owner_harness().await;
    let ctx = owner_ctx(&roots, &authority);
    let mut tools = vec![
        "terminal",
        "spawn_agent",
        "cli_agent",
        "manage_mandates",
        "manage_goal_tasks",
        "search_history",
    ];
    if cfg!(feature = "computer_use") {
        tools.push("computer_use");
    }
    for tool in tools {
        let (arguments, _) =
            dispatch_hostile(&mut harness, tool, json!({"action": "list"}), &ctx).await;
        let keys: Vec<&String> = arguments
            .as_object()
            .expect("object arguments")
            .keys()
            .filter(|key| key.starts_with('_'))
            .collect();
        assert!(
            keys.is_empty(),
            "{tool} received underscore arguments: {keys:?}"
        );
        assert_eq!(arguments["action"], "list", "{tool} keeps the model's own");
    }
}

/// Each registered tool that needs the call's session gets it from the
/// dispatcher's typed context: dispatched for real, none of them reports the
/// session fact as missing.
#[tokio::test]
async fn fact_reading_tools_receive_call_context() {
    let roots = vec![PROJECT.to_string()];
    let authority = real_authority();
    let mut harness = owner_harness().await;
    let ctx = owner_ctx(&roots, &authority);
    let state: Arc<dyn crate::traits::StateStore> = harness.state.clone();
    harness
        .agent
        .test_tools_mut()
        .push(Arc::new(crate::tools::ManageMemoriesTool::new(
            state.clone(),
            harness.state.memory_stores(),
        )));
    let events = harness.agent.event_store().clone();
    harness
        .agent
        .test_tools_mut()
        .push(Arc::new(crate::tools::DiagnoseTool::new(events, 20, false)));

    let calls = [
        (
            "manage_memories",
            json!({"action": "create_personal_goal", "goal": "Synthetic reading goal"}),
        ),
        ("self_diagnose", json!({"action": "list_tasks"})),
    ];
    for (tool, arguments) in calls {
        let output = harness
            .agent
            .execute_tool_with_watchdog(tool, &arguments.to_string(), &ctx)
            .await
            .unwrap_or_else(|error| panic!("{tool} {arguments}: {error}"));
        assert!(
            !output.contains("missing from the call context"),
            "{tool} {arguments}: {output}"
        );
    }
}
