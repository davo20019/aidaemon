//! A coding agent under a device ceiling (a run an e-ink device started or
//! steered): launched checked whatever full access the owner gave it, its
//! own risky requests asked on the phone, and no note from the capped run
//! into work nothing judges.

use super::*;

/// The turn's judge as the dispatcher hands it to the call, with the
/// request's device ceiling raised or not.
fn turn_judge(capped: bool) -> Arc<crate::approvals::DetachedNestedJudge> {
    judge_with(capped, HashMap::new(), Vec::new(), Default::default())
}

fn judge_with(
    capped: bool,
    tools: HashMap<String, Arc<dyn aidaemon_core::Tool>>,
    hooks: Vec<Arc<dyn aidaemon_core::Hook>>,
    approvals: Arc<crate::approvals::OwnerApprovals>,
) -> Arc<crate::approvals::DetachedNestedJudge> {
    let rules = crate::approvals::AppApprovalRules::default();
    if capped {
        rules.device_cap.raise();
    }
    Arc::new(crate::approvals::DetachedNestedJudge::new(
        tools,
        hooks,
        approvals,
        Arc::new(rules),
        aidaemon_core::SharedDeclines::default(),
        aidaemon_core::RunId::new(),
        "app:synthetic-ceiling".into(),
        "app:synthetic-ceiling".into(),
        None,
        Arc::new(AtomicBool::new(false)),
        false,
    ))
}

/// A `claude` executable in a folder of its own, given full access by the
/// owner under a level that allows it.
#[cfg(unix)]
async fn full_access_claude() -> (CliAgentTool, tempfile::NamedTempFile, tempfile::TempDir) {
    use std::os::unix::fs::PermissionsExt;
    let (tool, db) = setup_bash_tool().await;
    let bin = tempfile::tempdir().unwrap();
    let claude = bin.path().join("claude");
    std::fs::write(&claude, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&claude, std::fs::Permissions::from_mode(0o755)).unwrap();
    tool.tools.write().unwrap().clear();
    tool.tools.write().unwrap().insert(
        "claude".to_string(),
        CliToolEntry {
            command: claude.display().to_string(),
            args: vec!["-p".to_string()],
            description: "Claude Code".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );
    let config: toml::Table = toml::from_str(
        "[authority]\nowner_default = \"full_access\"\n\
         [cli_agents.permissions]\nclaude = \"full_access\"\n",
    )
    .unwrap();
    tool.set_permissions(crate::permissions::Permissions::from_config(&config));
    (tool, db, bin)
}

#[cfg(unix)]
async fn prepare_claude(
    tool: &CliAgentTool,
    folder: &str,
    judge: Arc<crate::approvals::DetachedNestedJudge>,
) -> PreparedRun {
    let write = vec![folder.to_string()];
    tool.prepare_run(
        "claude",
        "ceil1234",
        "fix the build",
        "fix the build",
        Some(folder),
        "app:synthetic-ceiling",
        None,
        None,
        CliWorkspaceMode::ReadWrite,
        &[],
        &write,
        true,
        &crate::helper_run::HelperSpec::default(),
        None,
        None,
        Some(judge),
        None,
        false,
        false,
        true,
        false,
        &[],
        &[],
        None,
    )
    .await
    .unwrap()
    .unwrap_or_else(|why| panic!("the launch should be prepared: {why}"))
}

/// Capped, an agent the owner gave full access runs checked: no bypass
/// flag, its requests answered by the run (a host). Uncapped it keeps full
/// access, as before.
#[cfg(unix)]
#[tokio::test]
async fn a_capped_run_launches_a_full_access_agent_checked() {
    let (tool, _db, _bin) = full_access_claude().await;
    let folder = tempfile::tempdir().unwrap();
    let folder = folder.path().canonicalize().unwrap().display().to_string();

    let open = prepare_claude(&tool, &folder, turn_judge(false)).await;
    assert_eq!(
        open.mode,
        aidaemon_coding_agents::PermissionMode::FullAccess
    );
    assert!(
        open.args.iter().any(|arg| arg == "bypassPermissions"),
        "{:?}",
        open.args
    );

    let held = prepare_claude(&tool, &folder, turn_judge(true)).await;
    assert_eq!(held.mode, aidaemon_coding_agents::PermissionMode::Checked);
    assert!(
        !held
            .args
            .iter()
            .any(|arg| arg == "bypassPermissions" || arg.starts_with("--dangerously")),
        "no bypass flag: {:?}",
        held.args
    );
    assert!(held.host.is_some(), "its requests come back to the run");
    assert!(held.detached_judge.is_some());
}

/// The launch holds to the ceiling by itself (no card) only when every
/// configured agent can run checked here; reading and stopping its runs
/// always do.
#[cfg(unix)]
#[tokio::test]
async fn a_launch_holds_to_the_ceiling_only_when_every_agent_fits() {
    let (tool, _db, _bin) = full_access_claude().await;
    let run = json!({"action": "run", "tool": "claude", "prompt": "fix it"}).to_string();
    assert!(tool.holds_launch_to_ceiling(&run));
    for action in ["list", "check", "cancel"] {
        assert!(tool.holds_launch_to_ceiling(&json!({"action": action}).to_string()));
    }
    // An agent that runs only unchecked: a launch may reach it (a fallback,
    // a follow-up), so the launch is asked about on the phone.
    tool.tools.write().unwrap().insert(
        "bash-agent".to_string(),
        CliToolEntry {
            command: "bash".to_string(),
            args: vec!["-c".to_string()],
            description: "Bash agent".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );
    assert!(!tool.holds_launch_to_ceiling(&run));
    let plain = turn_judge(true);
    let agent = tool.agent_named("bash-agent").unwrap();
    // Not held, so a launch the phone allowed runs as the owner set it.
    assert_eq!(
        tool.ceiling_mode_cap(plain.device_cap().is_capped(), &*agent, false, None),
        None
    );
}

/// A risky request the agent makes in a capped run (a push) waits for a
/// High, once-only card; allowing it for the session grants nothing
/// lasting. Uncapped, nothing asks.
#[tokio::test]
async fn a_risky_nested_request_in_a_capped_run_asks_the_phone_once() {
    use aidaemon_coding_agents::{ActionJudge as _, AgentAction};
    use aidaemon_core as sdk;

    struct Terminal;
    #[async_trait::async_trait]
    impl sdk::Tool for Terminal {
        fn spec(&self) -> sdk::ToolSpec {
            sdk::ToolSpec {
                name: "terminal".into(),
                description: "terminal".into(),
                parameters: json!({}),
            }
        }
        async fn call(
            &self,
            _arguments: Value,
            _ctx: &sdk::ToolContext,
        ) -> Result<sdk::ToolOutput, sdk::ToolError> {
            Ok(sdk::ToolOutput::text("ok"))
        }
    }

    for capped in [false, true] {
        let approvals = Arc::new(crate::approvals::OwnerApprovals::default());
        let (tx, mut rx) = mpsc::channel::<aidaemon_channels::hub::ApprovalRequest>(8);
        approvals.set_broker(crate::approvals::ApprovalBroker::new(tx));
        let cards = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = cards.clone();
        tokio::spawn(async move {
            while let Some(request) = rx.recv().await {
                seen.lock()
                    .unwrap()
                    .push((request.risk_level, request.kind));
                let _ = request
                    .response_tx
                    .send(aidaemon_tools::semantics::ApprovalResponse::AllowSession);
            }
        });
        let mut tools: HashMap<String, Arc<dyn sdk::Tool>> = HashMap::new();
        tools.insert("terminal".into(), Arc::new(Terminal));
        let probe = turn_judge(capped);
        let hook: Arc<dyn sdk::Hook> = Arc::new(
            crate::approvals::device_ceiling::DeviceCeilingHook::new(probe.device_cap().clone()),
        );
        let detached = judge_with(capped, tools, vec![hook], approvals);
        // The hook and the judge's rules share the request's cap in a turn;
        // here the hook's own handle carries it.
        let judge = crate::tools::agent_mode::DaemonJudge::for_run_with_detached(
            None,
            Some(detached),
            crate::tools::agent_mode::Asker {
                run: "cli-agent-ceiling".into(),
                agent: "Claude Code".into(),
            },
        );
        for _ in 0..2 {
            let verdict = judge
                .judge(AgentAction::RunCommand {
                    command: "git push origin main".into(),
                    working_dir: Some("/w/synthetic".into()),
                })
                .await;
            assert!(verdict.is_allowed(), "capped {capped}: {verdict:?}");
        }
        let cards = cards.lock().unwrap().clone();
        if capped {
            assert_eq!(cards.len(), 2, "a session answer is no lasting grant");
            for (risk, kind) in cards {
                assert!(risk >= aidaemon_tools::command_risk::RiskLevel::High);
                assert!(matches!(
                    kind,
                    aidaemon_tools::semantics::ApprovalKind::CommandOnce
                ));
            }
        } else {
            assert!(cards.is_empty(), "nothing asks uncapped: {cards:?}");
        }
    }
}

/// A note from a capped run never goes into a run with full access, which
/// nothing judges: it waits for that run instead. Uncapped, it goes in.
#[tokio::test]
async fn a_capped_note_never_enters_a_full_access_run() {
    use std::sync::Mutex as StdMutex;

    struct SteerableHost {
        notes: Arc<StdMutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl aidaemon_coding_agents::AgentHost for SteerableHost {
        async fn on_line(&self, line: &str) -> Vec<String> {
            vec![line.to_string()]
        }
        fn answers(&self) -> aidaemon_coding_agents::HostAnswers {
            Default::default()
        }
        fn accepts_mid_run(&self) -> bool {
            true
        }
        async fn send_note(&self, note: &str) -> Result<(), String> {
            self.notes.lock().unwrap().push(note.to_string());
            Ok(())
        }
        async fn close(&self) {}
    }

    for capped in [false, true] {
        let (tool, _db) = setup_bash_tool().await;
        tool.set_runs(Arc::new(crate::run_state::RunStates::default()));
        let chat = "app:synthetic-capped-note";
        let task_id = if capped { "capnote1" } else { "opennote" };
        let run_id = cli_run_id(task_id);
        let notes = Arc::new(StdMutex::new(Vec::new()));
        let host: Arc<dyn aidaemon_coding_agents::AgentHost> = Arc::new(SteerableHost {
            notes: notes.clone(),
        });
        let handles = tool.run_handles();
        handles.runs.begin(&run_id, chat, "Synthetic agent", None);
        handles.runs.set_agent(&run_id, "Synthetic agent");
        handles.runs.set_accepts_mid_run(&run_id, true);
        let shared = RunShared::default();
        shared.process.start(
            aidaemon_tools::backend::ProcessHandle::new(1, None, None),
            Some(host),
        );
        // It was started with full access by an uncapped run.
        shared.unchecked.store(true, Ordering::SeqCst);
        let call = RunCall {
            task_id: task_id.to_string(),
            tool_name: "bash-agent".to_string(),
            session_id: chat.to_string(),
            goal_id: None,
            delegated_task_id: None,
            label: "Synthetic agent".to_string(),
            short_summary: "Synthetic agent".to_string(),
            async_mode: true,
            status_tx: None,
            grant_session: chat.to_string(),
            new_folders: Vec::new(),
            began: chrono::Utc::now(),
            owner_follow_up: false,
            direct_agent: false,
            fresh_session: false,
            is_peer_run: false,
            continues_session: false,
            peer_binding: None,
            device_cap: None,
        };
        tool.running.lock().await.insert(
            task_id.to_string(),
            tracked(
                &call,
                &shared,
                Instant::now(),
                None,
                WriteScope::ReadOnly,
                10_000,
            ),
        );
        let call_id = format!("follow-up-synthetic-{task_id}");
        crate::helper_run::mark_owner_follow_up(&call_id);
        let mut facts = ToolCallContext {
            session_id: chat.to_string(),
            tool_call_id: Some(call_id),
            ..Default::default()
        };
        facts.extensions.insert(turn_judge(capped));
        let output = tool
            .call_with_status_under_contract(
                &json!({
                    "action": "run",
                    "continue_run": run_id,
                    "prompt": "also push it",
                    "async_mode": true,
                })
                .to_string(),
                None,
                false,
                false,
                &facts,
            )
            .await
            .unwrap_or_else(|error| error.to_string());
        if capped {
            assert!(notes.lock().unwrap().is_empty(), "{output}");
            assert!(!output.contains("Added your note"), "{output}");
        } else {
            assert!(output.contains("Added your note"), "{output}");
            assert_eq!(notes.lock().unwrap().len(), 1);
        }
        // Stop what the capped case queued behind the live run.
        tool.running.lock().await.clear();
    }
}

/// The run's terminal as the kernel sees it: `ls` a read whose effects it
/// declares in full, anything else a push off the machine.
struct SdkTerminal;

#[async_trait::async_trait]
impl aidaemon_core::Tool for SdkTerminal {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        aidaemon_core::ToolSpec {
            name: "terminal".into(),
            description: "terminal".into(),
            parameters: json!({}),
        }
    }
    fn annotations_for(&self, arguments: &Value) -> aidaemon_core::ToolAnnotations {
        if arguments["command"] == "ls" {
            aidaemon_core::ToolAnnotations::READ_ONLY
        } else {
            aidaemon_core::ToolAnnotations {
                egress: true,
                ..aidaemon_core::ToolAnnotations::WRITE
            }
        }
    }
    fn effects(&self, arguments: &Value) -> aidaemon_core::CallEffects {
        if arguments["command"] == "ls" {
            aidaemon_core::CallEffects::none()
        } else {
            aidaemon_core::CallEffects {
                keys: vec![aidaemon_core::EffectKey::egress(
                    "https",
                    "git.example.invalid",
                    None,
                )],
                complete: true,
                understood: true,
            }
        }
    }
    async fn call(
        &self,
        _arguments: Value,
        _ctx: &aidaemon_core::ToolContext,
    ) -> Result<aidaemon_core::ToolOutput, aidaemon_core::ToolError> {
        Ok(aidaemon_core::ToolOutput::text("ok"))
    }
}

/// The terminal's own ladder, as the approval rules consult it: every
/// command asks the owner at Medium (as the live card did).
struct MediumLadder;

#[async_trait::async_trait]
impl crate::tools::traits::ToolSemantics for MediumLadder {
    async fn owner_approval_request(
        &self,
        _call: &aidaemon_tools::semantics::ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> aidaemon_tools::semantics::OwnerApprovalNeed {
        aidaemon_tools::semantics::OwnerApprovalNeed::Ask(
            aidaemon_core::OwnerApprovalRequest::plain(
                "Run a command",
                "terminal run",
                aidaemon_core::ApprovalRisk::Medium,
                Vec::new(),
            ),
        )
    }
}

#[async_trait::async_trait]
impl Tool for MediumLadder {
    fn name(&self) -> &str {
        "terminal"
    }
    fn description(&self) -> &str {
        "terminal"
    }
    fn schema(&self) -> Value {
        json!({"name": "terminal", "description": "terminal", "parameters": {"type": "object"}})
    }
    async fn execute(
        &self,
        _arguments: &str,
        _facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        anyhow::bail!("not run in this test")
    }
}

/// The cards an agent's asks raise, judged as the run judges them (the
/// approval rules with the terminal's Medium ladder, then the device
/// ceiling), for an agent held from full access or not.
async fn cards_for(
    capped: bool,
    held_from_full_access: bool,
    command: &str,
) -> Vec<(aidaemon_tools::command_risk::RiskLevel, bool)> {
    use aidaemon_coding_agents::{ActionJudge as _, AgentAction};
    let approvals = Arc::new(crate::approvals::OwnerApprovals::default());
    let (tx, mut rx) = mpsc::channel::<aidaemon_channels::hub::ApprovalRequest>(8);
    approvals.set_broker(crate::approvals::ApprovalBroker::new(tx));
    let cards = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = cards.clone();
    tokio::spawn(async move {
        while let Some(request) = rx.recv().await {
            let once = matches!(
                request.kind,
                aidaemon_tools::semantics::ApprovalKind::CommandOnce
            );
            seen.lock().unwrap().push((request.risk_level, once));
            let _ = request
                .response_tx
                .send(aidaemon_tools::semantics::ApprovalResponse::AllowOnce);
        }
    });
    let ladder: Arc<dyn Tool> = Arc::new(MediumLadder);
    let rules = crate::approvals::AppApprovalRules {
        tools: crate::approvals::ApprovalTools::new([ladder]),
        ..Default::default()
    };
    if capped {
        rules.device_cap.raise();
    }
    let cap = rules.device_cap.clone();
    let rules = Arc::new(rules);
    let policy = aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never)
        .with_rules(rules.clone());
    let hooks: Vec<Arc<dyn aidaemon_core::Hook>> = vec![
        Arc::new(policy),
        Arc::new(crate::approvals::device_ceiling::DeviceCeilingHook::new(
            cap,
        )),
    ];
    let mut tools: HashMap<String, Arc<dyn aidaemon_core::Tool>> = HashMap::new();
    tools.insert("terminal".into(), Arc::new(SdkTerminal));
    let detached = Arc::new(crate::approvals::DetachedNestedJudge::new(
        tools,
        hooks,
        approvals,
        rules,
        aidaemon_core::SharedDeclines::default(),
        aidaemon_core::RunId::new(),
        "app:synthetic-ceiling".into(),
        "app:synthetic-ceiling".into(),
        None,
        Arc::new(AtomicBool::new(false)),
        false,
    ));
    let judge = crate::tools::agent_mode::DaemonJudge::for_run_with_detached(
        None,
        Some(detached),
        crate::tools::agent_mode::Asker {
            run: "cli-agent-held".into(),
            agent: "Antigravity".into(),
        },
    )
    .held_from_full_access(held_from_full_access);
    let verdict = judge
        .judge(AgentAction::RunCommand {
            command: command.into(),
            working_dir: Some("/w/synthetic".into()),
        })
        .await;
    assert!(verdict.is_allowed(), "{command}: {verdict:?}");
    cards.lock().unwrap().clone()
}

/// An agent the owner gave full access, held checked by the ceiling: a read
/// runs with no card (the full-access answer), a push asks once on a High
/// card only the phone can answer. An agent the owner set to ask for every
/// command keeps its cards; uncapped, nothing changes.
#[tokio::test]
async fn a_held_full_access_agent_is_asked_only_above_the_ceiling() {
    use aidaemon_tools::command_risk::RiskLevel;
    assert!(cards_for(true, true, "ls").await.is_empty());
    let push = cards_for(true, true, "git push origin main").await;
    assert_eq!(push.len(), 1, "{push:?}");
    assert!(push[0].0 >= RiskLevel::High && push[0].1, "{push:?}");
    // The owner's own every-command choice: the ladder's card as today.
    let every = cards_for(true, false, "ls").await;
    assert_eq!(every, vec![(RiskLevel::Medium, false)]);
    // Uncapped: the ladder's card, and no ceiling card.
    let open = cards_for(false, false, "git push origin main").await;
    assert_eq!(open, vec![(RiskLevel::Medium, false)]);
}

/// The launch records that the ceiling, not the owner, held the agent from
/// full access; an owner's lower choice is not marked.
#[cfg(unix)]
#[tokio::test]
async fn the_launch_records_an_agent_held_from_full_access() {
    let (tool, _db, _bin) = full_access_claude().await;
    let folder = tempfile::tempdir().unwrap();
    let folder = folder.path().canonicalize().unwrap().display().to_string();
    assert!(
        !prepare_claude(&tool, &folder, turn_judge(false))
            .await
            .held_from_full_access
    );
    assert!(
        prepare_claude(&tool, &folder, turn_judge(true))
            .await
            .held_from_full_access
    );
    let config: toml::Table = toml::from_str(
        "[authority]\nowner_default = \"full_access\"\n\
         [cli_agents.permissions]\nclaude = \"every_command\"\n",
    )
    .unwrap();
    let (every, _db2, _bin2) = full_access_claude().await;
    every.permissions.get().unwrap().reload(&config);
    let held = prepare_claude(&every, &folder, turn_judge(true)).await;
    assert_eq!(
        held.mode,
        aidaemon_coding_agents::PermissionMode::EveryCommand
    );
    assert!(!held.held_from_full_access);
}
