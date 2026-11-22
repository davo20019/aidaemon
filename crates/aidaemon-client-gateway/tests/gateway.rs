//! The gateway over real sockets, on a scripted agent: the HTTP mode, the
//! compact capability and a watch's risk ceiling, idempotent commands,
//! approvals settled by whoever answers first, auth and version checks, and
//! resuming or starting from a snapshot.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aidaemon_channels::hub::{
    ApprovalRequest, ChannelCapabilities, ChannelHub, HubChannel, MediaMessage, SessionChannelStore,
};
use aidaemon_channels::ports::{
    AgentIngress, AgentResponseEnvelope, ChannelAgentRuntime, InboundMessageRequest,
};
use aidaemon_channels::turns::{TurnDeps, TurnWatch};
use aidaemon_client_gateway::{
    ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps, SettingsDone, SettingsGate,
};
use aidaemon_client_protocol::*;
use aidaemon_store_sqlite::SqliteClientEventLog;
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalKind as CardKind, ApprovalResponse};
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;

/// Replies "Done." — or, for a message that says "approve", asks the owner
/// first (at `risk`) and says what the answer was.
struct ScriptedAgent {
    approvals: mpsc::Sender<ApprovalRequest>,
    risk: RiskLevel,
    turns: AtomicUsize,
    /// The files the last message came with.
    files: std::sync::Mutex<Vec<aidaemon_tools::semantics::MessageAttachment>>,
}

#[async_trait]
impl AgentIngress for ScriptedAgent {
    async fn handle_inbound_message(
        &self,
        request: InboundMessageRequest,
    ) -> anyhow::Result<AgentResponseEnvelope> {
        self.turns.fetch_add(1, Ordering::SeqCst);
        *self.files.lock().unwrap() = request.attachments.clone();
        if let Some(status) = request.status_tx.as_ref() {
            let _ = status
                .send(StatusUpdate::tool_start(
                    "terminal",
                    "building the synthetic report",
                    aidaemon_core::card::ActivityKind::Run,
                ))
                .await;
            for chunk in ["Do", "ne."] {
                let _ = status.send(StatusUpdate::ReplyText(chunk.into())).await;
            }
            // The turn takes in a message the owner sent while it ran.
            if request.user_text.contains("mid-run") {
                let _ = status
                    .send(StatusUpdate::MessageTaken {
                        text: "Make it rounder too".into(),
                        origin: "app".into(),
                    })
                    .await;
            }
        }
        let text = if request.user_text.contains("approve") {
            let (response_tx, response_rx) = oneshot::channel();
            self.approvals
                .send(ApprovalRequest {
                    command: "report --publish".into(),
                    session_id: request.session_id.clone(),
                    risk_level: self.risk,
                    warnings: Vec::new(),
                    permission_mode: PermissionMode::Default,
                    response_tx,
                    kind: CardKind::Command,
                    card: None,
                    expires_at_ms: None,
                })
                .await
                .unwrap();
            match response_rx.await {
                Ok(ApprovalResponse::Deny) => "Not published.".to_string(),
                Ok(response) => format!("Published ({response:?})."),
                Err(_) => "No answer.".to_string(),
            }
        } else {
            "Done.".to_string()
        };
        Ok(AgentResponseEnvelope {
            response_id: "r".into(),
            task_id: "t".into(),
            turn_id: None,
            text,
            referenced_receipts: Vec::new(),
            disposition: Default::default(),
            model: None,
        })
    }
}

#[async_trait]
impl ChannelAgentRuntime for ScriptedAgent {
    async fn terminalize_supervised_task(
        &self,
        _: &str,
        _: aidaemon_events::TaskStatus,
        _: String,
    ) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
    async fn cancel_active_finite_work_for_session(&self, _: &str) -> Vec<String> {
        Vec::new()
    }
    async fn current_model(&self) -> String {
        "scripted".into()
    }
    async fn context_debug_settings(
        &self,
        _: &str,
        _: &str,
    ) -> (bool, usize, usize, usize, Option<i64>) {
        (false, 0, 0, 0, None)
    }
    async fn set_model(&self, _: String) {}
    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }
    async fn clear_model_override(&self) {}
    async fn clear_session_context(&self, _: &str) -> anyhow::Result<()> {
        Ok(())
    }
    async fn clear_session(&self, _: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

struct NoSessions;
impl SessionChannelStore for NoSessions {}

struct Host;

#[async_trait]
impl ClientHost for Host {
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        Ok(vec![SessionSummary {
            archived: false,
            id: "telegram:synthetic-user-1".into(),
            channel: "telegram".into(),
            title: None,
            last_activity_ms: Some(5),
            writable: false,
            project_id: None,
            group: false,
        }])
    }
    async fn history(&self, _: &str, _: usize) -> anyhow::Result<Vec<HistoryMessage>> {
        Ok((1..=5)
            .map(|n| HistoryMessage {
                attachments: Vec::new(),
                id: format!("m{n}"),
                role: "user".into(),
                text: format!("message {n}"),
                at_ms: n,
                tools: Vec::new(),
                calls: Vec::new(),
                handed_off: false,
                served_by: None,
            })
            .collect())
    }
    async fn slash(&self, _: &str, command: &str, _: &str) -> Option<String> {
        (command == "clear").then(|| "Context cleared.".to_string())
    }
    /// Raising the permission level asks, as the daemon's Settings does.
    async fn settings_gate(&self, change: &SettingsChange) -> Result<SettingsGate, String> {
        match change {
            SettingsChange::SetPermissionLevel { .. } => Ok(SettingsGate::Ask {
                summary: "Give AIdaemon full access".into(),
                warnings: Vec::new(),
                risk: Risk::High,
                other_device: true,
            }),
            _ => Err("Not in this test".into()),
        }
    }
    async fn change_settings(&self, _: SettingsChange, _: bool) -> Result<SettingsDone, String> {
        Ok(SettingsDone {
            message: "AIdaemon's permission level is now Full access.".into(),
            applies: Some(Applies::Now),
        })
    }
    async fn create_project(
        &self,
        name: &str,
        path: Option<&str>,
        create_folder: bool,
    ) -> Result<ProjectSummary, ProtocolError> {
        if name.trim().is_empty() {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "A project needs a name",
            ));
        }
        if path == Some("/home/synthetic/new-site") && !create_folder {
            return Err(ProtocolError::new(
                ErrorCode::FolderMissing,
                "There is no folder at /home/synthetic/new-site on this computer",
            ));
        }
        Ok(ProjectSummary {
            id: "project-acme".into(),
            name: name.into(),
            path: path.map(str::to_string),
            channels: Vec::new(),
        })
    }
    async fn changes(&self, session_id: &str) -> anyhow::Result<SessionChanges> {
        Ok(SessionChanges {
            last_check: None,
            repos: vec![RepoChanges {
                root: format!("/home/synthetic/{session_id}"),
                branch: Some("main".into()),
                files: Vec::new(),
                patch: String::new(),
                truncated: false,
            }],
        })
    }
}

/// A chat that shows cards and answers them when told to.
struct Chat {
    answer: Mutex<Option<oneshot::Receiver<ApprovalResponse>>>,
    asked: AtomicUsize,
}

#[async_trait]
impl HubChannel for Chat {
    fn name(&self) -> String {
        "telegram".into()
    }
    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: true,
            media: true,
            max_message_len: 4096,
        }
    }
    async fn send_text(&self, _: &str, _: &str) -> anyhow::Result<()> {
        Ok(())
    }
    async fn send_media(&self, _: &str, _: &MediaMessage) -> anyhow::Result<()> {
        Ok(())
    }
    async fn request_approval(
        &self,
        _: &str,
        _: &str,
        _: RiskLevel,
        _: &[String],
        _: PermissionMode,
        _: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        let answer = self.answer.lock().await.take();
        match answer {
            Some(answer) => Ok(answer.await?),
            None => std::future::pending().await,
        }
    }
}

struct Fixture {
    gateway: Arc<ClientGateway>,
    agent: Arc<ScriptedAgent>,
    url: String,
    token: String,
    hub: Arc<ChannelHub>,
    deps: TurnDeps,
    _dir: tempfile::TempDir,
}

async fn fixture(risk: RiskLevel) -> Fixture {
    fixture_after(risk, &[]).await
}

/// The fixture over a log that a previous process left `earlier` events in.
async fn fixture_after(risk: RiskLevel, earlier: &[(&str, Event)]) -> Fixture {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let log = SqliteClientEventLog::open(pool).await.unwrap();
    for (at, (session, event)) in earlier.iter().enumerate() {
        let turn_id = match event {
            Event::TurnStarted { turn_id, .. } | Event::TurnCompleted { turn_id, .. } => {
                Some(turn_id.as_str())
            }
            Event::TaskUpdated { task } => Some(task.id.as_str()),
            _ => None,
        };
        log.append(
            "owner",
            at as u64 + 1,
            Some(session),
            event.kind(),
            turn_id,
            &serde_json::to_string(event).unwrap(),
        )
        .await
        .unwrap();
    }
    let (approvals_tx, approvals_rx) = mpsc::channel(16);
    let approvals_for_pairing = approvals_tx.clone();
    let agent = Arc::new(ScriptedAgent {
        approvals: approvals_tx,
        risk,
        turns: AtomicUsize::new(0),
        files: std::sync::Mutex::default(),
    });
    let session_map = aidaemon_channels::hub::SessionMap::default();
    let hub = Arc::new(ChannelHub::new(Vec::new(), session_map.clone()));
    tokio::spawn(hub.clone().approval_listener(approvals_rx));
    let deps = TurnDeps {
        agent: agent.clone(),
        session_map,
        sessions: Arc::new(NoSessions),
        task_registry: Arc::new(aidaemon_channels::tasks::TaskRegistry::new(50)),
        watchdog_stale_threshold_secs: 0,
        watch: TurnWatch::default(),
    };
    let dir = tempfile::tempdir().unwrap();
    let token_file = dir.path().join("client-token.sha256");
    let token = aidaemon_client_gateway::token::mint(&token_file).unwrap();
    let files = Arc::new(
        aidaemon_client_gateway::files::FileStore::open(
            dir.path().join("client-files"),
            dir.path().join("inbox"),
        )
        .unwrap(),
    );
    let gateway = ClientGateway::start(GatewayDeps {
        files: Some(files),
        turns: deps.clone(),
        hub: hub.clone(),
        host: Arc::new(Host),
        log,
        config: ClientGatewayConfig {
            enabled: true,
            ..ClientGatewayConfig::default()
        },
        endpoints: aidaemon_core::EndpointSource::fixed(Vec::new()),
        token_file,
        approvals: Some(approvals_for_pairing),
    })
    .await
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    let app = aidaemon_client_gateway::router(gateway.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Fixture {
        gateway,
        agent,
        url,
        token,
        hub,
        deps,
        _dir: dir,
    }
}

impl Fixture {
    async fn post(&self, headers: &[(&str, &str)], frame: &ClientFrame) -> ServerFrame {
        let client = reqwest::Client::new();
        let mut request = client
            .post(format!("http://{}/client/v1/commands", self.url))
            .bearer_auth(&self.token)
            .json(frame);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        request.send().await.unwrap().json().await.unwrap()
    }

    async fn events(&self, headers: &[(&str, &str)], since: Option<u64>) -> EventsPage {
        let client = reqwest::Client::new();
        let mut url = format!("http://{}/client/v1/events?wait=2", self.url);
        if let Some(since) = since {
            url.push_str(&format!("&since={since}"));
        }
        let mut request = client.get(url).bearer_auth(&self.token);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        request.send().await.unwrap().json().await.unwrap()
    }

    /// Long-poll until an event matching `want` arrives; every event seen.
    async fn poll_until(
        &self,
        headers: &[(&str, &str)],
        since: &mut u64,
        want: impl Fn(&Event) -> bool,
    ) -> Vec<Event> {
        let mut seen = Vec::new();
        for _ in 0..20 {
            let page = self.events(headers, Some(*since)).await;
            *since = page.next_since;
            for frame in page.frames {
                if let ServerFrame::Event { event, .. } = frame {
                    let done = want(&event);
                    seen.push(event);
                    if done {
                        return seen;
                    }
                }
            }
        }
        panic!("event never came; saw {seen:?}");
    }
}

fn command(request_id: &str, command: Command) -> ClientFrame {
    ClientFrame::Command {
        request_id: request_id.into(),
        command,
    }
}

fn send(text: &str) -> Command {
    Command::SendMessage {
        session_id: None,
        text: text.into(),
        attachments: Vec::new(),
        mid_run: None,
        project_id: None,
        model: None,
        effort: None,
    }
}

/// A message the running turn took in reaches clients as what the owner
/// said, before the turn's end: the waiting list drops it once the turn
/// takes it, and nothing else showed it (live 2026-11-17).
#[tokio::test]
async fn a_message_taken_into_a_running_turn_is_shown_as_the_owners() {
    let f = fixture(RiskLevel::Medium).await;
    f.post(&[], &command("r-mid", send("Draw a mascot (mid-run)")))
        .await;
    let mut since = 0;
    let events = f
        .poll_until(
            &[],
            &mut since,
            |e| matches!(e, Event::UserMessage { text, .. } if text == "Make it rounder too"),
        )
        .await;
    let said: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            Event::UserMessage { text, origin, .. } => {
                assert_eq!(origin, "app");
                Some(text.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(said, ["Draw a mascot (mid-run)", "Make it rounder too"]);
}

/// A turn the previous process left running is ended at start, so a
/// client resuming from the log stops showing it as working.
#[tokio::test]
async fn a_restart_ends_the_turns_the_previous_process_left_running() {
    let started = |turn: &str| Event::TurnStarted {
        turn_id: turn.into(),
        session_id: "app:synthetic-1".into(),
        origin: "app".into(),
    };
    let f = fixture_after(
        RiskLevel::Medium,
        &[
            ("app:synthetic-1", started("t-done")),
            (
                "app:synthetic-1",
                Event::TurnCompleted {
                    turn_id: "t-done".into(),
                    text: "Done".into(),
                    attachments: Vec::new(),
                    state: TurnState::Completed,
                    error: None,
                    served_by: None,
                },
            ),
            ("app:synthetic-1", started("t-cut")),
        ],
    )
    .await;
    let mut since = 3;
    let seen = f
        .poll_until(
            &[],
            &mut since,
            |event| matches!(event, Event::TurnCompleted { turn_id, .. } if turn_id == "t-cut"),
        )
        .await;
    let Some(Event::TurnCompleted { state, error, .. }) = seen.last() else {
        unreachable!()
    };
    assert_eq!(*state, TurnState::Cancelled);
    assert!(error.is_some());
    assert!(!seen.iter().any(
        |event| matches!(event, Event::TurnCompleted { turn_id, .. } if turn_id == "t-done" )
    ));
}

/// A task the previous process left running (a background build it
/// started) is ended at start too: a client counts any running task as
/// work, so it would keep offering Stop for nothing.
#[tokio::test]
async fn a_restart_ends_the_tasks_the_previous_process_left_running() {
    let task = |id: &str, state: TaskState| Event::TaskUpdated {
        task: TaskSummary {
            id: id.into(),
            session_id: "app:synthetic-1".into(),
            number: 1,
            description: "synthetic build".into(),
            state,
            started_at_ms: 1,
            finished_at_ms: None,
            error: None,
        },
    };
    let f = fixture_after(
        RiskLevel::Medium,
        &[
            ("app:synthetic-1", task("k-done", TaskState::Running)),
            ("app:synthetic-1", task("k-done", TaskState::Completed)),
            ("app:synthetic-1", task("k-cut", TaskState::Running)),
        ],
    )
    .await;
    let mut since = 3;
    let seen = f
        .poll_until(
            &[],
            &mut since,
            |event| matches!(event, Event::TaskUpdated { task } if task.id == "k-cut"),
        )
        .await;
    let Some(Event::TaskUpdated { task }) = seen.last() else {
        unreachable!()
    };
    assert_eq!(task.state, TaskState::Cancelled);
    assert!(task.error.is_some());
    assert!(
        !seen
            .iter()
            .any(|event| matches!(event, Event::TaskUpdated { task } if task.id == "k-done"))
    );
}

#[tokio::test]
async fn http_mode_runs_a_turn_and_long_polls_its_events() {
    let f = fixture(RiskLevel::Medium).await;
    let ack = f
        .post(&[], &command("r-1", send("Build the synthetic report")))
        .await;
    let ServerFrame::Ack {
        result:
            CommandResult::MessageAccepted {
                session_id,
                new_session: true,
            },
        ..
    } = ack.clone()
    else {
        panic!("{ack:?}");
    };
    assert!(session_id.starts_with("app:"));

    let mut since = 0;
    let events = f
        .poll_until(&[], &mut since, |e| {
            matches!(e, Event::TurnCompleted { .. })
        })
        .await;
    let kinds: Vec<&str> = events.iter().map(Event::kind).collect();
    assert_eq!(
        kinds,
        [
            "session_created",
            "user_message",
            "turn_started",
            "task_updated",
            "turn_completed"
        ],
        "HTTP mode carries durable events only"
    );
    let Event::TurnCompleted { text, state, .. } = events.last().unwrap() else {
        unreachable!()
    };
    assert_eq!((text.as_str(), *state), ("Done.", TurnState::Completed));

    // The same request again: the first answer, and no second turn.
    let again = f
        .post(&[], &command("r-1", send("Build the synthetic report")))
        .await;
    assert_eq!(again, ack);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(f.agent.turns.load(Ordering::SeqCst), 1);

    // Paging from the start replays the same events in order.
    let page = f.events(&[], Some(0)).await;
    let seqs: Vec<u64> = page
        .frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Event { seq, .. } => *seq,
            _ => None,
        })
        .collect();
    assert_eq!(seqs, (1..=seqs.len() as u64).collect::<Vec<_>>());

    // Without a usable `since`, a snapshot comes first.
    let page = f.events(&[], None).await;
    assert!(matches!(page.frames[0], ServerFrame::Snapshot { .. }));
    let page = f.events(&[], Some(10_000)).await;
    assert!(matches!(page.frames[0], ServerFrame::Snapshot { .. }));

    // Commands that read: glance, sessions, history, slash commands.
    let glance = f.post(&[], &command("r-2", Command::GetGlance)).await;
    let ServerFrame::Ack {
        result: CommandResult::Glance { glance },
        ..
    } = glance
    else {
        panic!("{glance:?}");
    };
    assert_eq!(glance.last_reply.unwrap().text, "Done.");
    let history = f
        .post(
            &[],
            &command(
                "r-3",
                Command::GetHistory {
                    session_id: session_id.clone(),
                    before: Some("m4".into()),
                    limit: 2,
                },
            ),
        )
        .await;
    let ServerFrame::Ack {
        result: CommandResult::History { messages, more, .. },
        ..
    } = history
    else {
        panic!("{history:?}");
    };
    let ids: Vec<&str> = messages.iter().map(|m| m.id.as_str()).collect();
    assert_eq!((ids, more), (vec!["m2", "m3"], true));
    let changes = f
        .post(
            &[],
            &command(
                "r-3b",
                Command::GetChanges {
                    session_id: "acme".into(),
                },
            ),
        )
        .await;
    let ServerFrame::Ack {
        result: CommandResult::Changes { changes },
        ..
    } = changes
    else {
        panic!("{changes:?}");
    };
    assert_eq!(changes.repos[0].root, "/home/synthetic/acme");
    // A host without automations lists none and refuses a change.
    let automations = f
        .post(&[], &command("r-auto-1", Command::ListAutomations))
        .await;
    assert!(
        matches!(
            &automations,
            ServerFrame::Ack {
                result: CommandResult::Automations { automations, changed: false, .. },
                ..
            } if automations.is_empty()
        ),
        "{automations:?}"
    );
    let refused = f
        .post(
            &[],
            &command(
                "r-auto-2",
                Command::ChangeAutomation {
                    automation_id: "goal-1".into(),
                    change: aidaemon_client_protocol::AutomationChange::Pause,
                },
            ),
        )
        .await;
    assert!(matches!(refused, ServerFrame::Error { .. }), "{refused:?}");
    let created = f
        .post(
            &[],
            &command(
                "r-3c",
                Command::CreateProject {
                    name: "Acme site".into(),
                    path: Some("/home/synthetic/acme-site".into()),
                    create_folder: false,
                },
            ),
        )
        .await;
    let ServerFrame::Ack {
        result: CommandResult::Project { project },
        ..
    } = created
    else {
        panic!("{created:?}");
    };
    assert_eq!(project.path.as_deref(), Some("/home/synthetic/acme-site"));
    let refused = f
        .post(
            &[],
            &command(
                "r-3d",
                Command::CreateProject {
                    name: " ".into(),
                    path: None,
                    create_folder: false,
                },
            ),
        )
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.message == "A project needs a name"),
        "{refused:?}"
    );
    // A missing folder is a typed refusal; the same command with
    // `create_folder` goes through.
    let missing = f
        .post(
            &[],
            &command(
                "r-3e",
                Command::CreateProject {
                    name: "New site".into(),
                    path: Some("/home/synthetic/new-site".into()),
                    create_folder: false,
                },
            ),
        )
        .await;
    assert!(
        matches!(&missing, ServerFrame::Error { error, .. } if error.code == ErrorCode::FolderMissing),
        "{missing:?}"
    );
    let made = f
        .post(
            &[],
            &command(
                "r-3f",
                Command::CreateProject {
                    name: "New site".into(),
                    path: Some("/home/synthetic/new-site".into()),
                    create_folder: true,
                },
            ),
        )
        .await;
    assert!(matches!(made, ServerFrame::Ack { .. }), "{made:?}");
    let cleared = f
        .post(
            &[],
            &command(
                "r-4",
                Command::ClearSession {
                    session_id: session_id.clone(),
                },
            ),
        )
        .await;
    assert!(matches!(
        cleared,
        ServerFrame::Ack {
            result: CommandResult::Reply { .. },
            ..
        }
    ));
    // A reply into another chat's conversation needs that chat connected:
    // this one is not, so it is refused, not lost.
    let refused = f
        .post(
            &[],
            &command(
                "r-5",
                Command::SendMessage {
                    session_id: Some("telegram:synthetic-user-1".into()),
                    text: "hi".into(),
                    attachments: Vec::new(),
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: None,
                },
            ),
        )
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::NotFound),
        "{refused:?}"
    );
}

#[tokio::test]
async fn http_mode_refuses_a_missing_or_wrong_token() {
    let f = fixture(RiskLevel::Medium).await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{}/client/v1/commands", f.url))
        .json(&command("r-1", Command::GetGlance))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    let response = client
        .get(format!("http://{}/client/v1/events?since=0", f.url))
        .bearer_auth("acg_not-the-token")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
}

/// Sign-in's HTTP over one iroh connection, as a phone far from home makes
/// it.
struct IrohHttp(aidaemon_remote::RemoteConnection);

impl IrohHttp {
    async fn call(
        &self,
        request: http::Request<http_body_util::Full<bytes::Bytes>>,
    ) -> Result<aidaemon_client_core::signin::HttpReply, aidaemon_client_core::signin::SignInError>
    {
        let response =
            self.0.send(request).await.map_err(|e| {
                aidaemon_client_core::signin::SignInError::Unreachable(e.to_string())
            })?;
        Ok(aidaemon_client_core::signin::HttpReply {
            status: response.status().as_u16(),
            body: String::from_utf8_lossy(response.body()).into_owned(),
        })
    }
}

fn path_of(url: &str) -> &str {
    &url[url.find("/client/").unwrap()..]
}

impl aidaemon_client_core::signin::Http for IrohHttp {
    async fn post_json(
        &self,
        url: &str,
        body: String,
    ) -> Result<aidaemon_client_core::signin::HttpReply, aidaemon_client_core::signin::SignInError>
    {
        self.call(
            http::Request::post(path_of(url))
                .header("content-type", "application/json")
                .body(http_body_util::Full::new(bytes::Bytes::from(body)))
                .unwrap(),
        )
        .await
    }

    async fn get(
        &self,
        url: &str,
    ) -> Result<aidaemon_client_core::signin::HttpReply, aidaemon_client_core::signin::SignInError>
    {
        self.call(
            http::Request::get(path_of(url))
                .body(http_body_util::Full::default())
                .unwrap(),
        )
        .await
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}

fn glance_request(token: &str) -> http::Request<http_body_util::Full<bytes::Bytes>> {
    http::Request::post("/client/v1/commands")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {token}"))
        .body(http_body_util::Full::new(bytes::Bytes::from(
            serde_json::to_vec(&command("r-iroh", Command::GetGlance)).unwrap(),
        )))
        .unwrap()
}

/// Over iroh a phone pairs and signs in with its key, and its session
/// works only from the endpoint that signed in for it: the same token from
/// another endpoint is refused, and so is the owner's local token.
#[tokio::test]
async fn over_iroh_a_device_pairs_signs_in_and_keeps_its_session_to_itself() {
    use aidaemon_remote::{Access, Relays, RemoteClient, RemoteListener, RemotePeer, SecretKey};
    let f = fixture(RiskLevel::Medium).await;
    let listener = RemoteListener::bind(
        SecretKey::generate(),
        &Relays::Disabled,
        Access::Any,
        aidaemon_client_gateway::remote_router::<RemotePeer>(f.gateway.clone()),
    )
    .await
    .unwrap();
    let daemon = listener.url().to_string();
    f.gateway.pairing().set_remote_url(Some(daemon.clone()));

    let phone = RemoteClient::bind(SecretKey::generate(), &Relays::Disabled)
        .await
        .unwrap();
    let conn = phone.connect(listener.addr()).await.unwrap();

    // The owner's local token is for the daemon's own machine.
    let refused = conn.send(glance_request(&f.token)).await.unwrap();
    assert_eq!(refused.status(), 401);

    // Pair over iroh from the offer's link, which names the daemon's
    // address; the owner approves.
    let offer = f.gateway.pairing().offer(None).await.unwrap();
    let link = aidaemon_client_core::signin::PairingLink::parse(&offer.link).unwrap();
    assert_eq!(link.remote.as_deref(), Some(daemon.as_str()));
    let request = aidaemon_client_core::signin::PairRequest {
        gateway: daemon.clone(),
        offer_id: link.offer_id.clone(),
        code: link.code.clone(),
        device_name: "Synthetic Phone".into(),
        device_kind: ClientKind::Mobile,
        fingerprint: link.fingerprint.clone(),
    };
    let http = IrohHttp(conn.clone());
    let pairing = tokio::spawn(async move {
        aidaemon_client_core::signin::pair_with(&http, &request, Duration::from_millis(20)).await
    });
    let mut approved = false;
    for _ in 0..100 {
        if f.gateway
            .pairing()
            .decide(&offer.offer_id, true)
            .await
            .is_ok()
        {
            approved = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(approved, "the redeem never reached the owner");
    let device = pairing.await.unwrap().unwrap();
    assert_eq!(device.gateway, daemon);

    // Sign in with the device key; the session works from this endpoint.
    let grant = aidaemon_client_core::signin::sign_in_with(&IrohHttp(conn.clone()), &device)
        .await
        .unwrap();
    let response = conn
        .send(glance_request(&grant.access_token))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let frame: ServerFrame = serde_json::from_slice(response.body()).unwrap();
    assert!(
        matches!(
            frame,
            ServerFrame::Ack {
                result: CommandResult::Glance { .. },
                ..
            }
        ),
        "{frame:?}"
    );
    let mut auth = http::HeaderMap::new();
    auth.insert(
        "authorization",
        format!("Bearer {}", grant.access_token).parse().unwrap(),
    );
    assert!(conn.websocket("/client/v1/ws", &auth).await.is_ok());
    // A socket that would give its token in `hello` is not taken.
    assert!(
        conn.websocket("/client/v1/ws", &http::HeaderMap::new())
            .await
            .is_err()
    );

    // The same token from another endpoint is refused.
    let thief = RemoteClient::bind(SecretKey::generate(), &Relays::Disabled)
        .await
        .unwrap();
    let stolen = thief.connect(listener.addr()).await.unwrap();
    let response = stolen
        .send(glance_request(&grant.access_token))
        .await
        .unwrap();
    assert_eq!(response.status(), 401);

    phone.close().await;
    thief.close().await;
    listener.close().await;
}

/// Pair a device over iroh (no relay) and return what it keeps.
async fn pair_over_iroh(
    f: &Fixture,
    listener: &aidaemon_remote::RemoteListener,
) -> aidaemon_client_core::StoredDevice {
    use aidaemon_remote::{Relays, RemoteClient, SecretKey};
    let phone = RemoteClient::bind(SecretKey::generate(), &Relays::Disabled)
        .await
        .unwrap();
    let conn = phone.connect(listener.addr()).await.unwrap();
    let offer = f.gateway.pairing().offer(None).await.unwrap();
    let request = aidaemon_client_core::signin::PairRequest {
        gateway: listener.url().to_string(),
        offer_id: offer.offer_id.clone(),
        code: offer.code.clone(),
        device_name: "Synthetic Phone".into(),
        device_kind: ClientKind::Mobile,
        fingerprint: None,
    };
    let http = IrohHttp(conn);
    let pairing = tokio::spawn(async move {
        aidaemon_client_core::signin::pair_with(&http, &request, Duration::from_millis(20)).await
    });
    for _ in 0..100 {
        if f.gateway
            .pairing()
            .decide(&offer.offer_id, true)
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    pairing.await.unwrap().unwrap()
}

/// What the app loads by URL goes through the core's local bridge: a
/// command with the device's token, a signed file link (no token: the
/// signed-in endpoint is enough) and the client socket's upgrade, while a
/// device that never signed in, or a path without the bridge's secret, gets
/// nothing.
#[tokio::test]
async fn the_local_bridge_carries_requests_files_and_sockets_over_iroh() {
    use aidaemon_client_core::bridge;
    use aidaemon_client_core::remote::{self, Dialer, RemoteHttp};
    use aidaemon_remote::{Access, Relays, RemoteListener, RemotePeer, SecretKey};
    let f = fixture(RiskLevel::Medium).await;
    let listener = RemoteListener::bind(
        SecretKey::generate(),
        &Relays::Disabled,
        Access::Any,
        aidaemon_client_gateway::remote_router::<RemotePeer>(f.gateway.clone()),
    )
    .await
    .unwrap();
    remote::learn(listener.addr());
    let device = pair_over_iroh(&f, &listener).await;
    let dialer = Dialer::for_device(&device).unwrap();
    let grant = aidaemon_client_core::signin::sign_in_with(
        &RemoteHttp::new(&device.gateway, dialer.clone()).unwrap(),
        &device,
    )
    .await
    .unwrap();
    let base = bridge::base(&device.gateway, &dialer).await.unwrap();
    assert!(base.starts_with("http://127.0.0.1:"), "{base}");
    assert_eq!(bridge::base(&device.gateway, &dialer).await.unwrap(), base);
    let http = reqwest::Client::new();

    let frame: ServerFrame = http
        .post(format!("{base}/client/v1/commands"))
        .bearer_auth(&grant.access_token)
        .json(&command("b-1", Command::GetGlance))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(matches!(frame, ServerFrame::Ack { .. }), "{frame:?}");

    // A file link with a wrong signature reaches the gateway (its own 404),
    // past the guard.
    let file = http
        .get(format!("{base}/client/v1/files/f-1/not-signed/a.png"))
        .send()
        .await
        .unwrap();
    assert_eq!(file.status(), 404);
    assert_eq!(file.text().await.unwrap(), "");
    // Without the secret, the bridge itself refuses.
    let port = base
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap();
    let guessed = http
        .get(format!(
            "http://{port}/client/v1/files/f-1/not-signed/a.png"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(guessed.status(), 404);
    assert_eq!(guessed.text().await.unwrap(), "not found");

    // The client socket through the bridge: hello, then the gateway talks.
    let mut request =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(format!(
            "{}/client/v1/ws",
            base.replacen("http://", "ws://", 1)
        ))
        .unwrap();
    request.headers_mut().insert(
        "authorization",
        format!("Bearer {}", grant.access_token).parse().unwrap(),
    );
    let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    let mut machine = aidaemon_client_core::machine::LinkMachine::new(
        aidaemon_client_core::machine::ClientIdentity {
            kind: ClientKind::Mobile,
            name: "test".into(),
            version: None,
        },
        None,
        Vec::new(),
    );
    for frame in machine.opened(0) {
        socket
            .send(Message::text(serde_json::to_string(&frame).unwrap()))
            .await
            .unwrap();
    }
    let reply = tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("the gateway answered hello")
        .unwrap()
        .unwrap();
    assert!(reply.is_text(), "{reply:?}");

    // A device that never signed in gets no file through its own bridge.
    let stranger = Dialer::Device(Box::new(SecretKey::generate()));
    let other = bridge::base(&device.gateway, &stranger).await.unwrap();
    assert_ne!(other, base);
    let refused = http
        .get(format!("{other}/client/v1/files/f-1/not-signed/a.png"))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 401);
}

/// Live, through n0's relays: the app's own pairing and link (sign-in, the
/// socket, hello and resume) against a daemon known only by its
/// `iroh://` address. Needs the internet:
/// `cargo test -p aidaemon-client-gateway --test gateway the_app_link -- --ignored`.
#[tokio::test]
#[ignore = "needs the internet and n0's public relays"]
async fn the_app_link_runs_over_iroh_through_a_relay() {
    use aidaemon_client_core::KeyStore as _;
    use aidaemon_client_core::link::{self, Auth, LinkConfig};
    use aidaemon_client_core::machine::{ClientIdentity, Update};
    use aidaemon_client_core::remote::{Dialer, RemoteHttp};
    use aidaemon_remote::{Access, Relays, RemoteListener, RemotePeer, SecretKey};
    let f = fixture(RiskLevel::Medium).await;
    let listener = RemoteListener::bind(
        SecretKey::generate(),
        &Relays::N0,
        Access::Any,
        aidaemon_client_gateway::remote_router::<RemotePeer>(f.gateway.clone()),
    )
    .await
    .unwrap();
    listener.online().await;
    let daemon = listener.url().to_string();
    println!("daemon at {daemon}");
    f.gateway.pairing().set_remote_url(Some(daemon.clone()));

    let offer = f.gateway.pairing().offer(None).await.unwrap();
    let link = aidaemon_client_core::signin::PairingLink::parse(&offer.link).unwrap();
    let request = aidaemon_client_core::signin::PairRequest {
        gateway: link.remote.clone().unwrap(),
        offer_id: link.offer_id.clone(),
        code: link.code.clone(),
        device_name: "Synthetic Phone".into(),
        device_kind: ClientKind::Mobile,
        fingerprint: link.fingerprint.clone(),
    };
    let pairing = tokio::spawn(async move {
        let http = RemoteHttp::new(&request.gateway, Dialer::Pairing).unwrap();
        aidaemon_client_core::signin::pair_with(&http, &request, Duration::from_millis(200)).await
    });
    let mut approved = false;
    for _ in 0..150 {
        if f.gateway
            .pairing()
            .decide(&offer.offer_id, true)
            .await
            .is_ok()
        {
            approved = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(approved, "the redeem never reached the owner");
    let device = pairing.await.unwrap().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let store = std::sync::Arc::new(aidaemon_client_core::FileKeyStore::new(
        dir.path().join("device.json"),
    ));
    store.save(&device).unwrap();
    let (updates_tx, mut updates) = tokio::sync::mpsc::unbounded_channel();
    let _commands = link::spawn(
        LinkConfig {
            url: aidaemon_client_core::machine::socket_url(&device.gateway),
            auth: Auth::Device(store),
            client: ClientIdentity {
                kind: ClientKind::Mobile,
                name: "test".into(),
                version: None,
            },
            device: None,
            capabilities: Vec::new(),
        },
        updates_tx,
    );
    let mut connected = false;
    let mut framed = false;
    while let Ok(Some(update)) = tokio::time::timeout(Duration::from_secs(30), updates.recv()).await
    {
        match update {
            Update::Connected => connected = true,
            Update::Frame(_) if connected => {
                framed = true;
                break;
            }
            Update::Disconnected(why) => panic!("{why:?}"),
            _ => {}
        }
    }
    assert!(
        connected && framed,
        "connected {connected}, framed {framed}"
    );

    // What the app loads by URL, through the local bridge over the relay.
    let base = aidaemon_client_core::local::reachable_base(
        &device.gateway,
        &Auth::Device(std::sync::Arc::new(
            aidaemon_client_core::FileKeyStore::new(dir.path().join("device.json")),
        )),
    )
    .await
    .unwrap();
    let file = reqwest::Client::new()
        .get(format!("{base}/client/v1/files/f-1/not-signed/a.png"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        file.status(),
        404,
        "the gateway's own answer, past the guard"
    );
    listener.close().await;
}

/// A watch gets compact frames and may only deny above its risk ceiling;
/// the first answer wins and a later one gets `conflict`.
#[tokio::test]
async fn a_compact_watch_denies_above_its_ceiling_and_the_first_answer_wins() {
    let f = fixture(RiskLevel::High).await;
    let watch = [
        ("x-aidaemon-client", "watch"),
        ("x-aidaemon-device", "Synthetic Watch"),
    ];
    f.post(&watch, &command("w-1", send("Please approve the report")))
        .await;
    let mut since = 0;
    let events = f
        .poll_until(&watch, &mut since, |e| {
            matches!(e, Event::ApprovalRequested { .. })
        })
        .await;
    let Event::ApprovalRequested { approval } = events.last().unwrap() else {
        unreachable!()
    };
    assert_eq!(approval.risk, Risk::High);
    assert_eq!(approval.detail, None, "compact frames leave out the detail");
    assert_eq!(
        approval.options,
        [
            ApprovalDecision::ApproveOnce,
            ApprovalDecision::ApproveAlways,
            ApprovalDecision::Deny
        ]
    );

    // The glance a watch reads is compact too.
    let glance = f.post(&watch, &command("w-g", Command::GetGlance)).await;
    let ServerFrame::Ack {
        result: CommandResult::Glance { glance },
        ..
    } = glance
    else {
        panic!("{glance:?}");
    };
    assert_eq!(glance.pending_approvals.len(), 1);
    assert_eq!(glance.running_tasks.len(), 1);

    let resolve = |id: &str, decision| {
        command(
            id,
            Command::ResolveApproval {
                approval_id: approval.id.clone(),
                decision,
            },
        )
    };
    let refused = f
        .post(&watch, &resolve("w-2", ApprovalDecision::ApproveOnce))
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
        "{refused:?}"
    );
    let denied = f
        .post(&watch, &resolve("w-3", ApprovalDecision::Deny))
        .await;
    assert!(
        matches!(
            denied,
            ServerFrame::Ack {
                result: CommandResult::ApprovalResolved {
                    outcome: ApprovalOutcome::Denied,
                    ..
                },
                ..
            }
        ),
        "{denied:?}"
    );
    // A phone answering afterwards is told who won.
    let late = f
        .post(&[], &resolve("p-1", ApprovalDecision::ApproveOnce))
        .await;
    let ServerFrame::Error { error, .. } = &late else {
        panic!("{late:?}");
    };
    assert_eq!(error.code, ErrorCode::Conflict);
    assert!(error.message.contains("denied"), "{}", error.message);

    let events = f
        .poll_until(&watch, &mut since, |e| {
            matches!(e, Event::TurnCompleted { .. })
        })
        .await;
    assert!(events.iter().any(|e| matches!(
        e,
        Event::ApprovalResolved {
            outcome: ApprovalOutcome::Denied,
            resolved_by: Some(ResolvedBy { channel, device: Some(device) }),
            ..
        } if channel == "app" && device == "Synthetic Watch"
    )));
    let Event::TurnCompleted { text, .. } = events.last().unwrap() else {
        unreachable!()
    };
    assert_eq!(text, "Not published.");
}

/// Over a socket, a compact client gets no turn streaming; a full one does.
#[tokio::test]
async fn compact_sockets_get_no_turn_streaming() {
    let f = fixture(RiskLevel::Medium).await;
    async fn open(f: &Fixture, compact: bool) -> Socket {
        let (mut socket, _) =
            tokio_tungstenite::connect_async(format!("ws://{}/client/v1/ws", f.url))
                .await
                .unwrap();
        let hello = ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: ClientKind::Tui,
                name: "test".into(),
                version: None,
                device: None,
            },
            capabilities: if compact {
                vec![Capability::Compact]
            } else {
                Vec::new()
            },
            resume_from: Some(0),
            token: Some(f.token.clone()),
        };
        socket
            .send(Message::text(serde_json::to_string(&hello).unwrap()))
            .await
            .unwrap();
        socket
    }
    let mut compact = open(&f, true).await;
    let mut full = open(&f, false).await;
    assert!(matches!(
        next(&mut compact).await,
        ServerFrame::Welcome { .. }
    ));
    assert!(matches!(next(&mut full).await, ServerFrame::Welcome { .. }));
    f.post(&[], &command("c-1", send("Build it"))).await;
    let saw_status = |frames: &[ServerFrame]| {
        frames.iter().any(|frame| {
            matches!(
                frame,
                ServerFrame::Event {
                    event: Event::TurnStatus { .. },
                    ..
                }
            )
        })
    };
    let compact_frames = until_completed(&mut compact).await;
    let full_frames = until_completed(&mut full).await;
    assert!(!saw_status(&compact_frames), "{compact_frames:?}");
    assert!(saw_status(&full_frames), "{full_frames:?}");
    // The reply streams as it is written: a full client gets it as
    // `turn_delta` chunks, a compact one gets none.
    let streamed = |frames: &[ServerFrame]| {
        frames
            .iter()
            .filter_map(|frame| match frame {
                ServerFrame::Event {
                    event: Event::TurnDelta { text, .. },
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>()
    };
    assert_eq!(streamed(&full_frames), "Done.", "{full_frames:?}");
    assert_eq!(streamed(&compact_frames), "", "{compact_frames:?}");
}

type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn next(socket: &mut Socket) -> ServerFrame {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(10), socket.next())
            .await
            .expect("a frame in time")
            .expect("socket open")
            .unwrap();
        if let Message::Text(text) = message {
            return serde_json::from_str(&text).unwrap();
        }
    }
}

async fn until_completed(socket: &mut Socket) -> Vec<ServerFrame> {
    let mut frames = Vec::new();
    loop {
        let frame = next(socket).await;
        let done = matches!(
            frame,
            ServerFrame::Event {
                event: Event::TurnCompleted { .. },
                ..
            }
        );
        frames.push(frame);
        if done {
            // Status updates travel on their own task and may trail the
            // reply by a moment.
            while let Ok(Some(Ok(Message::Text(text)))) =
                tokio::time::timeout(Duration::from_millis(300), socket.next()).await
            {
                frames.push(serde_json::from_str(&text).unwrap());
            }
            return frames;
        }
    }
}

#[tokio::test]
async fn a_socket_with_a_bad_token_or_major_is_refused() {
    let f = fixture(RiskLevel::Medium).await;
    for (token, version, code) in [
        (
            "acg_wrong",
            ProtocolVersion::CURRENT,
            ErrorCode::Unauthorized,
        ),
        (
            f.token.as_str(),
            ProtocolVersion { major: 2, minor: 0 },
            ErrorCode::VersionUnsupported,
        ),
    ] {
        let (mut socket, _) =
            tokio_tungstenite::connect_async(format!("ws://{}/client/v1/ws", f.url))
                .await
                .unwrap();
        let hello = ClientFrame::Hello {
            version,
            client: ClientInfo {
                kind: ClientKind::Script,
                name: "test".into(),
                version: None,
                device: None,
            },
            capabilities: Vec::new(),
            resume_from: None,
            token: Some(token.to_string()),
        };
        socket
            .send(Message::text(serde_json::to_string(&hello).unwrap()))
            .await
            .unwrap();
        let frame = next(&mut socket).await;
        assert!(
            matches!(&frame, ServerFrame::Error { error, .. } if error.code == code),
            "{frame:?}"
        );
    }
}

/// A card on another chat: whoever answers first settles it, and the
/// other side is told.
#[tokio::test]
async fn an_answer_from_the_chat_settles_the_card_for_clients() {
    let f = fixture(RiskLevel::Medium).await;
    let (answer_tx, answer_rx) = oneshot::channel();
    let chat = Arc::new(Chat {
        answer: Mutex::new(Some(answer_rx)),
        asked: AtomicUsize::new(0),
    });
    f.hub.register_channel(chat.clone()).await;
    let session = "telegram:synthetic-user-1";
    let runner = aidaemon_channels::turns::TurnRunner::new(f.deps.clone());
    runner.register_session(session, "telegram").await;
    runner
        .submit(
            Arc::new(Quiet),
            aidaemon_channels::turns::Turn {
                session_id: session.into(),
                text: "approve the report".into(),
                attachments: Vec::new(),
                user_role: aidaemon_tools::semantics::UserRole::Owner,
                channel_ctx: aidaemon_channels::context::ChannelContext::private("telegram"),
                dedup_key: Some("m1".into()),
                ingress_timing: None,
            },
        )
        .await;
    let mut since = 0;
    let events = f
        .poll_until(&[], &mut since, |e| {
            matches!(e, Event::ApprovalRequested { .. })
        })
        .await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::TurnStarted { origin, .. } if origin == "test"))
    );
    let Event::ApprovalRequested { approval } = events.last().unwrap() else {
        unreachable!()
    };
    assert_eq!(approval.channel.as_deref(), Some("telegram"));
    answer_tx.send(ApprovalResponse::AllowOnce).unwrap();
    let events = f
        .poll_until(&[], &mut since, |e| {
            matches!(e, Event::TurnCompleted { .. })
        })
        .await;
    assert!(events.iter().any(|e| matches!(
        e,
        Event::ApprovalResolved {
            outcome: ApprovalOutcome::ApprovedOnce,
            resolved_by: Some(ResolvedBy { channel, .. }),
            ..
        } if channel == "telegram"
    )));
    let late = f
        .post(
            &[],
            &command(
                "late",
                Command::ResolveApproval {
                    approval_id: approval.id.clone(),
                    decision: ApprovalDecision::Deny,
                },
            ),
        )
        .await;
    assert!(
        matches!(&late, ServerFrame::Error { error, .. } if error.code == ErrorCode::Conflict),
        "{late:?}"
    );
    assert_eq!(chat.asked.load(Ordering::SeqCst), 1);
    let _ = &f.gateway;
}

/// A surface that shows nothing (the test turn's own platform).
struct Quiet;

#[async_trait]
impl aidaemon_channels::turns::TurnSurface for Quiet {
    fn platform(&self) -> &'static str {
        "test"
    }
    async fn say(&self, _: &str) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn show_working(
        &self,
        _: tokio_util::sync::CancellationToken,
        _: Arc<std::sync::atomic::AtomicU64>,
        _: u64,
    ) {
    }
}

// ---------------------------------------------------------------------------
// Pairing
// ---------------------------------------------------------------------------

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature, SigningKey};
use p256::elliptic_curve::Generate;

/// A device: its key, and what it was given.
struct Device {
    key: SigningKey,
    device_id: String,
}

impl Fixture {
    fn http(&self) -> reqwest::Client {
        reqwest::Client::new()
    }

    async fn owner_post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> T {
        let response = self
            .http()
            .post(format!("http://{}{path}", self.url))
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(
            response.status().is_success(),
            "{path}: {}",
            response.status()
        );
        response.json().await.unwrap()
    }

    async fn redeem(
        &self,
        offer: &PairingOffer,
        key: &SigningKey,
        scopes: Vec<Scope>,
    ) -> reqwest::Response {
        self.http()
            .post(format!("http://{}/client/v1/enrollments/redeem", self.url))
            .json(&RedeemRequest {
                offer_id: offer.offer_id.clone(),
                code: offer.code.to_lowercase(),
                device_name: "Synthetic Phone".into(),
                device_kind: ClientKind::Web,
                public_key: URL_SAFE_NO_PAD
                    .encode(key.verifying_key().to_sec1_point(false).as_bytes()),
                scopes,
            })
            .send()
            .await
            .unwrap()
    }

    /// Pair a device, confirmed by the owner's decision call.
    async fn pair(&self, scopes: Vec<Scope>) -> Device {
        let offer: PairingOffer = self
            .owner_post("/client/v1/pairing/offers", serde_json::json!({}))
            .await;
        let key = SigningKey::generate_from_rng(&mut rand::rng());
        assert_eq!(self.redeem(&offer, &key, scopes).await.status(), 202);
        let enrollment: Enrollment = self
            .owner_post(
                &format!("/client/v1/pairing/offers/{}/decision", offer.offer_id),
                serde_json::json!({"approve": true}),
            )
            .await;
        assert_eq!(enrollment.state, EnrollmentState::Approved);
        Device {
            key,
            device_id: enrollment.device_id.unwrap(),
        }
    }

    async fn sign_in(&self, device: &Device) -> reqwest::Response {
        let response = self
            .http()
            .post(format!("http://{}/client/v1/sessions/challenge", self.url))
            .json(&ChallengeRequest {
                device_id: device.device_id.clone(),
            })
            .send()
            .await
            .unwrap();
        if !response.status().is_success() {
            return response;
        }
        let challenge: Challenge = response.json().await.unwrap();
        let signature: Signature = device
            .key
            .sign(challenge_message(&challenge.challenge_id, &challenge.nonce).as_bytes());
        self.http()
            .post(format!("http://{}/client/v1/sessions", self.url))
            .json(&SessionRequest {
                challenge_id: challenge.challenge_id,
                signature: URL_SAFE_NO_PAD.encode(signature.to_bytes()),
            })
            .send()
            .await
            .unwrap()
    }

    async fn post_as(&self, token: &str, frame: &ClientFrame) -> ServerFrame {
        self.http()
            .post(format!("http://{}/client/v1/commands", self.url))
            .bearer_auth(token)
            .json(frame)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn a_device_pairs_signs_in_and_is_held_to_its_scopes() {
    let f = fixture(RiskLevel::Medium).await;
    // A wrong code does not redeem.
    let offer: PairingOffer = f
        .owner_post("/client/v1/pairing/offers", serde_json::json!({}))
        .await;
    assert!(offer.link.starts_with("aidaemon-pair:?gateway="));
    let mut wrong = offer.clone();
    wrong.code = "AAAA-AAAA".into();
    let key = SigningKey::generate_from_rng(&mut rand::rng());
    assert_eq!(f.redeem(&wrong, &key, Vec::new()).await.status(), 401);

    // A screen that may only watch.
    let kiosk = f.pair(vec![Scope::Observe]).await;
    let grant: AccessGrant = f.sign_in(&kiosk).await.json().await.unwrap();
    assert_eq!(grant.scopes, vec![Scope::Observe]);
    let refused = f
        .post_as(&grant.access_token, &command("k-1", send("hi")))
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
        "{refused:?}"
    );
    let glance = f
        .post_as(&grant.access_token, &command("k-2", Command::GetGlance))
        .await;
    assert!(matches!(glance, ServerFrame::Ack { .. }), "{glance:?}");
    // Devices cannot manage devices.
    let response = f
        .http()
        .get(format!("http://{}/client/v1/devices", f.url))
        .bearer_auth(&grant.access_token)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 403);

    // A signature from another key does not sign in.
    let impostor = Device {
        key: SigningKey::generate_from_rng(&mut rand::rng()),
        device_id: kiosk.device_id.clone(),
    };
    assert_eq!(f.sign_in(&impostor).await.status(), 401);

    let devices: Vec<DeviceInfo> = f
        .http()
        .get(format!("http://{}/client/v1/devices", f.url))
        .bearer_auth(&f.token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].name, "Synthetic Phone");
    assert!(devices[0].last_seen_ms.is_some());
}

#[tokio::test]
async fn revoking_a_device_closes_its_socket_and_its_tokens() {
    let f = fixture(RiskLevel::Medium).await;
    let phone = f.pair(Vec::new()).await;
    let grant: AccessGrant = f.sign_in(&phone).await.json().await.unwrap();
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{}/client/v1/ws", f.url))
        .await
        .unwrap();
    let hello = ClientFrame::Hello {
        version: ProtocolVersion::CURRENT,
        client: ClientInfo {
            kind: ClientKind::Web,
            name: "web".into(),
            version: None,
            device: None,
        },
        capabilities: Vec::new(),
        resume_from: None,
        token: Some(grant.access_token.clone()),
    };
    socket
        .send(Message::text(serde_json::to_string(&hello).unwrap()))
        .await
        .unwrap();
    assert!(matches!(
        next(&mut socket).await,
        ServerFrame::Welcome { .. }
    ));
    let _: Vec<DeviceInfo> = f
        .owner_post(
            &format!("/client/v1/devices/{}/revoke", phone.device_id),
            serde_json::json!({}),
        )
        .await;
    loop {
        let frame = next(&mut socket).await;
        if let ServerFrame::Error { error, .. } = frame {
            assert_eq!(error.code, ErrorCode::Unauthorized);
            break;
        }
    }
    let response = f
        .http()
        .post(format!("http://{}/client/v1/commands", f.url))
        .bearer_auth(&grant.access_token)
        .json(&command("r-1", Command::GetGlance))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    assert_eq!(f.sign_in(&phone).await.status(), 401);
}

/// An offer made in a chat is confirmed there, with an approval card.
#[tokio::test]
async fn a_pairing_made_in_a_chat_is_confirmed_there() {
    let f = fixture(RiskLevel::Medium).await;
    let (answer_tx, answer_rx) = oneshot::channel();
    let chat = Arc::new(Chat {
        answer: Mutex::new(Some(answer_rx)),
        asked: AtomicUsize::new(0),
    });
    f.hub.register_channel(chat.clone()).await;
    aidaemon_channels::turns::TurnRunner::new(f.deps.clone())
        .register_session("telegram:synthetic-user-1", "telegram")
        .await;
    let offer: PairingOffer = f
        .owner_post(
            "/client/v1/pairing/offers",
            serde_json::json!({"confirm_in_session": "telegram:synthetic-user-1"}),
        )
        .await;
    let key = SigningKey::generate_from_rng(&mut rand::rng());
    let pending: Enrollment = f
        .redeem(&offer, &key, Vec::new())
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(pending.state, EnrollmentState::Pending);
    assert_eq!(pending.fingerprint, offer.fingerprint);
    for _ in 0..200 {
        if chat.asked.load(Ordering::SeqCst) > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    answer_tx.send(ApprovalResponse::AllowOnce).unwrap();
    let mut state = EnrollmentState::Pending;
    for _ in 0..200 {
        let enrollment: Enrollment = f
            .http()
            .get(format!(
                "http://{}/client/v1/enrollments/{}",
                f.url, offer.offer_id
            ))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        state = enrollment.state;
        if state != EnrollmentState::Pending {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(state, EnrollmentState::Approved);
}

/// Without an installed web build, `/client/v1/app` says so and how to get
/// it; the old built-in client's modules are gone.
#[tokio::test]
async fn the_app_path_explains_a_missing_web_build() {
    let f = fixture(RiskLevel::Medium).await;
    let page = f
        .http()
        .get(format!("http://{}/client/v1/app", f.url))
        .send()
        .await
        .unwrap();
    assert_eq!(page.status(), 200);
    assert!(
        page.headers()["content-security-policy"]
            .to_str()
            .unwrap()
            .contains("default-src 'none'")
    );
    let text = page.text().await.unwrap();
    assert!(text.contains("not installed"));
    assert!(text.contains("nsh package --target web"));
    assert!(!text.contains("<script"));
    let gone = f
        .http()
        .get(format!("http://{}/client/v1/app/app.js", f.url))
        .send()
        .await
        .unwrap();
    assert_eq!(gone.status(), 404);
}

/// A pairing link opens the gateway's own page, which hands the offer in
/// its fragment to the phone app or the web app; nothing else serves it.
#[tokio::test]
async fn the_pairing_page_is_served_by_the_gateway() {
    let f = fixture(RiskLevel::Medium).await;
    let page = f
        .http()
        .get(format!("http://{}/client/v1/pair", f.url))
        .send()
        .await
        .unwrap();
    assert_eq!(page.status(), 200);
    assert_eq!(page.headers()["referrer-policy"], "no-referrer");
    let text = page.text().await.unwrap();
    assert!(text.contains("scheme=aidaemon-pair"));
    assert!(text.contains("'/client/v1/app#offer='"));
    assert!(!text.contains("__WEB_APP__"));
}

/// The shared client core pairs a device with its own key, signs in with a
/// signed challenge and holds a socket on the grant, against this router.
#[tokio::test]
async fn the_client_core_pairs_and_signs_in_with_a_device_key() {
    use aidaemon_client_core::keystore::{KeyStore, MemoryKeyStore};
    use aidaemon_client_core::link::{self, Auth, ClientIdentity, LinkConfig, Update};
    use aidaemon_client_core::signin::{self, PairRequest};

    let f = fixture(RiskLevel::Medium).await;
    let offer: PairingOffer = f
        .owner_post("/client/v1/pairing/offers", serde_json::json!({}))
        .await;
    let base = format!("http://{}", f.url);
    let request = PairRequest {
        gateway: base.clone(),
        offer_id: offer.offer_id.clone(),
        code: offer.code.clone(),
        device_name: "Synthetic Desktop".into(),
        device_kind: ClientKind::Desktop,
        fingerprint: Some(offer.fingerprint.clone()),
    };
    let pairing = tokio::spawn(async move {
        signin::pair(&reqwest::Client::new(), &request, Duration::from_millis(20)).await
    });
    // The owner confirms once the device has redeemed the offer.
    let mut decided = false;
    for _ in 0..200 {
        let response = f
            .http()
            .post(format!(
                "{base}/client/v1/pairing/offers/{}/decision",
                offer.offer_id
            ))
            .bearer_auth(&f.token)
            .json(&serde_json::json!({"approve": true}))
            .send()
            .await
            .unwrap();
        if response.status().is_success() {
            decided = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(decided, "the offer was never redeemed");
    let device = pairing.await.unwrap().unwrap();
    assert_eq!(device.gateway, base);

    // A direct sign-in gets a grant for this device.
    let grant = signin::sign_in(&reqwest::Client::new(), &device)
        .await
        .unwrap();
    assert_eq!(grant.device_id, device.device_id);

    // The link signs in by itself and speaks the protocol on the grant.
    let store = Arc::new(MemoryKeyStore::default());
    store.save(&device).unwrap();
    let (updates_tx, mut updates) = mpsc::unbounded_channel();
    let commands = link::spawn(
        LinkConfig {
            url: format!("ws://{}/client/v1/ws", f.url),
            auth: Auth::Device(store),
            client: ClientIdentity {
                kind: ClientKind::Desktop,
                name: "synthetic-desktop".into(),
                version: None,
            },
            device: None,
            capabilities: Vec::new(),
        },
        updates_tx,
    );
    commands
        .send(("core-1".into(), Command::GetGlance))
        .unwrap();
    let mut welcomed = false;
    loop {
        let update = tokio::time::timeout(Duration::from_secs(10), updates.recv())
            .await
            .expect("an update in time")
            .expect("the link runs");
        match update {
            Update::Disconnected(disconnect) => panic!("dropped: {disconnect:?}"),
            Update::Frame(ServerFrame::Welcome { .. }) => welcomed = true,
            Update::Frame(ServerFrame::Ack { request_id, .. }) if request_id == "core-1" => break,
            _ => {}
        }
    }
    assert!(welcomed);
}

// ---------------------------------------------------------------------------
// The agent-driven view
// ---------------------------------------------------------------------------

/// A socket with these capabilities, past its welcome.
async fn socket_with(f: &Fixture, capabilities: Vec<Capability>) -> Socket {
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{}/client/v1/ws", f.url))
        .await
        .unwrap();
    let hello = ClientFrame::Hello {
        version: ProtocolVersion::CURRENT,
        client: ClientInfo {
            kind: ClientKind::Tui,
            name: "test".into(),
            version: None,
            device: None,
        },
        capabilities,
        resume_from: Some(0),
        token: Some(f.token.clone()),
    };
    socket
        .send(Message::text(serde_json::to_string(&hello).unwrap()))
        .await
        .unwrap();
    let ServerFrame::Welcome {
        server_capabilities,
        ..
    } = next(&mut socket).await
    else {
        panic!("welcome first");
    };
    assert!(server_capabilities.contains(&Capability::ViewActions));
    socket
}

/// Send `frame` on `socket` and read until its ack; every frame before it.
async fn command_on(socket: &mut Socket, frame: &ClientFrame) -> (CommandResult, Vec<ServerFrame>) {
    socket
        .send(Message::text(serde_json::to_string(frame).unwrap()))
        .await
        .unwrap();
    let mut seen = Vec::new();
    loop {
        match next(socket).await {
            ServerFrame::Ack { result, .. } => return (result, seen),
            other => seen.push(other),
        }
    }
}

/// Every view action the socket gets within `wait`.
async fn view_actions_within(socket: &mut Socket, wait: Duration) -> Vec<ViewAction> {
    let mut actions = Vec::new();
    let until = tokio::time::Instant::now() + wait;
    while let Ok(Some(Ok(message))) = tokio::time::timeout_at(until, socket.next()).await {
        if let Message::Text(text) = message
            && let ServerFrame::Event {
                event: Event::ViewAction { action, .. },
                seq,
                ..
            } = serde_json::from_str(&text).unwrap()
        {
            assert_eq!(seq, None, "view actions are never logged");
            actions.push(action);
        }
    }
    actions
}

/// The view action goes to the connection that sent the turn's message and
/// declared the capability; a client without it, or one that only watches,
/// is never sent the event, and a message from a client that cannot be
/// moved leaves the session without a target.
#[tokio::test]
async fn a_view_action_reaches_only_the_capable_sender() {
    let f = fixture(RiskLevel::Medium).await;
    let mut sender = socket_with(&f, vec![Capability::ViewActions]).await;
    let mut plain = socket_with(&f, Vec::new()).await;
    let mut watcher = socket_with(&f, vec![Capability::ViewActions]).await;
    let (result, _) =
        command_on(&mut sender, &command("v-1", send("Show me what's waiting"))).await;
    let CommandResult::MessageAccepted { session_id, .. } = result else {
        panic!("accepted");
    };
    assert!(f.gateway.views().can_show(&session_id));
    f.gateway
        .views()
        .show(
            &session_id,
            ViewAction::OpenSession {
                session_id: session_id.clone(),
            },
            "You asked to see it",
        )
        .unwrap();
    assert_eq!(
        view_actions_within(&mut sender, Duration::from_millis(500)).await,
        vec![ViewAction::OpenSession {
            session_id: session_id.clone()
        }]
    );
    assert!(
        view_actions_within(&mut plain, Duration::from_millis(300))
            .await
            .is_empty()
    );
    assert!(
        view_actions_within(&mut watcher, Duration::from_millis(300))
            .await
            .is_empty()
    );
    // Nothing of it reached the log, so no later replay can move a view.
    let logged = f.gateway.bus().after(0, 500).await.unwrap();
    assert!(!logged.iter().any(|frame| matches!(
        frame,
        ServerFrame::Event {
            event: Event::ViewAction { .. },
            ..
        }
    )));

    // A message into the same session over HTTP (no capability) clears the
    // target: the next turn cannot move anyone.
    let ack = f
        .post(
            &[],
            &command(
                "v-2",
                Command::SendMessage {
                    session_id: Some(session_id.clone()),
                    text: "Carry on".into(),
                    attachments: Vec::new(),
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: None,
                },
            ),
        )
        .await;
    assert!(matches!(ack, ServerFrame::Ack { .. }), "{ack:?}");
    assert!(!f.gateway.views().can_show(&session_id));
    assert_eq!(
        f.gateway
            .views()
            .show(&session_id, ViewAction::NewSession, "again"),
        Err(aidaemon_client_gateway::ViewError::NoClient)
    );
    // A session of another chat never has a target.
    assert!(!f.gateway.views().can_show("telegram:synthetic-user-1"));
}

/// No view action can resolve an approval: with a card waiting, every kind
/// of view action (pointing at that very card where it can) is delivered,
/// and the card is still pending, no answer is logged and the turn that
/// asked is still waiting for the owner.
#[tokio::test]
async fn no_view_action_can_resolve_an_approval() {
    let f = fixture(RiskLevel::Medium).await;
    let mut socket = socket_with(&f, vec![Capability::ViewActions]).await;
    let (result, _) = command_on(&mut socket, &command("v-1", send("Please approve this"))).await;
    let CommandResult::MessageAccepted { session_id, .. } = result else {
        panic!("accepted");
    };
    let approval_id = loop {
        if let ServerFrame::Event {
            event: Event::ApprovalRequested { approval },
            ..
        } = next(&mut socket).await
        {
            break approval.id;
        }
    };
    let every_action = [
        ViewAction::OpenSession {
            session_id: session_id.clone(),
        },
        ViewAction::ShowApproval {
            approval_id: approval_id.clone(),
        },
        ViewAction::ShowTask {
            task_id: "any".into(),
        },
        ViewAction::FocusPane {
            pane: Pane::Approvals,
        },
        ViewAction::NewSession,
        ViewAction::ShowDevices,
        ViewAction::ShowSurface {
            surface_id: "srf_synthetic".into(),
        },
        ViewAction::Unknown,
    ];
    // The list is every kind there is: adding one fails to compile here.
    for action in &every_action {
        match action {
            ViewAction::OpenSession { .. }
            | ViewAction::ShowApproval { .. }
            | ViewAction::ShowTask { .. }
            | ViewAction::FocusPane { .. }
            | ViewAction::NewSession
            | ViewAction::ShowDevices
            | ViewAction::ShowSurface { .. }
            | ViewAction::Unknown => {}
        }
    }
    for action in &every_action {
        f.gateway
            .views()
            .show(&session_id, action.clone(), "Look at this")
            .unwrap();
    }
    let delivered = view_actions_within(&mut socket, Duration::from_millis(800)).await;
    assert_eq!(delivered, every_action.to_vec());
    let (_, snapshot) = f.gateway.snapshot().await;
    assert!(
        snapshot
            .pending_approvals
            .iter()
            .any(|a| a.id == approval_id),
        "the card is still waiting"
    );
    // The turn waiting on it still runs: a client connecting now sees it.
    assert!(
        snapshot
            .live_turns
            .iter()
            .any(|turn| turn.session_id == session_id),
        "the waiting turn is live in the snapshot"
    );
    let logged = f.gateway.bus().after(0, 500).await.unwrap();
    assert!(!logged.iter().any(|frame| matches!(
        frame,
        ServerFrame::Event {
            event: Event::ApprovalResolved { .. } | Event::TurnCompleted { .. },
            ..
        }
    )));
    // Only the owner's own command answers it.
    let (result, _) = command_on(
        &mut socket,
        &command(
            "v-2",
            Command::ResolveApproval {
                approval_id: approval_id.clone(),
                decision: ApprovalDecision::Deny,
            },
        ),
    )
    .await;
    assert!(
        matches!(result, CommandResult::ApprovalResolved { .. }),
        "{result:?}"
    );
}

#[tokio::test]
async fn a_marked_up_picture_tells_the_agent_where_the_unmarked_one_is() {
    use base64::Engine as _;
    let f = fixture(RiskLevel::Medium).await;
    let upload = |id: &'static str, name: &'static str, marked_from: Option<String>| {
        command(
            id,
            Command::UploadFile {
                name: name.into(),
                mime: None,
                data: base64::engine::general_purpose::STANDARD.encode(
                    [b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".as_slice(), name.as_bytes()].concat(),
                ),
                marked_from,
            },
        )
    };
    let uploaded = |ack: ServerFrame| match ack {
        ServerFrame::Ack {
            result: CommandResult::Uploaded { attachment },
            ..
        } => attachment,
        other => panic!("{other:?}"),
    };
    let original = uploaded(f.post(&[], &upload("up-1", "shot.png", None)).await);
    let marked = uploaded(
        f.post(
            &[],
            &upload("up-2", "shot-marked.jpg", Some(original.id.clone())),
        )
        .await,
    );
    assert_eq!(marked.marked_from.as_deref(), Some(original.id.as_str()));
    // A file the gateway does not have is not noted.
    let stray = uploaded(
        f.post(&[], &upload("up-3", "other.png", Some("0".repeat(24))))
            .await,
    );
    assert_eq!(stray.marked_from, None);

    let ack = f
        .post(
            &[],
            &command(
                "r-marked",
                Command::SendMessage {
                    session_id: None,
                    text: "What is this part?".into(),
                    attachments: vec![marked.clone()],
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: None,
                },
            ),
        )
        .await;
    assert!(
        matches!(
            ack,
            ServerFrame::Ack {
                result: CommandResult::MessageAccepted { .. },
                ..
            }
        ),
        "{ack:?}"
    );
    let mut since = 0;
    let events = f
        .poll_until(&[], &mut since, |e| {
            matches!(e, Event::TurnCompleted { .. })
        })
        .await;
    // The agent's file names the unmarked picture.
    let got = f.agent.files.lock().unwrap().clone();
    assert_eq!(got.len(), 1);
    let unmarked = got[0].marked_up_from.clone().expect("noted as marked up");
    assert!(unmarked.ends_with("shot.png"), "{unmarked}");
    // Clients see the owner's words and the marked picture, still marked.
    let shown = events
        .iter()
        .find_map(|e| match e {
            Event::UserMessage {
                text, attachments, ..
            } => Some((text.clone(), attachments.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(shown.0, "What is this part?");
    assert_eq!(
        shown.1[0].marked_from.as_deref(),
        Some(original.id.as_str())
    );
}

#[tokio::test]
async fn an_uploaded_picture_goes_to_the_agent_and_back_to_clients() {
    use base64::Engine as _;
    let f = fixture(RiskLevel::Medium).await;
    let png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDRsynthetic";
    let ack = f
        .post(
            &[],
            &command(
                "up-1",
                Command::UploadFile {
                    name: "receipt.png".into(),
                    mime: None,
                    data: base64::engine::general_purpose::STANDARD.encode(png),
                    marked_from: None,
                },
            ),
        )
        .await;
    let ServerFrame::Ack {
        result: CommandResult::Uploaded { attachment },
        ..
    } = ack
    else {
        panic!("{ack:?}");
    };
    assert_eq!(attachment.mime.as_deref(), Some("image/png"));
    let url = attachment.url.clone().unwrap();

    // Served at its link with no token, as a picture; a wrong link is not.
    let fetched = reqwest::get(format!("http://{}{url}", f.url))
        .await
        .unwrap();
    assert_eq!(fetched.status(), 200);
    assert_eq!(fetched.headers()["content-type"], "image/png");
    assert!(
        fetched.headers()["content-disposition"]
            .to_str()
            .unwrap()
            .starts_with("inline")
    );
    assert_eq!(fetched.bytes().await.unwrap().as_ref(), png);
    let forged = url.replacen(&attachment.id, &"0".repeat(24), 1);
    assert_eq!(
        reqwest::get(format!("http://{}{forged}", f.url))
            .await
            .unwrap()
            .status(),
        404
    );

    // A message with only the picture.
    let ack = f
        .post(
            &[],
            &command(
                "r-files",
                Command::SendMessage {
                    session_id: None,
                    text: String::new(),
                    attachments: vec![attachment.clone()],
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: None,
                },
            ),
        )
        .await;
    assert!(
        matches!(
            ack,
            ServerFrame::Ack {
                result: CommandResult::MessageAccepted { .. },
                ..
            }
        ),
        "{ack:?}"
    );
    let mut since = 0;
    let events = f
        .poll_until(&[], &mut since, |e| {
            matches!(e, Event::TurnCompleted { .. })
        })
        .await;
    let shown = events
        .iter()
        .find_map(|e| match e {
            Event::UserMessage {
                text, attachments, ..
            } => Some((text.clone(), attachments.clone())),
            _ => None,
        })
        .unwrap();
    // Clients see the picture, not the note the agent read about it.
    assert_eq!(shown.0, "");
    assert_eq!(shown.1, vec![attachment]);
    let got = f.agent.files.lock().unwrap().clone();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].mime_type, "image/png");
    assert_eq!(std::fs::read(&got[0].local_path).unwrap(), png);

    // An unknown file is refused, and nothing is sent.
    let ack = f
        .post(
            &[],
            &command(
                "r-missing",
                Command::SendMessage {
                    session_id: None,
                    text: "and this".into(),
                    attachments: vec![AttachmentRef {
                        marked_from: None,
                        height: None,
                        width: None,
                        id: "0".repeat(24),
                        name: "gone.png".into(),
                        mime: None,
                        size_bytes: None,
                        url: None,
                    }],
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: None,
                },
            ),
        )
        .await;
    assert!(matches!(ack, ServerFrame::Error { .. }), "{ack:?}");
}

#[tokio::test]
async fn a_picture_the_agent_sends_reaches_clients() {
    let f = fixture(RiskLevel::Medium).await;
    let session = "app:synthetic-media";
    aidaemon_channels::turns::TurnRunner::new(f.deps.clone())
        .register_session(session, aidaemon_client_gateway::APP_CHANNEL)
        .await;
    f.hub
        .send_media(
            session,
            &MediaMessage {
                session_id: session.into(),
                caption: "The chart".into(),
                kind: aidaemon_channels::hub::MediaKind::Photo {
                    data: b"\x89PNG\r\n\x1a\n\0\0\0\rIHDRchart".to_vec(),
                },
                result_tx: None,
            },
        )
        .await
        .unwrap();
    let mut since = 0;
    let events = f
        .poll_until(
            &[],
            &mut since,
            |e| matches!(e, Event::AssistantMessage { attachments, .. } if !attachments.is_empty()),
        )
        .await;
    let Some(Event::AssistantMessage {
        text, attachments, ..
    }) = events.last()
    else {
        panic!()
    };
    assert_eq!(text, "The chart");
    assert!(attachments[0].is_image());
}

/// A conversation of another chat keeps that chat's model: setting one
/// there, alone or with a message, is refused; reading one answers
/// automatic.
#[tokio::test]
async fn only_app_conversations_take_a_model() {
    let f = fixture(RiskLevel::Medium).await;
    let pin = || ModelChoice::Pinned {
        provider: "main".into(),
        model: "synthetic-model-large".into(),
    };
    let refused = f
        .post(
            &[],
            &command(
                "m-1",
                Command::SetSessionModel {
                    session_id: "telegram:synthetic-user-1".into(),
                    model: Some(pin()),
                    effort: None,
                },
            ),
        )
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Invalid),
        "{refused:?}"
    );
    let refused = f
        .post(
            &[],
            &command(
                "m-2",
                Command::SendMessage {
                    session_id: Some("telegram:synthetic-user-1".into()),
                    text: "hello".into(),
                    attachments: Vec::new(),
                    mid_run: None,
                    project_id: None,
                    model: None,
                    effort: Some(Effort::High),
                },
            ),
        )
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Invalid),
        "{refused:?}"
    );
    let read = f
        .post(
            &[],
            &command(
                "m-3",
                Command::GetSessionModel {
                    session_id: "app:synthetic".into(),
                },
            ),
        )
        .await;
    assert!(
        matches!(&read, ServerFrame::Ack { result: CommandResult::SessionModel { session_model }, .. }
            if session_model.model == ModelChoice::Auto && session_model.effort == Effort::Auto),
        "{read:?}"
    );
}

// ---------------------------------------------------------------------------
// Live surfaces (protocol 1.30)
// ---------------------------------------------------------------------------

use aidaemon_client_protocol::media::{self, MediaClientMessage, MediaHeader, SurfaceInput};
use aidaemon_surfaces::testing::MockSurface;
use aidaemon_surfaces::{SurfaceRegistry, SurfacesConfig};

async fn with_surfaces(f: &Fixture) -> (SurfaceRegistry, MockSurface, String) {
    let registry = SurfaceRegistry::new(SurfacesConfig::default());
    let mock = MockSurface::new();
    mock.set_notifier(registry.notifier());
    registry.add_backend(Arc::new(mock.clone()));
    let id = mock.add("app:surface-test", "tab-1");
    registry.refresh().await;
    f.gateway.attach_surfaces(registry.clone());
    (registry, mock, id.to_string())
}

async fn media_socket(f: &Fixture, transport: &SurfaceTransport, ticket: &str) -> Socket {
    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{}{}", f.url, transport.url))
            .await
            .unwrap();
    let auth = MediaClientMessage::Auth {
        ticket: ticket.into(),
    };
    socket
        .send(Message::text(serde_json::to_string(&auth).unwrap()))
        .await
        .unwrap();
    socket
}

/// The next binary message's header and payload.
async fn media_next(socket: &mut Socket) -> (MediaHeader, Vec<u8>) {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
            .await
            .expect("a media message in time")
            .expect("the socket open")
            .unwrap();
        if let Message::Binary(bytes) = message {
            let (header, payload) = media::decode(&bytes).unwrap();
            return (header, payload.to_vec());
        }
    }
}

async fn media_until_frame(socket: &mut Socket) -> (u64, Vec<u8>) {
    loop {
        if let (MediaHeader::Frame { seq, .. }, payload) = media_next(socket).await {
            return (seq, payload);
        }
    }
}

async fn media_send(socket: &mut Socket, message: &MediaClientMessage) {
    socket
        .send(Message::text(serde_json::to_string(message).unwrap()))
        .await
        .unwrap();
}

fn opened(result: CommandResult) -> (String, SurfaceTransport) {
    let CommandResult::SurfaceOpened {
        live_id,
        mut transports,
        ..
    } = result
    else {
        panic!("surface_opened, got {result:?}");
    };
    (live_id, transports.remove(0))
}

#[tokio::test]
async fn a_surface_streams_the_newest_frame_and_takes_input_only_under_control() {
    let f = fixture(RiskLevel::Medium).await;
    let (registry, mock, surface_id) = with_surfaces(&f).await;
    let mut socket = socket_with(&f, vec![Capability::ViewActions, Capability::Surfaces]).await;
    let (listed, _) = command_on(&mut socket, &command("s-1", Command::ListSurfaces)).await;
    assert!(
        matches!(&listed, CommandResult::Surfaces { surfaces } if surfaces.len() == 1 && surfaces[0].id == surface_id),
        "{listed:?}"
    );
    let (result, _) = command_on(
        &mut socket,
        &command(
            "s-2",
            Command::OpenSurface {
                surface_id: surface_id.clone(),
                mode: SurfaceMode::View,
                accept: StreamAccept::default(),
            },
        ),
    )
    .await;
    let (live_id, transport) = opened(result);
    assert!(
        !transport.url.contains(&transport.ticket),
        "never in the URL"
    );

    // A wrong ticket is refused.
    let mut wrong = media_socket(&f, &transport, "tkt_wrong").await;
    assert!(matches!(
        media_next(&mut wrong).await.0,
        MediaHeader::Closed { .. }
    ));

    let mut media = media_socket(&f, &transport, &transport.ticket).await;
    assert!(matches!(
        media_next(&mut media).await.0,
        MediaHeader::Ready { .. }
    ));
    assert!(matches!(
        media_next(&mut media).await.0,
        MediaHeader::Controller {
            controller: SurfaceController::Agent
        }
    ));
    // The ticket works once.
    let mut again = media_socket(&f, &transport, &transport.ticket).await;
    assert!(matches!(
        media_next(&mut again).await.0,
        MediaHeader::Closed { .. }
    ));

    let surface = aidaemon_surfaces::SurfaceId::from_string(surface_id.clone());
    assert!(mock.push_frame(&surface, b"first"));
    let (seq, payload) = media_until_frame(&mut media).await;
    assert_eq!(payload, b"first");
    // Unacked: newer frames wait, and only the newest is sent after the ack.
    mock.push_frame(&surface, b"second");
    mock.push_frame(&surface, b"third");
    assert!(
        tokio::time::timeout(Duration::from_millis(300), media_until_frame(&mut media))
            .await
            .is_err(),
        "no frame before the ack"
    );
    media_send(&mut media, &MediaClientMessage::Ack { seq }).await;
    let (_, payload) = media_until_frame(&mut media).await;
    assert_eq!(payload, b"third");

    // Watching sends no input.
    let typed = MediaClientMessage::Input {
        event: SurfaceInput::Text {
            text: "synthetic-secret".into(),
        },
    };
    media_send(&mut media, &typed).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(mock.inputs().is_empty());

    // Under control it does.
    let (result, _) = command_on(
        &mut socket,
        &command(
            "s-3",
            Command::TakeControl {
                surface_id: surface_id.clone(),
            },
        ),
    )
    .await;
    assert!(
        matches!(&result, CommandResult::SurfaceControl { surface } if matches!(surface.controller, SurfaceController::Owner { .. })),
        "{result:?}"
    );
    media_send(&mut media, &typed).await;
    for _ in 0..50 {
        if !mock.inputs().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(mock.inputs().len(), 1);
    // After typing, the view is told where the typing went (never what).
    let focus = loop {
        if let (
            MediaHeader::Focus {
                what,
                label,
                rect,
                typing,
            },
            _,
        ) = media_next(&mut media).await
        {
            assert!(rect.is_some(), "the field's place comes with it");
            assert_eq!(typing, Some(media::FocusTyping::Text), "a field takes text");
            break (what, label);
        }
    };
    assert_eq!(
        focus,
        (Some("a text field".into()), Some("Username".into()))
    );

    // The phone turns while in control: the page is laid out for its new
    // room.
    media_send(&mut media, &MediaClientMessage::Fit { w: 800, h: 300 }).await;
    // ...and the field with the keyboard is placed again on the new layout,
    // so the phone's outline follows it.
    loop {
        if let (MediaHeader::Focus { what, rect, .. }, _) = media_next(&mut media).await {
            assert_eq!(what.as_deref(), Some("a text field"));
            assert!(rect.is_some(), "the field's new place comes with it");
            break;
        }
    }
    let turned = Some(aidaemon_surfaces::Size { w: 800, h: 300 });
    for _ in 0..50 {
        if mock.fits().iter().any(|(room, _)| *room == turned) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        mock.fits().iter().any(|(room, _)| *room == turned),
        "{:?}",
        mock.fits()
    );

    // Closing the view ends the stream; nothing typed was kept by the gateway.
    let (result, _) = command_on(
        &mut socket,
        &command("s-4", Command::CloseSurface { live_id }),
    )
    .await;
    assert!(matches!(result, CommandResult::Reply { .. }));
    let (_, snapshot) = f.gateway.snapshot().await;
    let snapshot = serde_json::to_string(&snapshot).unwrap();
    assert!(!snapshot.contains("synthetic-secret"));
    drop(registry);
}

#[tokio::test]
async fn surfaces_need_the_control_scope_and_a_scope_change_applies_at_once() {
    let f = fixture(RiskLevel::Medium).await;
    let (_registry, _mock, surface_id) = with_surfaces(&f).await;
    let phone = f
        .pair(vec![Scope::Observe, Scope::Chat, Scope::Approve])
        .await;
    let grant: AccessGrant = f.sign_in(&phone).await.json().await.unwrap();
    let open = command(
        "c-1",
        Command::OpenSurface {
            surface_id: surface_id.clone(),
            mode: SurfaceMode::View,
            accept: StreamAccept::default(),
        },
    );
    let refused = f.post_as(&grant.access_token, &open).await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
        "{refused:?}"
    );
    let snapshot = f
        .post_as(&grant.access_token, &command("c-2", Command::GetSnapshot))
        .await;
    assert!(
        matches!(&snapshot, ServerFrame::Ack { result: CommandResult::Snapshot { state, .. }, .. } if state.surfaces.is_empty()),
        "a device without control sees no surfaces"
    );
    // The owner grants it: the same token may now.
    let devices: Vec<DeviceInfo> = f
        .owner_post(
            &format!("/client/v1/devices/{}/scopes", phone.device_id),
            serde_json::json!({"scopes": ["observe", "chat", "approve", "control"]}),
        )
        .await;
    assert!(devices[0].scopes.contains(&Scope::Control));
    let allowed = f
        .post_as(
            &grant.access_token,
            &command(
                "c-3",
                Command::OpenSurface {
                    surface_id,
                    mode: SurfaceMode::View,
                    accept: StreamAccept::default(),
                },
            ),
        )
        .await;
    assert!(
        matches!(
            &allowed,
            ServerFrame::Ack {
                result: CommandResult::SurfaceOpened { .. },
                ..
            }
        ),
        "{allowed:?}"
    );
}

#[tokio::test]
async fn a_phone_pairs_with_control_by_default() {
    let f = fixture(RiskLevel::Medium).await;
    let phone = f.pair(Vec::new()).await;
    let grant: AccessGrant = f.sign_in(&phone).await.json().await.unwrap();
    assert!(grant.scopes.contains(&Scope::Control));
}

#[tokio::test]
async fn a_handoff_brings_the_surface_forward_and_control_waits_for_it_when_off() {
    let f = fixture(RiskLevel::Medium).await;
    let registry = SurfaceRegistry::new(SurfacesConfig {
        control_anytime: false,
        ..SurfacesConfig::default()
    });
    let mock = MockSurface::new();
    mock.set_notifier(registry.notifier());
    registry.add_backend(Arc::new(mock.clone()));
    f.gateway.attach_surfaces(registry.clone());
    let mut socket = socket_with(&f, vec![Capability::ViewActions, Capability::Surfaces]).await;
    // A message makes this socket the session's view target.
    let (accepted, _) = command_on(&mut socket, &command("h-0", send("open the site"))).await;
    let CommandResult::MessageAccepted { session_id, .. } = accepted else {
        panic!("{accepted:?}");
    };
    let surface = mock.add(&session_id, "tab-1");
    registry.refresh().await;
    let take = command(
        "h-1",
        Command::TakeControl {
            surface_id: surface.to_string(),
        },
    );
    let refused = f.post(&[], &take).await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
        "{refused:?}"
    );
    registry.begin_handoff(&surface).unwrap();
    let actions = view_actions_within(&mut socket, Duration::from_millis(800)).await;
    assert!(
        actions.contains(&ViewAction::ShowSurface {
            surface_id: surface.to_string()
        }),
        "{actions:?}"
    );
    let allowed = f
        .post(
            &[],
            &command(
                "h-2",
                Command::TakeControl {
                    surface_id: surface.to_string(),
                },
            ),
        )
        .await;
    assert!(matches!(allowed, ServerFrame::Ack { .. }), "{allowed:?}");
}

#[tokio::test]
async fn only_the_daemons_own_computer_changes_what_a_device_may_do() {
    let f = fixture(RiskLevel::Medium).await;
    let phone = f.pair(Vec::new()).await;
    let grant: AccessGrant = f.sign_in(&phone).await.json().await.unwrap();
    let listed = f
        .post_as(&grant.access_token, &command("d-1", Command::ListDevices))
        .await;
    assert!(
        matches!(&listed, ServerFrame::Ack { result: CommandResult::Devices { devices, can_change: false }, .. } if devices.len() == 1),
        "{listed:?}"
    );
    let narrow = Command::SetDeviceScopes {
        device_id: phone.device_id.clone(),
        scopes: vec![Scope::Observe, Scope::Chat, Scope::Approve],
    };
    let refused = f
        .post_as(&grant.access_token, &command("d-2", narrow.clone()))
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
        "{refused:?}"
    );
    let changed = f.post(&[], &command("d-3", narrow)).await;
    assert!(
        matches!(&changed, ServerFrame::Ack { result: CommandResult::Devices { devices, can_change: true }, .. }
            if !devices[0].scopes.contains(&Scope::Control)),
        "{changed:?}"
    );
}

#[tokio::test]
async fn pointing_sends_the_agent_what_is_there_as_the_owners_message() {
    let f = fixture(RiskLevel::Medium).await;
    let registry = SurfaceRegistry::new(SurfacesConfig::default());
    let mock = MockSurface::new();
    mock.set_notifier(registry.notifier());
    registry.add_backend(Arc::new(mock.clone()));
    f.gateway.attach_surfaces(registry.clone());
    let mut socket = socket_with(&f, vec![Capability::ViewActions, Capability::Surfaces]).await;
    let (accepted, _) = command_on(&mut socket, &command("p-0", send("open the site"))).await;
    let CommandResult::MessageAccepted { session_id, .. } = accepted else {
        panic!("{accepted:?}");
    };
    let surface = mock.add(&session_id, "tab-1");
    registry.refresh().await;
    let (pointed, seen) = command_on(
        &mut socket,
        &command(
            "p-1",
            Command::PointAt {
                surface_id: surface.to_string(),
                x: 0.4,
                y: 0.6,
                note: Some("Click this one".into()),
            },
        ),
    )
    .await;
    assert!(
        matches!(&pointed, CommandResult::MessageAccepted { session_id: s, .. } if *s == session_id),
        "{pointed:?}"
    );
    let mut said = seen;
    for _ in 0..20 {
        if said.iter().any(|f| matches!(f, ServerFrame::Event { event: Event::UserMessage { text, .. }, .. } if text.contains("Click this one"))) {
            break;
        }
        said.push(next(&mut socket).await);
    }
    let text = said
        .iter()
        .find_map(|f| match f {
            ServerFrame::Event {
                event: Event::UserMessage { text, .. },
                ..
            } if text.contains("Click this one") => Some(text.clone()),
            _ => None,
        })
        .expect("the owner's message");
    assert!(
        text.contains("a button “Sign in” on browser tab tab-1 (selector `#signin`)"),
        "{text}"
    );
}

/// A fake screen on "another computer": one picture, input recorded.
struct FakeScreen {
    inputs: Arc<std::sync::Mutex<Vec<String>>>,
    controlled: Arc<std::sync::Mutex<Vec<bool>>>,
}

impl aidaemon_client_core::screen_share::ScreenSource for FakeScreen {
    fn size(&self) -> (u32, u32) {
        (1920, 1080)
    }
    fn capture(
        &mut self,
        _max_w: u32,
        _max_h: u32,
        _quality: u8,
    ) -> Result<aidaemon_client_core::screen_share::Captured, String> {
        Ok(aidaemon_client_core::screen_share::Captured {
            codec: "jpeg",
            w: 4,
            h: 2,
            bytes: b"remote-frame".to_vec(),
        })
    }
    fn input(&mut self, event: SurfaceInput) -> Result<(), String> {
        let kind = match event {
            SurfaceInput::Text { text } => format!("text:{text}"),
            other => format!("{other:?}"),
        };
        self.inputs.lock().unwrap().push(kind);
        Ok(())
    }
    fn controlled(&mut self, on: bool) {
        self.controlled.lock().unwrap().push(on);
    }
}

#[tokio::test]
async fn another_computer_shares_its_screen_and_the_owner_drives_it() {
    let f = fixture(RiskLevel::Medium).await;
    let registry = SurfaceRegistry::new(SurfacesConfig::default());
    registry.start();
    f.gateway.attach_surfaces(registry.clone());
    let inputs = Arc::new(std::sync::Mutex::new(Vec::new()));
    let controlled = Arc::new(std::sync::Mutex::new(Vec::new()));
    let screen = FakeScreen {
        inputs: inputs.clone(),
        controlled: controlled.clone(),
    };
    let base = format!("http://{}", f.url);
    let auth = aidaemon_client_core::Auth::LocalToken(f.token.clone());
    tokio::spawn(async move {
        let _ = aidaemon_client_core::screen_share::share(
            &base,
            &auth,
            "Synthetic studio Mac",
            Box::new(screen),
            |_| {},
        )
        .await;
    });
    // The shared screen appears as a surface.
    let mut surface_id = None;
    for _ in 0..100 {
        registry.refresh().await;
        if let Some(found) = registry
            .list()
            .into_iter()
            .find(|s| s.title == "Synthetic studio Mac")
        {
            surface_id = Some(found.id.to_string());
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let surface_id = surface_id.expect("the shared screen is listed");

    let mut socket = socket_with(&f, vec![Capability::ViewActions, Capability::Surfaces]).await;
    let (result, _) = command_on(
        &mut socket,
        &command(
            "r-1",
            Command::OpenSurface {
                surface_id: surface_id.clone(),
                mode: SurfaceMode::Control,
                accept: StreamAccept::default(),
            },
        ),
    )
    .await;
    let (_, transport) = opened(result);
    let mut media = media_socket(&f, &transport, &transport.ticket).await;
    let (_, payload) = media_until_frame(&mut media).await;
    assert_eq!(payload, b"remote-frame");
    media_send(
        &mut media,
        &MediaClientMessage::Input {
            event: SurfaceInput::Text {
                text: "synthetic".into(),
            },
        },
    )
    .await;
    for _ in 0..100 {
        if !inputs.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(inputs.lock().unwrap().as_slice(), ["text:synthetic"]);
    assert_eq!(
        controlled.lock().unwrap().first(),
        Some(&true),
        "the computer is told"
    );
}

/// Opening a conversation shell, typing into it on its socket and reading
/// what it writes; a device that may not answer approvals is refused.
#[cfg(unix)]
#[tokio::test]
async fn a_conversation_shell_opens_on_its_socket_and_only_for_a_trusted_device() {
    use aidaemon_client_protocol::shell::{ShellClientMessage, ShellServerMessage};

    let f = fixture(RiskLevel::Medium).await;
    let shells = aidaemon_shells::ShellRegistry::new(aidaemon_shells::ShellsConfig::default());
    f.gateway.attach_shells(shells.clone());

    let mut main = socket_with(&f, vec![Capability::Shells]).await;
    let (result, _) = command_on(
        &mut main,
        &command(
            "sh-1",
            Command::OpenShell {
                session_id: "app:shell-test".into(),
                cols: 80,
                rows: 24,
            },
        ),
    )
    .await;
    let CommandResult::ShellOpened { shell, transport } = result else {
        panic!("expected shell_opened, got {result:?}");
    };
    assert_eq!(shell.session_id, "app:shell-test");
    assert!(transport.url.starts_with("/client/v1/shells/"));

    let (mut term, _) =
        tokio_tungstenite::connect_async(format!("ws://{}{}", f.url, transport.url))
            .await
            .unwrap();
    let say = |m: ShellClientMessage| Message::text(serde_json::to_string(&m).unwrap());
    term.send(say(ShellClientMessage::Auth {
        ticket: transport.ticket.clone(),
    }))
    .await
    .unwrap();
    term.send(say(ShellClientMessage::Input {
        data: "printf 'sum:%s\\n' $((40+2))\n".into(),
    }))
    .await
    .unwrap();
    let mut seen = String::new();
    let mut ready = false;
    let read = tokio::time::timeout(Duration::from_secs(30), async {
        while !seen.contains("sum:42") {
            let text = match term.next().await {
                Some(Ok(Message::Text(text))) => text,
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                other => panic!("the shell socket closed ({other:?}); it wrote {seen:?}"),
            };
            match serde_json::from_str::<ShellServerMessage>(&text).unwrap() {
                ShellServerMessage::Ready { .. } => ready = true,
                ShellServerMessage::Output { data } => seen.push_str(&data),
                other => panic!("unexpected {other:?}"),
            }
        }
    })
    .await;
    assert!(read.is_ok(), "no output; saw {seen:?}");
    assert!(ready, "ready comes first");

    // The ticket was single use.
    let (mut again, _) =
        tokio_tungstenite::connect_async(format!("ws://{}{}", f.url, transport.url))
            .await
            .unwrap();
    again
        .send(say(ShellClientMessage::Auth {
            ticket: transport.ticket,
        }))
        .await
        .unwrap();
    let Some(Ok(Message::Text(text))) = again.next().await else {
        panic!("expected a refusal");
    };
    assert!(matches!(
        serde_json::from_str::<ShellServerMessage>(&text).unwrap(),
        ShellServerMessage::Closed { .. }
    ));

    // A phone that may chat and approve but not control: refused.
    let phone = f
        .pair(vec![Scope::Observe, Scope::Chat, Scope::Approve])
        .await;
    let grant: AccessGrant = f.sign_in(&phone).await.json().await.unwrap();
    let refused = f
        .post_as(
            &grant.access_token,
            &command(
                "sh-2",
                Command::OpenShell {
                    session_id: "app:shell-test".into(),
                    cols: 80,
                    rows: 24,
                },
            ),
        )
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
        "{refused:?}"
    );

    // Closing it ends it.
    let (_, _) = command_on(
        &mut main,
        &command(
            "sh-3",
            Command::CloseShell {
                shell_id: shell.id.clone(),
            },
        ),
    )
    .await;
    assert!(shells.list().is_empty());
}

/// `/trustall on` from an app takes Settings' check: a device that is the
/// only one paired confirms it itself (a chat cannot answer the card, so it
/// does not count), and with another device paired, only that one may.
#[tokio::test]
async fn trust_all_on_from_an_app_asks_like_settings() {
    let f = fixture(RiskLevel::Medium).await;
    // A chat is connected, but it cannot answer this card.
    let (_answer_tx, answer_rx) = oneshot::channel();
    f.hub
        .register_channel(Arc::new(Chat {
            answer: Mutex::new(Some(answer_rx)),
            asked: AtomicUsize::new(0),
        }))
        .await;
    let phone = f.pair(vec![Scope::Chat, Scope::Control]).await;
    let grant: AccessGrant = f.sign_in(&phone).await.json().await.unwrap();
    let trust_all = |id: &str| {
        command(
            id,
            Command::RunSlash {
                session_id: "app:synthetic".into(),
                command: "/trustall".into(),
                args: "on".into(),
            },
        )
    };
    let reply = |frame: ServerFrame| match frame {
        ServerFrame::Ack {
            result: CommandResult::Reply { text },
            ..
        } => text,
        other => panic!("{other:?}"),
    };
    let alone = reply(f.post_as(&grant.access_token, &trust_all("t-1")).await);
    assert!(alone.contains("Confirm it on the card"), "{alone}");

    let _laptop = f.pair(vec![Scope::Chat]).await;
    let other = reply(f.post_as(&grant.access_token, &trust_all("t-2")).await);
    assert!(other.contains("another of your devices"), "{other}");
}

/// `/trustall on` from a chat: the owner's paired devices approve it, never
/// the chat; with no paired device there is no one to ask here.
#[tokio::test]
async fn trust_all_on_from_a_chat_asks_the_paired_devices() {
    let f = fixture(RiskLevel::Medium).await;
    let change = || SettingsChange::SetPermissionLevel {
        level: PermissionLevel::FullAccess,
    };
    let chat = "telegram:synthetic-user-1";
    assert!(
        f.gateway
            .ask_devices_for_chat(change(), chat)
            .await
            .is_none()
    );
    let _phone = f.pair(vec![Scope::Chat]).await;
    let asked = f
        .gateway
        .ask_devices_for_chat(change(), chat)
        .await
        .expect("a paired device is asked")
        .unwrap();
    assert!(asked.contains("Approve it in the AIdaemon app"), "{asked}");
}
