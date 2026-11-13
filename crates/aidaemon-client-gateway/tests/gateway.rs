//! The gateway over real sockets, on a scripted agent: the HTTP mode, the
//! compact capability and a watch's risk ceiling, idempotent commands,
//! approvals settled by whoever answers first, auth and version checks, and
//! resuming or starting from a snapshot.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aidaemon_channels::hub::{
    ApprovalRequest, ChannelCapabilities, ChannelHub, HubChannel, MediaMessage, SessionChannelStore,
};
use aidaemon_channels::ports::{
    AgentIngress, AgentResponseEnvelope, ChannelAgentRuntime, InboundMessageRequest,
};
use aidaemon_channels::turns::{TurnDeps, TurnWatch};
use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps};
use aidaemon_client_protocol::*;
use aidaemon_store_sqlite::SqliteClientEventLog;
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalKind as CardKind, ApprovalResponse};
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_tungstenite::tungstenite::Message;

/// Replies "Done." — or, for a message that says "approve", asks the owner
/// first (at `risk`) and says what the answer was.
struct ScriptedAgent {
    approvals: mpsc::Sender<ApprovalRequest>,
    risk: RiskLevel,
    turns: AtomicUsize,
}

#[async_trait]
impl AgentIngress for ScriptedAgent {
    async fn handle_inbound_message(
        &self,
        request: InboundMessageRequest,
    ) -> anyhow::Result<AgentResponseEnvelope> {
        self.turns.fetch_add(1, Ordering::SeqCst);
        if let Some(status) = request.status_tx.as_ref() {
            let _ = status
                .send(StatusUpdate::ToolStart {
                    name: "terminal".into(),
                    summary: "building the synthetic report".into(),
                })
                .await;
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
            id: "telegram:synthetic-user-1".into(),
            channel: "telegram".into(),
            title: None,
            last_activity_ms: Some(5),
            writable: false,
        }])
    }
    async fn history(&self, _: &str, _: usize) -> anyhow::Result<Vec<HistoryMessage>> {
        Ok((1..=5)
            .map(|n| HistoryMessage {
                id: format!("m{n}"),
                role: "user".into(),
                text: format!("message {n}"),
                at_ms: n,
            })
            .collect())
    }
    async fn slash(&self, _: &str, command: &str, _: &str) -> Option<String> {
        (command == "clear").then(|| "Context cleared.".to_string())
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
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let log = SqliteClientEventLog::open(pool).await.unwrap();
    let (approvals_tx, approvals_rx) = mpsc::channel(16);
    let agent = Arc::new(ScriptedAgent {
        approvals: approvals_tx,
        risk,
        turns: AtomicUsize::new(0),
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
    let gateway = ClientGateway::start(GatewayDeps {
        turns: deps.clone(),
        hub: hub.clone(),
        host: Arc::new(Host),
        log,
        config: ClientGatewayConfig {
            enabled: true,
            ..ClientGatewayConfig::default()
        },
        token_file,
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
    }
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
    // Another chat's session is read-only from here.
    let refused = f
        .post(
            &[],
            &command(
                "r-5",
                Command::SendMessage {
                    session_id: Some("telegram:synthetic-user-1".into()),
                    text: "hi".into(),
                    attachments: Vec::new(),
                },
            ),
        )
        .await;
    assert!(
        matches!(&refused, ServerFrame::Error { error, .. } if error.code == ErrorCode::Forbidden),
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
    assert!(events
        .iter()
        .any(|e| matches!(e, Event::TurnStarted { origin, .. } if origin == "test")));
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
