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
    let approvals_for_pairing = approvals_tx.clone();
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

// ---------------------------------------------------------------------------
// Pairing
// ---------------------------------------------------------------------------

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
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

/// The gateway serves the web client: its page and its script.
#[tokio::test]
async fn the_web_client_is_served() {
    let f = fixture(RiskLevel::Medium).await;
    let page = f
        .http()
        .get(format!("http://{}/client/v1/app", f.url))
        .send()
        .await
        .unwrap();
    assert_eq!(page.status(), 200);
    assert!(page.headers()["content-security-policy"]
        .to_str()
        .unwrap()
        .contains("default-src 'self'"));
    assert!(page.text().await.unwrap().contains("/client/v1/app/app.js"));
    let script = f
        .http()
        .get(format!("http://{}/client/v1/app/app.js", f.url))
        .send()
        .await
        .unwrap();
    assert!(script.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/javascript"));
    assert!(script.text().await.unwrap().contains("./ui.js"));
    // Each module the page imports is served; nothing else is.
    for (module, holds) in [
        ("link.js", "/sessions/challenge"),
        ("state.js", "approval_resolved"),
        ("platform.js", "__TAURI__"),
        ("ui.js", "resolve_approval"),
    ] {
        let response = f
            .http()
            .get(format!("http://{}/client/v1/app/{module}", f.url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{module}");
        assert!(response.text().await.unwrap().contains(holds), "{module}");
    }
    let missing = f
        .http()
        .get(format!("http://{}/client/v1/app/protocol.js", f.url))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);
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
        if let Message::Text(text) = message {
            if let ServerFrame::Event {
                event: Event::ViewAction { action, .. },
                seq,
                ..
            } = serde_json::from_str(&text).unwrap()
            {
                assert_eq!(seq, None, "view actions are never logged");
                actions.push(action);
            }
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
    assert!(view_actions_within(&mut plain, Duration::from_millis(300))
        .await
        .is_empty());
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
