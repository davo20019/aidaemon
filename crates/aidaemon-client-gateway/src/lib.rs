//! # aidaemon-client-gateway
//!
//! **Tier: internal.** Internal to aidaemon, no stability promise.
//!
//! The Client Gateway serves the AIdaemon Client Protocol v1
//! ([`aidaemon_client_protocol`]) so the owner's own apps (a terminal UI, a
//! web page, a phone, a watch) drive and watch the daemon:
//!
//! - an app's message runs as a turn on the same
//!   [`aidaemon_channels::turns::TurnRunner`] the chat bots use,
//!   with the same queueing, stop and approval behaviour ([`AppSurface`]);
//! - every chat's turns are reported as client events
//!   ([`TurnObserver`](aidaemon_channels::turns::TurnObserver)), so a turn
//!   started from Telegram shows in the app too;
//! - every approval card the hub routes is shown to clients
//!   ([`ApprovalTap`](aidaemon_channels::hub::ApprovalTap)) and the first
//!   answer, from a client or the chat, settles it;
//! - durable events go to the `client_events` log first, so a client that
//!   reconnects resumes from its last `seq` without loss or duplicates, or
//!   gets a snapshot when it is too far behind;
//! - commands are idempotent by `request_id` for 24 hours.
//!
//! Loopback only, with a local token from `aidaemon client-token`. The
//! application supplies what only it knows through [`ClientHost`].

mod approvals;
mod bus;
pub mod client_view;
mod config;
mod observer;
pub mod pairing;
mod server;
pub mod token;
pub mod views;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::hub::{ChannelHub, HubChannel};
use aidaemon_channels::stop::{self, StopRequest};
use aidaemon_channels::turns::{Turn, TurnDeps, TurnRunner};
use aidaemon_client_protocol::{
    ApprovalDecision, ClientKind, Command, CommandResult, ErrorCode, Event, Glance, HistoryMessage,
    LastReply, OwnerInfo, ProtocolError, Risk, Scope, ServerFrame, SessionSummary, Snapshot,
    TaskState, TaskSummary,
};
use aidaemon_store_sqlite::SqliteClientEventLog;
use aidaemon_tools::semantics::UserRole;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{info, warn};

pub use approvals::Approvals;
pub use bus::{EventBus, OWNER};
pub use config::ClientGatewayConfig;
pub use observer::{AppChannel, AppSurface, GatewayObserver, Ids};
pub use pairing::{Pairing, Principal};
pub use server::{router, serve};
pub use views::{ViewError, ViewTargets};

/// The hub channel and turn platform name of the app.
pub const APP_CHANNEL: &str = "app";
/// Sessions started from a client: `app:<id>`.
pub const APP_SESSION_PREFIX: &str = "app:";
/// How long a command's reply is remembered for a retry.
const REPLY_MEMORY: Duration = Duration::from_secs(24 * 60 * 60);
/// How often the log is trimmed.
const UPKEEP_EVERY: Duration = Duration::from_secs(60 * 60);

/// What only the application knows: its sessions, their transcripts and
/// its slash commands.
#[async_trait]
pub trait ClientHost: Send + Sync {
    /// Every session the daemon knows, any order.
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>>;
    /// Up to `limit` of the session's newest user and assistant messages,
    /// oldest first.
    async fn history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<HistoryMessage>>;
    /// Run slash command `command` (without the slash) as the owner; `None`
    /// when there is no such command.
    async fn slash(&self, session_id: &str, command: &str, args: &str) -> Option<String>;
    fn owner(&self) -> OwnerInfo {
        OwnerInfo::default()
    }
}

/// Hands events to the bus in the order they were raised, from code that
/// cannot wait (the turn observer, the approval tap).
#[derive(Clone)]
pub struct Publisher(mpsc::UnboundedSender<(Option<String>, Event)>);

impl Publisher {
    pub fn publish(&self, session_id: Option<String>, event: Event) {
        let _ = self.0.send((session_id, event));
    }
}

/// The gateway's address as pairing links name it: the first origin of
/// `endpoints`, else `fallback` (config `public_url` or loopback).
#[derive(Clone)]
pub(crate) struct PublicUrl {
    pub endpoints: aidaemon_core::EndpointSource,
    pub fallback: String,
}

impl PublicUrl {
    pub fn get(&self) -> String {
        self.endpoints
            .current()
            .into_iter()
            .next()
            .map(|origin| origin.trim_end_matches('/').to_string())
            .unwrap_or_else(|| self.fallback.clone())
    }
}

/// Everything the gateway runs on.
pub struct GatewayDeps {
    /// The turn runner's services (the same the bots run on).
    pub turns: TurnDeps,
    pub hub: Arc<ChannelHub>,
    pub host: Arc<dyn ClientHost>,
    /// The log, on the daemon's database.
    pub log: SqliteClientEventLog,
    pub config: ClientGatewayConfig,
    /// Where devices reach the gateway: a live tunnel origin, then
    /// `config.public_url`. `EndpointSource::fixed(vec![])` serves the
    /// configured or loopback address only.
    pub endpoints: aidaemon_core::EndpointSource,
    /// The local token's hash.
    pub token_file: PathBuf,
    /// Where pairing confirmations go: the hub's approval queue (the same
    /// one the agent's cards use). `None`: pairings are confirmed only
    /// through the owner's decision call.
    pub approvals: Option<mpsc::Sender<aidaemon_channels::hub::ApprovalRequest>>,
}

/// Who sent a command.
#[derive(Debug, Clone)]
pub struct Caller {
    pub kind: Option<ClientKind>,
    pub device: Option<String>,
    pub compact: bool,
    /// The token's owner: the local owner or a paired device.
    pub principal: Principal,
    /// The caller's socket, when it declared `view_actions`: a message it
    /// sends makes it the session's view target.
    pub view_connection: Option<u64>,
}

impl Caller {
    /// The owner at the machine, with nothing else known.
    pub fn owner() -> Self {
        Self {
            kind: None,
            device: None,
            compact: false,
            principal: Principal::owner(),
            view_connection: None,
        }
    }
}

/// The running gateway.
pub struct ClientGateway {
    pub(crate) bus: Arc<EventBus>,
    pub(crate) publisher: Publisher,
    pub(crate) approvals: Arc<Approvals>,
    pub(crate) ids: Ids,
    runner: TurnRunner,
    turns: TurnDeps,
    host: Arc<dyn ClientHost>,
    active: Arc<Mutex<HashSet<String>>>,
    /// Chat conversations' contexts, for replies from the app.
    contexts: observer::Contexts,
    hub: Arc<ChannelHub>,
    pub(crate) config: ClientGatewayConfig,
    endpoints: aidaemon_core::EndpointSource,
    pub(crate) token_file: PathBuf,
    pub(crate) pairing: Arc<Pairing>,
    pub(crate) views: Arc<ViewTargets>,
    /// Requests being handled now, so a retry waits for the first answer.
    in_flight: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl ClientGateway {
    /// Wire the gateway into the daemon: see every turn and approval card,
    /// take app sessions on the hub, and start the event writer and the
    /// log's upkeep. Serving is separate ([`serve`]).
    pub async fn start(deps: GatewayDeps) -> anyhow::Result<Arc<Self>> {
        let boot = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        let ids = Ids { boot: boot.clone() };
        let bus = Arc::new(EventBus::new(deps.log));
        let (tx, mut rx) = mpsc::unbounded_channel::<(Option<String>, Event)>();
        let publisher = Publisher(tx);
        {
            let bus = bus.clone();
            tokio::spawn(async move {
                while let Some((session_id, event)) = rx.recv().await {
                    bus.publish(session_id, event).await;
                }
            });
        }

        let approvals = Arc::new(Approvals::new(publisher.clone(), boot));
        // Cards do not survive a restart: settle those a previous process
        // left open in the log.
        match bus
            .log()
            .open_refs(OWNER, "approval_requested", "approval_resolved")
            .await
        {
            Ok(rows) => approvals.withdraw_stale(
                rows.into_iter()
                    .filter_map(|row| row.ref_id.map(|id| (id, row.session_id)))
                    .collect(),
            ),
            Err(error) => warn!(%error, "Could not read open approvals from the client log"),
        }

        let active = Arc::new(Mutex::new(HashSet::new()));
        let contexts = observer::Contexts::default();
        deps.turns.watch.set(Arc::new(GatewayObserver {
            publisher: publisher.clone(),
            ids: ids.clone(),
            active: active.clone(),
            running: Mutex::default(),
            contexts: contexts.clone(),
        }));
        deps.hub.set_approval_tap(approvals.clone());
        deps.hub
            .register_channel(Arc::new(AppChannel {
                publisher: publisher.clone(),
            }) as Arc<dyn HubChannel>)
            .await;

        let pairing = Arc::new(Pairing::new(
            bus.log().clone(),
            deps.approvals,
            PublicUrl {
                endpoints: deps.endpoints.clone(),
                fallback: deps.config.public_url(),
            },
            deps.config.watch_risk_ceiling,
        ));
        let gateway = Arc::new(Self {
            pairing,
            bus,
            publisher,
            approvals,
            ids,
            runner: TurnRunner::new(deps.turns.clone()),
            turns: deps.turns,
            host: deps.host,
            active,
            contexts,
            hub: deps.hub.clone(),
            config: deps.config,
            endpoints: deps.endpoints,
            token_file: deps.token_file,
            views: Arc::default(),
            in_flight: Mutex::default(),
        });
        let upkeep = Arc::downgrade(&gateway);
        tokio::spawn(async move {
            loop {
                let Some(gateway) = upkeep.upgrade() else {
                    return;
                };
                gateway.upkeep().await;
                drop(gateway);
                tokio::time::sleep(UPKEEP_EVERY).await;
            }
        });
        Ok(gateway)
    }

    /// Trim the log to its retention and forget old replies.
    pub async fn upkeep(&self) {
        let now = bus::now_ms();
        let day_ms = 24 * 60 * 60 * 1000;
        let before = now.saturating_sub(self.config.retention_days.saturating_mul(day_ms));
        match self
            .bus
            .log()
            .prune(OWNER, before, self.config.max_events)
            .await
        {
            Ok(0) => {}
            Ok(removed) => info!(removed, "Trimmed the client event log"),
            Err(error) => warn!(%error, "Could not trim the client event log"),
        }
        let forget = now.saturating_sub(u64::try_from(REPLY_MEMORY.as_millis()).unwrap_or(0));
        if let Err(error) = self.bus.log().forget_replies_before(forget).await {
            warn!(%error, "Could not forget old client replies");
        }
    }

    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    pub fn owner(&self) -> OwnerInfo {
        self.host.owner()
    }

    pub fn pairing(&self) -> &Arc<Pairing> {
        &self.pairing
    }

    /// The connections the agent may move, by session (the agent-driven
    /// view).
    pub fn views(&self) -> &Arc<ViewTargets> {
        &self.views
    }

    /// The path of the web client pairing links open
    /// ([`ClientGatewayConfig::web_app_path`]).
    pub fn web_app_path(&self) -> &'static str {
        self.config.web_app_path()
    }

    /// Where devices reach the gateway now: the live tunnel origin, else
    /// `public_url` from config, else the loopback address.
    pub fn public_url(&self) -> String {
        PublicUrl {
            endpoints: self.endpoints.clone(),
            fallback: self.config.public_url(),
        }
        .get()
    }

    /// Who `token` belongs to: the owner's local token, or a paired
    /// device's live access token.
    pub fn principal(&self, token: &str) -> Option<Principal> {
        if crate::token::verify(&self.token_file, token) {
            return Some(Principal::owner());
        }
        self.pairing.principal(token)
    }

    /// Answer `command` once per `request_id`: a retry gets the first reply.
    pub async fn command(
        &self,
        request_id: &str,
        command: Command,
        caller: &Caller,
    ) -> ServerFrame {
        let gate = {
            let mut in_flight = self.in_flight.lock().unwrap_or_else(|p| p.into_inner());
            in_flight
                .entry(request_id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let reply = {
            let _turn = gate.lock().await;
            self.command_once(request_id, command, caller).await
        };
        let mut in_flight = self.in_flight.lock().unwrap_or_else(|p| p.into_inner());
        if in_flight
            .get(request_id)
            .is_some_and(|gate| Arc::strong_count(gate) <= 2)
        {
            in_flight.remove(request_id);
        }
        reply
    }

    async fn command_once(
        &self,
        request_id: &str,
        command: Command,
        caller: &Caller,
    ) -> ServerFrame {
        let log = self.bus.log();
        match log.remembered_reply(OWNER, request_id).await {
            Ok(Some(reply)) => {
                if let Ok(frame) = serde_json::from_str::<ServerFrame>(&reply) {
                    return frame;
                }
            }
            Ok(None) => {}
            Err(error) => warn!(%error, "Could not read a remembered client reply"),
        }
        let reply = match self.dispatch(request_id, command, caller).await {
            Ok(result) => ServerFrame::Ack {
                request_id: request_id.to_string(),
                result,
            },
            Err(error) => ServerFrame::Error {
                request_id: Some(request_id.to_string()),
                error,
            },
        };
        // A reply worth retrying for is not remembered.
        let keep = !matches!(&reply, ServerFrame::Error { error, .. } if error.retryable);
        if keep && let Ok(text) = serde_json::to_string(&reply) {
            match log
                .remember_reply(OWNER, request_id, bus::now_ms(), &text)
                .await
            {
                Ok(kept) => {
                    if let Ok(frame) = serde_json::from_str(&kept) {
                        return frame;
                    }
                }
                Err(error) => warn!(%error, "Could not remember a client reply"),
            }
        }
        reply
    }

    async fn dispatch(
        &self,
        request_id: &str,
        command: Command,
        caller: &Caller,
    ) -> Result<CommandResult, ProtocolError> {
        let needs = match &command {
            Command::SendMessage { .. }
            | Command::StopTurn { .. }
            | Command::CancelTask { .. }
            | Command::ClearSession { .. }
            | Command::RunSlash { .. } => Some(Scope::Chat),
            Command::ResolveApproval { .. } => Some(Scope::Approve),
            Command::ListSessions { .. }
            | Command::GetHistory { .. }
            | Command::GetGlance
            | Command::GetSnapshot => Some(Scope::Observe),
            Command::Unknown => None,
        };
        if let Some(scope) = needs
            && !caller.principal.may(scope)
        {
            return Err(ProtocolError::new(
                ErrorCode::Forbidden,
                format!("This device may not {scope:?}"),
            ));
        }
        match command {
            Command::SendMessage {
                session_id,
                text,
                attachments,
            } => {
                self.send_message(
                    request_id,
                    session_id,
                    text,
                    attachments.is_empty(),
                    caller.view_connection,
                )
                .await
            }
            Command::StopTurn { session_id } => {
                let report = stop::stop(
                    &self.turns.task_registry,
                    self.turns.agent.as_ref(),
                    &session_id,
                    StopRequest::All,
                )
                .await;
                Ok(CommandResult::Stopped { report })
            }
            Command::CancelTask { task_id } => {
                let Some((session_id, number)) = self.ids.parse_task(&task_id) else {
                    return Err(ProtocolError::new(
                        ErrorCode::NotFound,
                        format!("No task {task_id} is running"),
                    ));
                };
                let report = stop::stop(
                    &self.turns.task_registry,
                    self.turns.agent.as_ref(),
                    &session_id,
                    StopRequest::Numbers(vec![number]),
                )
                .await;
                Ok(CommandResult::Stopped { report })
            }
            Command::ResolveApproval {
                approval_id,
                decision,
            } => {
                let watch = (caller.kind == Some(ClientKind::Watch))
                    .then_some(self.config.watch_risk_ceiling);
                let ceiling = match (watch, caller.principal.ceiling) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                let device = caller
                    .device
                    .clone()
                    .or_else(|| caller.principal.device_name.clone());
                self.resolve_approval(&approval_id, decision, device, ceiling)
            }
            Command::ListSessions { cursor, limit } => self.list_sessions(cursor, limit).await,
            Command::GetHistory {
                session_id,
                before,
                limit,
            } => self.history(session_id, before, limit).await,
            Command::ClearSession { session_id } => self.slash(&session_id, "clear", "").await,
            Command::RunSlash {
                session_id,
                command,
                args,
            } => {
                let command = command.trim().trim_start_matches('/').to_string();
                self.slash(&session_id, &command, &args).await
            }
            Command::GetGlance => Ok(CommandResult::Glance {
                glance: self.glance(caller.compact).await,
            }),
            Command::GetSnapshot => {
                let (seq, state) = self.snapshot().await;
                Ok(CommandResult::Snapshot { seq, state })
            }
            Command::Unknown => Err(ProtocolError::new(
                ErrorCode::Invalid,
                "Unknown command for this gateway version",
            )),
        }
    }

    fn resolve_approval(
        &self,
        approval_id: &str,
        decision: ApprovalDecision,
        device: Option<String>,
        ceiling: Option<Risk>,
    ) -> Result<CommandResult, ProtocolError> {
        let outcome = self
            .approvals
            .resolve(approval_id, decision, device, ceiling)?;
        Ok(CommandResult::ApprovalResolved {
            approval_id: approval_id.to_string(),
            outcome,
        })
    }

    /// A message from the app into a chat's own conversation: echoed to the
    /// chat, then run as a turn of that conversation, under the context its
    /// channel last gave it (its visibility decides what memory the turn may
    /// use). A conversation not seen this boot runs as public, so nothing
    /// private reaches a shared channel. The answer goes to the chat, and
    /// the app sees the turn as its own.
    async fn reply_in_chat(
        &self,
        request_id: &str,
        session_id: String,
        text: String,
    ) -> Result<CommandResult, ProtocolError> {
        let known = self
            .contexts
            .lock()
            .ok()
            .and_then(|contexts| contexts.get(&session_id).cloned());
        let channel_ctx = match known {
            Some(context) => context,
            None => {
                let platform = self
                    .hub
                    .session_map()
                    .read()
                    .await
                    .get(&session_id)
                    .cloned()
                    .ok_or_else(|| {
                        ProtocolError::new(
                            ErrorCode::NotFound,
                            "That conversation's chat is not connected right now",
                        )
                    })?;
                let mut context = aidaemon_channels::context::ChannelContext::private(&platform);
                context.visibility = aidaemon_core::ChannelVisibility::Public;
                context
            }
        };
        self.hub
            .send_text(&session_id, &format!("↪ From the AIdaemon app: {text}"))
            .await
            .map_err(|error| {
                ProtocolError::new(
                    ErrorCode::Internal,
                    format!("Could not post to that chat: {error}"),
                )
            })?;
        let turn = Turn {
            session_id: session_id.clone(),
            text,
            attachments: Vec::new(),
            user_role: UserRole::Owner,
            channel_ctx,
            dedup_key: Some(request_id.to_string()),
            ingress_timing: None,
        };
        let surface = Arc::new(observer::ChatReplySurface {
            publisher: self.publisher.clone(),
            hub: self.hub.clone(),
            session_id: session_id.clone(),
        });
        let runner = self.runner.clone();
        tokio::spawn(async move { runner.submit(surface, turn).await });
        Ok(CommandResult::MessageAccepted {
            session_id,
            new_session: false,
        })
    }

    async fn send_message(
        &self,
        request_id: &str,
        session_id: Option<String>,
        text: String,
        no_attachments: bool,
        view_connection: Option<u64>,
    ) -> Result<CommandResult, ProtocolError> {
        if !no_attachments {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "This gateway does not take attachments yet",
            ));
        }
        if text.trim().is_empty() {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "The message is empty",
            ));
        }
        let (session_id, new_session) = match session_id {
            None => (
                format!("{APP_SESSION_PREFIX}{}", uuid::Uuid::new_v4().simple()),
                true,
            ),
            Some(session_id) if session_id.starts_with(APP_SESSION_PREFIX) => (session_id, false),
            Some(session_id) => {
                return self.reply_in_chat(request_id, session_id, text).await;
            }
        };
        self.runner.register_session(&session_id, APP_CHANNEL).await;
        // The turn this message starts may move the sender's view, and only
        // the sender's, when it declared the capability.
        self.views.message_from(&session_id, view_connection);
        if new_session {
            self.publisher.publish(
                Some(session_id.clone()),
                Event::SessionCreated {
                    session: SessionSummary {
                        id: session_id.clone(),
                        channel: APP_CHANNEL.to_string(),
                        title: Some(aidaemon_channels::presentation::word_preview(&text, 60)),
                        last_activity_ms: Some(bus::now_ms()),
                        writable: true,
                    },
                },
            );
        }
        let mut channel_ctx = aidaemon_channels::context::ChannelContext::private(APP_CHANNEL);
        channel_ctx.channel_id = Some(session_id.clone());
        let turn = Turn {
            session_id: session_id.clone(),
            text,
            attachments: Vec::new(),
            user_role: UserRole::Owner,
            channel_ctx,
            dedup_key: Some(request_id.to_string()),
            ingress_timing: None,
        };
        let surface = Arc::new(AppSurface {
            publisher: self.publisher.clone(),
            session_id: session_id.clone(),
        });
        // Submitting may first read the message against running work, which
        // can take seconds: the ack does not wait for it.
        let runner = self.runner.clone();
        tokio::spawn(async move { runner.submit(surface, turn).await });
        Ok(CommandResult::MessageAccepted {
            session_id,
            new_session,
        })
    }

    async fn list_sessions(
        &self,
        cursor: Option<String>,
        limit: u32,
    ) -> Result<CommandResult, ProtocolError> {
        let mut sessions = self.host.sessions().await.map_err(internal)?;
        sessions.sort_by(|a, b| {
            b.last_activity_ms
                .cmp(&a.last_activity_ms)
                .then_with(|| a.id.cmp(&b.id))
        });
        let start: usize = match cursor {
            Some(cursor) => cursor
                .parse()
                .map_err(|_| ProtocolError::new(ErrorCode::Invalid, "Bad cursor"))?,
            None => 0,
        };
        let limit = usize::try_from(limit.clamp(1, 500)).unwrap_or(50);
        let end = start.saturating_add(limit).min(sessions.len());
        let page = sessions.get(start..end).unwrap_or_default().to_vec();
        Ok(CommandResult::Sessions {
            sessions: page,
            next_cursor: (end < sessions.len()).then(|| end.to_string()),
        })
    }

    async fn history(
        &self,
        session_id: String,
        before: Option<String>,
        limit: u32,
    ) -> Result<CommandResult, ProtocolError> {
        let limit = usize::try_from(limit.clamp(1, 500)).unwrap_or(50);
        // Read enough to page back past `before`.
        let mut messages = self
            .host
            .history(&session_id, 2_000)
            .await
            .map_err(internal)?;
        if let Some(before) = before {
            match messages.iter().position(|m| m.id == before) {
                Some(at) => messages.truncate(at),
                None => {
                    return Err(ProtocolError::new(
                        ErrorCode::NotFound,
                        format!("No message {before} in this session's history"),
                    ));
                }
            }
        }
        let more = messages.len() > limit;
        let messages = messages.split_off(messages.len().saturating_sub(limit));
        Ok(CommandResult::History {
            session_id,
            messages,
            more,
        })
    }

    async fn slash(
        &self,
        session_id: &str,
        command: &str,
        args: &str,
    ) -> Result<CommandResult, ProtocolError> {
        match self.host.slash(session_id, command, args).await {
            Some(text) => Ok(CommandResult::Reply { text }),
            None => Err(ProtocolError::new(
                ErrorCode::NotFound,
                format!("No command /{command}"),
            )),
        }
    }

    /// Everything running for every session that ran a turn this boot.
    pub async fn running_tasks(&self) -> Vec<TaskSummary> {
        let sessions: Vec<String> = self
            .active
            .lock()
            .map(|active| active.iter().cloned().collect())
            .unwrap_or_default();
        let mut tasks = Vec::new();
        for session_id in sessions {
            let list = stop::running_work(
                &self.turns.task_registry,
                self.turns.agent.as_ref(),
                &session_id,
            )
            .await;
            for entry in list.entries {
                tasks.push(TaskSummary {
                    id: self.ids.task(&session_id, entry.number),
                    session_id: session_id.clone(),
                    number: entry.number,
                    description: entry.title.clone(),
                    state: TaskState::Running,
                    started_at_ms: u64::try_from(entry.started_at.timestamp_millis()).unwrap_or(0),
                    finished_at_ms: None,
                    error: None,
                });
            }
        }
        tasks.sort_by_key(|task| task.started_at_ms);
        tasks
    }

    /// The state a client starts from when it cannot resume, and the `seq`
    /// it is current to. The `seq` is read first, so events after it may
    /// repeat what the state shows but none are missed.
    pub async fn snapshot(&self) -> (u64, Snapshot) {
        let seq = self.bus.latest().await;
        let sessions = match self.host.sessions().await {
            Ok(sessions) => sessions,
            Err(error) => {
                warn!(%error, "Could not list sessions for a snapshot");
                Vec::new()
            }
        };
        (
            seq,
            Snapshot {
                sessions,
                running_tasks: self.running_tasks().await,
                pending_approvals: self.approvals.pending(),
            },
        )
    }

    pub async fn glance(&self, compact: bool) -> Glance {
        let last_reply = match self.bus.log().latest_of_kind(OWNER, "turn_completed").await {
            Ok(Some(row)) => match serde_json::from_str::<Event>(&row.payload) {
                Ok(Event::TurnCompleted { turn_id, text, .. }) => Some(LastReply {
                    session_id: row.session_id.unwrap_or_default(),
                    turn_id,
                    text,
                    at_ms: row.at_ms,
                }),
                _ => None,
            },
            _ => None,
        };
        let pending = self.approvals.pending();
        Glance {
            running_tasks: self.running_tasks().await,
            pending_approvals: if compact {
                pending.into_iter().map(|a| a.compact()).collect()
            } else {
                pending
            },
            last_reply,
        }
    }
}

fn internal(error: anyhow::Error) -> ProtocolError {
    warn!(%error, "Client command failed");
    ProtocolError::new(
        ErrorCode::Internal,
        "The daemon could not complete the command",
    )
}

#[cfg(test)]
mod public_url_tests {
    use super::PublicUrl;

    #[test]
    fn live_origin_then_config_then_loopback() {
        let (endpoints, publisher) = aidaemon_core::EndpointSource::live(Vec::new());
        let url = PublicUrl {
            endpoints,
            fallback: "http://127.0.0.1:8788".into(),
        };
        assert_eq!(url.get(), "http://127.0.0.1:8788");
        publisher.set(Some("https://app.example.com/".into()));
        assert_eq!(url.get(), "https://app.example.com");
        publisher.set(None);
        assert_eq!(url.get(), "http://127.0.0.1:8788");
    }
}
