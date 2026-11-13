//! # aidaemon-client-gateway
//!
//! **Tier: internal.** Internal to aidaemon, no stability promise.
//!
//! The Client Gateway serves the AIdaemon Client Protocol v1
//! ([`aidaemon_client_protocol`]) so the owner's own apps (a terminal UI, a
//! web page, a phone, a watch) drive and watch the daemon:
//!
//! - an app's message runs as a turn on the same
//!   [`TurnRunner`](aidaemon_channels::turns::TurnRunner) the chat bots use,
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
mod config;
mod observer;
mod server;
pub mod token;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::hub::{ChannelHub, HubChannel};
use aidaemon_channels::stop::{self, StopRequest};
use aidaemon_channels::turns::{Turn, TurnDeps, TurnRunner};
use aidaemon_client_protocol::{
    ApprovalDecision, ClientKind, Command, CommandResult, ErrorCode, Event, Glance, HistoryMessage,
    LastReply, OwnerInfo, ProtocolError, Risk, ServerFrame, SessionSummary, Snapshot, TaskState,
    TaskSummary,
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
pub use server::{router, serve};

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

/// Everything the gateway runs on.
pub struct GatewayDeps {
    /// The turn runner's services (the same the bots run on).
    pub turns: TurnDeps,
    pub hub: Arc<ChannelHub>,
    pub host: Arc<dyn ClientHost>,
    /// The log, on the daemon's database.
    pub log: SqliteClientEventLog,
    pub config: ClientGatewayConfig,
    /// The local token's hash.
    pub token_file: PathBuf,
}

/// Who sent a command.
#[derive(Debug, Clone, Default)]
pub struct Caller {
    pub kind: Option<ClientKind>,
    pub device: Option<String>,
    pub compact: bool,
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
    pub(crate) config: ClientGatewayConfig,
    pub(crate) token_file: PathBuf,
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
        deps.turns.watch.set(Arc::new(GatewayObserver {
            publisher: publisher.clone(),
            ids: ids.clone(),
            active: active.clone(),
            running: Mutex::default(),
        }));
        deps.hub.set_approval_tap(approvals.clone());
        deps.hub
            .register_channel(Arc::new(AppChannel {
                publisher: publisher.clone(),
            }) as Arc<dyn HubChannel>)
            .await;

        let gateway = Arc::new(Self {
            bus,
            publisher,
            approvals,
            ids,
            runner: TurnRunner::new(deps.turns.clone()),
            turns: deps.turns,
            host: deps.host,
            active,
            config: deps.config,
            token_file: deps.token_file,
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
        if keep {
            if let Ok(text) = serde_json::to_string(&reply) {
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
        }
        reply
    }

    async fn dispatch(
        &self,
        request_id: &str,
        command: Command,
        caller: &Caller,
    ) -> Result<CommandResult, ProtocolError> {
        match command {
            Command::SendMessage {
                session_id,
                text,
                attachments,
            } => {
                self.send_message(request_id, session_id, text, attachments.is_empty())
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
                let ceiling = (caller.kind == Some(ClientKind::Watch))
                    .then_some(self.config.watch_risk_ceiling);
                self.resolve_approval(&approval_id, decision, caller.device.clone(), ceiling)
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

    async fn send_message(
        &self,
        request_id: &str,
        session_id: Option<String>,
        text: String,
        no_attachments: bool,
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
            Some(_) => {
                return Err(ProtocolError::new(
                    ErrorCode::Forbidden,
                    "Sessions of other chats are read-only here: post from that chat, or start \
                     a new session",
                ))
            }
        };
        self.runner.register_session(&session_id, APP_CHANNEL).await;
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
                    ))
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
