//! # aidaemon-client-gateway
//!
//! **Tier: internal.** Internal to aidaemon, no stability promise.
//!
//! The Client Gateway serves the AIDaemon Client Protocol v1
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

/// A tool call's kind of work as the app is told it: the daemon's card
/// kind, one to one.
pub fn tool_kind(kind: aidaemon_core::card::ActivityKind) -> aidaemon_client_protocol::ToolKind {
    use aidaemon_client_protocol::ToolKind;
    use aidaemon_core::card::ActivityKind;
    match kind {
        ActivityKind::Read => ToolKind::Read,
        ActivityKind::Search => ToolKind::Search,
        ActivityKind::Write => ToolKind::Write,
        ActivityKind::Run => ToolKind::Run,
        ActivityKind::Web => ToolKind::Web,
        ActivityKind::Delegate => ToolKind::Delegate,
        ActivityKind::Recall => ToolKind::Recall,
        ActivityKind::Remember => ToolKind::Remember,
        ActivityKind::Remind => ToolKind::Remind,
        ActivityKind::Check => ToolKind::Check,
        ActivityKind::Change => ToolKind::Change,
    }
}
mod bus;
pub mod client_feature;
pub mod client_view;
mod config;
pub mod files;
mod observer;
mod pair_page;
pub mod pairing;
pub mod project_files;
mod remote;
pub mod remote_screens;
mod server;
pub mod setup;
pub mod shells;
pub mod surfaces;
pub mod token;
pub mod views;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::hub::{ChannelHub, HubChannel};
use aidaemon_channels::stop::{self, StopRequest};
use aidaemon_channels::turns::{MidRunPlacement, Turn, TurnDeps, TurnRunner};
use aidaemon_client_protocol::{
    Applies, ApprovalDecision, ClientKind, Command, CommandResult, ErrorCode, Event, Glance,
    HistoryMessage, LastReply, MemoryItem, MidRun, OwnerInfo, ProjectSummary, ProtocolError, Risk,
    Scope, SearchHit, ServerFrame, SessionChanges, SessionSummary, Settings, SettingsChange,
    SettingsOutcome, SetupAction, SetupState, SlashCommand, Snapshot, TaskState, TaskSummary,
    TurnState,
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
pub use remote::remote_router;
pub use server::{exposed_router, local_router, router, serve};
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
    /// The connected provider that offers `model`, with it, for a reply
    /// written by it (protocol 1.35); `None` when no provider lists it.
    /// Called inline as a turn ends: answer from memory and config only.
    fn served_by(&self, _model: &str) -> Option<aidaemon_client_protocol::ServedBy> {
        None
    }
    /// How the daemon itself is doing; `None` when this host does not say.
    async fn system(&self) -> Option<aidaemon_client_protocol::SystemStatus> {
        None
    }
    /// Up to `limit` messages of any session holding every word of
    /// `query`, newest first.
    async fn search(&self, _query: &str, _limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        Ok(Vec::new())
    }
    /// The slash commands the owner can run.
    fn commands(&self) -> Vec<SlashCommand> {
        Vec::new()
    }
    /// What the daemon remembers about the owner.
    async fn memories(&self) -> anyhow::Result<Vec<MemoryItem>> {
        Ok(Vec::new())
    }
    /// Forget memory `id`; `false` when there is no such memory.
    async fn forget(&self, _id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// The uncommitted changes in the git repositories session
    /// `session_id` worked in; empty when it worked in none.
    async fn changes(&self, _session_id: &str) -> anyhow::Result<SessionChanges> {
        Ok(SessionChanges::default())
    }

    /// The folder conversation `session_id`'s shell starts in (its
    /// project's); `None`: the owner's home folder.
    async fn shell_folder(&self, _session_id: &str) -> anyhow::Result<Option<std::path::PathBuf>> {
        Ok(None)
    }

    /// Name conversation `session_id` (`None` or blank: back to the name
    /// taken from what was said) (protocol 1.17). `Err`: why not.
    async fn rename_session(&self, _session_id: &str, _title: Option<&str>) -> Result<(), String> {
        Err("This daemon does not rename conversations".to_string())
    }

    /// Archive conversation `session_id`, or bring it back (protocol 1.17).
    async fn archive_session(&self, _session_id: &str, _archived: bool) -> Result<(), String> {
        Err("This daemon does not archive conversations".to_string())
    }

    /// Undo the uncommitted changes to `path` in repository `root`, one of
    /// session `session_id`'s; the changes after (protocol 1.17).
    async fn revert_file(
        &self,
        _session_id: &str,
        _root: &str,
        _path: &str,
    ) -> Result<SessionChanges, String> {
        Err("This daemon does not undo changes".to_string())
    }

    /// Whether `path` holds secrets (keys, credentials, env files, the
    /// daemon's own config): a project's Files view shows its name, never
    /// its text (protocol 1.32).
    fn is_secret_path(&self, path: &std::path::Path) -> bool {
        aidaemon_tools::path_class::DaemonPaths::default().is_secret(path)
    }

    /// The owner's projects.
    async fn projects(&self) -> anyhow::Result<Vec<ProjectSummary>> {
        Ok(Vec::new())
    }

    /// Make a project; with `create_folder`, make its missing folder too.
    /// `Err`: it could not be made, and why in plain words
    /// (`FolderMissing` when only the folder is missing).
    async fn create_project(
        &self,
        _name: &str,
        _path: Option<&str>,
        _create_folder: bool,
    ) -> Result<ProjectSummary, ProtocolError> {
        Err(ProtocolError::new(
            ErrorCode::Invalid,
            "This daemon has no projects",
        ))
    }

    /// Rename a project or change its folder (`Some("")` removes it).
    /// `Err`: nothing changed, and why.
    async fn update_project(
        &self,
        _id: &str,
        _name: Option<&str>,
        _path: Option<&str>,
        _create_folder: bool,
    ) -> Result<ProjectSummary, ProtocolError> {
        Err(ProtocolError::new(
            ErrorCode::Invalid,
            "This daemon has no projects",
        ))
    }

    /// Delete a project: its conversations, goals and runs go to no
    /// project and its bots to none; its folder stays. `Err`: nothing
    /// changed, and why.
    async fn delete_project(&self, _id: &str) -> Result<(), ProtocolError> {
        Err(ProtocolError::new(
            ErrorCode::Invalid,
            "This daemon has no projects",
        ))
    }

    /// The providers and models a conversation can be set to answer with
    /// (protocol 1.16).
    async fn model_options(&self) -> anyhow::Result<aidaemon_client_protocol::ModelOptions> {
        Ok(Default::default())
    }

    /// The model and effort conversation `session_id` answers with.
    async fn session_model(
        &self,
        session_id: &str,
    ) -> anyhow::Result<aidaemon_client_protocol::SessionModel> {
        Ok(aidaemon_client_protocol::SessionModel {
            session_id: session_id.to_string(),
            ..Default::default()
        })
    }

    /// Set what conversation `session_id` answers with from its next turn;
    /// a `None` stays as it is. `Err`: nothing changed, and why.
    async fn set_session_model(
        &self,
        _session_id: &str,
        _model: Option<aidaemon_client_protocol::ModelChoice>,
        _effort: Option<aidaemon_client_protocol::Effort>,
    ) -> Result<aidaemon_client_protocol::SessionModel, String> {
        Err("This daemon cannot change a conversation's model".to_string())
    }

    /// Put session `session_id` in project `project_id` (`None`: in no
    /// project). `Err`: it stayed where it was, and why.
    async fn set_session_project(
        &self,
        _session_id: &str,
        _project_id: Option<&str>,
    ) -> Result<(), String> {
        Err("This daemon has no projects".to_string())
    }

    /// Send chat bot `channel`'s direct-message chats to project
    /// `project_id` (`None`: to no project).
    async fn set_channel_project(
        &self,
        _channel: &str,
        _project_id: Option<&str>,
    ) -> Result<(), String> {
        Err("This daemon has no projects".to_string())
    }

    /// Keep `run` as it ended, for its conversation's history
    /// ([`Self::past_runs`]). Called inline as the run ends: hand the write
    /// off rather than wait on it.
    fn keep_run(&self, _run: &aidaemon_client_protocol::RunSummary) {}

    /// Conversation `session_id`'s ended runs as [`Self::keep_run`] kept
    /// them, oldest first.
    async fn past_runs(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Vec<aidaemon_client_protocol::RunSummary>> {
        Ok(Vec::new())
    }

    /// What run `run_id` did, as its record keeps it (`get_run`).
    async fn run_detail(
        &self,
        run_id: &str,
    ) -> anyhow::Result<aidaemon_client_protocol::RunDetail> {
        Ok(aidaemon_client_protocol::RunDetail {
            run_id: run_id.to_string(),
            agent_session_id: None,
            can_continue: false,
            model: None,
            working_dir: None,
            task: String::new(),
            result: None,
            steps: Vec::new(),
            plan: Vec::new(),
            permissions: Vec::new(),
            usage: None,
        })
    }

    /// Everything that runs on its own, next to run first.
    async fn automations(
        &self,
    ) -> anyhow::Result<Vec<aidaemon_client_protocol::AutomationSummary>> {
        Ok(Vec::new())
    }

    /// Change automation `id` as its owner: whether it changed, and what
    /// to tell them. `Err`: it could not be tried.
    async fn change_automation(
        &self,
        _id: &str,
        _change: aidaemon_client_protocol::AutomationChange,
    ) -> Result<(bool, String), String> {
        Err("This daemon has no automations".to_string())
    }

    /// Rename, reschedule or re-instruct one automation (protocol 1.15); a
    /// `None` field stays. Returns whether it changed, and what happened.
    async fn edit_automation(
        &self,
        _id: &str,
        _name: Option<&str>,
        _when: Option<&str>,
        _instructions: Option<&str>,
        _notify: Option<aidaemon_client_protocol::AutomationNotify>,
    ) -> Result<(bool, String), String> {
        Err("This daemon has no automations".to_string())
    }

    /// The daemon's settings; `None` when this host does not show them.
    async fn settings(&self) -> anyhow::Result<Option<Settings>> {
        Ok(None)
    }

    /// What `change` needs before it runs: nothing, or the owner's
    /// approval. `Err`: it cannot be made (why, in plain words).
    async fn settings_gate(&self, _change: &SettingsChange) -> Result<SettingsGate, String> {
        Err("This daemon does not change its settings from apps".to_string())
    }

    /// Make `change`; `approved`: the owner approved it (what
    /// [`Self::settings_gate`] asked for). `Err`: nothing was changed, and why.
    async fn change_settings(
        &self,
        _change: SettingsChange,
        _approved: bool,
    ) -> Result<SettingsDone, String> {
        Err("This daemon does not change its settings from apps".to_string())
    }

    /// A signed-in client sent a command: it reaches the gateway.
    fn client_seen(&self) {}

    /// First-run setup still to finish; `None` when there is none.
    async fn setup(&self) -> Option<SetupState> {
        None
    }

    /// Take a step of first-run setup. `Err`: it could not be taken, and
    /// why, in plain words.
    async fn setup_step(&self, _action: SetupAction) -> Result<SetupState, String> {
        Err("This daemon is already set up".to_string())
    }
}

/// What a settings change needs before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsGate {
    Free,
    /// The owner's approval, with what the card says.
    Ask {
        summary: String,
        warnings: Vec<String>,
        risk: Risk,
        /// Only another device (or the owner at the computer) may approve:
        /// it widens what the daemon may do.
        other_device: bool,
    },
}

/// A settings change made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsDone {
    pub message: String,
    pub applies: Option<Applies>,
}

/// The session a settings change's approval card belongs to.
pub const SETTINGS_SESSION: &str = "app:settings";

/// A new app conversation's id.
pub fn new_app_session_id() -> String {
    format!("{APP_SESSION_PREFIX}{}", uuid::Uuid::new_v4().simple())
}

/// How long a settings change waits for its approval.
const SETTINGS_APPROVAL_WAIT: Duration = Duration::from_secs(10 * 60);

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
    /// Files with messages (uploads, and pictures and documents the agent
    /// sends). `None`: the gateway takes no attachments.
    pub files: Option<Arc<files::FileStore>>,
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
    /// Turns running now, as far as each has got.
    live: observer::LiveTurns,
    hub: Arc<ChannelHub>,
    pub(crate) config: ClientGatewayConfig,
    endpoints: aidaemon_core::EndpointSource,
    pub(crate) token_file: PathBuf,
    pub(crate) pairing: Arc<Pairing>,
    pub(crate) views: Arc<ViewTargets>,
    /// Requests being handled now, so a retry waits for the first answer.
    in_flight: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    pub(crate) files: Option<Arc<files::FileStore>>,
    /// Live surfaces, once the daemon attaches its registry.
    pub(crate) surfaces: std::sync::OnceLock<Arc<surfaces::SurfaceDesk>>,
    /// Conversation shells, once the daemon attaches its registry.
    pub(crate) shells: std::sync::OnceLock<Arc<shells::ShellDesk>>,
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
        // Turns do not survive a restart either: end those a previous
        // process left running in the log, so a client that resumes stops
        // showing them as working (and offering Stop).
        match bus
            .log()
            .open_refs(OWNER, "turn_started", "turn_completed")
            .await
        {
            Ok(rows) => {
                for row in rows {
                    let Some(turn_id) = row.ref_id else { continue };
                    bus.publish(
                        row.session_id,
                        Event::TurnCompleted {
                            turn_id,
                            text: String::new(),
                            attachments: Vec::new(),
                            state: TurnState::Cancelled,
                            error: Some("The daemon restarted before this turn finished.".into()),
                            served_by: None,
                        },
                    )
                    .await;
                }
            }
            Err(error) => warn!(%error, "Could not read open turns from the client log"),
        }
        // Nor do tasks: one whose last update says it runs ends cancelled,
        // or a client keeps counting it as work and offering Stop.
        match bus.log().latest_per_ref(OWNER, "task_updated").await {
            Ok(rows) => {
                for row in rows {
                    let Ok(Event::TaskUpdated { mut task }) =
                        serde_json::from_str::<Event>(&row.payload)
                    else {
                        continue;
                    };
                    if !matches!(task.state, TaskState::Running | TaskState::Cancelling) {
                        continue;
                    }
                    task.state = TaskState::Cancelled;
                    task.finished_at_ms = Some(bus::now_ms());
                    task.error = Some("The daemon restarted before this task finished.".into());
                    bus.publish(row.session_id, Event::TaskUpdated { task })
                        .await;
                }
            }
            Err(error) => warn!(%error, "Could not read open tasks from the client log"),
        }

        let active = Arc::new(Mutex::new(HashSet::new()));
        let contexts = observer::Contexts::default();
        let live = observer::LiveTurns::default();
        deps.turns.watch.set(Arc::new(GatewayObserver {
            publisher: publisher.clone(),
            ids: ids.clone(),
            active: active.clone(),
            running: Mutex::default(),
            live: live.clone(),
            contexts: contexts.clone(),
            files: deps.files.clone(),
            host: deps.host.clone(),
        }));
        deps.hub.set_approval_tap(approvals.clone());
        let kept = observer::Kept::default();
        if let Some(files) = deps.files.clone() {
            deps.hub.set_media_tap(Arc::new(observer::MediaToClients {
                publisher: publisher.clone(),
                files,
                kept: kept.clone(),
            }));
        }
        deps.hub
            .register_channel(Arc::new(AppChannel {
                publisher: publisher.clone(),
                files: deps.files.clone(),
                kept,
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
            live,
            hub: deps.hub.clone(),
            config: deps.config,
            endpoints: deps.endpoints,
            token_file: deps.token_file,
            views: Arc::default(),
            in_flight: Mutex::default(),
            files: deps.files,
            surfaces: std::sync::OnceLock::new(),
            shells: std::sync::OnceLock::new(),
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

    /// The address devices reach the gateway at: a live tunnel origin, else
    /// `public_url`, else the loopback address.
    pub fn public_base(&self) -> String {
        PublicUrl {
            endpoints: self.endpoints.clone(),
            fallback: self.config.public_url(),
        }
        .get()
    }

    /// Offer live surfaces to the owner's apps (protocol 1.30): their
    /// events, the surface commands and the frame socket. Once only.
    pub fn attach_surfaces(&self, registry: aidaemon_surfaces::SurfaceRegistry) {
        let desk = surfaces::SurfaceDesk::new(registry);
        if self.surfaces.set(desk.clone()).is_ok() {
            desk.spawn_pump(self.publisher.clone(), self.views.clone());
        }
    }

    /// Offer conversation shells to the owner's apps (protocol 1.42):
    /// their events, the shell commands and the shell socket. Once only.
    pub fn attach_shells(&self, registry: aidaemon_shells::ShellRegistry) {
        let desk = shells::ShellDesk::new(registry);
        if self.shells.set(desk.clone()).is_ok() {
            desk.spawn_pump(self.publisher.clone());
        }
    }

    /// The shells desk, when shells are attached and turned on.
    pub(crate) fn shell_desk(&self) -> Option<Arc<shells::ShellDesk>> {
        self.shells.get().filter(|desk| desk.enabled()).cloned()
    }

    /// The surfaces desk, when surfaces are attached and turned on.
    pub(crate) fn surface_desk(&self) -> Option<Arc<surfaces::SurfaceDesk>> {
        self.surfaces.get().filter(|desk| desk.enabled()).cloned()
    }

    pub fn owner(&self) -> OwnerInfo {
        self.host.owner()
    }

    /// First-run setup still to finish.
    pub async fn setup(&self) -> Option<SetupState> {
        self.host.setup().await
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
        // Remembered per caller: one device never reads the reply another
        // got (history, memories, a live-view ticket) by presenting its
        // request id.
        let remembered_as = match caller.principal.device_id.as_deref() {
            Some(device_id) => format!("{device_id}/{request_id}"),
            None => request_id.to_string(),
        };
        match log.remembered_reply(OWNER, &remembered_as).await {
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
                .remember_reply(OWNER, &remembered_as, bus::now_ms(), &text)
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
            | Command::GetSnapshot
            | Command::GetSystem
            | Command::SearchMessages { .. }
            | Command::ListCommands
            | Command::ListMemories
            | Command::GetSettings
            | Command::GetSetup
            | Command::GetChanges { .. }
            | Command::ListProjects
            | Command::ListAutomations
            | Command::GetRun { .. }
            | Command::ListModels
            | Command::GetSessionModel { .. } => Some(Scope::Observe),
            Command::SetSessionModel { .. } => Some(Scope::Chat),
            Command::ListQueued { .. } => Some(Scope::Observe),
            Command::RenameSession { .. }
            | Command::ArchiveSession { .. }
            | Command::RevertFile { .. }
            | Command::EditQueued { .. } => Some(Scope::Chat),
            // A project's source is more than a watch should read; a device
            // that may chat could ask the agent for it anyway.
            Command::ListProjectFiles { .. } | Command::ReadProjectFile { .. } => Some(Scope::Chat),
            Command::CreateProject { .. }
            | Command::UpdateProject { .. }
            | Command::DeleteProject { .. }
            | Command::SetSessionProject { .. }
            | Command::SetChannelProject { .. }
            | Command::ChangeAutomation { .. }
            | Command::EditAutomation { .. } => Some(Scope::Chat),
            Command::ForgetMemory { .. }
            | Command::UploadFile { .. }
            | Command::ChangeSettings { .. }
            | Command::Setup { .. } => Some(Scope::Chat),
            Command::ListDevices => Some(Scope::Observe),
            // The agent's screens (their sites and titles) are for a device
            // that may watch them.
            Command::ListSurfaces => Some(Scope::Control),
            Command::PointAt { .. } | Command::ShareScreen { .. } => Some(Scope::Control),
            Command::SetDeviceScopes { .. } => None,
            Command::OpenSurface { .. }
            | Command::CloseSurface { .. }
            | Command::TakeControl { .. }
            | Command::ReleaseControl { .. } => Some(Scope::Control),
            // A shell runs anything here: checked below, beyond a scope.
            Command::OpenShell { .. } | Command::CloseShell { .. } => Some(Scope::Control),
            Command::Unknown => None,
        };
        if let Some(scope) = needs
            && !self.pairing.may_now(&caller.principal, scope)
        {
            return Err(ProtocolError::new(
                ErrorCode::Forbidden,
                format!("This device may not {scope:?}"),
            ));
        }
        if matches!(
            command,
            Command::OpenShell { .. } | Command::CloseShell { .. }
        ) && !self.pairing.may_shell_now(&caller.principal)
        {
            return Err(ProtocolError::new(
                ErrorCode::Forbidden,
                "Only a device that may answer every approval may use a shell",
            ));
        }
        // A setup step can connect a chat bot (a new way in, with an owner
        // of its own) and restarts the daemon: that is for the owner's own
        // computer or a device that may answer any approval, never one with
        // only Chat or a risk ceiling.
        if matches!(command, Command::Setup { .. })
            && !caller.principal.is_owner()
            && (!self.pairing.may_now(&caller.principal, Scope::Approve)
                || caller.principal.ceiling.is_some())
        {
            return Err(ProtocolError::new(
                ErrorCode::Forbidden,
                "This device may not change the daemon's setup",
            ));
        }
        self.host.client_seen();
        match command {
            Command::SendMessage {
                session_id,
                text,
                attachments,
                mid_run,
                project_id,
                model,
                effort,
                in_run,
            } => {
                let placement = match mid_run {
                    Some(MidRun::Steer) => Some(MidRunPlacement::Steer),
                    Some(MidRun::Queue) => Some(MidRunPlacement::Queue),
                    Some(MidRun::Unknown) | None => None,
                };
                self.send_message(
                    request_id,
                    session_id,
                    text,
                    attachments,
                    caller.view_connection,
                    placement,
                    project_id,
                    (model, effort),
                    in_run,
                )
                .await
            }
            Command::RenameSession { session_id, title } => {
                self.host
                    .rename_session(&session_id, title.as_deref())
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
                self.publish_session(&session_id).await;
                Ok(CommandResult::Reply {
                    text: "Renamed".to_string(),
                })
            }
            Command::ArchiveSession {
                session_id,
                archived,
            } => {
                self.host
                    .archive_session(&session_id, archived)
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
                self.publish_session(&session_id).await;
                if archived && let Some(desk) = self.shells.get() {
                    desk.registry().close_session(&session_id);
                }
                Ok(CommandResult::Reply {
                    text: if archived { "Archived" } else { "Brought back" }.to_string(),
                })
            }
            Command::RevertFile {
                session_id,
                root,
                path,
            } => Ok(CommandResult::Changes {
                changes: self
                    .host
                    .revert_file(&session_id, &root, &path)
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?,
            }),
            Command::ListQueued { session_id } => Ok(self.queued(session_id).await),
            Command::EditQueued {
                session_id,
                message_id,
                now: true,
                ..
            } => {
                if !self
                    .turns
                    .task_registry
                    .send_queued_now(&session_id, message_id)
                    .await
                {
                    return Err(ProtocolError::new(
                        ErrorCode::Conflict,
                        "That message is no longer waiting for its own turn",
                    ));
                }
                Ok(self.queued(session_id).await)
            }
            Command::EditQueued {
                session_id,
                message_id,
                text,
                now: false,
            } => {
                let text = text.map(|t| t.trim().to_string());
                if text.as_deref().is_some_and(str::is_empty) {
                    return Err(ProtocolError::new(
                        ErrorCode::Invalid,
                        "A waiting message needs some text; remove it instead",
                    ));
                }
                if !self
                    .turns
                    .task_registry
                    .edit_queued_message(&session_id, message_id, text)
                    .await
                {
                    return Err(ProtocolError::new(
                        ErrorCode::Conflict,
                        "That message is no longer waiting: it has gone to the agent",
                    ));
                }
                Ok(self.queued(session_id).await)
            }
            Command::ListModels => Ok(CommandResult::Models {
                options: self.host.model_options().await.map_err(internal)?,
            }),
            Command::GetSessionModel { session_id } => Ok(CommandResult::SessionModel {
                session_model: self
                    .host
                    .session_model(&session_id)
                    .await
                    .map_err(internal)?,
            }),
            Command::SetSessionModel {
                session_id,
                model,
                effort,
            } => {
                if !session_id.starts_with(APP_SESSION_PREFIX) {
                    return Err(ProtocolError::new(
                        ErrorCode::Invalid,
                        "Only the app's own conversations can be set to a model",
                    ));
                }
                Ok(CommandResult::SessionModel {
                    session_model: self
                        .host
                        .set_session_model(&session_id, model, effort)
                        .await
                        .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?,
                })
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
                // A task's id, or a run's (protocol 1.46: a run's card stops
                // it by the id its `run_updated` carries).
                let found = match self.ids.parse_task(&task_id) {
                    Some(found) => Some(found),
                    None => self.running_run(&task_id).await,
                };
                let Some((session_id, number)) = found else {
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
                // Who answered: a card that goes away is traceable to the
                // client and device that answered it.
                tracing::info!(
                    approval_id = %approval_id,
                    ?decision,
                    client = ?caller.kind,
                    device = device.as_deref().unwrap_or("this machine"),
                    "An approval was answered from a client"
                );
                self.resolve_approval(
                    &approval_id,
                    decision,
                    device,
                    caller.principal.device_id.as_deref(),
                    ceiling,
                )
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
                // Exact command words: `/trustall on` raises AIdaemon's
                // level, so it takes the same check (and card) as Settings.
                if command == "trustall" && args.trim().eq_ignore_ascii_case("on") {
                    return self.trust_all_on(caller).await;
                }
                self.slash(&session_id, &command, &args).await
            }
            Command::GetGlance => Ok(CommandResult::Glance {
                glance: self.glance(caller.compact).await,
            }),
            Command::GetSnapshot => {
                let (seq, mut state) = self.snapshot().await;
                if !self.pairing.may_now(&caller.principal, Scope::Control) {
                    state.surfaces.clear();
                }
                if !self.pairing.may_shell_now(&caller.principal) {
                    state.shells.clear();
                }
                Ok(CommandResult::Snapshot { seq, state })
            }
            Command::GetSystem => match self.host.system().await {
                Some(system) => Ok(CommandResult::System { system }),
                None => Err(ProtocolError::new(
                    ErrorCode::NotFound,
                    "This daemon does not report its status",
                )),
            },
            Command::SearchMessages { query, limit } => {
                let query = query.trim();
                if query.is_empty() {
                    return Ok(CommandResult::SearchResults { hits: Vec::new() });
                }
                let limit = (limit as usize).clamp(1, 100);
                let hits = self.host.search(query, limit).await.map_err(|e| {
                    warn!(error = %e, "Message search failed");
                    ProtocolError::new(ErrorCode::Internal, "Search failed")
                })?;
                Ok(CommandResult::SearchResults { hits })
            }
            Command::ListCommands => Ok(CommandResult::Commands {
                commands: self.host.commands(),
            }),
            Command::ListMemories => {
                let memories = self.host.memories().await.map_err(|e| {
                    warn!(error = %e, "Listing memories failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not read memories")
                })?;
                Ok(CommandResult::Memories { memories })
            }
            Command::UploadFile {
                name,
                mime,
                data,
                marked_from,
            } => self.upload(name, mime, data, marked_from),
            Command::ForgetMemory { memory_id } => match self.host.forget(&memory_id).await {
                Ok(true) => Ok(CommandResult::Forgotten { memory_id }),
                Ok(false) => Err(ProtocolError::new(ErrorCode::NotFound, "No such memory")),
                Err(e) => {
                    warn!(error = %e, "Forgetting a memory failed");
                    Err(ProtocolError::new(
                        ErrorCode::Internal,
                        "Could not forget it",
                    ))
                }
            },
            Command::ListProjects => {
                let projects = self.host.projects().await.map_err(|e| {
                    warn!(error = %e, "Listing projects failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not read the projects")
                })?;
                Ok(CommandResult::Projects { projects })
            }
            Command::CreateProject {
                name,
                path,
                create_folder,
            } => {
                let project = self
                    .host
                    .create_project(&name, path.as_deref(), create_folder)
                    .await?;
                self.publisher.publish(
                    None,
                    Event::ProjectUpdated {
                        project: project.clone(),
                    },
                );
                Ok(CommandResult::Project { project })
            }
            Command::UpdateProject {
                project_id,
                name,
                path,
                create_folder,
            } => {
                let project = self
                    .host
                    .update_project(&project_id, name.as_deref(), path.as_deref(), create_folder)
                    .await?;
                self.publisher.publish(
                    None,
                    Event::ProjectUpdated {
                        project: project.clone(),
                    },
                );
                Ok(CommandResult::Project { project })
            }
            Command::DeleteProject { project_id } => {
                // Its conversations, read before they move, are sent again
                // in no project.
                let moved: Vec<String> = match self.host.sessions().await {
                    Ok(sessions) => sessions
                        .into_iter()
                        .filter(|s| s.project_id.as_deref() == Some(project_id.as_str()))
                        .map(|s| s.id)
                        .collect(),
                    Err(e) => {
                        warn!(error = %e, "Reading sessions before a project delete failed");
                        Vec::new()
                    }
                };
                self.host.delete_project(&project_id).await?;
                self.publisher.publish(
                    None,
                    Event::ProjectDeleted {
                        project_id: project_id.clone(),
                    },
                );
                for session in moved {
                    self.publish_session(&session).await;
                }
                let projects = self.host.projects().await.map_err(|e| {
                    warn!(error = %e, "Listing projects failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not read the projects")
                })?;
                Ok(CommandResult::Projects { projects })
            }
            Command::SetSessionProject {
                session_id,
                project_id,
            } => {
                self.host
                    .set_session_project(&session_id, project_id.as_deref())
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
                self.publish_session(&session_id).await;
                Ok(CommandResult::Reply {
                    text: match project_id {
                        Some(_) => "Moved to the project".to_string(),
                        None => "Taken out of its project".to_string(),
                    },
                })
            }
            Command::SetChannelProject {
                channel,
                project_id,
            } => {
                self.host
                    .set_channel_project(&channel, project_id.as_deref())
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
                let projects = self.host.projects().await.map_err(|e| {
                    warn!(error = %e, "Listing projects failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not list the projects")
                })?;
                for project in &projects {
                    self.publisher.publish(
                        None,
                        Event::ProjectUpdated {
                            project: project.clone(),
                        },
                    );
                }
                Ok(CommandResult::Projects { projects })
            }
            Command::GetRun { run_id } => {
                let run = self.host.run_detail(&run_id).await.map_err(|e| {
                    warn!(error = %e, "Reading a run failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not read the run")
                })?;
                Ok(CommandResult::Run { run })
            }
            Command::ListAutomations => {
                let automations = self.host.automations().await.map_err(|e| {
                    warn!(error = %e, "Listing automations failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not list the automations")
                })?;
                Ok(CommandResult::Automations {
                    automations,
                    note: None,
                    changed: false,
                })
            }
            Command::ChangeAutomation {
                automation_id,
                change,
            } => {
                let (changed, note) = self
                    .host
                    .change_automation(&automation_id, change)
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
                let automations = self.host.automations().await.map_err(|e| {
                    warn!(error = %e, "Listing automations failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not list the automations")
                })?;
                Ok(CommandResult::Automations {
                    automations,
                    note: Some(note),
                    changed,
                })
            }
            Command::EditAutomation {
                automation_id,
                name,
                when,
                instructions,
                notify,
            } => {
                let (changed, note) = self
                    .host
                    .edit_automation(
                        &automation_id,
                        name.as_deref(),
                        when.as_deref(),
                        instructions.as_deref(),
                        notify,
                    )
                    .await
                    .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
                let automations = self.host.automations().await.map_err(|e| {
                    warn!(error = %e, "Listing automations failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not list the automations")
                })?;
                Ok(CommandResult::Automations {
                    automations,
                    note: Some(note),
                    changed,
                })
            }
            Command::ListProjectFiles {
                project_id,
                path,
                ignored,
            } => {
                let folder = self.project_folder(&project_id).await?;
                let host = self.host.clone();
                let dir = tokio::task::spawn_blocking(move || {
                    project_files::list(
                        &folder,
                        &project_id,
                        path.as_deref().unwrap_or(""),
                        ignored,
                        &|p| host.is_secret_path(p),
                    )
                })
                .await
                .map_err(|_| {
                    ProtocolError::new(ErrorCode::Internal, "Could not read the folder")
                })??;
                Ok(CommandResult::ProjectDir { dir })
            }
            Command::ReadProjectFile { project_id, path } => {
                let folder = self.project_folder(&project_id).await?;
                let host = self.host.clone();
                let files = self.files.clone();
                let file = tokio::task::spawn_blocking(move || {
                    project_files::read_linked(
                        &folder,
                        &project_id,
                        &path,
                        &|p| host.is_secret_path(p),
                        &|shown| {
                            let mut link = files
                                .as_ref()?
                                .register(shown.path, shown.name, shown.mime, shown.size_bytes)
                                .map_err(|e| warn!(error = %e, "Could not link a project file"))
                                .ok()?;
                            // A new version is a new address, so no cached
                            // copy of the old one shows.
                            let version = shown.version.trim_start_matches("sha256:");
                            link.url = link.url.map(|url| {
                                format!("{url}?v={}", &version[..version.len().min(16)])
                            });
                            Some(link)
                        },
                    )
                })
                .await
                .map_err(|_| {
                    ProtocolError::new(ErrorCode::Internal, "Could not read the file")
                })??;
                Ok(CommandResult::ProjectFile { file })
            }
            Command::GetChanges { session_id } => {
                let changes = self.host.changes(&session_id).await.map_err(|e| {
                    warn!(error = %e, "Reading a session's changes failed");
                    ProtocolError::new(ErrorCode::Internal, "Could not read the changes")
                })?;
                Ok(CommandResult::Changes { changes })
            }
            Command::GetSettings => match self.host.settings().await {
                Ok(Some(settings)) => Ok(CommandResult::Settings { settings }),
                Ok(None) => Err(ProtocolError::new(
                    ErrorCode::NotFound,
                    "This daemon does not show its settings",
                )),
                Err(e) => {
                    warn!(error = %e, "Reading settings failed");
                    Err(ProtocolError::new(
                        ErrorCode::Internal,
                        "Could not read the settings",
                    ))
                }
            },
            Command::ChangeSettings { change } => self.change_settings(change, caller).await,
            Command::GetSetup => Ok(CommandResult::Setup {
                setup: self.host.setup().await.unwrap_or_else(|| SetupState {
                    step: aidaemon_client_protocol::SetupStep::Done,
                    ..SetupState::default()
                }),
            }),
            Command::Setup { action } => {
                let setup = self
                    .host
                    .setup_step(action)
                    .await
                    .map_err(|message| ProtocolError::new(ErrorCode::Invalid, message))?;
                self.publisher.publish(
                    None,
                    Event::SetupUpdated {
                        setup: setup.clone(),
                    },
                );
                Ok(CommandResult::Setup { setup })
            }
            Command::PointAt {
                surface_id,
                x,
                y,
                note,
            } => {
                // It sends a message too.
                if !self.pairing.may_now(&caller.principal, Scope::Chat) {
                    return Err(ProtocolError::new(
                        ErrorCode::Forbidden,
                        "This device may not Chat",
                    ));
                }
                let (session, text) = self
                    .need_surfaces()?
                    .point_message(surface_id, x, y, note)
                    .await?;
                self.send_message(
                    request_id,
                    Some(session),
                    text,
                    Vec::new(),
                    caller.view_connection,
                    None,
                    None,
                    (None, None),
                    None,
                )
                .await
            }
            Command::ShareScreen {
                name,
                width,
                height,
            } => self
                .need_surfaces()?
                .share(&caller.principal, name, width, height),
            Command::ListDevices => self.devices_for(caller).await,
            Command::SetDeviceScopes { device_id, scopes } => {
                // Only the daemon's own computer: a paired device never
                // changes what it, or another device, may do.
                if caller.principal.device_id.is_some() {
                    return Err(ProtocolError::new(
                        ErrorCode::Forbidden,
                        "Change what devices may do on the computer AIDaemon runs on",
                    ));
                }
                if !self.pairing.set_scopes(&device_id, &scopes).await? {
                    return Err(ProtocolError::new(ErrorCode::NotFound, "No such device"));
                }
                self.devices_for(caller).await
            }
            Command::ListSurfaces => Ok(CommandResult::Surfaces {
                surfaces: self
                    .surface_desk()
                    .map(|desk| desk.list())
                    .unwrap_or_default(),
            }),
            Command::OpenSurface {
                surface_id,
                mode,
                accept,
            } => {
                self.need_surfaces()?
                    .open(&caller.principal, surface_id, mode, accept)
                    .await
            }
            Command::OpenShell {
                session_id,
                cols,
                rows,
            } => {
                let desk = self.shell_desk().ok_or_else(|| {
                    ProtocolError::new(ErrorCode::Forbidden, "Shells are turned off")
                })?;
                let folder = match self.host.shell_folder(&session_id).await {
                    Ok(Some(folder)) => folder,
                    Ok(None) => dirs::home_dir().ok_or_else(|| {
                        ProtocolError::new(ErrorCode::Internal, "No home folder for the shell")
                    })?,
                    Err(error) => {
                        warn!(%error, "Finding a conversation's folder failed");
                        return Err(ProtocolError::new(
                            ErrorCode::Internal,
                            "Could not find the conversation's folder",
                        ));
                    }
                };
                desk.open(&caller.principal, &session_id, &folder, cols, rows)
            }
            Command::CloseShell { shell_id } => self
                .shell_desk()
                .ok_or_else(|| ProtocolError::new(ErrorCode::Forbidden, "Shells are turned off"))?
                .close(&shell_id),
            Command::CloseSurface { live_id } => {
                self.need_surfaces()?.close(&caller.principal, &live_id)
            }
            Command::TakeControl { surface_id } => {
                self.need_surfaces()?
                    .take(&caller.principal, surface_id)
                    .await
            }
            Command::ReleaseControl { surface_id } => {
                self.need_surfaces()?.release(&caller.principal, surface_id)
            }
            Command::Unknown => Err(ProtocolError::new(
                ErrorCode::Invalid,
                "Unknown command for this gateway version",
            )),
        }
    }

    async fn devices_for(&self, caller: &Caller) -> Result<CommandResult, ProtocolError> {
        Ok(CommandResult::Devices {
            devices: self.pairing.devices().await?,
            can_change: caller.principal.device_id.is_none(),
        })
    }

    fn need_surfaces(&self) -> Result<Arc<surfaces::SurfaceDesk>, ProtocolError> {
        self.surface_desk().ok_or_else(|| {
            ProtocolError::new(
                ErrorCode::Forbidden,
                "Watching the agent's screens is turned off in Settings",
            )
        })
    }

    /// A settings change from a client: made now, or after the owner
    /// approves it. Every client is told to read the settings again.
    async fn change_settings(
        &self,
        change: SettingsChange,
        caller: &Caller,
    ) -> Result<CommandResult, ProtocolError> {
        let invalid = |message: String| ProtocolError::new(ErrorCode::Invalid, message);
        // Only a client on the daemon's machine (the local token) can finish
        // a sign-in on its loopback port; any other pastes the address.
        let change = match change {
            SettingsChange::SignIn { paste } => SettingsChange::SignIn {
                paste: paste || !caller.principal.is_owner(),
            },
            other => other,
        };
        match self.host.settings_gate(&change).await.map_err(invalid)? {
            SettingsGate::Free => {
                let done = self
                    .host
                    .change_settings(change, false)
                    .await
                    .map_err(invalid)?;
                self.settings_updated(true, done.message.clone());
                Ok(CommandResult::SettingsChanged {
                    outcome: SettingsOutcome::Done {
                        message: done.message,
                        applies: done.applies,
                    },
                })
            }
            SettingsGate::Ask {
                summary,
                warnings,
                risk,
                other_device,
            } => {
                let mut warnings = warnings;
                let mut not_from = other_device
                    .then(|| caller.principal.device_id.clone())
                    .flatten();
                // With no other device to ask, the device asking confirms
                // it itself, on a card that says what widens.
                if let Some(device) = not_from.as_deref()
                    && !self.another_approver(device).await
                {
                    not_from = None;
                    warnings.push(
                        "No other device is paired, so this device confirms it. Only allow it \
                         if you asked for it just now."
                            .to_string(),
                    );
                }
                let other_device = not_from.is_some();
                let approval_id = self
                    .ask_settings_change(change, summary, warnings, risk, not_from, None)
                    .await;
                Ok(CommandResult::SettingsChanged {
                    outcome: SettingsOutcome::AwaitingApproval {
                        approval_id,
                        other_device,
                    },
                })
            }
        }
    }

    /// Open a card for `change` on the owner's apps (not on device
    /// `not_from`), and make it when one approves; what happened is
    /// announced to the apps and, when `chat` names the chat that asked,
    /// told there too. Returns the card's id.
    async fn ask_settings_change(
        &self,
        change: SettingsChange,
        summary: String,
        warnings: Vec<String>,
        risk: Risk,
        not_from: Option<String>,
        chat: Option<String>,
    ) -> String {
        self.runner
            .register_session(SETTINGS_SESSION, APP_CHANNEL)
            .await;
        let (approval_id, answer) = self.approvals.ask(approvals::OwnAsk {
            session_id: SETTINGS_SESSION.to_string(),
            summary,
            warnings,
            risk,
            expires_at_ms: bus::now_ms().saturating_add(SETTINGS_APPROVAL_WAIT.as_millis() as u64),
            not_from,
        });
        let host = self.host.clone();
        let approvals = self.approvals.clone();
        let publisher = self.publisher.clone();
        let hub = self.hub.clone();
        let id = approval_id.clone();
        tokio::spawn(async move {
            let answer = tokio::time::timeout(SETTINGS_APPROVAL_WAIT, answer)
                .await
                .ok()
                .and_then(Result::ok);
            let approved = matches!(
                answer,
                Some((
                    aidaemon_tools::semantics::ApprovalResponse::AllowOnce
                        | aidaemon_tools::semantics::ApprovalResponse::AllowSession
                        | aidaemon_tools::semantics::ApprovalResponse::AllowAlways,
                    _
                ))
            );
            let answered = answer.is_some();
            approvals.settle(&id, answer);
            let (ok, message) = if approved {
                match host.change_settings(change, true).await {
                    Ok(done) => (true, done.message),
                    Err(reason) => (false, reason),
                }
            } else if answered {
                (false, "Declined: nothing was changed.".to_string())
            } else {
                (
                    false,
                    "Nobody approved it in time: nothing was changed.".to_string(),
                )
            };
            if let Some(chat) = chat
                && let Err(error) = hub.send_text(&chat, &message).await
            {
                warn!(%error, "Could not tell the chat how its settings change ended");
            }
            publisher.publish(None, Event::SettingsUpdated { ok, message });
        });
        approval_id
    }

    /// Ask the owner's paired devices to approve `change` for a chat that
    /// cannot answer an app's card itself (a slash command sent in
    /// Telegram): the chat is never the one that approves. `None` when no
    /// paired device can answer; else what to tell the chat now, or why
    /// nothing was asked.
    pub async fn ask_devices_for_chat(
        &self,
        change: SettingsChange,
        chat_session: &str,
    ) -> Option<Result<String, String>> {
        let devices = self.pairing.devices().await.unwrap_or_default();
        if !devices.iter().any(|device| !device.revoked) {
            return None;
        }
        Some(match self.host.settings_gate(&change).await {
            Err(reason) => Err(reason),
            Ok(SettingsGate::Free) => match self.host.change_settings(change, false).await {
                Ok(done) => {
                    self.settings_updated(true, done.message.clone());
                    Ok(done.message)
                }
                Err(reason) => Err(reason),
            },
            Ok(SettingsGate::Ask {
                summary,
                warnings,
                risk,
                ..
            }) => {
                self.ask_settings_change(
                    change,
                    summary,
                    warnings,
                    risk,
                    None,
                    Some(chat_session.to_string()),
                )
                .await;
                Ok(
                    "Approve it in the AIdaemon app on one of your paired devices; I'll say \
                     here when it's done."
                        .to_string(),
                )
            }
        })
    }

    /// Whether a device other than `device_id` can answer an approval this
    /// gateway opens: another paired device. A chat cannot (these cards
    /// reach the owner's apps only), so it does not count.
    async fn another_approver(&self, device_id: &str) -> bool {
        let devices = self.pairing.devices().await.unwrap_or_default();
        devices
            .iter()
            .any(|d| !d.revoked && d.device_id != device_id)
    }

    /// Start an app conversation with a message from the daemon itself (the
    /// first-run greeting), with prompts the owner may press. The host keeps
    /// the message in `session_id`'s history first ([`new_app_session_id`]).
    pub async fn post_greeting(
        &self,
        session_id: &str,
        title: &str,
        text: &str,
        suggestions: Vec<String>,
    ) {
        let session_id = session_id.to_string();
        self.runner.register_session(&session_id, APP_CHANNEL).await;
        self.publisher.publish(
            Some(session_id.clone()),
            Event::SessionCreated {
                session: SessionSummary {
                    archived: false,
                    id: session_id.clone(),
                    channel: APP_CHANNEL.to_string(),
                    title: Some(title.to_string()),
                    last_activity_ms: Some(bus::now_ms()),
                    writable: true,
                    project_id: None,
                    group: false,
                },
            },
        );
        self.publisher.publish(
            Some(session_id.clone()),
            Event::AssistantMessage {
                attachments: Vec::new(),
                session_id: session_id.clone(),
                text: text.to_string(),
                suggestions,
            },
        );
    }

    /// The agent needs `feature`, which is off: every client offers to turn
    /// it on.
    pub fn feature_requested(
        &self,
        session_id: &str,
        feature: aidaemon_client_protocol::Feature,
        reason: &str,
    ) {
        self.publisher.publish(
            Some(session_id.to_string()),
            Event::FeatureRequested {
                session_id: session_id.to_string(),
                feature,
                reason: reason.to_string(),
            },
        );
    }

    /// Tell every client where setup stands.
    pub fn setup_updated(&self, setup: SetupState) {
        self.publisher.publish(None, Event::SetupUpdated { setup });
    }

    /// Tell every client to read the settings again: `message` says what
    /// changed (a sign-in that finished on its own, say).
    /// `upload_file`: keep the file for a message to send.
    fn upload(
        &self,
        name: String,
        mime: Option<String>,
        data: String,
        marked_from: Option<String>,
    ) -> Result<CommandResult, ProtocolError> {
        use base64::Engine as _;
        let Some(files) = &self.files else {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "This gateway does not take files",
            ));
        };
        let too_big = || {
            ProtocolError::new(
                ErrorCode::Invalid,
                format!(
                    "The file is too large: up to {} MB",
                    files::MAX_UPLOAD_BYTES / (1024 * 1024)
                ),
            )
        };
        if data.len() / 4 * 3 > files::MAX_UPLOAD_BYTES + 3 {
            return Err(too_big());
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data.trim())
            .map_err(|_| ProtocolError::new(ErrorCode::Invalid, "The file is not base64"))?;
        if bytes.is_empty() {
            return Err(ProtocolError::new(ErrorCode::Invalid, "The file is empty"));
        }
        if bytes.len() > files::MAX_UPLOAD_BYTES {
            return Err(too_big());
        }
        let mut attachment = files.save(&name, mime.as_deref(), &bytes).map_err(|e| {
            warn!(error = %e, "Saving an uploaded file failed");
            ProtocolError::new(ErrorCode::Internal, "Could not keep the file")
        })?;
        if let Some(original) = marked_from {
            match files.mark_up(&attachment.id, &original) {
                Ok(true) => attachment.marked_from = Some(original),
                Ok(false) => warn!("A marked-up picture names a file the gateway does not have"),
                Err(e) => warn!(error = %e, "Noting a marked-up picture failed"),
            }
        }
        Ok(CommandResult::Uploaded { attachment })
    }

    /// The uploaded files a message names, as the agent takes them.
    fn message_files(
        &self,
        attachments: &[aidaemon_client_protocol::AttachmentRef],
    ) -> Result<Vec<aidaemon_tools::semantics::MessageAttachment>, ProtocolError> {
        if attachments.is_empty() {
            return Ok(Vec::new());
        }
        let Some(files) = &self.files else {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "This gateway does not take files",
            ));
        };
        attachments
            .iter()
            .map(|attachment| {
                let record = files.get(&attachment.id).ok_or_else(|| {
                    ProtocolError::new(
                        ErrorCode::NotFound,
                        format!("No uploaded file {}: upload it again", attachment.id),
                    )
                })?;
                let mut file = aidaemon_channels::attachments::message_attachment(
                    record.path,
                    record.name,
                    record.mime,
                    record.size_bytes,
                );
                // Drawn over another picture: the agent is told so, and
                // where the unmarked one is.
                file.marked_up_from = record
                    .marked_from
                    .and_then(|original| files.get(&original))
                    .map(|original| original.path.to_string_lossy().into_owned());
                Ok(file)
            })
            .collect()
    }

    pub fn settings_updated(&self, ok: bool, message: String) {
        self.publisher
            .publish(None, Event::SettingsUpdated { ok, message });
    }

    fn resolve_approval(
        &self,
        approval_id: &str,
        decision: ApprovalDecision,
        device: Option<String>,
        device_id: Option<&str>,
        ceiling: Option<Risk>,
    ) -> Result<CommandResult, ProtocolError> {
        let outcome = self
            .approvals
            .resolve(approval_id, decision, device, device_id, ceiling)?;
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
        placement: Option<MidRunPlacement>,
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
                // Not seen since the daemon started: a chat recorded as a
                // direct message stays private; anything else is treated
                // as seen by others.
                let direct = self
                    .host
                    .sessions()
                    .await
                    .ok()
                    .and_then(|all| all.into_iter().find(|s| s.id == session_id))
                    .is_some_and(|s| !s.group);
                let mut context = aidaemon_channels::context::ChannelContext::private(&platform);
                if !direct {
                    context.visibility = aidaemon_core::ChannelVisibility::Public;
                }
                context
            }
        };
        self.hub
            .send_text(&session_id, &format!("↪ From the AIDaemon app: {text}"))
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
        tokio::spawn(async move { runner.submit_placed(surface, turn, placement).await });
        Ok(CommandResult::MessageAccepted {
            session_id,
            new_session: false,
        })
    }

    /// Tell every client how session `session_id` looks now (after a move
    /// between projects).
    /// `session_id`'s waiting messages, as `queued` gives them.
    /// Project `id`'s folder, for its Files view.
    async fn project_folder(&self, id: &str) -> Result<std::path::PathBuf, ProtocolError> {
        let projects = self.host.projects().await.map_err(|e| {
            warn!(error = %e, "Listing projects failed");
            ProtocolError::new(ErrorCode::Internal, "Could not read the projects")
        })?;
        let project = projects
            .into_iter()
            .find(|p| p.id == id)
            .ok_or_else(|| ProtocolError::new(ErrorCode::NotFound, "No such project"))?;
        project
            .path
            .filter(|p| !p.trim().is_empty())
            .map(std::path::PathBuf::from)
            .ok_or_else(|| ProtocolError::new(ErrorCode::Invalid, "This project has no folder"))
    }

    async fn queued(&self, session_id: String) -> CommandResult {
        let messages = self
            .turns
            .task_registry
            .queued_messages(&session_id)
            .await
            .into_iter()
            .map(
                |(id, text, at, after)| aidaemon_client_protocol::QueuedMessage {
                    id,
                    text,
                    queued_at_ms: u64::try_from(at.timestamp_millis()).unwrap_or(0),
                    after,
                },
            )
            .collect();
        CommandResult::Queued {
            session_id,
            messages,
        }
    }

    async fn publish_session(&self, session_id: &str) {
        match self.host.sessions().await {
            Ok(sessions) => {
                if let Some(session) = sessions.into_iter().find(|s| s.id == session_id) {
                    self.publisher.publish(
                        Some(session_id.to_string()),
                        Event::SessionUpdated { session },
                    );
                }
            }
            Err(e) => warn!(error = %e, "Reading sessions after a project move failed"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn send_message(
        &self,
        request_id: &str,
        session_id: Option<String>,
        text: String,
        attachments: Vec<aidaemon_client_protocol::AttachmentRef>,
        view_connection: Option<u64>,
        placement: Option<MidRunPlacement>,
        project_id: Option<String>,
        (model, effort): (
            Option<aidaemon_client_protocol::ModelChoice>,
            Option<aidaemon_client_protocol::Effort>,
        ),
        in_run: Option<String>,
    ) -> Result<CommandResult, ProtocolError> {
        let files = self.message_files(&attachments)?;
        if text.trim().is_empty() && files.is_empty() {
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
            Some(_) if model.is_some() || effort.is_some() || in_run.is_some() => {
                return Err(ProtocolError::new(
                    ErrorCode::Invalid,
                    "Only the app's own conversations take a model or a run's thread",
                ));
            }
            Some(session_id) => {
                if !files.is_empty() {
                    return Err(ProtocolError::new(
                        ErrorCode::Invalid,
                        "Files go only to the app's own conversations for now; send them \
                         in a new conversation",
                    ));
                }
                return self
                    .reply_in_chat(request_id, session_id, text, placement)
                    .await;
            }
        };
        // The model and effort this message sets apply from its own turn,
        // so they are stored before it starts.
        if model.is_some() || effort.is_some() {
            self.host
                .set_session_model(&session_id, model, effort)
                .await
                .map_err(|why| ProtocolError::new(ErrorCode::Invalid, why))?;
        }
        self.runner.register_session(&session_id, APP_CHANNEL).await;
        // A new conversation started inside a project is placed there
        // before its first turn, so that turn already works in its folder.
        let project_id = match project_id.filter(|_| new_session) {
            Some(project_id) => match self
                .host
                .set_session_project(&session_id, Some(&project_id))
                .await
            {
                Ok(()) => Some(project_id),
                Err(why) => {
                    return Err(ProtocolError::new(ErrorCode::Invalid, why));
                }
            },
            None => None,
        };
        // The turn this message starts may move the sender's view, and only
        // the sender's, when it declared the capability.
        self.views.message_from(&session_id, view_connection);
        if new_session {
            self.publisher.publish(
                Some(session_id.clone()),
                Event::SessionCreated {
                    session: SessionSummary {
                        archived: false,
                        id: session_id.clone(),
                        channel: APP_CHANNEL.to_string(),
                        title: Some(if text.trim().is_empty() {
                            files
                                .first()
                                .map(|f| f.filename.clone())
                                .unwrap_or_default()
                        } else {
                            aidaemon_channels::presentation::word_preview(&text, 60)
                        }),
                        last_activity_ms: Some(bus::now_ms()),
                        writable: true,
                        project_id,
                        group: false,
                    },
                },
            );
        }
        let mut channel_ctx = aidaemon_channels::context::ChannelContext::private(APP_CHANNEL);
        channel_ctx.channel_id = Some(session_id.clone());
        // Typed: the thread the app showed it in. Which run it is, and
        // whether it is this conversation's, the agent's tool checks.
        channel_ctx.in_run = in_run.filter(|_| !new_session);
        let text = if files.is_empty() {
            text
        } else {
            aidaemon_channels::attachments::build_inbound_text(&text, &files)
        };
        let turn = Turn {
            session_id: session_id.clone(),
            text,
            attachments: files,
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
        tokio::spawn(async move { runner.submit_placed(surface, turn, placement).await });
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
        // Its ended runs, from the first message shown on: older ones
        // belong to older pages.
        let from = messages.first().map_or(0, |m| m.at_ms);
        let runs = match self.host.past_runs(&session_id).await {
            Ok(runs) => runs
                .into_iter()
                .filter(|run| run.ended_at_ms.unwrap_or(run.started_at_ms) >= from)
                .collect(),
            Err(error) => {
                tracing::warn!(%error, "Could not read a conversation's past runs");
                Vec::new()
            }
        };
        Ok(CommandResult::History {
            session_id,
            messages,
            more,
            runs,
        })
    }

    /// `/trustall on` from an app: Settings' change to Full access, through
    /// its check, answered as the command's reply.
    async fn trust_all_on(&self, caller: &Caller) -> Result<CommandResult, ProtocolError> {
        let change = SettingsChange::SetPermissionLevel {
            level: aidaemon_client_protocol::PermissionLevel::FullAccess,
        };
        let text = match self.change_settings(change, caller).await {
            Ok(CommandResult::SettingsChanged {
                outcome: SettingsOutcome::Done { message, .. },
            }) => message,
            Ok(CommandResult::SettingsChanged {
                outcome: SettingsOutcome::AwaitingApproval { other_device, .. },
            }) => if other_device {
                "Approve it on another of your devices: raising AIdaemon to Full access \
                 needs a second device."
            } else {
                "Confirm it on the card: raising AIdaemon to Full access asks first."
            }
            .to_string(),
            Ok(_) => "Nothing was changed.".to_string(),
            Err(error) => error.message,
        };
        Ok(CommandResult::Reply { text })
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
                live_turns: self.live_turns(),
                surfaces: self
                    .surface_desk()
                    .map(|desk| desk.list())
                    .unwrap_or_default(),
                shells: self
                    .shell_desk()
                    .map(|desk| desk.list())
                    .unwrap_or_default(),
                runs: self.runs().await,
            },
        )
    }

    /// The conversation and running-work number of run `run_id`, while it
    /// runs: what a stop of it takes.
    async fn running_run(&self, run_id: &str) -> Option<(String, u64)> {
        let sessions: Vec<String> = self
            .active
            .lock()
            .map(|active| active.iter().cloned().collect())
            .unwrap_or_default();
        for session_id in sessions {
            let list = stop::running_work(
                &self.turns.task_registry,
                self.turns.agent.as_ref(),
                &session_id,
            )
            .await;
            if let Some(entry) = list
                .entries
                .iter()
                .find(|entry| entry.run.as_deref() == Some(run_id))
            {
                return Some((session_id, entry.number));
            }
        }
        None
    }

    /// The helper runs of every session that ran a turn this boot, running
    /// or ended and still on record, oldest first.
    async fn runs(&self) -> Vec<aidaemon_client_protocol::RunSummary> {
        let sessions: Vec<String> = self
            .active
            .lock()
            .map(|active| active.iter().cloned().collect())
            .unwrap_or_default();
        let mut runs = Vec::new();
        for session_id in sessions {
            for run in self.turns.agent.chat_runs(&session_id).await.runs {
                runs.push(observer::run_summary(
                    &session_id,
                    &run.id,
                    &run.view,
                    observer::RunRecord {
                        activity: run.activity.as_deref(),
                        agent: run.agent.as_deref(),
                        group: run.group.as_deref(),
                        edited: &run.edited,
                        continues: run.continues.as_deref(),
                        started_at: run.started_at,
                        ended_at: run.ended_at,
                    },
                ));
            }
        }
        runs.sort_by_key(|run| run.started_at_ms);
        runs
    }

    /// Turns running now, oldest first, as far as each has got.
    fn live_turns(&self) -> Vec<aidaemon_client_protocol::LiveTurnState> {
        let mut turns: Vec<_> = self
            .live
            .lock()
            .map(|live| live.values().cloned().collect())
            .unwrap_or_default();
        turns.sort_by_key(|turn: &aidaemon_client_protocol::LiveTurnState| turn.started_at_ms);
        turns
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
