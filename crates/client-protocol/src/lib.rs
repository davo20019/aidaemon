//! # aidaemon-client-protocol
//!
//! The wire types of the AIDaemon Client Protocol v1, spoken between the
//! daemon's Client Gateway (`/client/v1`) and the owner's own clients: a
//! terminal UI, a web page, a desktop or phone app, a watch.
//!
//! Every frame is one JSON object with a `type` tag. Three families:
//!
//! - commands, client to daemon ([`ClientFrame::Command`]), each with a
//!   client-chosen `request_id`;
//! - replies, matched by that id: exactly one [`ServerFrame::Ack`] or
//!   [`ServerFrame::Error`] per command;
//! - events, daemon to client ([`ServerFrame::Event`]), ordered by a
//!   per-owner, gap-free `seq`. Ephemeral events (turn status) carry no
//!   `seq` and are never replayed.
//!
//! The same commands go over plain HTTP (`POST /client/v1/commands` takes a
//! [`ClientFrame::Command`] and answers with an ack or error frame) and the
//! same events long-poll (`GET /client/v1/events?since=<seq>&wait=<s>`
//! answers with an [`EventsPage`]).
//!
//! These types are the source of truth: [`schema`] generates the JSON Schema
//! checked in at `schemas/client-protocol/v1.schema.json`, from which
//! non-Rust clients generate theirs. Internal daemon types never go on the
//! wire directly; the gateway maps them onto these.
//!
//! Compatibility: within a major version, clients ignore unknown fields and
//! unknown event types (the Rust enums read them as `Unknown`). Adding an
//! optional field or a new event type is a minor bump; renaming, removing or
//! changing the meaning of a field is a major one.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub mod media;
pub mod participants;
pub mod shell;

/// The major version this crate speaks.
pub const PROTOCOL_MAJOR: u16 = 1;
/// The minor version this crate speaks.
pub const PROTOCOL_MINOR: u16 = 53;

/// A protocol version. Same major: compatible.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    /// The version this crate speaks.
    pub const CURRENT: ProtocolVersion = ProtocolVersion {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };

    /// Whether a peer speaking `self` can talk to one speaking `other`.
    pub fn compatible_with(self, other: ProtocolVersion) -> bool {
        self.major == other.major
    }
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// A frame from a client to the daemon.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientFrame {
    /// The first frame on a socket.
    Hello {
        version: ProtocolVersion,
        client: ClientInfo,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        capabilities: Vec<Capability>,
        /// The last `seq` the client applied; `None` for a client with no
        /// cache. Within the retention window the gateway replays every
        /// event after it, otherwise it sends a snapshot.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resume_from: Option<u64>,
        /// The access token, for clients that cannot set an
        /// `Authorization` header on a WebSocket (browsers).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token: Option<String>,
    },
    /// A command. Answered by exactly one ack or error with the same
    /// `request_id`; a retried id gets the original answer.
    Command {
        request_id: String,
        command: Command,
    },
    Ping,
}

/// A frame from the daemon to a client.
// Wire values live briefly between parse and use; boxing would only make
// them clumsier to build and match.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    /// The answer to `hello`.
    Welcome {
        version: ProtocolVersion,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        server_capabilities: Vec<Capability>,
        owner: OwnerInfo,
        /// The newest `seq` in the log when the socket opened.
        latest_seq: u64,
        /// First-run setup still to finish (protocol 1.5). `None`: the
        /// daemon is set up.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        setup: Option<SetupState>,
    },
    /// Everything a client needs when it cannot resume: drop the cache and
    /// start from this state. Events after `seq` follow.
    Snapshot {
        seq: u64,
        state: Snapshot,
    },
    Event {
        /// Per owner, gap-free; `None` for an ephemeral event.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seq: Option<u64>,
        at_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        event: Event,
    },
    Ack {
        request_id: String,
        result: CommandResult,
    },
    Error {
        /// `None` for an error that answers no command (a bad frame).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        error: ProtocolError,
    },
    Pong,
}

/// What a client is.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ClientInfo {
    pub kind: ClientKind,
    /// The client program's name ("aidaemon-tui").
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// The device's name as the owner knows it ("Pixel 9").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// The kind of client, so the daemon can tailor what it sends and the owner
/// can see which device acted.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClientKind {
    Tui,
    Web,
    Desktop,
    Mobile,
    /// A smartwatch: short actions, compact frames, a risk ceiling on what
    /// it may approve.
    Watch,
    /// A script or other program.
    Script,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// An optional feature, named instead of a version bump. A client lists the
/// ones it wants in `hello`; the server lists the ones it offers in
/// `welcome`, and only those it offers apply.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Short frames: an approval's summary without its detail or card, no
    /// `turn_delta` or `turn_status`. The default for a watch.
    Compact,
    /// Streaming `turn_delta` events.
    Deltas,
    /// `goal_updated` events.
    Goals,
    /// Files with messages: `upload_file` before `send_message`, and the
    /// files of messages and replies, each with a `url` (protocol 1.11).
    Attachments,
    /// Push notifications through a relay.
    Push,
    /// The agent may move this connection's view (`view_action` events):
    /// open a session, show an approval or a piece of work, focus a pane.
    /// Sent only to the connection whose message started the turn, never
    /// replayed, and never able to answer, pair, send or stop anything.
    ViewActions,
    /// Live surfaces (protocol 1.30): `snapshot.surfaces`,
    /// `surface_updated` and `surface_closed` events, and the surface
    /// commands. Watching and controlling a surface also needs
    /// [`Scope::Control`].
    Surfaces,
    /// Conversation shells (protocol 1.42): `snapshot.shells`,
    /// `shell_updated` and `shell_closed` events, and the shell commands.
    /// Opening one also needs [`Scope::Control`] and a device that may
    /// answer every approval.
    Shells,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// The daemon's owner, as the gateway knows them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct OwnerInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// What a client asks the daemon to do.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    /// Start a session (no `session_id`) or continue one.
    SendMessage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        text: String,
        /// Files uploaded beforehand; needs the `attachments` capability.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<AttachmentRef>,
        /// What to do with it when the session is already working
        /// (protocol 1.3). `None`: the agent reads it and decides.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mid_run: Option<MidRun>,
        /// The project a new session starts in (protocol 1.10); ignored
        /// when continuing one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<String>,
        /// The model the conversation answers with from this message on
        /// (protocol 1.16); absent: as it was. Only the app's own
        /// conversations take one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<ModelChoice>,
        /// How hard it thinks from this message on (protocol 1.16);
        /// absent: as it was.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effort: Option<Effort>,
        /// The run whose thread it was posted in (protocol 1.48): the owner
        /// follows up with that run's coding agent, which goes on in its
        /// own session ([`RunSummary::continues`]). Only the app's own
        /// conversations take one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        in_run: Option<String>,
    },
    /// Stop everything the session has running (`/cancel all`).
    StopTurn { session_id: String },
    ResolveApproval {
        approval_id: String,
        decision: ApprovalDecision,
    },
    /// Stop one entry of the running-work list.
    CancelTask { task_id: String },
    /// What one run did, as its record keeps it (protocol 1.46): a coding
    /// agent's steps, plan, permission decisions and usage. Answers with
    /// `run`.
    GetRun { run_id: String },
    ListSessions {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cursor: Option<String>,
        #[serde(default = "default_limit")]
        limit: u32,
    },
    GetHistory {
        session_id: String,
        /// Messages before this one (its id); `None` for the newest.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        before: Option<String>,
        #[serde(default = "default_limit")]
        limit: u32,
    },
    /// Start the session afresh (`/clear`), keeping its history on record.
    ClearSession { session_id: String },
    /// An existing slash command, until it has a typed command of its own.
    RunSlash {
        session_id: String,
        /// Without the slash ("tasks").
        command: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        args: String,
    },
    /// Running tasks, pending approvals and the last reply, in one call.
    GetGlance,
    /// The state a snapshot frame carries, on demand.
    GetSnapshot,
    /// How the daemon itself is doing: version, models, usage, schedules,
    /// background jobs, queues and health checks (protocol 1.2).
    GetSystem,
    /// Messages of every session whose text holds all of `query`'s words,
    /// newest first (protocol 1.3).
    SearchMessages {
        query: String,
        #[serde(default = "default_limit")]
        limit: u32,
    },
    /// The slash commands the owner can run, for a client's completion
    /// list (protocol 1.3).
    ListCommands,
    /// What the daemon remembers about the owner (protocol 1.3).
    ListMemories,
    /// Upload a file to send with a message (capability `attachments`,
    /// protocol 1.11): its bytes as standard base64. The reply's attachment
    /// goes in `send_message`'s `attachments`.
    UploadFile {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime: Option<String>,
        data: String,
        /// The picture the owner drew marks over to make this one: its
        /// attachment `id`. The agent is told the marks are the owner's and
        /// where the unmarked picture is (protocol 1.25).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        marked_from: Option<String>,
    },
    /// Forget one remembered fact, by its `id` from `list_memories`.
    ForgetMemory { memory_id: String },
    /// The daemon's settings: every value its config file holds (secrets
    /// only as set or not), the settings it knows that are unset, its bots
    /// and MCP servers, and what waits for a reload or restart
    /// (protocol 1.4).
    GetSettings,
    /// Change the daemon's settings (protocol 1.4). A security-sensitive
    /// change asks for approval first, and the reply says so; its result
    /// then arrives as a `settings_updated` event.
    ChangeSettings { change: SettingsChange },
    /// Where first-run setup stands (protocol 1.5).
    GetSetup,
    /// A step of first-run setup (protocol 1.5). A daemon still in setup
    /// answers only this, `get_setup` and pings.
    Setup { action: SetupAction },
    /// The uncommitted changes in the git repositories a session worked
    /// in (protocol 1.9), for a client's code view.
    GetChanges { session_id: String },
    /// The owner's projects (protocol 1.10).
    ListProjects,
    /// A new project (protocol 1.10): a name, and the folder its
    /// conversations work in (any folder; none for a project without one).
    CreateProject {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        /// Make `path` when it does not exist (protocol 1.24), after the
        /// owner confirmed it on a `folder_missing` error. Only folders
        /// under the daemon's home folder are made.
        #[serde(default, skip_serializing_if = "is_false")]
        create_folder: bool,
    },
    /// Rename a project or change its folder (protocol 1.10). `name` and
    /// `path` change only when given; an empty `path` removes the folder.
    UpdateProject {
        project_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        /// As `CreateProject`'s (protocol 1.24).
        #[serde(default, skip_serializing_if = "is_false")]
        create_folder: bool,
    },
    /// Delete a project (protocol 1.27): its conversations, goals and
    /// runs move to no project and its chat bots to none; its folder is
    /// left as it is. Answers with `projects`.
    DeleteProject { project_id: String },
    /// Put a session in a project, or take it out of every project
    /// (`project_id` absent) (protocol 1.10).
    SetSessionProject {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<String>,
    },
    /// Send a chat bot's direct-message chats to a project, or to none
    /// (`project_id` absent) (protocol 1.12). `channel` is a session's
    /// `channel` (`telegram:coding_bot`). Answers with `projects`.
    SetChannelProject {
        channel: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<String>,
    },
    /// Everything that runs on its own: scheduled work and reminders,
    /// watching automations and health checks (protocol 1.13). Answers with
    /// `automations`.
    ListAutomations,
    /// Pause, resume, cancel or run one automation now (protocol 1.13).
    /// Answers with `automations`: the list as it is after, and what
    /// happened.
    ChangeAutomation {
        automation_id: String,
        change: AutomationChange,
    },
    /// Change what an automation is called, when it runs or what its runs
    /// do (protocol 1.15); a field left out stays as it is. Only one whose
    /// summary carries `edit` takes it. Answers with `automations`.
    EditAutomation {
        automation_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// When it runs, in words ("every weekday at 9am").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        when: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        /// What its runs tell the owner (protocol 1.15).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        notify: Option<AutomationNotify>,
    },
    /// The providers and models a conversation can be set to answer with
    /// (protocol 1.16). Answers with `models`.
    ListModels,
    /// The model and effort a conversation answers with (protocol 1.16).
    /// Answers with `session_model`.
    GetSessionModel { session_id: String },
    /// Set the model and effort an app conversation answers with from its
    /// next turn (protocol 1.16); a field left out stays as it is. Answers
    /// with `session_model`.
    SetSessionModel {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<ModelChoice>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effort: Option<Effort>,
    },
    /// Name a conversation (protocol 1.17); `title` absent or blank: back
    /// to the name taken from what was said. Answers with `reply`; the
    /// session's `session_updated` follows.
    RenameSession {
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// Name one of a conversation's agents (protocol 1.48): the agent
    /// session whose thread is `thread` (`RunSummary::thread`, or its first
    /// run's id), "Designer". `None` or blank: back to what it was started
    /// for. Answers with `reply`; `agent_renamed` follows.
    RenameAgent {
        session_id: String,
        thread: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Archive a conversation, or bring it back (protocol 1.17). Answers
    /// with `reply`; `session_updated` follows.
    ArchiveSession { session_id: String, archived: bool },
    /// Undo the uncommitted changes to one file of a session's changes
    /// (protocol 1.17): a changed or deleted file goes back to its last
    /// commit, a new one is removed. `root` and `path` as `get_changes`
    /// gave them. Answers with `changes`, as they are after.
    RevertFile {
        session_id: String,
        root: String,
        path: String,
    },
    /// One folder of a project's files (protocol 1.32), read-only: what is
    /// in it, folders first. `path` is relative to the project's folder,
    /// `/`-separated; absent or empty: the folder itself. Files git ignores
    /// are left out unless `ignored`; `.git` never shows. Needs scope
    /// `chat`. Answers with `project_dir`.
    ListProjectFiles {
        project_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        ignored: bool,
    },
    /// One file of a project (protocol 1.32), read-only; `path` as
    /// `list_project_files`'s. A file that holds secrets, a binary one or
    /// one too large comes back without its text, and says why. Needs
    /// scope `chat`. Answers with `project_file`.
    ReadProjectFile { project_id: String, path: String },
    /// The messages waiting for a session's running work to end (protocol
    /// 1.17). Answers with `queued`.
    ListQueued { session_id: String },
    /// Change a waiting message's text, or take it out (`text` absent)
    /// (protocol 1.17). With `now` (protocol 1.41), a message sent to wait
    /// for its own turn (`after`) is given to the running work instead, as
    /// if it had been sent to be read now; `text` is then ignored and
    /// nothing is taken out. Answers with `queued`, as it is after.
    EditQueued {
        session_id: String,
        message_id: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        now: bool,
    },
    /// Every live surface now (protocol 1.30, capability `surfaces`).
    /// Answers with `surfaces`.
    ListSurfaces,
    /// Watch a surface, and with `mode: control` take control of it
    /// (protocol 1.30; needs scope `control`). Answers with
    /// `surface_opened`: where to connect for frames.
    OpenSurface {
        surface_id: String,
        mode: SurfaceMode,
        #[serde(default)]
        accept: StreamAccept,
    },
    /// Stop watching (protocol 1.30). Answers with `reply`.
    CloseSurface { live_id: String },
    /// The conversation's shell, started in its folder when none runs
    /// (protocol 1.42, capability `shells`; scope `control` and a device
    /// that may answer every approval). `cols` and `rows` are the
    /// terminal's size in cells. Answers with `shell_opened`: where to
    /// connect for its output.
    OpenShell {
        session_id: String,
        cols: u16,
        rows: u16,
    },
    /// End a conversation's shell (protocol 1.42). Answers with `reply`;
    /// `shell_closed` follows.
    CloseShell { shell_id: String },
    /// Take control of a surface (protocol 1.30; scope `control`).
    /// Answers with `surface_control`.
    TakeControl { surface_id: String },
    /// Give a surface back to the agent (protocol 1.30). Answers with
    /// `surface_control`.
    ReleaseControl { surface_id: String },
    /// Point the agent at what is at `(x, y)` (fractions of the frame) on a
    /// surface, with an optional `note` (protocol 1.31; scopes `chat` and
    /// `control`). The daemon says what is there and sends it, as the
    /// owner's message, to the surface's conversation. Answers with
    /// `message_accepted`.
    PointAt {
        surface_id: String,
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    /// Offer this device's own screen as a live surface (protocol 1.31;
    /// scope `control`): another computer the owner paired shares its
    /// display. Answers with `screen_shared`: where to connect to send
    /// frames and take input.
    ShareScreen {
        /// What the owner calls this screen ("Studio Mac").
        name: String,
        width: u32,
        height: u32,
    },
    /// The paired devices (protocol 1.31). Answers with `devices`.
    ListDevices,
    /// What a paired device may do (protocol 1.31): only from the daemon's
    /// own computer (its local token), so a device never widens its own
    /// reach. Answers with `devices`, as they are after.
    SetDeviceScopes {
        device_id: String,
        scopes: Vec<Scope>,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// How a client opens a surface (protocol 1.30).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceMode {
    /// Watch; input is off.
    #[default]
    View,
    /// Watch and take control.
    Control,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What a viewer can take (protocol 1.30). Every field is a ceiling; the
/// daemon may send less.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct StreamAccept {
    /// Codecs the client can draw, best first (`jpeg`, `png`). Empty: `jpeg`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codecs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_w: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_h: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fps: Option<u32>,
    /// 10 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<u8>,
    /// The room the viewer has for the picture, in its points (a phone's
    /// screen less the controls). While this device holds control, a
    /// surface that can lay itself out (a browser page) takes this size,
    /// so the picture fills the screen; it goes back to its own size when
    /// control goes back. Both or neither.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_w: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_h: Option<u32>,
}

/// A screen the agent works on that the owner can watch and control
/// (protocol 1.30).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SurfaceInfo {
    pub id: String,
    pub kind: SurfaceKind,
    /// The conversation it belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// From the page; show it as data, never as the app's own words.
    pub title: String,
    /// The host only, never a path or a query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    pub width: u32,
    pub height: u32,
    pub controller: SurfaceController,
    /// The agent asked the owner to do something here and waits.
    #[serde(default)]
    pub handoff_pending: bool,
    /// Whether it takes pointer, key and text input.
    #[serde(default)]
    pub interactive: bool,
    /// A popup's parent surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opener: Option<String>,
}

/// What kind of screen a surface is (protocol 1.30).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    /// A tab of the agent's browser.
    BrowserTab,
    /// A whole display of the daemon's computer.
    Desktop,
    Window,
    Node,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Who acts on a surface now (protocol 1.30).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SurfaceController {
    Agent,
    /// The owner, from one device.
    Owner {
        device_id: String,
        /// The device's name, when the gateway knows it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device: Option<String>,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Where a client connects for a live session's frames (protocol 1.30).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SurfaceTransport {
    pub kind: TransportKind,
    /// A path on the gateway's own origin.
    pub url: String,
    /// Sent as the first message on the socket, never in the URL. Single
    /// use, short lived, bound to this device and live session.
    pub ticket: String,
    pub codecs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransportKind {
    /// A WebSocket of binary frames (see [`media`]).
    WsFrames,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A conversation's shell (protocol 1.42): a terminal on the daemon's
/// computer, started in the conversation's folder, that the owner and the
/// agent share.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ShellInfo {
    pub id: String,
    pub session_id: String,
    /// The folder it started in.
    pub folder: String,
    /// What the program called its window; show it as data.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    pub cols: u16,
    pub rows: u16,
    /// Absent while it runs; its exit code once it ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<u32>,
    pub started_at_ms: u64,
}

/// Where a client connects for a shell's output (protocol 1.42); see
/// [`shell`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ShellTransport {
    /// A path on the gateway's own origin.
    pub url: String,
    /// Sent as the first message on the socket, never in the URL. Single
    /// use, short lived, bound to this device and shell.
    pub ticket: String,
}

/// The model a conversation answers with (protocol 1.16).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ModelChoice {
    /// The daemon picks, turn by turn, as it does for every chat.
    #[default]
    Auto,
    /// This model of this provider (ids from `list_models`), every turn.
    Pinned { provider: String, model: String },
    /// This coding agent (an id from `list_models`' `agents`), every turn
    /// (protocol 1.49): each message goes straight to it in the
    /// conversation's project folder, with no model call first. Only a
    /// conversation in a project with a folder takes one.
    Agent { agent: String },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// How hard the model thinks before it answers (protocol 1.16).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// The model's own setting, raised by the daemon when it needs more.
    #[default]
    Auto,
    Low,
    Medium,
    High,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What a conversation can be set to answer with (protocol 1.16).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ModelOptions {
    /// The connected providers, the main one first, each listing only the
    /// models no earlier provider lists.
    pub providers: Vec<ModelProvider>,
    /// The coding agents installed and ready on the daemon's computer, the
    /// one it prefers first (protocol 1.49); an `agent` choice names one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agents: Vec<AgentOption>,
}

/// A coding agent a code conversation can be set to (protocol 1.49).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AgentOption {
    /// What an `agent` choice names it by.
    pub id: String,
    /// In plain words ("Claude Code", "Codex").
    pub name: String,
    /// The mark a client shows for it (protocol 1.53): the id of its
    /// maker's model provider ("anthropic", "openai", "gemini"). `None`: a
    /// generic agent mark.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark: Option<String>,
}

/// A connected provider, as a model picker lists it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ModelProvider {
    /// What a `pinned` choice names it by.
    pub id: String,
    /// In plain words ("Anthropic", "ChatGPT").
    pub name: String,
    pub billing: Billing,
    pub models: Vec<ModelOption>,
}

/// A model a conversation can be set to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ModelOption {
    pub id: String,
    /// The efforts it takes besides `auto`; empty: it takes none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub efforts: Vec<Effort>,
}

/// The model and effort a conversation answers with (protocol 1.16).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SessionModel {
    pub session_id: String,
    pub model: ModelChoice,
    pub effort: Effort,
}

/// Where a message sent while its session works goes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MidRun {
    /// Into the work running now: the agent takes it into account.
    Steer,
    /// After the work running now, as a turn of its own.
    Queue,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_limit() -> u32 {
    50
}

/// A file with a message: one a client uploaded before sending it, or one
/// the daemon sent (a picture, a document).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AttachmentRef {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Where to fetch it, on the gateway's own origin
    /// (`/client/v1/files/...`). It needs no token: the path itself is the
    /// permission, so treat it like the file (protocol 1.11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// A picture's size in pixels, so a client lays it out before it
    /// loads (protocol 1.12).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// The picture the owner marked up to make this one, by its `id`
    /// (`upload_file`'s `marked_from`; protocol 1.25).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marked_from: Option<String>,
}

impl AttachmentRef {
    /// Whether it is a picture a client can show.
    pub fn is_image(&self) -> bool {
        self.mime
            .as_deref()
            .is_some_and(|m| m.starts_with("image/"))
    }
}

/// The answer to a command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommandResult {
    /// `send_message`: the message was taken; what happens next arrives as
    /// events of `session_id`.
    MessageAccepted {
        session_id: String,
        /// The message started a session.
        #[serde(default)]
        new_session: bool,
    },
    /// `stop_turn` and `cancel_task`: what was stopped, in plain words, as
    /// read back afterwards.
    Stopped {
        report: String,
    },
    /// `resolve_approval`: this answer settled the approval.
    ApprovalResolved {
        approval_id: String,
        outcome: ApprovalOutcome,
    },
    Sessions {
        sessions: Vec<SessionSummary>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        next_cursor: Option<String>,
    },
    History {
        session_id: String,
        messages: Vec<HistoryMessage>,
        /// More, older messages exist.
        #[serde(default)]
        more: bool,
        /// The conversation's ended runs as the daemon kept them when each
        /// ended (protocol 1.46): which agents ran and how each ended, past
        /// a restart. A run the daemon still holds comes in its snapshot
        /// and events instead, which win.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        runs: Vec<RunSummary>,
        /// The names the owner gave its agents (protocol 1.48), by thread
        /// (`rename_agent`).
        #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
        agent_names: std::collections::BTreeMap<String, String>,
    },
    /// `clear_session` and `run_slash`: the command's reply.
    Reply {
        text: String,
    },
    Glance {
        glance: Glance,
    },
    Snapshot {
        seq: u64,
        state: Snapshot,
    },
    System {
        system: SystemStatus,
    },
    /// `search_messages`.
    SearchResults {
        hits: Vec<SearchHit>,
    },
    /// `list_commands`.
    Commands {
        commands: Vec<SlashCommand>,
    },
    /// `list_memories`.
    Memories {
        memories: Vec<MemoryItem>,
    },
    /// `forget_memory`: the fact is no longer recalled.
    Forgotten {
        memory_id: String,
    },
    /// `upload_file`: the file is kept; send it with `send_message`.
    Uploaded {
        attachment: AttachmentRef,
    },
    /// `get_settings`.
    Settings {
        settings: Settings,
    },
    /// `change_settings`: done, or waiting for an approval.
    SettingsChanged {
        outcome: SettingsOutcome,
    },
    /// `get_setup` and `setup`: where setup stands now.
    Setup {
        setup: SetupState,
    },
    /// `get_changes`.
    Changes {
        changes: SessionChanges,
    },
    /// `list_project_files` (protocol 1.32).
    ProjectDir {
        dir: ProjectDir,
    },
    /// `read_project_file` (protocol 1.32).
    ProjectFile {
        file: ProjectFile,
    },
    /// `list_projects` and `delete_project`.
    Projects {
        projects: Vec<ProjectSummary>,
    },
    /// `get_run`.
    Run {
        run: RunDetail,
    },
    /// `list_automations` and `change_automation`: every automation, next
    /// to run first; after a change, what happened (`note`) and whether it
    /// changed.
    Automations {
        automations: Vec<AutomationSummary>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        changed: bool,
    },
    /// `create_project` and `update_project`: the project as it is now.
    Project {
        project: ProjectSummary,
    },
    /// `list_models` (protocol 1.16).
    Models {
        options: ModelOptions,
    },
    /// `get_session_model` and `set_session_model` (protocol 1.16).
    SessionModel {
        session_model: SessionModel,
    },
    /// `list_queued` and `edit_queued`: the session's waiting messages,
    /// oldest first (protocol 1.17).
    Queued {
        session_id: String,
        messages: Vec<QueuedMessage>,
    },
    /// `list_surfaces` (protocol 1.30).
    Surfaces {
        surfaces: Vec<SurfaceInfo>,
    },
    /// `open_surface` (protocol 1.30): connect to one of `transports`.
    SurfaceOpened {
        live_id: String,
        surface: SurfaceInfo,
        transports: Vec<SurfaceTransport>,
    },
    /// `take_control` and `release_control` (protocol 1.30): the surface
    /// as it is after.
    SurfaceControl {
        surface: SurfaceInfo,
    },
    /// `open_shell` (protocol 1.42): connect to `transport`.
    ShellOpened {
        shell: ShellInfo,
        transport: ShellTransport,
    },
    /// `share_screen` (protocol 1.31): connect to `url` (a path on the
    /// gateway) and send `ticket` first.
    ScreenShared {
        surface_id: String,
        url: String,
        ticket: String,
    },
    /// `list_devices` and `set_device_scopes` (protocol 1.31).
    Devices {
        devices: Vec<DeviceInfo>,
        /// This client may change what devices may do (it is on the
        /// daemon's own computer).
        #[serde(default)]
        can_change: bool,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A message that matched a search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SearchHit {
    pub session_id: String,
    pub message_id: String,
    /// `user` or `assistant`.
    pub role: String,
    /// The text around the first match, plain.
    pub snippet: String,
    pub at_ms: u64,
}

/// A slash command, as a completion list shows it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SlashCommand {
    /// Without the slash ("tasks").
    pub name: String,
    pub description: String,
}

/// One remembered fact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct MemoryItem {
    pub id: String,
    pub category: String,
    pub key: String,
    pub value: String,
    /// Where it came from, in a word or two.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    pub updated_ms: u64,
}

// ---------------------------------------------------------------------------
// Code changes (protocol 1.9)
// ---------------------------------------------------------------------------

/// What a session changed in code: the git repositories its tools worked
/// in, each with its uncommitted changes. Empty when it worked in none.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SessionChanges {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repos: Vec<RepoChanges>,
    /// The last command the session ran after its last file change (a test
    /// or build, usually), and how it ended (protocol 1.12). `None` when no
    /// command ran since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_check: Option<CheckRun>,
}

/// Something that runs on its own (protocol 1.13).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AutomationSummary {
    pub id: String,
    pub kind: AutomationKind,
    /// The name the owner gave it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// What it does, in the owner's words.
    pub what: String,
    /// When it runs ("Every weekday at 9:00 AM ET").
    pub how_often: String,
    /// `active`, `paused`, `failed`, `awaiting_input`, `pending_confirmation`...
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_run_ms: Option<u64>,
    /// Its latest runs, newest first (a few): what it did, not only what
    /// it will do (protocol 1.14).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_runs: Vec<AutomationRun>,
    /// The rest of its card, one `"<label>: <text>"` line each: what it may
    /// do, when it tells you, when it ends, its workspace and budget
    /// (protocol 1.15).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub details: Vec<String>,
    /// What can be changed here, as it is now; absent when it changes only
    /// by asking in a conversation (protocol 1.15).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<AutomationEdit>,
    /// The conversation its runs work in (protocol 1.31): a run waiting for
    /// the owner on a live surface shows there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// An automation's changeable fields as they are now (protocol 1.15).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AutomationEdit {
    /// When it runs, in the words it was set with ("every weekday at 9am").
    pub when: String,
    /// What its runs do; absent for a reminder, whose message is fixed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// What its runs tell the owner; absent for a reminder, which always
    /// does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notify: Option<AutomationNotify>,
}

/// What a scheduled automation's runs tell the owner (protocol 1.15).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AutomationNotify {
    /// Every run's result.
    EveryRun,
    /// Only a run that found something new, or a problem.
    Changes,
    /// Only a run that failed or needs the owner.
    Problems,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// One run of an automation (protocol 1.14).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AutomationRun {
    pub started_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_ms: Option<u64>,
    pub outcome: RunOutcome,
    /// What it found or did, or why it stopped, in a line or two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// How a run ended, or that it has not yet.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    /// Waiting to start, or running now.
    Running,
    Done,
    Failed,
    /// It stopped on something it needs (an answer, access, a fix).
    Blocked,
    Cancelled,
    /// It was due but never ran: the daemon was not running then, and it
    /// was too late to catch up when it came back (protocol 1.28). Not a
    /// failure. An older client reads it as `Unknown`.
    Missed,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AutomationKind {
    /// Runs an instruction or a reminder at set times.
    Scheduled,
    /// Works toward an objective within limits the owner approved.
    Watching,
    /// Checks a service, port, command or file on a schedule.
    Probe,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AutomationChange {
    Pause,
    Resume,
    Cancel,
    /// Scheduled work only.
    RunNow,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A command a session ran to check its change (protocol 1.12).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct CheckRun {
    pub command: String,
    /// It exited with 0.
    pub passed: bool,
    /// Its exit code when it failed with one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub at_ms: u64,
}

/// One repository's uncommitted changes against `HEAD`, the owner's and
/// other sessions' included: a working tree has no record of who made
/// them. `by_session` marks the files this session's tools wrote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RepoChanges {
    /// The repository's top directory, as an absolute path.
    pub root: String,
    /// The checked-out branch; `None` on a detached `HEAD`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileChange>,
    /// The unified diff of `files`, untracked files included.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub patch: String,
    /// `patch` was cut short to stay small enough to send.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// One changed file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct FileChange {
    /// Relative to the repository's root, `/`-separated.
    pub path: String,
    pub status: FileStatus,
    /// Lines added and removed; 0 and 0 for a binary file.
    pub added: u32,
    pub removed: u32,
    /// This session's file tools wrote it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub by_session: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
    /// New, and not yet added to git.
    Untracked,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

// ---------------------------------------------------------------------------
// Project files (protocol 1.32)
// ---------------------------------------------------------------------------

/// One folder of a project's files.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProjectDir {
    pub project_id: String,
    /// Relative to the project's folder, `/`-separated; empty for the
    /// folder itself.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub path: String,
    /// Folders first, then files, each by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<ProjectEntry>,
    /// The folder holds more than were sent.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// The checked-out branch when the folder is in a git repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
}

/// A file or folder in a [`ProjectDir`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProjectEntry {
    pub name: String,
    pub kind: EntryKind,
    /// A file's size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_ms: Option<u64>,
    /// How a file differs from git's `HEAD`; `None` when it does not, or
    /// the folder is not in a repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<FileStatus>,
    /// A folder holding uncommitted changes.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub changed: bool,
    /// Git ignores it (sent only when asked for).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ignored: bool,
    /// It holds secrets: its text is never sent.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub secret: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Folder,
    File,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// One file of a project, as it is now.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProjectFile {
    pub project_id: String,
    pub path: String,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_ms: Option<u64>,
    /// A hash of the whole file's bytes (`sha256:<hex>`): what a later
    /// change to the file names as the version it was made against, so a
    /// file changed meanwhile is not overwritten.
    pub version: String,
    /// Its text (UTF-8); `None` when `withheld` says why not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// `text` is only the file's start: the file is longer than is sent.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub withheld: Option<Withheld>,
    /// How it differs from git's `HEAD`, as in [`ProjectEntry::status`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<FileStatus>,
    /// Where to fetch the file itself, for one that is not text but can be
    /// shown (a picture, a PDF): a signed link, as a message's files have.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<AttachmentRef>,
}

/// Why a [`ProjectFile`]'s text is not sent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Withheld {
    /// Keys, credentials, env files: never shown.
    Secret,
    /// Not text.
    Binary,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

// ---------------------------------------------------------------------------
// Settings (protocol 1.4)
// ---------------------------------------------------------------------------

/// A secret's text: sent once, to be stored, and never shown or logged.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(transparent)]
pub struct SecretText(pub String);

impl std::fmt::Debug for SecretText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretText(***)")
    }
}

/// The daemon's settings, as the Settings page shows them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Settings {
    /// The config file the settings live in.
    pub config_path: String,
    /// The settings, by config table.
    pub groups: Vec<SettingGroup>,
    /// Bots connected at runtime (stored in the database, not the file).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bots: Vec<BotInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerInfo>,
    /// Saved changes that take effect on a reload.
    #[serde(default)]
    pub reload_pending: bool,
    /// Saved changes that take effect on a restart.
    #[serde(default)]
    pub restart_pending: bool,
    /// What is scheduled or last happened to the database or the gateway's
    /// address, in plain words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notices: Vec<String>,
    /// The database file.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub database_path: String,
    /// Whether the database is encrypted (its key can be rotated).
    #[serde(default)]
    pub database_encrypted: bool,
    /// The providers a failover entry can name (protocol 1.6), so the
    /// client offers a list rather than a preset id typed by hand.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub providers: Vec<ProviderOption>,
    /// The providers the daemon uses, main first, then each failover in
    /// the order it is tried (protocol 1.8).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub connected: Vec<ConnectedProvider>,
    /// A ChatGPT subscription sign-in is stored on the daemon's machine.
    #[serde(default)]
    pub chatgpt_signed_in: bool,
    /// A ChatGPT sign-in started from Settings, waiting for the owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sign_in: Option<SignIn>,
    /// The last key checked from Settings, with its account's models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked: Option<CheckedKey>,
    /// The coding agents the daemon can hand work to, each with how much
    /// it may do on its own (protocol 1.36).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coding_agents: Vec<CodingAgent>,
    /// AIdaemon's own permission level: the most it, its sub-agents and
    /// every coding agent may do (protocol 1.43). `None`: a daemon from
    /// before levels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_level: Option<PermissionLevel>,
}

/// AIdaemon's permission level (protocol 1.43).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PermissionLevel {
    /// Only tools that read; anything else is refused.
    ReadOnly,
    /// Every tool, with risky terminal commands asked first.
    Standard,
    /// Every terminal command runs unasked (the owner's `/trustall`).
    FullAccess,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A coding agent the daemon can hand work to (Claude Code, Codex, ...)
/// and its permission mode (protocol 1.36).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct CodingAgent {
    /// Its name in the daemon (`claude`, `codex`, or one the owner added).
    pub name: String,
    /// The program it launches, in plain words ("Claude Code", "Codex").
    pub program: String,
    /// Its mode; `None`: it may not run delegated work until the owner
    /// picks one of `modes`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<AgentMode>,
    /// The modes it can run in, safest first: checking needs an agent
    /// whose approval requests the daemon can answer.
    pub modes: Vec<AgentMode>,
    /// Why some modes are not offered, in plain words; empty when all are.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// Whether it can do work now, as it last answered (protocol 1.38).
    #[serde(default)]
    pub readiness: AgentReadiness,
    /// For one that is not usable: why, in its own words when it gave them,
    /// and how the owner fixes it (protocol 1.38).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub readiness_reason: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub readiness_fix: String,
    /// The owner chose "same as AIdaemon": it follows the level
    /// (protocol 1.43).
    #[serde(default)]
    pub same_as_aidaemon: bool,
    /// The modes the owner may pick for it under AIdaemon's level, safest
    /// first: never more than AIdaemon itself (protocol 1.43).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub within_level: Vec<AgentMode>,
    /// The owner's pick is above the level, so it runs as `mode`, the most
    /// the level allows (protocol 1.43).
    #[serde(default)]
    pub capped: bool,
    /// Whose account it works with, in plain words: its own sign-in,
    /// never AIdaemon's providers (protocol 1.47).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub account: String,
    /// It can be given a model to work with (protocol 1.47).
    #[serde(default)]
    pub takes_model: bool,
    /// The model the owner chose for it; `None`: its own default
    /// (protocol 1.47).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The model it uses when it is given none, as it said when last
    /// asked; `None`: it did not say (protocol 1.47).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// The models it said it offers, in its order; empty when it lists
    /// none (any id it knows may still be typed) (protocol 1.47).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<AgentModelOption>,
}

/// A model a coding agent offers (protocol 1.47).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AgentModelOption {
    /// What selects it.
    pub id: String,
    /// Its name, as the agent gives it.
    pub name: String,
}

/// Whether a coding agent can do work now (protocol 1.38).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentReadiness {
    /// It answered that it is ready.
    Ready,
    /// It answered that it is not (not signed in, not served for this
    /// account): the daemon does not hand it work.
    NotUsable,
    /// It has not been asked, or could not say.
    #[default]
    Unchecked,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// How much a coding agent may do on its own (protocol 1.36).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// Free inside the project folder (its sandbox); anything beyond it
    /// is asked first and checked by the daemon's rules, and the owner is
    /// asked only what those do not cover.
    Checked,
    /// On the machine itself, with every command and file change (but a
    /// few read-only commands) asked first and checked the same way.
    EveryCommand,
    /// On the machine itself, nothing asked: the daemon's rules do not
    /// see what it does.
    FullAccess,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A provider the daemon uses: the main one or a failover.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ConnectedProvider {
    /// `None`: the main provider; else failover `index` (from 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failover: Option<u32>,
    /// The catalog preset it matches; empty when none does.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub preset: String,
    /// In plain words ("Anthropic", "OpenAI-compatible · example.com").
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub base_url: String,
    /// Its default model first, then the others it uses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// The model it uses for the hardest work, the owner's choice or the
    /// catalog's when they named none (protocol 1.20). Empty when unknown.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub hard_model: String,
    /// The model it uses for quick jobs, chosen the same way (protocol 1.20).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub quick_model: String,
    /// The models its provider offers, from the daemon's catalog, for the
    /// client to offer as choices besides `models` (protocol 1.21). Empty
    /// when the catalog cannot know them (an OpenAI-compatible server,
    /// whose models depend on its address) or the daemon is older.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub offered_models: Vec<String>,
    pub credential: CredentialState,
    /// The setting its key is stored under (`provider.api_key`,
    /// `provider.fallbacks[0].api_key`); empty for a sign-in.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub key_setting: String,
    /// What it has used and has left, when its provider reports that
    /// (protocol 1.37): a ChatGPT plan's limits, an OpenRouter balance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ProviderUsage>,
}

/// What a provider reports of its account's use (protocol 1.37).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProviderUsage {
    /// The plan, as the provider names it ("plus", "pro"); empty when unsaid.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub plan: String,
    /// Rolling limits, shortest window first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub windows: Vec<UsageWindow>,
    /// Money spent and left, for an account billed from a balance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balance: Option<UsageBalance>,
    /// When the daemon read it (ms since the epoch).
    #[serde(default)]
    pub read_at_ms: u64,
}

/// One rolling usage limit (protocol 1.37).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct UsageWindow {
    /// How long the window is, in minutes (300: five hours; 10080: a week).
    pub minutes: u32,
    /// How much of it is used, 0 to 100.
    pub used_percent: u8,
    /// When it starts over (ms since the epoch), when the provider says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resets_at_ms: Option<u64>,
}

/// Money spent and left, in US cents (protocol 1.37).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct UsageBalance {
    /// Spent so far.
    pub used_cents: u64,
    /// The most that may be spent; `None`: no cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit_cents: Option<u64>,
    /// Left to spend; `None`: no cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining_cents: Option<u64>,
}

/// How a connected provider is signed in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CredentialState {
    /// Its key is in the keychain.
    KeyStored,
    /// Its key is written in config.toml.
    KeyInFile,
    /// It needs a key and has none.
    KeyMissing,
    /// The ChatGPT subscription is signed in.
    SignedIn,
    /// The ChatGPT subscription is not signed in.
    SignedOut,
    /// It needs no key.
    NotNeeded,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// The settings of one config table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SettingGroup {
    /// The table's name in the file (`provider`, `memory`, ...).
    pub name: String,
    pub settings: Vec<Setting>,
}

/// One setting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Setting {
    /// Its dotted path, a list entry as `a[2]` (`provider.models.default`).
    pub key: String,
    /// Its value as TOML (`"gpt-x"`, `true`, `[1, 2]`); `None` when it is
    /// not set (the default applies) or is a secret.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    pub kind: SettingKind,
    /// For a secret: whether it is set.
    #[serde(default)]
    pub secret_set: bool,
    /// What it does, from the example config.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The example config's value, as TOML.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
    pub applies: Applies,
    /// Changing it asks for approval.
    #[serde(default)]
    pub guarded: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SettingKind {
    Text,
    Number,
    Bool,
    List,
    /// A table or a list of tables, edited as TOML.
    Table,
    /// Stored in the keychain; set with `SettingsChange::Secret`.
    Secret,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// When a saved change takes effect.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Applies {
    /// Already in effect.
    Now,
    /// On `apply` with `reload`.
    Reload,
    /// On `apply` with `restart`.
    Restart,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A bot connected at runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct BotInfo {
    pub id: i64,
    /// `telegram`, `discord` or `slack`.
    pub platform: String,
    /// Who may talk to it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_user_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub created_at: String,
}

/// An MCP server the daemon runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct McpServerInfo {
    pub name: String,
    /// The command line it runs.
    pub command: String,
    pub enabled: bool,
    /// The tools it offers while running.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// From the config file (changed as settings) rather than added at
    /// runtime.
    #[serde(default)]
    pub from_config: bool,
}

/// One change to the daemon's settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SettingsChange {
    /// Set `key` to `value` (TOML), or remove it (`None`) so its default
    /// applies.
    Set {
        key: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<String>,
    },
    /// Store a secret in the keychain and point `key` at it.
    Secret {
        key: String,
        value: SecretText,
    },
    /// Make a provider preset the primary provider. `key` (protocol 1.8):
    /// its API key, stored in the main provider's keychain slot first.
    SwitchProvider {
        provider: ProviderChoice,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key: Option<SecretText>,
    },
    /// Add a failover provider after the others. `key` (protocol 1.8): its
    /// API key, stored in the new failover's keychain slot first.
    AddFailover {
        provider: ProviderChoice,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key: Option<SecretText>,
    },
    /// Check a provider's key (nothing is saved) and read the models its
    /// account offers: the settings' `checked` (protocol 1.8).
    CheckKey {
        provider: ProviderChoice,
        key: SecretText,
    },
    /// Start a ChatGPT subscription sign-in: the settings' `sign_in`.
    /// `paste`: the browser is on another device (protocol 1.8).
    SignIn {
        #[serde(default)]
        paste: bool,
    },
    /// The address the sign-in's browser ended on (protocol 1.8).
    FinishSignIn {
        redirect: String,
    },
    /// Drop a sign-in under way (protocol 1.8).
    CancelSignIn,
    /// Remove failover provider `index` (from 0).
    RemoveFailover {
        index: u32,
        /// The `name` the client showed at `index` (protocol 1.23): the
        /// daemon refuses when another provider is there now. Empty: no
        /// check (an older client).
        #[serde(default, skip_serializing_if = "String::is_empty")]
        expect: String,
    },
    /// Move failover `index` to place `to` (both from 0), its keys with it:
    /// failovers are tried in order (protocol 1.22).
    MoveFailover {
        index: u32,
        to: u32,
        /// As `RemoveFailover`'s `expect` (protocol 1.23).
        #[serde(default, skip_serializing_if = "String::is_empty")]
        expect: String,
    },
    /// Put back the config file as it was before the last change.
    RestoreBackup,
    /// Remove a bot connected at runtime.
    RemoveBot {
        bot_id: i64,
    },
    Mcp {
        name: String,
        action: McpAction,
    },
    /// Put saved changes into effect.
    Apply {
        how: ApplyHow,
    },
    /// Database upkeep, done at the next restart.
    Database {
        action: DatabaseAction,
    },
    /// Delete every sign-in the agent's own browser profile kept (protocol
    /// 1.30). Asks first; open tabs close.
    ForgetSignIns,
    /// Set coding agent `agent`'s permission mode (protocol 1.36). Full
    /// access asks first.
    SetAgentMode {
        agent: String,
        mode: AgentMode,
    },
    /// Ask every coding agent again whether it can do work now (protocol
    /// 1.38).
    CheckAgents,
    /// Set AIdaemon's permission level (protocol 1.43). A wider level asks
    /// first.
    SetPermissionLevel {
        level: PermissionLevel,
    },
    /// Have coding agent `agent` follow AIdaemon's level (protocol 1.43).
    AgentSameAsAidaemon {
        agent: String,
    },
    /// Set the model coding agent `agent` works with; `None`: its own
    /// default (protocol 1.47).
    SetAgentModel {
        agent: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A provider, by preset (`openai`, `anthropic`, `openrouter`, ...).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProviderChoice {
    pub preset: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub base_url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub primary_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub fast_model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub smart_model: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpAction {
    Enable,
    Disable,
    Restart,
    Remove,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApplyHow {
    /// Reload the config in place (providers and models).
    Reload,
    /// Restart the daemon. Clients reconnect.
    Restart,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatabaseAction {
    /// Move the database file to `path` (absolute, or relative to the
    /// config file).
    MoveTo { path: String },
    /// Re-encrypt the database under a new key.
    RotateKey,
    /// Drop what is scheduled.
    Cancel,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What a `change_settings` did.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SettingsOutcome {
    /// Done: `message` says what changed.
    Done {
        message: String,
        /// When a saved change takes effect.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        applies: Option<Applies>,
    },
    /// Waiting for an approval (shown like any other); the result arrives
    /// as a `settings_updated` event.
    AwaitingApproval {
        approval_id: String,
        /// Only another device (or the owner at the computer) may approve.
        #[serde(default)]
        other_device: bool,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

// ---------------------------------------------------------------------------
// First-run setup (protocol 1.5)
// ---------------------------------------------------------------------------

/// Where first-run setup stands.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SetupState {
    pub step: SetupStep,
    /// Models already on this machine (a sign-in, a key in the
    /// environment), each tried once.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub found: Vec<FoundModel>,
    /// Every provider the owner may pick, subscription sign-in first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub providers: Vec<ProviderOption>,
    /// The model that passed its test, in plain words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// A subscription sign-in under way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sign_in: Option<SignIn>,
    /// A key the provider accepted, and the models its account offers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked: Option<CheckedKey>,
    /// The chats this daemon can be reached on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ChannelOption>,
    /// A chat connected in this setup, waiting for (or past) its first
    /// message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<ChannelWait>,
    /// What the owner should know (where secrets were stored, why a check
    /// failed), in plain words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notices: Vec<String>,
    /// The daemon runs on a machine of its own (a server), not this
    /// computer: what it says about sleep and sign-in differs.
    #[serde(default)]
    pub on_server: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SetupStep {
    /// Connect a model (required).
    #[default]
    Model,
    /// The model works and the daemon is starting with it.
    Starting,
    /// Reach the agent from a phone (may be skipped).
    Channel,
    /// Set up.
    Done,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// How a provider is paid for.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Billing {
    /// Uses a subscription the owner already pays for; it has limits.
    Plan,
    /// Billed per token by the provider.
    PerToken,
    /// Runs on the owner's own machine (Ollama): nothing is billed.
    Local,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A model connection already on this machine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct FoundModel {
    /// The provider preset (`openai_chatgpt`, `openai`, ...).
    pub preset: String,
    /// The provider and model, in plain words.
    pub name: String,
    /// Where it was found ("ChatGPT sign-in on this computer").
    pub source: String,
    pub billing: Billing,
    /// It answered a test call.
    pub works: bool,
    /// Why not, when it did not.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

/// A provider the owner may connect.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProviderOption {
    pub preset: String,
    pub name: String,
    pub billing: Billing,
    /// Connected with an API key (else with a sign-in).
    pub needs_key: bool,
    /// Where to get one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub key_url: String,
    /// The owner names the address (a custom endpoint).
    #[serde(default)]
    pub needs_base_url: bool,
    /// The model it usually starts with (the account's list decides).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
}

/// A key its provider accepted, with what the account offers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct CheckedKey {
    pub preset: String,
    /// The account's models, as the provider lists them.
    pub models: Vec<String>,
    /// The one to preselect, when the account has the provider's usual
    /// starting model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested: Option<String>,
}

/// A subscription sign-in waiting for the owner.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SignIn {
    /// Open this in a browser.
    pub url: String,
    /// The browser cannot return to the daemon by itself: paste the address
    /// it ends on (`finish_sign_in`).
    #[serde(default)]
    pub paste: bool,
}

/// A chat the daemon can be reached on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ChannelOption {
    /// `telegram`, `slack` or `discord`.
    pub platform: String,
    pub name: String,
    /// Where to make a bot and get its token.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub help_url: String,
    /// It takes a second token (Slack's app token).
    #[serde(default)]
    pub needs_app_token: bool,
}

/// A chat connected during setup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ChannelWait {
    pub platform: String,
    /// The bot, as the owner finds it (`@synthetic_bot`).
    pub bot: String,
    /// The owner's first message arrived and was answered there.
    #[serde(default)]
    pub connected: bool,
    /// The one-time code whose direct message claims the bot (`None` for a
    /// bot connected before). Only this screen shows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_code: Option<String>,
}

/// One step of first-run setup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SetupAction {
    /// Look for models on this machine again, and test them.
    Detect,
    /// Use `found` entry `index` (from 0).
    UseFound { index: u32 },
    /// Start the subscription sign-in. `paste`: the browser is on another
    /// device and cannot reach the daemon's sign-in listener.
    SignIn {
        #[serde(default)]
        paste: bool,
    },
    /// The address the sign-in's browser ended on.
    FinishSignIn { redirect: String },
    /// Check a provider's key (nothing is saved) and read the models its
    /// account offers: the answer's `checked`.
    CheckKey {
        provider: ProviderChoice,
        key: SecretText,
    },
    /// Connect a provider with an API key (`None` for one that needs none)
    /// and a model; it is tested before anything is saved.
    ConnectModel {
        provider: ProviderChoice,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        key: Option<SecretText>,
    },
    /// Connect a chat bot by its token.
    ConnectChannel {
        platform: String,
        token: SecretText,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        app_token: Option<SecretText>,
    },
    /// Leave the chat step for later.
    SkipChannel,
    /// The chat step is done.
    Finish,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A feature the agent can ask to have turned on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    /// Browsing websites.
    Browser,
    /// Understanding voice notes.
    Voice,
    /// Seeing and using this computer's screen.
    ComputerUse,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Something that happened. Durable unless noted.
// Wire values live briefly between parse and use; boxing would only make
// them clumsier to build and match.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    SessionCreated {
        session: SessionSummary,
    },
    SessionUpdated {
        session: SessionSummary,
    },
    /// The owner (or someone in a shared chat) said something.
    UserMessage {
        session_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<AttachmentRef>,
        /// Where it was said: `app`, `telegram`, `slack`...
        origin: String,
    },
    /// A message from the daemon outside a turn's reply: a queued-message
    /// notice, an answer about running work, a background job's report.
    AssistantMessage {
        session_id: String,
        text: String,
        /// Files sent with it: a picture the agent made or took, a document
        /// (protocol 1.11).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<AttachmentRef>,
        /// Things the owner might ask next, shown as prompts to press
        /// (protocol 1.5).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        suggestions: Vec<String>,
    },
    TurnStarted {
        turn_id: String,
        session_id: String,
        origin: String,
    },
    /// Ephemeral: a chunk of the reply as it streams (capability `deltas`).
    TurnDelta {
        turn_id: String,
        text: String,
    },
    /// Ephemeral: what the turn is doing.
    TurnStatus {
        turn_id: String,
        status: TurnStatus,
    },
    /// The turn waits for the owner to do something outside the chat
    /// (finish an OAuth sign-in in a browser).
    UserActionRequired {
        turn_id: String,
        message: String,
    },
    TurnCompleted {
        turn_id: String,
        /// The final reply, as Markdown; empty when there was none.
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<AttachmentRef>,
        state: TurnState,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// The provider and model that wrote the reply (protocol 1.35);
        /// `None` when there was none or it is not known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        served_by: Option<ServedBy>,
    },
    ApprovalRequested {
        approval: Approval,
    },
    ApprovalResolved {
        approval_id: String,
        outcome: ApprovalOutcome,
        /// Who answered; `None` when nobody did (expired, withdrawn).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_by: Option<ResolvedBy>,
    },
    TaskUpdated {
        task: TaskSummary,
    },
    /// A helper run started or ended (protocol 1.46): a sub-agent, a coding
    /// agent the daemon launched or a background command, from the daemon's
    /// own record of the run, so a client shows work that goes on after the
    /// turn that started it is over.
    RunUpdated {
        run: RunSummary,
    },
    /// Ephemeral (protocol 1.46): what a running run is doing now. Sent at
    /// most every few seconds per run; its start and end are `run_updated`.
    RunStep {
        run_id: String,
        session_id: String,
        /// The step in plain words; empty when it says nothing worth showing.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        step: String,
        at_ms: u64,
    },
    /// The owner named one of a conversation's agents (protocol 1.48,
    /// `rename_agent`); `None`: the name was taken away.
    AgentRenamed {
        session_id: String,
        thread: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A project was created or changed (protocol 1.10).
    ProjectUpdated {
        project: ProjectSummary,
    },
    /// A project was deleted (protocol 1.27); its conversations are in no
    /// project now.
    ProjectDeleted {
        project_id: String,
    },
    /// Only with capability `goals`.
    GoalUpdated {
        goal: GoalSummary,
    },
    /// Ephemeral, and only to the one connection that declared
    /// `view_actions` and sent the turn's message: the agent asks this
    /// client to move its view. The client shows a short note of what moved
    /// and why. A view action only changes what is on screen; it never
    /// answers an approval, pairs or revokes a device, sends a message or
    /// stops work.
    ViewAction {
        action: ViewAction,
        /// Why, in a few plain words, for the note the client shows.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        reason: String,
    },
    /// Ephemeral: the settings changed (or a change waiting for approval
    /// ended): read them again (protocol 1.4).
    SettingsUpdated {
        ok: bool,
        message: String,
    },
    /// Ephemeral: first-run setup moved on (a sign-in finished, a model
    /// passed its test, a channel heard its first message); protocol 1.5.
    SetupUpdated {
        setup: SetupState,
    },
    /// The agent needs a feature that is off for what it was asked to do
    /// (protocol 1.5). The client offers to turn it on; nothing changes
    /// until the owner does.
    FeatureRequested {
        session_id: String,
        feature: Feature,
        /// What it is needed for, in the agent's words.
        reason: String,
    },
    /// Ephemeral, capability `surfaces` (protocol 1.30): a surface appeared
    /// or changed (title, site, controller, handoff).
    SurfaceUpdated {
        surface: SurfaceInfo,
    },
    /// Ephemeral, capability `surfaces` (protocol 1.30).
    SurfaceClosed {
        surface_id: String,
    },
    /// Ephemeral, capability `shells` (protocol 1.42): a shell started,
    /// was retitled or ended.
    ShellUpdated {
        shell: ShellInfo,
    },
    /// Ephemeral, capability `shells` (protocol 1.42): a shell is gone.
    ShellClosed {
        shell_id: String,
        session_id: String,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

impl Event {
    /// Whether the event is kept in the log and replayed on resume.
    pub fn is_durable(&self) -> bool {
        !matches!(
            self,
            Event::TurnDelta { .. }
                | Event::TurnStatus { .. }
                | Event::ViewAction { .. }
                | Event::SettingsUpdated { .. }
                | Event::SetupUpdated { .. }
                | Event::SurfaceUpdated { .. }
                | Event::SurfaceClosed { .. }
                | Event::ShellUpdated { .. }
                | Event::ShellClosed { .. }
                | Event::RunStep { .. }
        )
    }

    /// The event's `type` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Event::SessionCreated { .. } => "session_created",
            Event::SessionUpdated { .. } => "session_updated",
            Event::UserMessage { .. } => "user_message",
            Event::AssistantMessage { .. } => "assistant_message",
            Event::TurnStarted { .. } => "turn_started",
            Event::TurnDelta { .. } => "turn_delta",
            Event::TurnStatus { .. } => "turn_status",
            Event::UserActionRequired { .. } => "user_action_required",
            Event::TurnCompleted { .. } => "turn_completed",
            Event::ApprovalRequested { .. } => "approval_requested",
            Event::ApprovalResolved { .. } => "approval_resolved",
            Event::TaskUpdated { .. } => "task_updated",
            Event::RunUpdated { .. } => "run_updated",
            Event::RunStep { .. } => "run_step",
            Event::AgentRenamed { .. } => "agent_renamed",
            Event::ProjectUpdated { .. } => "project_updated",
            Event::ProjectDeleted { .. } => "project_deleted",
            Event::GoalUpdated { .. } => "goal_updated",
            Event::ViewAction { .. } => "view_action",
            Event::SettingsUpdated { .. } => "settings_updated",
            Event::SetupUpdated { .. } => "setup_updated",
            Event::FeatureRequested { .. } => "feature_requested",
            Event::SurfaceUpdated { .. } => "surface_updated",
            Event::SurfaceClosed { .. } => "surface_closed",
            Event::ShellUpdated { .. } => "shell_updated",
            Event::ShellClosed { .. } => "shell_closed",
            Event::Unknown => "unknown",
        }
    }
}

/// Where the agent asks a client to take its view. Every variant only
/// changes what is on screen: none carries a decision, a message or a
/// device, so none can answer an approval or act for the owner. Kinds a
/// client does not know read as `Unknown` and are ignored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewAction {
    /// Open a session's transcript.
    OpenSession { session_id: String },
    /// Bring a waiting approval's card into view (not answer it).
    ShowApproval { approval_id: String },
    /// Show an entry of the running-work list.
    ShowTask { task_id: String },
    /// Move the keyboard to a pane.
    FocusPane { pane: Pane },
    /// Clear the composer's target, so the next message starts a session.
    NewSession,
    /// Show the paired devices.
    ShowDevices,
    /// Bring a surface's viewer forward (protocol 1.30). It opens no
    /// stream and takes no control.
    ShowSurface { surface_id: String },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// A part of a client's screen.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Pane {
    Sessions,
    Chat,
    Approvals,
    Tasks,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TurnState {
    /// The reply was delivered.
    Completed,
    /// The agent had nothing to say.
    Empty,
    Failed,
    Cancelled,
    /// The work goes on in the background; its report comes later.
    HandedOff,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What a running turn is doing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnStatus {
    Thinking,
    /// Built with [`TurnStatus::tool_started`]: non-exhaustive so a new
    /// field changes one constructor, not every sender.
    #[non_exhaustive]
    ToolStarted {
        name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
        /// What kind of work it is (protocol 1.29).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        activity: Option<ToolKind>,
        /// The call's arguments as history shows them, redacted and cut to
        /// size (protocol 1.40); empty when the daemon gave none.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        input: String,
    },
    ToolProgress {
        name: String,
        chunk: String,
    },
    /// Built with [`TurnStatus::tool_completed`].
    #[non_exhaustive]
    ToolCompleted {
        name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
        /// The call's result as history shows it, redacted and cut to size
        /// (protocol 1.40); empty when the daemon gave none.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        output: String,
    },
    /// A call ended by handing its work to something that keeps running (a
    /// helper or a background run), built with [`TurnStatus::tool_handed_off`]
    /// (protocol 1.50). The work has not finished, so it never reads as a
    /// finished call: `text` says the work goes on, and `output` is the
    /// call's own result (its launch line) for the call's row.
    #[non_exhaustive]
    ToolHandedOff {
        name: String,
        text: String,
        /// The call's result as history shows it, redacted and cut to size;
        /// empty when the daemon gave none.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        output: String,
    },
    PlanStep {
        /// 0-based.
        step: u32,
        total: u32,
        description: String,
        state: StepState,
    },
    /// A rendered checklist of the turn's plan, as Markdown.
    Checklist {
        text: String,
    },
    /// Anything else worth a line ("helper started: …").
    Note {
        text: String,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

impl TurnStatus {
    /// A call starting; [`Self::with_detail`] adds its shown arguments.
    pub fn tool_started(
        name: impl Into<String>,
        summary: impl Into<String>,
        activity: Option<ToolKind>,
    ) -> Self {
        TurnStatus::ToolStarted {
            name: name.into(),
            summary: summary.into(),
            activity,
            input: String::new(),
        }
    }

    /// A call finished; [`Self::with_detail`] adds its shown result.
    pub fn tool_completed(name: impl Into<String>, summary: impl Into<String>) -> Self {
        TurnStatus::ToolCompleted {
            name: name.into(),
            summary: summary.into(),
            output: String::new(),
        }
    }

    /// A call that handed its work off; [`Self::with_detail`] adds its
    /// shown result.
    pub fn tool_handed_off(name: impl Into<String>, text: impl Into<String>) -> Self {
        TurnStatus::ToolHandedOff {
            name: name.into(),
            text: text.into(),
            output: String::new(),
        }
    }

    /// A started call with its shown arguments, or a finished or handed-off
    /// one with its shown result. Other statuses are unchanged.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        match &mut self {
            TurnStatus::ToolStarted { input: shown, .. }
            | TurnStatus::ToolCompleted { output: shown, .. }
            | TurnStatus::ToolHandedOff { output: shown, .. } => *shown = detail.into(),
            _ => {}
        }
        self
    }

    /// What a tool row shows for this status: a started call's arguments or
    /// a completed or handed-off call's result, else its summary (a handed-off
    /// call's words) when the daemon sent none (an older daemon). `None` for
    /// a status that is not a call.
    pub fn call_detail(&self) -> Option<&str> {
        let (shown, summary) = match self {
            TurnStatus::ToolStarted { input, summary, .. } => (input, summary),
            TurnStatus::ToolCompleted {
                output, summary, ..
            } => (output, summary),
            TurnStatus::ToolHandedOff { output, text, .. } => (output, text),
            _ => return None,
        };
        Some(if shown.trim().is_empty() {
            summary
        } else {
            shown
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StepState {
    Started,
    Completed,
    Failed,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

// ---------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------

/// A question the owner must answer before work goes on. Pending until
/// exactly one of: answered, expired (nobody answered in time: the call is
/// not run, and it is not a refusal) or withdrawn (its turn stopped, or the
/// daemon restarted).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Approval {
    pub id: String,
    pub session_id: String,
    pub kind: ApprovalKind,
    pub risk: Risk,
    /// One plain sentence: what will happen.
    pub summary: String,
    /// The exact command or arguments. Left out in compact frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// The card's typed facts, when the asker gave them. Left out in
    /// compact frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub card: Option<ApprovalCard>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// The answers offered, in display order.
    pub options: Vec<ApprovalDecision>,
    pub requested_at_ms: u64,
    /// When it expires unanswered; `None` when the asker set no deadline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
    /// The chat the card is also shown in (`telegram`...), if any besides
    /// the app.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
}

impl Approval {
    /// The approval as compact frames carry it: no detail, no card.
    pub fn compact(mut self) -> Self {
        self.detail = None;
        self.card = None;
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalKind {
    /// Allow a call (a command, a request, a tool use).
    Command,
    /// Confirm a scheduled goal.
    GoalConfirmation,
    /// Confirm an Autopilot posture.
    AutopilotConfirmation,
    /// The agent asks the owner to do something themselves on a surface
    /// (protocol 1.30): `approve_once` means done, `deny` means not now.
    Handoff,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// How much is at stake, from the daemon's typed rating. Ordered.
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Risk {
    Low,
    Medium,
    High,
    Critical,
}

/// An answer to an approval.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    ApproveOnce,
    /// Allow it for the rest of this session.
    ApproveSession,
    /// Allow this kind of call from now on.
    ApproveAlways,
    Deny,
}

/// How an approval ended.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalOutcome {
    ApprovedOnce,
    ApprovedSession,
    ApprovedAlways,
    Denied,
    /// Nobody answered in time. The call did not run; this is not a
    /// refusal.
    Expired,
    /// The question went away unanswered: its turn was stopped, the card
    /// could not be shown, or the daemon restarted.
    Withdrawn,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

impl ApprovalOutcome {
    /// The outcome an answer gives.
    pub fn from_decision(decision: ApprovalDecision) -> Self {
        match decision {
            ApprovalDecision::ApproveOnce => Self::ApprovedOnce,
            ApprovalDecision::ApproveSession => Self::ApprovedSession,
            ApprovalDecision::ApproveAlways => Self::ApprovedAlways,
            ApprovalDecision::Deny => Self::Denied,
        }
    }
}

/// Who answered an approval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ResolvedBy {
    /// `app` for a client of this protocol, else the chat (`telegram`...).
    pub channel: String,
    /// The client's device, when `channel` is `app`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// An approval card's content as facts: who asks, what will happen, why the
/// owner is asked and the one effect worth a warning. Plain words only;
/// never authority.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ApprovalCard {
    /// The helper asking; `None` for the assistant itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asked_by: Option<String>,
    /// The run asking (protocol 1.48), when a coding agent a run launched
    /// asks: the agent a client shows as waiting on the owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    pub action: CardAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<AskReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impact: Option<Impact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CardAction {
    /// Run a command on the owner's computer.
    Command {
        line: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Folders a read-only command only looks inside.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        looks_inside: Vec<String>,
    },
    /// Let the browser open a site on this computer.
    OpenLocalSite { site: String, once_minutes: u64 },
    /// Use a tool; `what` completes "wants to …".
    Use {
        what: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Click a button on a web site in the browser (protocol 1.34): one
    /// whose effect the page's script decides. Allow once covers this
    /// click; allowing it for the session covers buttons on `site` (an
    /// origin) until the browser session ends. Forms, passwords and
    /// payment fields still ask. Older clients read it as `Unknown` and
    /// show the approval's `summary` and `detail`.
    ClickOnSite {
        site: String,
        /// The selector the click targets.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target: Option<String>,
    },
    /// Do something on a surface yourself (protocol 1.30): sign in, enter
    /// a code. `what` is the agent's few words; `site` the host only.
    TakeOver {
        surface_id: String,
        what: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        site: Option<String>,
        /// Whether a sign-in made there is kept for later tasks.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keeps_sign_in: Option<bool>,
    },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AskReason {
    /// The work read content from outside and the clean check could not
    /// confirm this step fits the request.
    ReadOutsideContent {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        check_says: Option<String>,
    },
    /// The owner said no to an earlier step of this run.
    AfterADecline,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Impact {
    Deletes {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        targets: Vec<String>,
    },
    Sends,
    Spends,
    WritesOutsideWorkArea,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

// ---------------------------------------------------------------------------
// Sessions, tasks, goals
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SessionSummary {
    pub id: String,
    /// The chat it belongs to: `app`, `telegram`...
    pub channel: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_activity_ms: Option<u64>,
    /// Whether clients may post into it. Sessions of other chats are
    /// read-mostly in v1: watch, stop and answer approvals there, but post
    /// from that chat.
    #[serde(default)]
    pub writable: bool,
    /// The project it belongs to (protocol 1.10); `None`: no project.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Other people are in it: a group, or a channel of a workspace
    /// (protocol 1.12). What the app posts there, everyone there sees.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub group: bool,
    /// The owner archived it (protocol 1.17): lists show it apart.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub archived: bool,
    /// The coding agent it is set to (protocol 1.52): an id from
    /// `list_models`' `agents`; each message goes straight to it
    /// ([`ModelChoice::Agent`]). `None`: AIDaemon answers. A change comes
    /// as `session_updated`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

/// A message waiting for its session's running work to end (protocol
/// 1.17).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct QueuedMessage {
    /// What `edit_queued` names it by, while it waits.
    pub id: u64,
    pub text: String,
    pub queued_at_ms: u64,
    /// Sent to wait for its own turn (the app's Queue), not to be read by
    /// the running work (protocol 1.18).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub after: bool,
}

/// A project: a name, and the folder its conversations work in
/// (protocol 1.10).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    /// An absolute path on the daemon's computer: a repository, a site, a
    /// folder of designs. `None`: the project has no folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The chat bots (session `channel`s, `telegram:coding_bot`) whose
    /// direct-message chats go to this project unless placed elsewhere
    /// (protocol 1.12). Group chats are never placed by a bot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<String>,
}

/// One message of a session's transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct HistoryMessage {
    pub id: String,
    /// `user` or `assistant`.
    pub role: String,
    pub text: String,
    pub at_ms: u64,
    /// For a reply: the tools its turn used, in order (protocol 1.3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// For a reply: what each of those tools was given and gave back, in
    /// the same order (protocol 1.19). Trimmed and secrets redacted; empty
    /// from a daemon that does not send it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<ToolCallInfo>,
    /// Files sent with it (protocol 1.11).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<AttachmentRef>,
    /// A reply that only says its turn's work went on in the background
    /// (the turn ended handed off; protocol 1.29). The report comes later
    /// as its own reply.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub handed_off: bool,
    /// For a reply: the provider and model that wrote it (protocol 1.35).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by: Option<ServedBy>,
}

/// The provider and model that wrote a reply (protocol 1.35).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ServedBy {
    /// The provider's catalog preset (`openai_chatgpt`, `anthropic`...),
    /// for its mark; empty for one not from the catalog.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub preset: String,
    /// The provider in plain words, as Settings names it ("ChatGPT").
    pub provider: String,
    pub model: String,
}

/// What kind of work a tool call was, as the daemon's cards show it
/// (protocol 1.29): from the tool's own declared activity, never its name.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Reading a file or document.
    Read,
    /// Searching files or listing what is there.
    Search,
    /// Writing or editing files.
    Write,
    /// Running a command or program.
    Run,
    /// Searching or reading the web.
    Web,
    /// Handing work to a helper.
    Delegate,
    /// Reading what was remembered.
    Recall,
    /// Saving or changing what is remembered.
    Remember,
    /// Setting a reminder or a schedule.
    Remind,
    /// Any other look at the state of things.
    Check,
    /// Any other change.
    Change,
    /// A kind added after this client was built.
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// One tool call of a reply's turn, as the owner's app shows it when a
/// tool is opened (protocol 1.19).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ToolCallInfo {
    pub name: String,
    /// What it was given: its arguments, trimmed, secrets redacted.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub input: String,
    /// What it gave back, trimmed, secrets redacted. Empty: no result
    /// was recorded.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub output: String,
    /// What kind of work it was (protocol 1.29); `None` from an older
    /// daemon or a tool it no longer has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
}

/// One entry of a session's running-work list: a turn, a helper (sub-agent)
/// or a background command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct TaskSummary {
    /// Stable across the daemon's life; `cancel_task` takes it.
    pub id: String,
    pub session_id: String,
    /// The short number `/cancel <n>` takes in the chat.
    pub number: u64,
    pub description: String,
    pub state: TaskState,
    pub started_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One helper run as the daemon records it (protocol 1.46): a sub-agent, a
/// coding agent it launched (Claude Code, Codex, ...) or a background
/// command. Every field comes from the run's typed record, never from what
/// a model wrote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunSummary {
    /// Stable while the daemon runs; the key every update finds it by.
    /// `cancel_task` takes it while the run runs.
    pub id: String,
    /// The conversation it reports to.
    pub session_id: String,
    /// The name the owner reads for it.
    pub label: String,
    /// The coding agent it is ("Claude Code"), for a run that launched one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    pub phase: RunPhase,
    /// It goes on after the turn that started it ended.
    pub background: bool,
    /// What it is doing now, in plain words; empty when nothing worth
    /// showing (or once it ended).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub step: String,
    pub started_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at_ms: Option<u64>,
    /// Once it ended, how long it took, already worded by the daemon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub took: Option<String>,
    /// Once it failed or stopped, why, in plain words from its record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// It ended because the owner stopped it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stopped_by_owner: bool,
    /// The request (its turn's id) that started it: the runs one request
    /// started share it, and a client shows them together.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// The files the coding agent it launched changed, as full paths, in
    /// the order first changed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edited: Vec<String>,
    /// The earlier run it continues (protocol 1.48): the same coding
    /// agent's session, resumed with a follow-up. A client shows the runs
    /// one continues as one thread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continues: Option<String>,
    /// The first run of its thread (protocol 1.48): the agent session it is
    /// part of, one participant of the conversation. Absent: it starts its
    /// own (its id is the thread's). Every client groups runs by it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
    /// Its agent kept its session and it ended: a follow-up can go on from
    /// it (protocol 1.48; [`RunDetail::can_continue`] once its record says
    /// the session id too).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub can_continue: bool,
    /// What its coding agent ended with (protocol 1.48): its reply, its
    /// typed report and where the report disagrees with what it was seen
    /// doing. Set once it ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<RunResult>,
    /// What it was asked, and whether the owner asked it directly
    /// (protocol 1.48): a client shows a follow-up the owner sent an agent
    /// as the owner's message to that agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asked: Option<RunAsked>,
}

/// What a run was asked (protocol 1.48).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunAsked {
    /// The task as given, without the daemon's own rules; shortened.
    pub text: String,
    /// The owner sent it to the agent directly (a follow-up), rather than
    /// aidaemon delegating it.
    #[serde(default)]
    pub by_owner: bool,
}

/// What a run's coding agent ended with (protocol 1.48).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunResult {
    /// Its whole reply (bounded only against a runaway agent): its
    /// report's summary when it reported in fields, else its final message.
    pub text: String,
    /// Its report, when its agent was held to the report's fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<RunReport>,
    /// Where its report disagrees with its recorded actions; the record
    /// wins.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mismatches: Vec<RunMismatch>,
}

/// A coding agent's report: its claims, each one checked
/// ([`RunResult::mismatches`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunReport {
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changed_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<RunCheck>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_questions: Vec<String>,
    /// It can't go on without the owner.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub needs_owner: bool,
}

/// A check a coding agent says it ran.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunCheck {
    pub command: String,
    pub passed: bool,
}

/// A claim the run's recorded actions don't bear out.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunMismatch {
    /// It says it changed the file; no change to it was recorded.
    ChangeNotSeen { path: String },
    /// A change to the file was recorded; it doesn't say so.
    ChangeNotClaimed { path: String },
    /// It says it ran the check; no run of that program was recorded.
    CheckNotRun { command: String },
    /// It says the check passed; the recorded run failed.
    CheckFailed { command: String },
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// What one run did (protocol 1.46, `get_run`), from its record. Only a
/// coding agent's run keeps steps; any other has its summary alone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct RunDetail {
    pub run_id: String,
    /// The coding agent's own session id (Claude Code's session id, Codex's
    /// thread id, an ACP session id): what resumes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_session_id: Option<String>,
    /// A follow-up can go on from it (protocol 1.48): its agent kept its
    /// session. `send_message` with `in_run` set to it continues it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub can_continue: bool,
    /// The model it worked with, as it reported it (protocol 1.47).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The folder it worked in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    /// What it was asked, shortened.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub task: String,
    /// What it reported at its end, shortened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Its steps, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<RunStepItem>,
    /// Its plan, as it last wrote it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plan: Vec<RunPlanItem>,
    /// What it asked permission for and the answer, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub permissions: Vec<RunPermission>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<RunUsage>,
}

/// One step of a run, in plain words from its typed event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunStepItem {
    pub words: String,
    pub state: RunStepState,
    /// When it started, ms since the epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_ms: Option<u64>,
    /// One of the agent's own helpers ran it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub by_helper: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunStepState {
    /// Started; no end is on record (yet).
    Started,
    Done,
    Failed,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunPlanItem {
    pub text: String,
    pub status: RunPlanStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunPlanStatus {
    Pending,
    InProgress,
    Done,
    #[serde(other)]
    Unknown,
}

/// One permission a run asked for and how it was answered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RunPermission {
    /// What it asked, as the owner reads it ("write: /w/notes.md").
    pub label: String,
    pub outcome: RunPermissionOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunPermissionOutcome {
    Allowed,
    Refused,
    Declined,
    Unanswered,
    #[serde(other)]
    Unknown,
}

/// Tokens and cost, as the agent counted them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct RunUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

/// Where a run is (protocol 1.46).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunPhase {
    Running,
    /// Its own end record says it did the work.
    Succeeded,
    /// It ended without doing the work.
    Failed,
    /// It ended before the work was done (stopped, cancelled, stalled).
    Stopped,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

impl RunPhase {
    pub fn is_running(self) -> bool {
        self == Self::Running
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Running,
    Cancelling,
    Completed,
    Failed,
    Cancelled,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct GoalSummary {
    pub id: String,
    pub description: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// What a client needs when it cannot resume.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Snapshot {
    pub sessions: Vec<SessionSummary>,
    pub running_tasks: Vec<TaskSummary>,
    pub pending_approvals: Vec<Approval>,
    /// Turns running now, as far as each has got (protocol 1.7): a client
    /// that (re)connects mid-turn shows what it is doing, not only that it
    /// runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub live_turns: Vec<LiveTurnState>,
    /// Live surfaces (protocol 1.30, capability `surfaces`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub surfaces: Vec<SurfaceInfo>,
    /// Conversation shells (protocol 1.42, capability `shells`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shells: Vec<ShellInfo>,
    /// Helper runs the daemon still holds a record of, running or recently
    /// ended (protocol 1.46).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<RunSummary>,
}

/// A running turn as far as it has got: the text streamed so far, what it
/// is doing now and the tools it started, in order.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct LiveTurnState {
    pub session_id: String,
    pub turn_id: String,
    pub started_at_ms: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub streamed: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<TurnStatus>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// What each of those tools was given and, once done, gave back, in
    /// the same order (protocol 1.39), so a client that joins mid-turn shows
    /// them as one that watched it. Empty from an older daemon.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<ToolCallInfo>,
}

/// How the daemon itself is doing, for the app's System area.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct SystemStatus {
    pub version: String,
    pub uptime_secs: u64,
    /// The model normal requests use, and the ones tried when it fails.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_models: Vec<String>,
    /// Tokens per day and model, newest day first (the last 7 days).
    #[serde(default)]
    pub usage: Vec<UsageDay>,
    /// Model calls over the last 24 hours; `None` when not recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_calls: Option<ModelCalls>,
    /// Scheduled goals, soonest first.
    #[serde(default)]
    pub schedules: Vec<ScheduleInfo>,
    /// The daemon's own background jobs (memory upkeep, pruning, ...).
    #[serde(default)]
    pub jobs: Vec<JobInfo>,
    #[serde(default)]
    pub queues: Vec<QueueInfo>,
    /// The owner's health checks of their own services.
    #[serde(default)]
    pub checks: Vec<HealthCheckInfo>,
    /// Background goal runs of the last 24 hours, newest first (protocol
    /// 1.3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<GoalRunInfo>,
    /// The owner's devices (Nodes: a K10 board, a sensor) and how each
    /// reaches the daemon (protocol 1.44).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<NodeInfo>,
}

/// One of the owner's devices (a Node), as the System area shows it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct NodeInfo {
    pub node_id: String,
    pub name: String,
    /// What it is (`k10`, `simulator`, ...).
    pub kind: String,
    /// It sent a heartbeat recently.
    pub online: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firmware_version: Option<String>,
    /// How its latest heartbeat arrived; `None` before one since the
    /// daemon started.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection: Option<NodeConnection>,
}

/// How a device reaches the daemon.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NodeConnection {
    /// Straight to this computer over the local network.
    LocalNetwork,
    /// Through a tunnel (Cloudflare).
    Tunnel,
    /// Through a USB relay or a program on this computer.
    Local,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// One run of a background goal, judged by what it left behind.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct GoalRunInfo {
    pub goal_id: String,
    pub description: String,
    /// `running`, `completed`, `failed`, `cancelled`...
    pub status: String,
    /// What the run reported; `None` when it reported nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    pub started_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_ms: Option<u64>,
    /// The session the goal reports into, to open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct UsageDay {
    /// `YYYY-MM-DD`, UTC.
    pub day: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub requests: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ModelCalls {
    pub total: u64,
    pub failed: u64,
    /// Calls answered by a fallback model.
    pub fell_back: u64,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub max_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ScheduleInfo {
    pub goal_id: String,
    pub description: String,
    /// The goal's status ("active", "paused", ...).
    pub status: String,
    /// The schedule as the owner wrote it ("every weekday at 9").
    pub schedule: String,
    pub paused: bool,
    pub one_shot: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_run_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct JobInfo {
    pub name: String,
    pub every_secs: u64,
    pub running: bool,
    pub consecutive_failures: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct QueueInfo {
    pub name: String,
    pub depth: u64,
    pub capacity: u64,
    pub dropped: u64,
    pub failed: u64,
    pub overloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct HealthCheckInfo {
    pub name: String,
    pub target: String,
    pub status: CheckStatus,
    pub paused: bool,
    pub consecutive_failures: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Healthy,
    Degraded,
    Unhealthy,
    /// Not run yet.
    Unknown,
}

/// A watch's one-call view.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Glance {
    pub running_tasks: Vec<TaskSummary>,
    pub pending_approvals: Vec<Approval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_reply: Option<LastReply>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct LastReply {
    pub session_id: String,
    pub turn_id: String,
    pub text: String,
    pub at_ms: u64,
}

// ---------------------------------------------------------------------------
// Errors and HTTP mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ProtocolError {
    pub code: ErrorCode,
    pub message: String,
    /// Whether the same request may succeed later.
    pub retryable: bool,
}

impl ProtocolError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: matches!(code, ErrorCode::RateLimited | ErrorCode::Internal),
        }
    }
}

/// The closed set of error codes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Unauthorized,
    /// Authenticated, but not allowed: an approval above this device's risk
    /// ceiling, a post into another chat's session.
    Forbidden,
    VersionUnsupported,
    NotFound,
    /// The state moved on: an approval already resolved (the message names
    /// the winning outcome).
    Conflict,
    RateLimited,
    Invalid,
    Internal,
    /// A project's folder does not exist (protocol 1.24). The same command
    /// with `create_folder` makes it.
    FolderMissing,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// `GET /client/v1/events` answers with this: the event frames after
/// `since` (or a snapshot frame first, when `since` is out of the retention
/// window), and the `since` to ask with next.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct EventsPage {
    pub frames: Vec<ServerFrame>,
    pub next_since: u64,
}

// ---------------------------------------------------------------------------
// Pairing and device sessions (HTTP)
// ---------------------------------------------------------------------------
//
// A device pairs once: the owner makes an offer (`/pair_device` in a chat,
// or `aidaemon pair`), which shows a code and a link (as a QR code); the
// device redeems it with its public key (`POST /client/v1/enrollments/redeem`)
// and the owner confirms on a chat they already use (or in the terminal
// that made the offer). Each connection then proves the key: the device
// asks for a challenge (`POST /client/v1/sessions/challenge`), signs
// [`challenge_message`] and trades the signature for a short-lived access
// token (`POST /client/v1/sessions`), used as `Authorization: Bearer` or in
// `hello.token`.

/// What a device's token may do.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Read sessions, history, events, tasks and approvals.
    Observe,
    /// Send messages, stop and cancel work, run commands.
    Chat,
    /// Answer approvals (within the device's risk ceiling).
    Approve,
    /// Watch and control live surfaces: the agent's browser, the screen
    /// (protocol 1.30).
    Control,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Every scope: what a phone or computer gets by default.
pub const ALL_SCOPES: [Scope; 4] = [Scope::Observe, Scope::Chat, Scope::Approve, Scope::Control];

/// The owner's offer to pair a device (owner only:
/// `POST /client/v1/pairing/offers`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct OfferRequest {
    /// The chat where the owner confirms the device; `None` for a code
    /// shown on the owner's own screen, which pairs the device that redeems
    /// it at once (older daemons: confirm with
    /// `POST /client/v1/pairing/offers/{id}/decision`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm_in_session: Option<String>,
}

/// A single-use pairing offer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct PairingOffer {
    pub offer_id: String,
    /// What the owner types on the device ("K7Q2-9FZD").
    pub code: String,
    pub expires_at_ms: u64,
    /// The same as a link, for a QR code:
    /// `aidaemon-pair:?gateway=<url>&offer=<id>&code=<code>&fp=<fingerprint>`.
    pub link: String,
    /// Identifies this daemon; the device checks the redeem reply matches.
    pub fingerprint: String,
    /// Where devices reach the gateway now (a live tunnel origin, the
    /// configured address, or loopback). Absent from older daemons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_url: Option<String>,
    /// The web app on a site of its own (`https://app.aidaemon.ai` unless the
    /// owner names another), when the daemon is reachable by key over iroh:
    /// a phone with no app pairs there ([`PairingOffer::phone_link`])
    /// (protocol 1.45).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_app: Option<String>,
}

/// The gateway's page that opens the phone app on a pairing: tapped in a
/// chat or read from a QR code by a phone's camera (which open `https`
/// links, not the app's own scheme), it hands the offer to the app, or to
/// its store page when the app is missing, or pairs the browser instead.
pub const PAIRING_PAGE_PATH: &str = "/client/v1/pair";

impl PairingOffer {
    /// [`PairingOffer::link`] as a link to [`PAIRING_PAGE_PATH`] on the
    /// gateway's own address. The offer rides in the fragment, which
    /// browsers never send to the server. An older daemon that does not
    /// say its address, or one only its own machine can reach (loopback),
    /// gets the app link itself: the phone's camera opens the app, which
    /// pairs over the link's `remote` address.
    pub fn page_link(&self) -> String {
        let Some(base) = self
            .public_url
            .as_deref()
            .map(|url| url.trim().trim_end_matches('/'))
            .filter(|url| !url.is_empty() && !is_loopback_url(url))
        else {
            return self.link.clone();
        };
        let query = self
            .link
            .split_once('?')
            .map_or(self.link.as_str(), |(_, query)| query);
        format!("{base}{PAIRING_PAGE_PATH}#{query}")
    }

    /// The link a phone opens, for a QR code: the web app on its own site
    /// ([`PairingOffer::web_app`]) with this offer and the daemon's iroh
    /// address, which pairs the browser there, or opens the AIDaemon app
    /// when the phone has it (the site names the app). Else, with no iroh
    /// address or no web app, [`PairingOffer::page_link`].
    pub fn phone_link(&self) -> String {
        match self.web_app.as_deref().map(str::trim) {
            Some(web_app) if !web_app.is_empty() && self.link.contains("&remote=") => {
                self.web_link(web_app, "/")
            }
            _ => self.page_link(),
        }
    }

    /// The web client at `base` + `app_path` with this offer in its
    /// fragment, which pairs a browser, and the phone app when pasted
    /// there: it carries the daemon's peer-to-peer address (`remote`, from
    /// [`PairingOffer::link`]) too, so the app pairs over it.
    pub fn web_link(&self, base: &str, app_path: &str) -> String {
        let mut link = format!(
            "{}{app_path}#offer={}&code={}&fp={}",
            base.trim().trim_end_matches('/'),
            self.offer_id,
            self.code,
            self.fingerprint
        );
        // The query of our own `aidaemon-pair:` link (a machine format).
        if let Some(remote) = self
            .link
            .split_once('?')
            .into_iter()
            .flat_map(|(_, query)| query.split('&'))
            .find(|pair| pair.starts_with("remote="))
        {
            link.push('&');
            link.push_str(remote);
        }
        link
    }
}

/// Whether `url`'s host is this machine only (`127.x`, `localhost`,
/// `[::1]`): a URL, so its host is read from its own syntax.
/// Whether `url` names this machine only (`localhost`, `127.0.0.1`, `::1`):
/// no other device can open it.
pub fn is_loopback_url(url: &str) -> bool {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split('/').next().unwrap_or(rest);
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(v6),
        None => host.split(':').next().unwrap_or(host),
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// `POST /client/v1/enrollments/redeem`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RedeemRequest {
    pub offer_id: String,
    pub code: String,
    /// The device as the owner will see it ("Pixel 9").
    pub device_name: String,
    pub device_kind: ClientKind,
    /// The device's P-256 public key, SEC1 uncompressed, base64url.
    pub public_key: String,
    /// The scopes asked for; empty: all of them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<Scope>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnrollmentState {
    /// Not redeemed yet.
    Open,
    /// Redeemed; waiting for the owner.
    Pending,
    Approved,
    Denied,
    Expired,
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Where a pairing stands (`GET /client/v1/enrollments/{offer_id}`, and
/// the owner's `GET /client/v1/pairing/offers/{offer_id}`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Enrollment {
    pub offer_id: String,
    pub state: EnrollmentState,
    /// Once approved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_kind: Option<ClientKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<Scope>,
    pub fingerprint: String,
}

/// The owner's answer to a pending pairing made without a chat.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct PairingDecision {
    pub approve: bool,
}

/// `POST /client/v1/sessions/challenge`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ChallengeRequest {
    pub device_id: String,
    /// The endpoint key (64 hex digits) this device dials the daemon's
    /// peer-to-peer address with, recorded once the sign-in verifies so
    /// that address admits it. Absent from clients that have none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Challenge {
    pub challenge_id: String,
    pub nonce: String,
    pub expires_at_ms: u64,
}

/// The exact bytes a device signs (ECDSA P-256, SHA-256) for a challenge,
/// as clients before protocol 1.51 sign them: bound to no daemon, so a
/// gateway takes it only from a client on its own computer.
pub fn challenge_message(challenge_id: &str, nonce: &str) -> String {
    format!("aidaemon-client-v1:{challenge_id}:{nonce}")
}

/// The exact bytes a device signs for a challenge (protocol 1.51): bound to
/// the daemon the device meant to reach (`audience`, from
/// [`sign_in_audience`] on the address it dialed) and to the endpoint key
/// its challenge request named (`-` for none). A daemon that relays another
/// daemon's challenge to a device gets a signature naming itself, which the
/// other daemon refuses.
pub fn challenge_message_v2(
    challenge_id: &str,
    nonce: &str,
    audience: &str,
    endpoint: Option<&str>,
) -> String {
    let endpoint = endpoint
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .unwrap_or("-");
    format!("aidaemon-client-v2:{challenge_id}:{nonce}:{audience}:{endpoint}")
}

/// Who a device signs in to, as [`challenge_message_v2`] names it: for a
/// daemon dialed by key (`iroh://<key>@…`), `iroh:<key>` (64 lowercase hex
/// digits); for an HTTP gateway, its origin (`https://host[:port]`,
/// lowercase, default port dropped). `None` for anything else. Machine
/// formats: URLs and our own peer-to-peer address.
pub fn sign_in_audience(gateway: &str) -> Option<String> {
    let (scheme, rest) = gateway.trim().split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    let authority = rest.split(['/', '?', '#']).next()?;
    if scheme == "iroh" {
        let key = authority.split('@').next()?.to_ascii_lowercase();
        return (key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| format!("iroh:{key}"));
    }
    if !matches!(scheme.as_str(), "http" | "https") || authority.contains('@') {
        return None;
    }
    let authority = authority.to_ascii_lowercase();
    if authority.is_empty() {
        return None;
    }
    let default_port = if scheme == "https" { ":443" } else { ":80" };
    let host = authority.strip_suffix(default_port).unwrap_or(&authority);
    Some(format!("{scheme}://{host}"))
}

/// The header a socket upgrade over a peer-to-peer transport may carry
/// instead of a token: a [`SocketSignIn`] as JSON, which signs the device
/// in and opens the socket in one round trip (protocol 1.48). A daemon that
/// refuses it (an older one, a clock too far off, a proof seen before)
/// answers 401, and the client signs in with a challenge instead.
pub const SOCKET_SIGN_IN_HEADER: &str = "x-aidaemon-sign-in";

/// How far a [`SocketSignIn`]'s time may lie behind the daemon's clock.
pub const SOCKET_SIGN_IN_MAX_AGE_MS: u64 = 60_000;

/// How far it may lie ahead (the device's clock runs fast).
pub const SOCKET_SIGN_IN_MAX_AHEAD_MS: u64 = 30_000;

/// A device's sign-in carried by its socket upgrade
/// ([`SOCKET_SIGN_IN_HEADER`], protocol 1.48). In place of a challenge the
/// daemon picked, it signs its own time and a random value, so the daemon
/// takes each proof once, only while it is fresh, and only for itself and
/// the endpoint key the transport says the upgrade came from
/// ([`socket_sign_in_message`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SocketSignIn {
    pub device_id: String,
    /// The device's clock, milliseconds since the Unix epoch.
    pub issued_at_ms: u64,
    /// 32 to 64 random hex digits, never used twice.
    pub nonce: String,
    /// The signature of [`socket_sign_in_message`] as `r || s` (64 bytes),
    /// base64url.
    pub signature: String,
}

/// The exact bytes a device signs (ECDSA P-256, SHA-256) for a
/// [`SocketSignIn`]: its own prefix, so no challenge signature passes for
/// one nor one for a challenge's; the daemon's endpoint key and the
/// device's (both 64 hex digits), so it is good at one daemon and from one
/// endpoint only; the device, its time and the random value.
pub fn socket_sign_in_message(
    daemon: &str,
    endpoint: &str,
    device_id: &str,
    issued_at_ms: u64,
    nonce: &str,
) -> String {
    format!("aidaemon-socket-sign-in-v1:{daemon}:{endpoint}:{device_id}:{issued_at_ms}:{nonce}")
}

/// `POST /client/v1/sessions`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SessionRequest {
    pub challenge_id: String,
    /// The signature as `r || s` (64 bytes), base64url.
    pub signature: String,
    /// The daemon the signature names (protocol 1.51): it signs
    /// [`challenge_message_v2`] for this audience. Absent: the older
    /// [`challenge_message`], which a gateway takes from its own computer
    /// only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audience: Option<String>,
}

/// A short-lived access token for one device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AccessGrant {
    pub access_token: String,
    pub expires_at_ms: u64,
    pub device_id: String,
    pub scopes: Vec<Scope>,
    /// The highest risk this device may approve; `None`: no limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_ceiling: Option<Risk>,
}

/// A paired device (owner only: `GET /client/v1/devices`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct DeviceInfo {
    pub device_id: String,
    pub name: String,
    pub kind: ClientKind,
    pub scopes: Vec<Scope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_ceiling: Option<Risk>,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen_ms: Option<u64>,
    #[serde(default)]
    pub revoked: bool,
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// The protocol's JSON Schema (draft 2020-12): a frame is a
/// [`ClientFrame`] or a [`ServerFrame`]; every type is under `$defs`,
/// including the HTTP mode's [`EventsPage`] and the pairing types.
pub fn schema() -> serde_json::Value {
    let mut generator = schemars::generate::SchemaSettings::draft2020_12().into_generator();
    let client = generator.subschema_for::<ClientFrame>();
    let server = generator.subschema_for::<ServerFrame>();
    let _ = generator.subschema_for::<EventsPage>();
    let _ = generator.subschema_for::<OfferRequest>();
    let _ = generator.subschema_for::<PairingOffer>();
    let _ = generator.subschema_for::<RedeemRequest>();
    let _ = generator.subschema_for::<Enrollment>();
    let _ = generator.subschema_for::<PairingDecision>();
    let _ = generator.subschema_for::<ChallengeRequest>();
    let _ = generator.subschema_for::<Challenge>();
    let _ = generator.subschema_for::<SessionRequest>();
    let _ = generator.subschema_for::<SocketSignIn>();
    let _ = generator.subschema_for::<AccessGrant>();
    let _ = generator.subschema_for::<Vec<DeviceInfo>>();
    let _ = generator.subschema_for::<media::MediaHeader>();
    let _ = generator.subschema_for::<media::MediaClientMessage>();
    let _ = generator.subschema_for::<media::ProviderCommand>();
    let _ = generator.subschema_for::<shell::ShellClientMessage>();
    let _ = generator.subschema_for::<shell::ShellServerMessage>();
    let mut defs = serde_json::Value::Object(generator.take_definitions(true));
    drop_empty_alternatives(&mut defs);
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aidaemon.ai/schemas/client-protocol/v1.schema.json",
        "title": "AIDaemon Client Protocol v1",
        "oneOf": [client, server],
        "$defs": defs,
    })
}

/// Remove the `{"enum": []}` alternatives schemars leaves where a skipped
/// variant (`Unknown`) was: they match nothing, and code generators choke
/// on them.
fn drop_empty_alternatives(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for key in ["oneOf", "anyOf"] {
                if let Some(serde_json::Value::Array(items)) = map.get_mut(key) {
                    items.retain(|item| {
                        !item
                            .get("enum")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(Vec::is_empty)
                    });
                }
            }
            map.values_mut().for_each(drop_empty_alternatives);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(drop_empty_alternatives),
        _ => {}
    }
}

/// [`schema`] as the checked-in file holds it: pretty, with a final newline.
pub fn schema_file() -> String {
    let mut text = serde_json::to_string_pretty(&schema()).expect("a JSON value always serializes");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_socket_sign_in_signs_its_own_message_never_a_challenges() {
        let daemon = "a".repeat(64);
        let endpoint = "b".repeat(64);
        let message = socket_sign_in_message(&daemon, &endpoint, "dv-1", 1_767_225_600_000, "0f0f");
        assert_eq!(
            message,
            format!("aidaemon-socket-sign-in-v1:{daemon}:{endpoint}:dv-1:1767225600000:0f0f")
        );
        // No challenge message can be the same bytes: their prefixes differ.
        assert!(!message.starts_with("aidaemon-client-v1:"));
        assert!(!challenge_message("ch-1", "n").starts_with("aidaemon-socket-sign-in-v1:"));
    }

    #[test]
    fn a_challenge_signature_names_the_daemon_and_the_endpoint() {
        let key = "ab".repeat(32);
        assert_eq!(
            sign_in_audience(&format!(
                "iroh://{}@relay.example.invalid",
                key.to_uppercase()
            )),
            Some(format!("iroh:{key}"))
        );
        assert_eq!(
            sign_in_audience("HTTPS://App.Example.com:443/client/v1/pair"),
            Some("https://app.example.com".into())
        );
        assert_eq!(
            sign_in_audience("http://127.0.0.1:8788"),
            Some("http://127.0.0.1:8788".into())
        );
        assert_eq!(sign_in_audience("https://user@app.example.com"), None);
        assert_eq!(sign_in_audience("iroh://short@relay"), None);
        assert_eq!(sign_in_audience("ftp://app.example.com"), None);
        assert_eq!(
            challenge_message_v2("ch-1", "n", "https://app.example.com", None),
            "aidaemon-client-v2:ch-1:n:https://app.example.com:-"
        );
        assert_ne!(
            challenge_message_v2("ch-1", "n", "https://a.example.com", Some("e")),
            challenge_message_v2("ch-1", "n", "https://b.example.com", Some("e"))
        );
        assert!(!challenge_message_v2("ch-1", "n", "x", None).starts_with("aidaemon-client-v1:"));
    }

    #[test]
    fn a_phone_pairs_at_the_web_app_only_when_it_can_reach_the_daemon_by_key() {
        let mut offer = PairingOffer {
            offer_id: "of-1".into(),
            code: "K7Q2-9FZD".into(),
            expires_at_ms: 0,
            link: "aidaemon-pair:?gateway=http%3A%2F%2F127.0.0.1%3A8788&offer=of-1&code=K7Q2-9FZD&fp=3b1f".into(),
            fingerprint: "3b1f".into(),
            public_url: Some("http://127.0.0.1:8788".into()),
            web_app: Some("https://app.example.invalid".into()),
        };
        // No iroh address: the web app could not reach the daemon.
        assert_eq!(offer.phone_link(), offer.page_link());
        offer
            .link
            .push_str("&remote=iroh%3A%2F%2Fk%40relay.example.invalid");
        assert_eq!(
            offer.phone_link(),
            "https://app.example.invalid/#offer=of-1&code=K7Q2-9FZD&fp=3b1f&remote=iroh%3A%2F%2Fk%40relay.example.invalid"
        );
        // The owner turned the web app off.
        offer.web_app = Some(String::new());
        assert_eq!(offer.phone_link(), offer.page_link());
    }

    #[test]
    fn a_pairing_page_link_is_on_the_gateway_with_the_offer_in_its_fragment() {
        let mut offer = PairingOffer {
            offer_id: "of-1".into(),
            code: "K7Q2-9FZD".into(),
            expires_at_ms: 0,
            link: "aidaemon-pair:?gateway=https%3A%2F%2Fd.example.invalid&offer=of-1&code=K7Q2-9FZD&fp=3b1f".into(),
            fingerprint: "3b1f".into(),
            public_url: Some("https://d.example.invalid/".into()),
            web_app: None,
        };
        assert_eq!(
            offer.page_link(),
            "https://d.example.invalid/client/v1/pair#gateway=https%3A%2F%2Fd.example.invalid&offer=of-1&code=K7Q2-9FZD&fp=3b1f"
        );
        offer.public_url = None;
        assert_eq!(offer.page_link(), offer.link);
        // A loopback address is no page for a phone.
        for loopback in [
            "http://127.0.0.1:8788",
            "http://localhost:8788/",
            "http://[::1]:8788",
        ] {
            offer.public_url = Some(loopback.into());
            assert_eq!(offer.page_link(), offer.link, "{loopback}");
        }
        assert_eq!(
            offer.web_link("https://d.example.invalid/", "/client/v1/next/"),
            "https://d.example.invalid/client/v1/next/#offer=of-1&code=K7Q2-9FZD&fp=3b1f"
        );
        offer
            .link
            .push_str("&remote=iroh%3A%2F%2Fk%40relay.example.invalid");
        assert!(
            offer
                .web_link("https://d.example.invalid", "/client/v1/next/")
                .ends_with("&fp=3b1f&remote=iroh%3A%2F%2Fk%40relay.example.invalid")
        );
        offer.public_url = Some("http://192.168.1.20:8788".into());
        assert!(
            offer
                .page_link()
                .starts_with("http://192.168.1.20:8788/client/v1/pair#")
        );
    }

    #[test]
    fn unknown_event_types_and_fields_are_ignored() {
        let frame: ServerFrame = serde_json::from_str(
            r#"{"type":"event","seq":3,"at_ms":1,"event":{"type":"from_the_future","x":1},"later":true}"#,
        )
        .unwrap();
        assert!(matches!(
            frame,
            ServerFrame::Event {
                seq: Some(3),
                event: Event::Unknown,
                ..
            }
        ));
    }

    #[test]
    fn versions_with_one_major_are_compatible() {
        let v = ProtocolVersion { major: 1, minor: 7 };
        assert!(ProtocolVersion::CURRENT.compatible_with(v));
        assert!(!ProtocolVersion::CURRENT.compatible_with(ProtocolVersion { major: 2, minor: 0 }));
    }

    #[test]
    fn view_actions_are_ephemeral_and_unknown_kinds_are_ignored() {
        let event = Event::ViewAction {
            action: ViewAction::NewSession,
            reason: String::new(),
        };
        assert!(!event.is_durable());
        let action: ViewAction =
            serde_json::from_str(r#"{"kind":"open_the_pod_bay_doors"}"#).unwrap();
        assert_eq!(action, ViewAction::Unknown);
        let pane: ViewAction =
            serde_json::from_str(r#"{"kind":"focus_pane","pane":"holodeck"}"#).unwrap();
        assert_eq!(
            pane,
            ViewAction::FocusPane {
                pane: Pane::Unknown
            }
        );
    }

    /// A run outcome added later (`missed`, protocol 1.28) still decodes
    /// in a client that predates it, as `Unknown`: the frame is never lost.
    #[test]
    fn a_missed_run_decodes_and_an_unknown_outcome_is_kept() {
        let run: AutomationRun =
            serde_json::from_str(r#"{"started_ms":1,"outcome":"missed"}"#).unwrap();
        assert_eq!(run.outcome, RunOutcome::Missed);
        // What a client built before 1.28 does with it: its enum had no
        // `missed`, so the `#[serde(other)]` arm takes it.
        #[derive(Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum OlderOutcome {
            Done,
            #[serde(other)]
            Unknown,
        }
        let older: OlderOutcome = serde_json::from_str(r#""missed""#).unwrap();
        assert!(matches!(older, OlderOutcome::Unknown));
        assert!(matches!(
            serde_json::from_str::<OlderOutcome>(r#""done""#).unwrap(),
            OlderOutcome::Done
        ));
        let later: RunOutcome = serde_json::from_str(r#""postponed""#).unwrap();
        assert_eq!(later, RunOutcome::Unknown);
    }

    #[test]
    fn risks_are_ordered() {
        assert!(
            Risk::Low < Risk::Medium && Risk::Medium < Risk::High && Risk::High < Risk::Critical
        );
    }

    #[test]
    fn only_turn_streaming_is_ephemeral() {
        assert!(
            !Event::TurnStatus {
                turn_id: "t".into(),
                status: TurnStatus::Thinking
            }
            .is_durable()
        );
        assert!(
            Event::AssistantMessage {
                attachments: Vec::new(),
                session_id: "s".into(),
                text: "x".into(),
                suggestions: Vec::new(),
            }
            .is_durable()
        );
    }
}
