//! # aidaemon-client-protocol
//!
//! The wire types of the AIdaemon Client Protocol v1, spoken between the
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

/// The major version this crate speaks.
pub const PROTOCOL_MAJOR: u16 = 1;
/// The minor version this crate speaks.
pub const PROTOCOL_MINOR: u16 = 2;

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
    /// Files with messages, through upload slots.
    Attachments,
    /// Push notifications through a relay.
    Push,
    /// The agent may move this connection's view (`view_action` events):
    /// open a session, show an approval or a piece of work, focus a pane.
    /// Sent only to the connection whose message started the turn, never
    /// replayed, and never able to answer, pair, send or stop anything.
    ViewActions,
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
    },
    /// Stop everything the session has running (`/cancel all`).
    StopTurn { session_id: String },
    ResolveApproval {
        approval_id: String,
        decision: ApprovalDecision,
    },
    /// Stop one entry of the running-work list.
    CancelTask { task_id: String },
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
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

fn default_limit() -> u32 {
    50
}

/// A file a client uploaded before sending its message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct AttachmentRef {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
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
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

impl Event {
    /// Whether the event is kept in the log and replayed on resume.
    pub fn is_durable(&self) -> bool {
        !matches!(
            self,
            Event::TurnDelta { .. } | Event::TurnStatus { .. } | Event::ViewAction { .. }
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
            Event::GoalUpdated { .. } => "goal_updated",
            Event::ViewAction { .. } => "view_action",
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
    ToolStarted {
        name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
    },
    ToolProgress {
        name: String,
        chunk: String,
    },
    ToolCompleted {
        name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        summary: String,
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
}

/// One message of a session's transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct HistoryMessage {
    pub id: String,
    /// `user` or `assistant`.
    pub role: String,
    pub text: String,
    pub at_ms: u64,
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
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Every scope: what a phone or computer gets by default.
pub const ALL_SCOPES: [Scope; 3] = [Scope::Observe, Scope::Chat, Scope::Approve];

/// The owner's offer to pair a device (owner only:
/// `POST /client/v1/pairing/offers`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct OfferRequest {
    /// The chat where the owner confirms the device; `None` to confirm
    /// with `POST /client/v1/pairing/offers/{id}/decision`.
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct Challenge {
    pub challenge_id: String,
    pub nonce: String,
    pub expires_at_ms: u64,
}

/// The exact bytes a device signs (ECDSA P-256, SHA-256) for a challenge.
pub fn challenge_message(challenge_id: &str, nonce: &str) -> String {
    format!("aidaemon-client-v1:{challenge_id}:{nonce}")
}

/// `POST /client/v1/sessions`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SessionRequest {
    pub challenge_id: String,
    /// The signature as `r || s` (64 bytes), base64url.
    pub signature: String,
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
    let _ = generator.subschema_for::<AccessGrant>();
    let _ = generator.subschema_for::<Vec<DeviceInfo>>();
    let mut defs = serde_json::Value::Object(generator.take_definitions(true));
    drop_empty_alternatives(&mut defs);
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aidaemon.ai/schemas/client-protocol/v1.schema.json",
        "title": "AIdaemon Client Protocol v1",
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
                session_id: "s".into(),
                text: "x".into()
            }
            .is_durable()
        );
    }
}
