/* Generated from schemas/client-protocol/v1.schema.json by `npm run gen`. Do not edit. */

/**
 * How much is at stake, from the daemon's typed rating. Ordered.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Risk".
 */
export type Risk = "low" | "medium" | "high" | "critical";
/**
 * What a device's token may do.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Scope".
 */
export type Scope = "observe" | "chat" | "approve";
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "CardAction".
 */
export type CardAction =
  | {
      kind: "command";
      line: string;
      /**
       * Folders a read-only command only looks inside.
       */
      looks_inside?: string[];
      title?: string | null;
    }
  | {
      kind: "open_local_site";
      once_minutes: number;
      site: string;
    }
  | {
      detail?: string | null;
      kind: "use";
      what: string;
    };
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Impact".
 */
export type Impact =
  | {
      kind: "deletes";
      targets?: string[];
    }
  | {
      kind: "sends";
    }
  | {
      kind: "spends";
    }
  | {
      kind: "writes_outside_work_area";
    };
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "AskReason".
 */
export type AskReason =
  | {
      check_says?: string | null;
      kind: "read_outside_content";
    }
  | {
      kind: "after_a_decline";
    };
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ApprovalKind".
 */
export type ApprovalKind = "command" | "goal_confirmation" | "autopilot_confirmation";
/**
 * An answer to an approval.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ApprovalDecision".
 */
export type ApprovalDecision = ("approve_once" | "deny") | "approve_session" | "approve_always";
/**
 * How an approval ended.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ApprovalOutcome".
 */
export type ApprovalOutcome =
  ("approved_once" | "approved_session" | "approved_always" | "denied") | "expired" | "withdrawn";
/**
 * An optional feature, named instead of a version bump. A client lists the
 * ones it wants in `hello`; the server lists the ones it offers in
 * `welcome`, and only those it offers apply.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Capability".
 */
export type Capability = "compact" | "deltas" | "goals" | "attachments" | "push" | "view_actions";
/**
 * A frame from a client to the daemon.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ClientFrame".
 */
export type ClientFrame =
  | {
      capabilities?: Capability[];
      client: ClientInfo;
      /**
       * The last `seq` the client applied; `None` for a client with no
       * cache. Within the retention window the gateway replays every
       * event after it, otherwise it sends a snapshot.
       */
      resume_from?: number | null;
      /**
       * The access token, for clients that cannot set an
       * `Authorization` header on a WebSocket (browsers).
       */
      token?: string | null;
      type: "hello";
      version: ProtocolVersion;
    }
  | {
      command: Command;
      request_id: string;
      type: "command";
    }
  | {
      type: "ping";
    };
/**
 * The kind of client, so the daemon can tailor what it sends and the owner
 * can see which device acted.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ClientKind".
 */
export type ClientKind = ("tui" | "web" | "desktop" | "mobile") | "watch" | "script";
/**
 * What a client asks the daemon to do.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Command".
 */
export type Command =
  | {
      /**
       * Files uploaded beforehand; needs the `attachments` capability.
       */
      attachments?: AttachmentRef[];
      session_id?: string | null;
      text: string;
      type: "send_message";
    }
  | {
      session_id: string;
      type: "stop_turn";
    }
  | {
      approval_id: string;
      decision: ApprovalDecision;
      type: "resolve_approval";
    }
  | {
      task_id: string;
      type: "cancel_task";
    }
  | {
      cursor?: string | null;
      limit?: number;
      type: "list_sessions";
    }
  | {
      /**
       * Messages before this one (its id); `None` for the newest.
       */
      before?: string | null;
      limit?: number;
      session_id: string;
      type: "get_history";
    }
  | {
      session_id: string;
      type: "clear_session";
    }
  | {
      args?: string;
      /**
       * Without the slash ("tasks").
       */
      command: string;
      session_id: string;
      type: "run_slash";
    }
  | {
      type: "get_glance";
    }
  | {
      type: "get_snapshot";
    };
/**
 * The answer to a command.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "CommandResult".
 */
export type CommandResult =
  | {
      /**
       * The message started a session.
       */
      new_session?: boolean;
      session_id: string;
      type: "message_accepted";
    }
  | {
      report: string;
      type: "stopped";
    }
  | {
      approval_id: string;
      outcome: ApprovalOutcome;
      type: "approval_resolved";
    }
  | {
      next_cursor?: string | null;
      sessions: SessionSummary[];
      type: "sessions";
    }
  | {
      messages: HistoryMessage[];
      /**
       * More, older messages exist.
       */
      more?: boolean;
      session_id: string;
      type: "history";
    }
  | {
      text: string;
      type: "reply";
    }
  | {
      glance: Glance;
      type: "glance";
    }
  | {
      seq: number;
      state: Snapshot;
      type: "snapshot";
    };
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "TaskState".
 */
export type TaskState = "running" | "cancelling" | "completed" | "failed" | "cancelled";
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "EnrollmentState".
 */
export type EnrollmentState = ("approved" | "denied" | "expired") | "open" | "pending";
/**
 * The closed set of error codes.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ErrorCode".
 */
export type ErrorCode =
  | ("unauthorized" | "version_unsupported" | "not_found" | "rate_limited" | "invalid" | "internal")
  | "forbidden"
  | "conflict";
/**
 * Something that happened. Durable unless noted.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Event".
 */
export type Event =
  | {
      session: SessionSummary;
      type: "session_created";
    }
  | {
      session: SessionSummary;
      type: "session_updated";
    }
  | {
      attachments?: AttachmentRef[];
      /**
       * Where it was said: `app`, `telegram`, `slack`...
       */
      origin: string;
      session_id: string;
      text: string;
      type: "user_message";
    }
  | {
      session_id: string;
      text: string;
      type: "assistant_message";
    }
  | {
      origin: string;
      session_id: string;
      turn_id: string;
      type: "turn_started";
    }
  | {
      text: string;
      turn_id: string;
      type: "turn_delta";
    }
  | {
      status: TurnStatus;
      turn_id: string;
      type: "turn_status";
    }
  | {
      message: string;
      turn_id: string;
      type: "user_action_required";
    }
  | {
      attachments?: AttachmentRef[];
      error?: string | null;
      state: TurnState;
      /**
       * The final reply, as Markdown; empty when there was none.
       */
      text: string;
      turn_id: string;
      type: "turn_completed";
    }
  | {
      approval: Approval;
      type: "approval_requested";
    }
  | {
      approval_id: string;
      outcome: ApprovalOutcome;
      /**
       * Who answered; `None` when nobody did (expired, withdrawn).
       */
      resolved_by?: ResolvedBy | null;
      type: "approval_resolved";
    }
  | {
      task: TaskSummary;
      type: "task_updated";
    }
  | {
      goal: GoalSummary;
      type: "goal_updated";
    }
  | {
      action: ViewAction;
      /**
       * Why, in a few plain words, for the note the client shows.
       */
      reason?: string;
      type: "view_action";
    };
/**
 * What a running turn is doing.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "TurnStatus".
 */
export type TurnStatus =
  | {
      kind: "thinking";
    }
  | {
      kind: "tool_started";
      name: string;
      summary?: string;
    }
  | {
      chunk: string;
      kind: "tool_progress";
      name: string;
    }
  | {
      kind: "tool_completed";
      name: string;
      summary?: string;
    }
  | {
      description: string;
      kind: "plan_step";
      state: StepState;
      /**
       * 0-based.
       */
      step: number;
      total: number;
    }
  | {
      kind: "checklist";
      text: string;
    }
  | {
      kind: "note";
      text: string;
    };
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "StepState".
 */
export type StepState = "started" | "completed" | "failed";
/**
 * How a turn ended.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "TurnState".
 */
export type TurnState = ("failed" | "cancelled") | "completed" | "empty" | "handed_off";
/**
 * Where the agent asks a client to take its view. Every variant only
 * changes what is on screen: none carries a decision, a message or a
 * device, so none can answer an approval or act for the owner. Kinds a
 * client does not know read as `Unknown` and are ignored.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ViewAction".
 */
export type ViewAction =
  | {
      kind: "open_session";
      session_id: string;
    }
  | {
      approval_id: string;
      kind: "show_approval";
    }
  | {
      kind: "show_task";
      task_id: string;
    }
  | {
      kind: "focus_pane";
      pane: Pane;
    }
  | {
      kind: "new_session";
    }
  | {
      kind: "show_devices";
    };
/**
 * A part of a client's screen.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Pane".
 */
export type Pane = "sessions" | "chat" | "approvals" | "tasks";
/**
 * A frame from the daemon to a client.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ServerFrame".
 */
export type ServerFrame =
  | {
      /**
       * The newest `seq` in the log when the socket opened.
       */
      latest_seq: number;
      owner: OwnerInfo;
      server_capabilities?: Capability[];
      type: "welcome";
      version: ProtocolVersion;
    }
  | {
      seq: number;
      state: Snapshot;
      type: "snapshot";
    }
  | {
      at_ms: number;
      event: Event;
      /**
       * Per owner, gap-free; `None` for an ephemeral event.
       */
      seq?: number | null;
      session_id?: string | null;
      type: "event";
    }
  | {
      request_id: string;
      result: CommandResult;
      type: "ack";
    }
  | {
      error: ProtocolError;
      /**
       * `None` for an error that answers no command (a bad frame).
       */
      request_id?: string | null;
      type: "error";
    }
  | {
      type: "pong";
    };

/**
 * An index of every protocol type (not a wire value).
 */
export interface ProtocolTypes {
  AccessGrant?: AccessGrant;
  Approval?: Approval;
  ApprovalCard?: ApprovalCard;
  ApprovalDecision?: ApprovalDecision;
  ApprovalKind?: ApprovalKind;
  ApprovalOutcome?: ApprovalOutcome;
  AskReason?: AskReason;
  AttachmentRef?: AttachmentRef;
  Capability?: Capability;
  CardAction?: CardAction;
  Challenge?: Challenge;
  ChallengeRequest?: ChallengeRequest;
  ClientFrame?: ClientFrame;
  ClientInfo?: ClientInfo;
  ClientKind?: ClientKind;
  Command?: Command;
  CommandResult?: CommandResult;
  DeviceInfo?: DeviceInfo;
  Enrollment?: Enrollment;
  EnrollmentState?: EnrollmentState;
  ErrorCode?: ErrorCode;
  Event?: Event;
  EventsPage?: EventsPage;
  Glance?: Glance;
  GoalSummary?: GoalSummary;
  HistoryMessage?: HistoryMessage;
  Impact?: Impact;
  LastReply?: LastReply;
  OfferRequest?: OfferRequest;
  OwnerInfo?: OwnerInfo;
  PairingDecision?: PairingDecision;
  PairingOffer?: PairingOffer;
  Pane?: Pane;
  ProtocolError?: ProtocolError;
  ProtocolVersion?: ProtocolVersion;
  RedeemRequest?: RedeemRequest;
  ResolvedBy?: ResolvedBy;
  Risk?: Risk;
  Scope?: Scope;
  ServerFrame?: ServerFrame;
  SessionRequest?: SessionRequest;
  SessionSummary?: SessionSummary;
  Snapshot?: Snapshot;
  StepState?: StepState;
  TaskState?: TaskState;
  TaskSummary?: TaskSummary;
  TurnState?: TurnState;
  TurnStatus?: TurnStatus;
  ViewAction?: ViewAction;
}
/**
 * A short-lived access token for one device.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "AccessGrant".
 */
export interface AccessGrant {
  access_token: string;
  device_id: string;
  expires_at_ms: number;
  /**
   * The highest risk this device may approve; `None`: no limit.
   */
  risk_ceiling?: Risk | null;
  scopes: Scope[];
}
/**
 * A question the owner must answer before work goes on. Pending until
 * exactly one of: answered, expired (nobody answered in time: the call is
 * not run, and it is not a refusal) or withdrawn (its turn stopped, or the
 * daemon restarted).
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Approval".
 */
export interface Approval {
  /**
   * The card's typed facts, when the asker gave them. Left out in
   * compact frames.
   */
  card?: ApprovalCard | null;
  /**
   * The chat the card is also shown in (`telegram`...), if any besides
   * the app.
   */
  channel?: string | null;
  /**
   * The exact command or arguments. Left out in compact frames.
   */
  detail?: string | null;
  /**
   * When it expires unanswered; `None` when the asker set no deadline.
   */
  expires_at_ms?: number | null;
  id: string;
  kind: ApprovalKind;
  /**
   * The answers offered, in display order.
   */
  options: ApprovalDecision[];
  requested_at_ms: number;
  risk: Risk;
  session_id: string;
  /**
   * One plain sentence: what will happen.
   */
  summary: string;
  warnings?: string[];
}
/**
 * An approval card's content as facts: who asks, what will happen, why the
 * owner is asked and the one effect worth a warning. Plain words only;
 * never authority.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ApprovalCard".
 */
export interface ApprovalCard {
  action: CardAction;
  /**
   * The helper asking; `None` for the assistant itself.
   */
  asked_by?: string | null;
  impact?: Impact | null;
  reason?: AskReason | null;
}
/**
 * A file a client uploaded before sending its message.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "AttachmentRef".
 */
export interface AttachmentRef {
  id: string;
  mime?: string | null;
  name: string;
  size_bytes?: number | null;
}
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Challenge".
 */
export interface Challenge {
  challenge_id: string;
  expires_at_ms: number;
  nonce: string;
}
/**
 * `POST /client/v1/sessions/challenge`.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ChallengeRequest".
 */
export interface ChallengeRequest {
  device_id: string;
}
/**
 * What a client is.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ClientInfo".
 */
export interface ClientInfo {
  /**
   * The device's name as the owner knows it ("Pixel 9").
   */
  device?: string | null;
  kind: ClientKind;
  /**
   * The client program's name ("aidaemon-tui").
   */
  name: string;
  version?: string | null;
}
/**
 * A protocol version. Same major: compatible.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ProtocolVersion".
 */
export interface ProtocolVersion {
  major: number;
  minor: number;
}
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "SessionSummary".
 */
export interface SessionSummary {
  /**
   * The chat it belongs to: `app`, `telegram`...
   */
  channel: string;
  id: string;
  last_activity_ms?: number | null;
  title?: string | null;
  /**
   * Whether clients may post into it. Sessions of other chats are
   * read-mostly in v1: watch, stop and answer approvals there, but post
   * from that chat.
   */
  writable?: boolean;
}
/**
 * One message of a session's transcript.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "HistoryMessage".
 */
export interface HistoryMessage {
  at_ms: number;
  id: string;
  /**
   * `user` or `assistant`.
   */
  role: string;
  text: string;
}
/**
 * A watch's one-call view.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Glance".
 */
export interface Glance {
  last_reply?: LastReply | null;
  pending_approvals: Approval[];
  running_tasks: TaskSummary[];
}
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "LastReply".
 */
export interface LastReply {
  at_ms: number;
  session_id: string;
  text: string;
  turn_id: string;
}
/**
 * One entry of a session's running-work list: a turn, a helper (sub-agent)
 * or a background command.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "TaskSummary".
 */
export interface TaskSummary {
  description: string;
  error?: string | null;
  finished_at_ms?: number | null;
  /**
   * Stable across the daemon's life; `cancel_task` takes it.
   */
  id: string;
  /**
   * The short number `/cancel <n>` takes in the chat.
   */
  number: number;
  session_id: string;
  started_at_ms: number;
  state: TaskState;
}
/**
 * What a client needs when it cannot resume.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Snapshot".
 */
export interface Snapshot {
  pending_approvals: Approval[];
  running_tasks: TaskSummary[];
  sessions: SessionSummary[];
}
/**
 * A paired device (owner only: `GET /client/v1/devices`).
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "DeviceInfo".
 */
export interface DeviceInfo {
  created_at_ms: number;
  device_id: string;
  kind: ClientKind;
  last_seen_ms?: number | null;
  name: string;
  revoked?: boolean;
  risk_ceiling?: Risk | null;
  scopes: Scope[];
}
/**
 * Where a pairing stands (`GET /client/v1/enrollments/{offer_id}`, and
 * the owner's `GET /client/v1/pairing/offers/{offer_id}`).
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "Enrollment".
 */
export interface Enrollment {
  /**
   * Once approved.
   */
  device_id?: string | null;
  device_kind?: ClientKind | null;
  device_name?: string | null;
  fingerprint: string;
  offer_id: string;
  scopes?: Scope[];
  state: EnrollmentState;
}
/**
 * Who answered an approval.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ResolvedBy".
 */
export interface ResolvedBy {
  /**
   * `app` for a client of this protocol, else the chat (`telegram`...).
   */
  channel: string;
  /**
   * The client's device, when `channel` is `app`.
   */
  device?: string | null;
}
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "GoalSummary".
 */
export interface GoalSummary {
  description: string;
  id: string;
  session_id?: string | null;
  status: string;
}
/**
 * `GET /client/v1/events` answers with this: the event frames after
 * `since` (or a snapshot frame first, when `since` is out of the retention
 * window), and the `since` to ask with next.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "EventsPage".
 */
export interface EventsPage {
  frames: ServerFrame[];
  next_since: number;
}
/**
 * The daemon's owner, as the gateway knows them.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "OwnerInfo".
 */
export interface OwnerInfo {
  name?: string | null;
}
/**
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "ProtocolError".
 */
export interface ProtocolError {
  code: ErrorCode;
  message: string;
  /**
   * Whether the same request may succeed later.
   */
  retryable: boolean;
}
/**
 * The owner's offer to pair a device (owner only:
 * `POST /client/v1/pairing/offers`).
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "OfferRequest".
 */
export interface OfferRequest {
  /**
   * The chat where the owner confirms the device; `None` to confirm
   * with `POST /client/v1/pairing/offers/{id}/decision`.
   */
  confirm_in_session?: string | null;
}
/**
 * The owner's answer to a pending pairing made without a chat.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "PairingDecision".
 */
export interface PairingDecision {
  approve: boolean;
}
/**
 * A single-use pairing offer.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "PairingOffer".
 */
export interface PairingOffer {
  /**
   * What the owner types on the device ("K7Q2-9FZD").
   */
  code: string;
  expires_at_ms: number;
  /**
   * Identifies this daemon; the device checks the redeem reply matches.
   */
  fingerprint: string;
  /**
   * The same as a link, for a QR code:
   * `aidaemon-pair:?gateway=<url>&offer=<id>&code=<code>&fp=<fingerprint>`.
   */
  link: string;
  offer_id: string;
  /**
   * Where devices reach the gateway now (a live tunnel origin, the
   * configured address, or loopback). Absent from older daemons.
   */
  public_url?: string | null;
}
/**
 * `POST /client/v1/enrollments/redeem`.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "RedeemRequest".
 */
export interface RedeemRequest {
  code: string;
  device_kind: ClientKind;
  /**
   * The device as the owner will see it ("Pixel 9").
   */
  device_name: string;
  offer_id: string;
  /**
   * The device's P-256 public key, SEC1 uncompressed, base64url.
   */
  public_key: string;
  /**
   * The scopes asked for; empty: all of them.
   */
  scopes?: Scope[];
}
/**
 * `POST /client/v1/sessions`.
 *
 * This interface was referenced by `ProtocolTypes`'s JSON-Schema
 * via the `definition` "SessionRequest".
 */
export interface SessionRequest {
  challenge_id: string;
  /**
   * The signature as `r || s` (64 bytes), base64url.
   */
  signature: string;
}
