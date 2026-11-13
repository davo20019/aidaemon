// What the client knows, built only from server frames: sessions, each
// session's transcript, the approvals waiting, the work running, and the
// last `seq` applied. No DOM here: the same reducer runs in the browser, in
// the desktop shell and under Node for the shared fixtures
// (crates/aidaemon-client-core/tests/fixtures/reducer), which hold it to the
// Rust reducer in aidaemon-client-core. Its JSON shape is the Rust `State`'s.

import type {
  Approval,
  CommandResult,
  Event,
  HistoryMessage,
  ServerFrame,
  SessionSummary,
  Snapshot,
  TaskSummary,
  TurnStatus,
  ViewAction,
} from "./protocol";

export type Line =
  | { kind: "user"; text: string; origin: string }
  | { kind: "assistant"; text: string }
  | { kind: "note"; text: string };

/** A turn running now: its streamed text and its latest status. */
export interface LiveTurn {
  turn_id: string;
  streamed: string;
  status: string | null;
}

export interface SessionView {
  summary: SessionSummary | null;
  lines: Line[];
  live: LiveTurn | null;
}

export interface State {
  /** The last durable `seq` applied. */
  last_seq: number | null;
  sessions: Record<string, SessionView>;
  /** Session ids, newest activity first, as the list shows them. */
  order: string[];
  approvals: Record<string, Approval>;
  tasks: Record<string, TaskSummary>;
  /** The latest error or notice from the gateway. */
  notice: string | null;
  /**
   * The latest view move the agent asked this client for, until the screen
   * takes it (takeViewRequest). Never touches approvals, work or transcripts.
   */
  view_request: ViewRequest | null;
}

export interface ViewRequest {
  action: ViewAction;
  reason: string;
}

/** The view actions this client knows; any other kind is ignored. */
const VIEW_KINDS = new Set(["open_session", "show_approval", "show_task", "focus_pane", "new_session", "show_devices"]);

export function newState(): State {
  return { last_seq: null, sessions: {}, order: [], approvals: {}, tasks: {}, notice: null, view_request: null };
}

/** A turn status in a few words. */
export function statusText(status: TurnStatus): string {
  switch (status.kind) {
    case "thinking":
      return "Thinking...";
    case "tool_started":
      return status.summary ? `${status.name}: ${status.summary}...` : `${status.name}...`;
    case "tool_progress":
      return `${status.name}: ${[...status.chunk].slice(0, 80).join("")}`;
    case "tool_completed":
      return `✓ ${status.name} ${status.summary ?? ""}`;
    case "plan_step":
      return `Step ${status.step + 1} of ${status.total}: ${status.description}`;
    case "checklist":
      return "Updated the plan";
    case "note":
      return status.text;
  }
  return "Working...";
}

function session(state: State, id: string): SessionView {
  let view = state.sessions[id];
  if (!view) {
    state.order.unshift(id);
    view = state.sessions[id] = { summary: null, lines: [], live: null };
  }
  return view;
}

/** Move `id` to the top of the list. */
function touch(state: State, id: string) {
  session(state, id);
  state.order = state.order.filter((other) => other !== id);
  state.order.unshift(id);
}

function reset(state: State, seq: number, snapshot: Snapshot) {
  const transcripts = state.sessions;
  state.sessions = {};
  state.order = [];
  state.approvals = {};
  state.tasks = {};
  for (const summary of snapshot.sessions) {
    const old = transcripts[summary.id];
    state.sessions[summary.id] = { summary, lines: old ? old.lines : [], live: null };
    state.order.push(summary.id);
  }
  for (const approval of snapshot.pending_approvals) state.approvals[approval.id] = approval;
  for (const task of snapshot.running_tasks) state.tasks[task.id] = task;
  state.last_seq = seq;
}

/**
 * Apply one frame from the gateway. Durable events at or below the last
 * `seq` applied are ignored, so a replay never doubles a line.
 */
export function apply(state: State, frame: ServerFrame): void {
  switch (frame.type) {
    case "snapshot":
      reset(state, frame.seq, frame.state);
      return;
    case "event": {
      const seq = frame.seq;
      if (seq != null) {
        if (state.last_seq != null && seq <= state.last_seq) return;
        state.last_seq = seq;
      }
      applyEvent(state, frame.session_id ?? null, frame.event);
      return;
    }
    case "ack":
      applyResult(state, frame.result);
      return;
    case "error":
      state.notice = frame.error.message;
      return;
  }
}

function applyResult(state: State, result: CommandResult) {
  switch (result.type) {
    case "snapshot":
      reset(state, result.seq, result.state);
      return;
    case "sessions":
      for (const summary of result.sessions) session(state, summary.id).summary = summary;
      return;
    case "history":
      setHistory(state, result.session_id, result.messages);
      return;
    case "stopped":
      state.notice = result.report;
      return;
    case "reply":
      state.notice = result.text;
      return;
  }
}

function setHistory(state: State, sessionId: string, messages: HistoryMessage[]) {
  const view = session(state, sessionId);
  // History is older than anything live: it goes first.
  const lines: Line[] = messages.map((m) =>
    m.role === "user" ? { kind: "user", text: m.text, origin: "" } : { kind: "assistant", text: m.text },
  );
  view.lines = lines.concat(view.lines);
}

function liveTurn(state: State, turnId: string): SessionView | undefined {
  return Object.values(state.sessions).find((view) => view.live?.turn_id === turnId);
}

function applyEvent(state: State, sessionId: string | null, event: Event) {
  switch (event.type) {
    case "session_created":
    case "session_updated":
      touch(state, event.session.id);
      session(state, event.session.id).summary = event.session;
      return;
    case "user_message":
      touch(state, event.session_id);
      session(state, event.session_id).lines.push({ kind: "user", text: event.text, origin: event.origin });
      return;
    case "assistant_message":
      session(state, event.session_id).lines.push({ kind: "assistant", text: event.text });
      return;
    case "turn_started":
      session(state, event.session_id).live = { turn_id: event.turn_id, streamed: "", status: null };
      return;
    case "turn_delta": {
      const view = liveTurn(state, event.turn_id);
      if (view?.live) view.live.streamed += event.text;
      return;
    }
    case "turn_status": {
      const view = liveTurn(state, event.turn_id);
      if (view?.live) view.live.status = statusText(event.status);
      return;
    }
    case "user_action_required":
      if (sessionId != null) session(state, sessionId).lines.push({ kind: "note", text: event.message });
      return;
    case "turn_completed": {
      const view = liveTurn(state, event.turn_id);
      if (!view) {
        // Its start was before this client's window.
        if (sessionId != null && event.text) {
          session(state, sessionId).lines.push({ kind: "assistant", text: event.text });
        }
        return;
      }
      view.live = null;
      if (event.text) view.lines.push({ kind: "assistant", text: event.text });
      if (event.state === "cancelled") view.lines.push({ kind: "note", text: "Stopped." });
      if (event.state === "failed") view.lines.push({ kind: "note", text: `Failed: ${event.error ?? ""}` });
      return;
    }
    case "approval_requested":
      state.approvals[event.approval.id] = event.approval;
      return;
    case "approval_resolved":
      delete state.approvals[event.approval_id];
      return;
    case "view_action":
      // Kinds this client does not know are ignored.
      if (VIEW_KINDS.has(event.action.kind)) state.view_request = { action: event.action, reason: event.reason ?? "" };
      return;
    case "task_updated":
      if (event.task.state === "running" || event.task.state === "cancelling") {
        state.tasks[event.task.id] = event.task;
      } else {
        delete state.tasks[event.task.id];
      }
      return;
  }
}

/** The view move asked for, once. */
export function takeViewRequest(state: State): ViewRequest | null {
  const request = state.view_request;
  state.view_request = null;
  return request;
}

/** The oldest approval waiting, which the card shows. */
export function nextApproval(state: State): Approval | undefined {
  return Object.values(state.approvals).sort((a, b) => a.requested_at_ms - b.requested_at_ms)[0];
}

/** Running tasks, oldest first. */
export function running(state: State): TaskSummary[] {
  return Object.values(state.tasks).sort((a, b) => a.started_at_ms - b.started_at_ms);
}

function short(id: string): string {
  const chars = [...id];
  return chars.length > 24 ? `${chars.slice(0, 23).join("")}…` : id;
}

export function title(state: State, id: string): string {
  const summary = state.sessions[id]?.summary;
  if (summary?.title) return summary.title;
  if (summary) return `${short(id)} (${summary.channel})`;
  return short(id);
}
