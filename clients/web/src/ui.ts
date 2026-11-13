// The screen: sessions on the left, the chat in the middle with the oldest
// waiting approval as a card above the composer, running work on the right,
// and one palette (Ctrl/Cmd+K) for sessions and commands. The composer owns
// the keyboard; the common actions are one key away: Enter sends, Esc stops
// the session's work, y/s/a/n answer the card (only with the composer empty
// and the card on screen long enough to have been read), Alt+N starts a
// session. The status row always says what state the connection is in.
//
// The agent may move the view (`view_action` events, capability
// `view_actions`): open a session, bring a card up, show a piece of work,
// focus a pane. Each move shows a one-line note of what moved and why, with
// Back (Alt+B); Esc or the next key dismisses it. While the owner has text
// in the composer a move is only offered (Go, Alt+G), so a message never
// lands in a session the agent picked. A move never answers anything: a
// card it brings up is armed afresh, and it can only ever ask for history.

import type { Approval, ApprovalDecision, Command, ServerFrame, ViewAction } from "./protocol";
import type { Connection, LinkState } from "./link.js";
import type { Platform } from "./platform.js";
import {
  apply as reduce,
  newState,
  nextApproval,
  running,
  takeViewRequest,
  title as titleOf,
  type State,
  type ViewRequest,
} from "./state.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** How long a card must be on screen, unchanged, before a key answers it. */
export const ARM_AFTER_MS = 600;

const DECISIONS: Record<ApprovalDecision, { key: string; label: string }> = {
  approve_once: { key: "y", label: "Allow once" },
  approve_session: { key: "s", label: "Allow this session" },
  approve_always: { key: "a", label: "Always allow" },
  deny: { key: "n", label: "Deny" },
};

/** Slash commands the client handles itself; the rest run in the chat. */
const LOCAL_SLASH: Record<string, string> = {
  new: "start a new session",
  stop: "stop this session's work",
  sessions: "switch session",
  devices: "list paired devices",
  help: "show the keys",
};

function el(tag: string, props: Record<string, string> = {}, ...children: (Node | string)[]) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) node.setAttribute(k, v);
  node.append(...children);
  return node;
}

/** Where the view was before a move, for Back. */
interface Back {
  selected: string | null;
  shownApproval: string | null;
}

interface Note {
  text: string;
  back: Back | null;
  /** A move offered while the owner typed, taken with Go. */
  held: ViewRequest | null;
  since: number;
}

/** How long a note of a view move stays up. */
const NOTE_FOR_MS = 15_000;

interface PaletteItem {
  label: string;
  hint?: string;
  run(): void;
}

export class Screen {
  readonly state: State = newState();
  selected: string | null = null;
  private awaitingNew = false;
  private fetched = new Set<string>();
  private link: LinkState = { phase: "connecting" };
  private connection: Connection | null = null;
  /** The card on screen and since when, for arming its keys. */
  private card: { id: string; since: number } | null = null;
  private paletteIndex = 0;
  private flash: string | null = null;
  private note: Note | null = null;
  /** The approval a view move brought up, shown instead of the oldest. */
  private shownApproval: string | null = null;

  constructor(private platform: Platform) {
    this.state.last_seq = platform.loadSeq();
  }

  // --- Wiring ------------------------------------------------------------------

  async start() {
    this.bind();
    if (await this.platform.isPaired()) this.connect();
    else this.showPairing();
  }

  private connect() {
    $("pair").hidden = true;
    $("app").hidden = false;
    $("unpair").hidden = false;
    this.connection?.stop();
    this.connection = this.platform.connection(["view_actions"]);
    this.connection.start(
      {
        onState: (state) => this.linkState(state),
        onFrame: (frame) => this.frame(frame),
      },
      () => this.state.last_seq,
    );
    this.render();
    $("text").focus();
  }

  private linkState(state: LinkState) {
    this.link = state;
    if (state.phase === "connected") this.send({ type: "list_sessions", limit: 200 });
    if (state.phase === "not_paired") {
      this.connection?.stop();
      void this.platform.unpair().then(() => this.showPairing(state.reason));
      return;
    }
    this.render();
  }

  send(command: Command) {
    if (!this.connection || this.link.phase !== "connected") {
      this.say("Not connected yet: it is sent once the daemon answers.");
    }
    this.connection?.send(command);
  }

  frame(frame: ServerFrame) {
    const notice = this.state.notice;
    const waiting = new Set(Object.keys(this.state.approvals));
    reduce(this.state, frame);
    if (frame.type === "ack" && frame.result.type === "message_accepted" && frame.result.new_session && this.awaitingNew) {
      this.awaitingNew = false;
      this.selected = frame.result.session_id;
      this.fetched.add(frame.result.session_id);
    }
    this.platform.saveSeq(this.state.last_seq);
    const request = takeViewRequest(this.state);
    if (request) this.viewRequest(request);
    if (this.state.notice && this.state.notice !== notice) this.say(this.state.notice);
    for (const approval of Object.values(this.state.approvals)) {
      if (!waiting.has(approval.id)) this.platform.notify("Approval needed", approval.summary);
    }
    this.render();
  }

  private say(text: string) {
    this.flash = text;
    this.renderStatus();
  }

  // --- View moves --------------------------------------------------------------

  /** The card on screen: the one a move brought up while it waits, else the oldest. */
  shown(): Approval | undefined {
    return (this.shownApproval ? this.state.approvals[this.shownApproval] : undefined) ?? nextApproval(this.state);
  }

  private describe(action: ViewAction): string {
    switch (action.kind) {
      case "open_session":
        return `open ${this.title(action.session_id)}`;
      case "show_approval": {
        const approval = this.state.approvals[action.approval_id];
        return approval ? `the approval "${approval.summary}"` : "an approval that is no longer waiting";
      }
      case "show_task": {
        const task = this.state.tasks[action.task_id];
        return task ? `work #${task.number} ${task.description}` : "work that is no longer running";
      }
      case "focus_pane":
        return `the ${action.pane} pane`;
      case "new_session":
        return "a new session";
      case "show_devices":
        return "your paired devices";
    }
  }

  /** A move the agent asked for: taken now, or offered while the owner types. */
  viewRequest(request: ViewRequest) {
    const input = $("text") as HTMLTextAreaElement;
    const because = request.reason.trim() ? ` (${request.reason.trim().replace(/\.$/, "")})` : "";
    if (input.value || !$("palette").hidden) {
      this.note = { text: `The assistant suggests: ${this.describe(request.action)}${because}.`, back: null, held: request, since: Date.now() };
      return this.renderNote();
    }
    this.moveView(request, because);
  }

  private moveView(request: ViewRequest, because: string) {
    const back: Back = { selected: this.selected, shownApproval: this.shownApproval };
    const what = this.describe(request.action);
    let text = `Moved to ${what}`;
    const action = request.action;
    switch (action.kind) {
      case "open_session":
        this.select(action.session_id);
        break;
      case "show_approval":
        if (this.state.approvals[action.approval_id]) {
          this.shownApproval = action.approval_id;
          // A card a move brings up is armed afresh.
          this.card = { id: action.approval_id, since: Date.now() };
          text = `Showing ${what}: read it, then y / n`;
          $("cards").scrollIntoView({ block: "nearest" });
        } else {
          text = `Could not show ${what}`;
        }
        break;
      case "show_task": {
        const row = document.querySelector(`[data-task="${CSS.escape(action.task_id)}"]`);
        if (row) {
          row.classList.add("flash");
          row.scrollIntoView({ block: "nearest" });
        } else {
          text = `Could not show ${what}`;
        }
        break;
      }
      case "focus_pane":
        if (action.pane === "sessions") ($("sessionlist").querySelector("button") ?? $("text")).focus();
        else if (action.pane === "tasks") ($("tasklist").querySelector("button") ?? $("text")).focus();
        else $("text").focus();
        if (action.pane === "approvals") text = "The approval card is above the composer";
        break;
      case "new_session":
        this.select(null);
        break;
      case "show_devices": {
        // Listing them is the owner's own command: one Enter away.
        const input = $("text") as HTMLTextAreaElement;
        if (!input.value) input.value = "/devices";
        input.focus();
        text = "Press Enter to list your paired devices";
        break;
      }
      default:
        return;
    }
    this.note = { text: `${text}${because}.`, back, held: null, since: Date.now() };
    this.render();
  }

  private goBack() {
    const back = this.note?.back;
    if (!back) return;
    this.note = null;
    this.shownApproval = back.shownApproval;
    this.select(back.selected);
  }

  private takeHeld() {
    const held = this.note?.held;
    if (!held) return;
    this.note = null;
    this.moveView(held, held.reason.trim() ? ` (${held.reason.trim()})` : "");
  }

  private renderNote() {
    const note = this.note;
    $("note").hidden = !note;
    if (!note) return;
    $("notetext").textContent = note.text;
    const button = $("noteback") as HTMLButtonElement;
    button.hidden = !note.back && !note.held;
    button.textContent = note.held ? "Go (Alt+G)" : "Back (Alt+B)";
    button.onclick = () => (note.held ? this.takeHeld() : this.goBack());
    $("noteclose").onclick = () => this.dismissNote();
  }

  // --- Actions -----------------------------------------------------------------

  select(id: string | null) {
    this.selected = id;
    if (id && !this.fetched.has(id)) {
      this.fetched.add(id);
      this.send({ type: "get_history", session_id: id, limit: 100 });
    }
    this.render();
  }

  newSession() {
    this.select(null);
    $("text").focus();
  }

  /** Stop the selected session's work; whether there was any to stop. */
  stop(): boolean {
    const id = this.selected;
    if (!id) return false;
    const busy = !!this.state.sessions[id]?.live || Object.values(this.state.tasks).some((t) => t.session_id === id);
    if (!busy) return false;
    this.send({ type: "stop_turn", session_id: id });
    return true;
  }

  /** The card on screen, when its keys may answer it now. */
  armedCard(now = Date.now()): Approval | undefined {
    const approval = this.shown();
    if (!approval || !this.card || this.card.id !== approval.id) return undefined;
    return now - this.card.since >= ARM_AFTER_MS ? approval : undefined;
  }

  answer(approval: Approval, decision: ApprovalDecision) {
    this.send({ type: "resolve_approval", approval_id: approval.id, decision });
  }

  private submit() {
    const input = $("text") as HTMLTextAreaElement;
    const text = input.value.trim();
    if (!text) return;
    if (text.startsWith("/")) {
      const [name, ...rest] = text.slice(1).split(" ");
      input.value = "";
      return this.slash(name, rest.join(" "));
    }
    const writable = !this.selected || this.state.sessions[this.selected]?.summary?.writable !== false;
    if (!writable) return this.say("That session belongs to another chat: reply there, or Alt+N for a new one.");
    this.awaitingNew = !this.selected;
    this.send({ type: "send_message", session_id: this.selected ?? undefined, text });
    input.value = "";
    this.renderStatus();
  }

  private slash(name: string, args: string) {
    switch (name) {
      case "new":
        return this.newSession();
      case "stop":
        if (!this.stop()) this.say("Nothing is running in this session.");
        return;
      case "sessions":
        return this.openPalette();
      case "help":
        return this.say(
          "Enter send · Shift+Enter new line · Esc stop · y/s/a/n answer the card · Ctrl+K sessions and commands · Alt+N new session",
        );
    }
    const session = this.selected ?? this.state.order[0];
    if (!session) return this.say("Open a session first: commands run in a session.");
    this.send({ type: "run_slash", session_id: session, command: name, args });
  }

  // --- Palette -------------------------------------------------------------------

  private paletteItems(): PaletteItem[] {
    const items: PaletteItem[] = [
      { label: "New session", hint: "Alt+N", run: () => this.newSession() },
      { label: "Stop this session's work", hint: "Esc", run: () => void (this.stop() || this.say("Nothing is running in this session.")) },
    ];
    const start = this.platform.startDaemon?.bind(this.platform);
    if (start && this.link.phase === "reconnecting" && this.link.kind === "daemon_not_running") {
      items.unshift({
        label: "Start the daemon",
        hint: "service manager",
        run: () => void start().then((text) => this.say(text), (error) => this.say(String(error))),
      });
    }
    const approval = nextApproval(this.state);
    if (approval) {
      items.push({
        label: `Go to the waiting approval: ${approval.summary}`,
        run: () => this.select(approval.session_id),
      });
    }
    items.push(
      { label: "Show paired devices", run: () => this.slash("devices", "") },
      {
        label: "Turn on notifications",
        run: () => void (typeof Notification !== "undefined" && Notification.requestPermission()),
      },
      { label: this.platform.name === "desktop" ? "Unpair this app" : "Unpair this browser", run: () => void this.unpair() },
    );
    for (const id of this.state.order) {
      items.push({ label: titleOf(this.state, id), hint: this.state.sessions[id]?.summary?.channel ?? "", run: () => this.select(id) });
    }
    return items;
  }

  private filtered(): PaletteItem[] {
    const query = ($("paletteinput") as HTMLInputElement).value.trim().toLowerCase();
    const words = query.split(/\s+/).filter(Boolean);
    return this.paletteItems().filter((item) => {
      const text = `${item.label} ${item.hint ?? ""}`.toLowerCase();
      return words.every((word) => text.includes(word));
    });
  }

  openPalette() {
    $("palette").hidden = false;
    ($("paletteinput") as HTMLInputElement).value = "";
    this.paletteIndex = 0;
    this.renderPalette();
    $("paletteinput").focus();
  }

  closePalette() {
    $("palette").hidden = true;
    $("text").focus();
  }

  private renderPalette() {
    const items = this.filtered();
    this.paletteIndex = Math.max(0, Math.min(this.paletteIndex, items.length - 1));
    $("paletteitems").replaceChildren(
      ...items.map((item, i) => {
        const li = el("li", { class: i === this.paletteIndex ? "on" : "" }, item.label, el("small", {}, item.hint ?? ""));
        li.onclick = () => {
          this.closePalette();
          item.run();
        };
        return li;
      }),
    );
    if (!items.length) $("paletteitems").replaceChildren(el("li", { class: "empty" }, "Nothing matches"));
  }

  // --- Keys ------------------------------------------------------------------------

  private bind() {
    const input = $("text") as HTMLTextAreaElement;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this.submit();
      } else if (e.key === "Tab" && input.value.startsWith("/") && !input.value.includes(" ")) {
        const match = this.completions(input.value)[0];
        if (match) {
          e.preventDefault();
          input.value = `/${match} `;
          this.renderStatus();
        }
      }
    });
    input.addEventListener("input", () => this.renderStatus());
    $("paletteinput").addEventListener("input", () => {
      this.paletteIndex = 0;
      this.renderPalette();
    });
    document.addEventListener("keydown", (e) => this.key(e));
    $("palettebtn").onclick = () => this.openPalette();
    document.querySelector<HTMLButtonElement>("[data-new]")!.onclick = () => this.newSession();
    $("unpair").onclick = () => void this.unpair();
    setInterval(() => this.tick(), 250);
  }

  /** What a key does outside the composer's own editing. */
  key(e: KeyboardEvent) {
    const paletteOpen = !$("palette").hidden;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      return paletteOpen ? this.closePalette() : this.openPalette();
    }
    if (paletteOpen) {
      const items = this.filtered();
      if (e.key === "Escape") return this.closePalette();
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        this.paletteIndex += e.key === "ArrowDown" ? 1 : -1;
        return this.renderPalette();
      }
      if (e.key === "Enter") {
        e.preventDefault();
        const item = items[this.paletteIndex];
        this.closePalette();
        item?.run();
      }
      return;
    }
    if (e.altKey && e.code === "KeyB" && this.note?.back) {
      e.preventDefault();
      return this.goBack();
    }
    if (e.altKey && e.code === "KeyG" && this.note?.held) {
      e.preventDefault();
      return this.takeHeld();
    }
    if (e.altKey && e.code === "KeyN") {
      e.preventDefault();
      return this.newSession();
    }
    const input = $("text") as HTMLTextAreaElement;
    const inField = e.target instanceof HTMLInputElement || (e.target instanceof HTMLTextAreaElement && e.target !== input);
    if (inField) return;
    if (e.key === "Escape") {
      if (this.dismissNote()) return;
    } else if (this.note && !["Shift", "Alt", "Control", "Meta"].includes(e.key)) {
      // Any other key dismisses the note and does what it does.
      this.dismissNote();
    }
    if (e.key === "Escape") {
      if (input.value) return;
      if (this.stop()) e.preventDefault();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey || input.value) return;
    const approval = this.armedCard();
    if (!approval) return;
    const decision = approval.options.find((d) => DECISIONS[d].key === e.key.toLowerCase());
    if (decision) {
      e.preventDefault();
      this.answer(approval, decision);
    }
  }

  /** Dismiss the note of a view move; whether there was one. */
  dismissNote(): boolean {
    if (!this.note) return false;
    this.note = null;
    this.renderNote();
    return true;
  }

  private completions(text: string): string[] {
    const typed = text.slice(1).toLowerCase();
    return Object.keys(LOCAL_SLASH).filter((name) => name.startsWith(typed));
  }

  private tick() {
    // A card becomes armed after it has been read: show its keys as live.
    const approval = this.shown();
    if (approval && this.card?.id === approval.id && !$("cards").classList.contains("armed") && this.armedCard()) {
      this.renderCard();
    }
    if (this.note && Date.now() - this.note.since > NOTE_FOR_MS) this.dismissNote();
    this.renderStatus();
  }

  private async unpair() {
    this.connection?.stop();
    await this.platform.unpair();
    location.reload();
  }

  // --- Drawing ---------------------------------------------------------------------

  render() {
    this.renderNote();
    this.renderSessions();
    this.renderChat();
    this.renderCard();
    this.renderTasks();
    this.renderStatus();
  }

  private title(id: string) {
    return titleOf(this.state, id);
  }

  private renderSessions() {
    $("sessionlist").replaceChildren(
      ...this.state.order.map((id) => {
        const busy = !!this.state.sessions[id]?.live;
        const b = el("button", { class: id === this.selected ? "on" : "" }, `${busy ? "● " : ""}${this.title(id)}`);
        b.onclick = () => this.select(id);
        return b;
      }),
    );
  }

  private renderChat() {
    const log = $("log");
    const view = this.selected ? this.state.sessions[this.selected] : undefined;
    $("chattitle").textContent = this.selected ? this.title(this.selected) : "New session";
    const lines = (view?.lines ?? []).map((l) => {
      const who =
        l.kind === "user" ? (l.origin && l.origin !== "app" ? `You (${l.origin})` : "You") : l.kind === "assistant" ? "Assistant" : "";
      return el("p", { class: l.kind === "note" ? "msg note" : "msg" }, el("b", {}, who), l.text);
    });
    if (view?.live) {
      lines.push(
        el("p", { class: "msg live" }, view.live.streamed, view.live.streamed ? "\n" : "", el("span", { class: "spin" }, view.live.status ?? "Working..."), " · Esc to stop"),
      );
    }
    if (!view) {
      const first = this.state.order.length === 0;
      lines.push(
        el(
          "p",
          { class: "empty" },
          first
            ? "Nothing here yet. Type a message to start your first session."
            : "Type a message to start a new session, or Ctrl+K to open one.",
        ),
      );
    }
    log.replaceChildren(...lines);
    log.scrollTop = log.scrollHeight;
    const writable = !this.selected || this.state.sessions[this.selected]?.summary?.writable !== false;
    ($("text") as HTMLTextAreaElement).placeholder = !this.selected
      ? "Message (starts a new session)"
      : writable
        ? "Message"
        : "Read-only here: this session belongs to another chat";
  }

  private renderCard() {
    const approval = this.shown();
    const box = $("cards");
    if (!approval) {
      this.card = null;
      box.replaceChildren();
      box.className = "";
      return;
    }
    if (this.card?.id !== approval.id) this.card = { id: approval.id, since: Date.now() };
    const armed = !!this.armedCard();
    box.className = armed ? "armed" : "";
    const card = el("div", { class: `card ${approval.risk}` });
    const count = Object.keys(this.state.approvals).length;
    card.append(
      el("div", { class: "cardhead" }, `Approval needed · ${approval.risk} risk${count > 1 ? ` · ${count} waiting` : ""}`),
      el("strong", {}, approval.summary),
    );
    const impact = approval.card?.impact;
    if (impact) {
      const text =
        impact.kind === "deletes"
          ? `Deletes ${(impact.targets ?? []).join(", ") || "files"}`
          : impact.kind === "sends"
            ? "Sends something to another service"
            : impact.kind === "spends"
              ? "Spends money"
              : "Writes outside the work area";
      card.append(el("div", { class: "why" }, `⚠ ${text}`));
    }
    const reason = approval.card?.reason;
    if (reason?.kind === "read_outside_content") {
      card.append(el("div", { class: "why" }, reason.check_says ?? "It read outside content, so this step needs your OK."));
    }
    if (reason?.kind === "after_a_decline") card.append(el("div", { class: "why" }, "You said no to an earlier step, so this one asks first."));
    for (const w of approval.warnings ?? []) card.append(el("div", { class: "why" }, `• ${w}`));
    if (approval.detail) card.append(el("code", {}, approval.detail));
    card.append(el("div", { class: "why" }, `In ${this.title(approval.session_id)}${approval.channel ? `, also shown in ${approval.channel}` : ""}`));
    const buttons = el("div", { class: "choices" });
    for (const decision of approval.options) {
      const { key, label } = DECISIONS[decision];
      const button = el("button", { class: decision === "deny" ? "" : "yes" }, el("kbd", {}, key), ` ${label}`);
      button.onclick = () => this.answer(approval, decision);
      buttons.append(button);
    }
    card.append(buttons);
    box.replaceChildren(card);
  }

  private renderTasks() {
    const tasks = running(this.state);
    $("tasklist").replaceChildren(
      ...(tasks.length
        ? tasks.map((t) => {
            const cancel = el("button", {}, "Cancel");
            cancel.onclick = () => this.send({ type: "cancel_task", task_id: t.id });
            return el("div", { class: "task", "data-task": t.id }, `#${t.number} ${t.description}`, el("br"), el("small", {}, this.title(t.session_id)), el("br"), cancel);
          })
        : [el("p", { class: "empty" }, "Nothing running.")]),
    );
  }

  /** The connection state in words. */
  linkText(): string {
    const link = this.link;
    switch (link.phase) {
      case "connecting":
        return "Connecting…";
      case "connected":
        return "Connected";
      case "not_paired":
        return "Not paired";
      case "reconnecting": {
        const seconds = Math.max(1, Math.round(link.retryInMs / 1000));
        if (link.kind === "daemon_not_running") {
          return `Daemon not running · retrying in ${seconds} s${this.platform.startDaemon ? " · Ctrl+K to start it" : ""}`;
        }
        if (link.kind === "unauthorized") return `Signed out: ${link.reason}`;
        if (link.kind === "version_unsupported") return `Daemon too new or too old: ${link.reason}`;
        return `Reconnecting in ${seconds} s · ${link.reason}`;
      }
    }
  }

  private renderStatus() {
    const pill = $("conn");
    pill.textContent = this.linkText();
    pill.className = `pill ${this.link.phase}`;
    const waiting = Object.keys(this.state.approvals).length;
    const work = Object.keys(this.state.tasks).length;
    $("waiting").textContent = `${waiting ? `${waiting} approval${waiting > 1 ? "s" : ""} waiting` : "nothing waiting"}${work ? ` · ${work} running` : ""}`;
    const input = $("text") as HTMLTextAreaElement;
    let hint: string;
    if (input.value.startsWith("/") && !input.value.includes(" ")) {
      const names = this.completions(input.value);
      hint = names.length
        ? names.map((n) => `/${n}: ${LOCAL_SLASH[n]}`).join(" · ") + " · Tab completes; others run in the chat"
        : "Runs in the chat as a slash command";
    } else if (this.flash) {
      hint = this.flash;
    } else if (this.armedCard() && !input.value) {
      const keys = this.armedCard()!.options.map((d) => DECISIONS[d].key);
      hint = `Press ${keys.join(" / ")} to answer the card`;
    } else {
      hint = "Enter send · Esc stop · Ctrl+K sessions and commands · Alt+N new session";
    }
    $("hint").textContent = hint;
  }

  // --- Pairing ---------------------------------------------------------------------

  private showPairing(reason?: string) {
    $("app").hidden = true;
    $("unpair").hidden = true;
    $("pair").hidden = false;
    this.link = { phase: "not_paired", reason: reason ?? "" };
    this.renderStatus();
    const given = this.platform.pairingLink();
    if (given.code) ($("code") as HTMLInputElement).value = given.code;
    $("gatewayrow").hidden = this.platform.name !== "desktop";
    if (reason) $("pairnote").textContent = `This device needs pairing again (${reason}).`;
    $("pairform").onsubmit = async (e) => {
      e.preventDefault();
      const code = ($("code") as HTMLInputElement).value.trim();
      const deviceName = ($("devname") as HTMLInputElement).value.trim();
      const gateway = ($("gateway") as HTMLInputElement).value.trim() || undefined;
      try {
        const offerId = given.offerId ?? ($("offer") as HTMLInputElement).value.trim();
        if (!offerId) throw new Error("Open the link from /pair_device (it names the offer)");
        await this.platform.pair({ offerId, code, deviceName, fingerprint: given.fingerprint, gateway }, (text) => {
          $("pairnote").textContent = text;
        });
        this.platform.forgetPairingLink();
        this.connect();
      } catch (error) {
        $("pairnote").textContent = (error as Error).message;
      }
    };
    $("offerrow").hidden = !!given.offerId;
  }
}
