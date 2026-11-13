// The AIdaemon web client. Pairs this browser with a P-256 key that never
// leaves it (WebCrypto, non-extractable, kept in IndexedDB), signs in with a
// challenge for a short-lived token, then speaks the client protocol over a
// WebSocket: sessions, chat, approval cards and running work. Reconnects
// resume from the last event applied.

import type {
  AccessGrant,
  Approval,
  ApprovalDecision,
  Challenge,
  ClientFrame,
  Command,
  Enrollment,
  ServerFrame,
} from "./protocol";
import { apply as reduce, newState, nextApproval, running, title as titleOf } from "./state.js";

const API = "/client/v1";
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// --- Device identity --------------------------------------------------------

const DB = "aidaemon-client";
const DEVICE_KEY = "aidaemon.device";
const SEQ_KEY = "aidaemon.lastSeq";

function idb<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open(DB, 1);
    open.onupgradeneeded = () => open.result.createObjectStore("keys");
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const request = run(open.result.transaction("keys", mode).objectStore("keys"));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    };
  });
}

const loadKey = () => idb<CryptoKeyPair | undefined>("readonly", (s) => s.get("device"));
const saveKey = (key: CryptoKeyPair) => idb("readwrite", (s) => s.put(key, "device"));
const dropKey = () => idb("readwrite", (s) => s.delete("device"));

function b64url(bytes: ArrayBuffer): string {
  let text = "";
  for (const b of new Uint8Array(bytes)) text += String.fromCharCode(b);
  return btoa(text).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

async function post<T>(path: string, body: unknown, token?: string): Promise<T> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (token) headers.authorization = `Bearer ${token}`;
  const response = await fetch(API + path, { method: "POST", headers, body: JSON.stringify(body) });
  const value = await response.json();
  if (!response.ok) throw new Error(value?.error?.message ?? response.statusText);
  return value as T;
}

async function signIn(): Promise<AccessGrant> {
  const deviceId = localStorage.getItem(DEVICE_KEY);
  const key = await loadKey();
  if (!deviceId || !key) throw new Error("not paired");
  const challenge = await post<Challenge>("/sessions/challenge", { device_id: deviceId });
  const message = new TextEncoder().encode(`aidaemon-client-v1:${challenge.challenge_id}:${challenge.nonce}`);
  const signature = await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, key.privateKey, message);
  return post<AccessGrant>("/sessions", { challenge_id: challenge.challenge_id, signature: b64url(signature) });
}

async function pair(offerId: string, code: string, name: string, fingerprint?: string) {
  const key = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
  const publicKey = b64url(await crypto.subtle.exportKey("raw", key.publicKey));
  const note = $("pairnote");
  let enrollment = await post<Enrollment>("/enrollments/redeem", {
    offer_id: offerId,
    code,
    device_name: name,
    device_kind: "web",
    public_key: publicKey,
  });
  if (fingerprint && enrollment.fingerprint !== fingerprint) {
    throw new Error("This is not the daemon the code came from");
  }
  note.textContent = "Confirm the pairing in the chat you asked from…";
  while (enrollment.state === "pending" || enrollment.state === "open") {
    await new Promise((r) => setTimeout(r, 1500));
    const response = await fetch(`${API}/enrollments/${encodeURIComponent(offerId)}`);
    enrollment = (await response.json()) as Enrollment;
  }
  if (enrollment.state !== "approved" || !enrollment.device_id) {
    throw new Error(`Pairing ${enrollment.state}`);
  }
  await saveKey(key);
  localStorage.setItem(DEVICE_KEY, enrollment.device_id);
  localStorage.removeItem(SEQ_KEY);
}

// --- State -------------------------------------------------------------------

const state = newState();
state.last_seq = Number(localStorage.getItem(SEQ_KEY) ?? "") || null;
let selected: string | null = null;
let awaitingNew = false;

function apply(frame: ServerFrame) {
  const notice = state.notice;
  reduce(state, frame);
  if (frame.type === "ack" && frame.result.type === "message_accepted" && frame.result.new_session && awaitingNew) {
    awaitingNew = false;
    selected = frame.result.session_id;
  }
  if (state.last_seq != null) localStorage.setItem(SEQ_KEY, String(state.last_seq));
  if (state.notice && state.notice !== notice) status(state.notice);
  render();
}

// --- Connection --------------------------------------------------------------

let socket: WebSocket | null = null;
let grant: AccessGrant | null = null;
let retry = 500;
const fetched = new Set<string>();

function status(text: string) {
  $("status").textContent = text;
}

function send(command: Command) {
  const frame: ClientFrame = { type: "command", request_id: crypto.randomUUID(), command };
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(frame));
  else status("Not connected; try again in a moment");
}

async function connect() {
  try {
    if (!grant || grant.expires_at_ms - Date.now() < 60_000) grant = await signIn();
  } catch (error) {
    if (String(error).includes("not paired")) {
      localStorage.removeItem(DEVICE_KEY);
      return showPairing();
    }
    status(`Sign-in failed: ${(error as Error).message}`);
    setTimeout(connect, (retry = Math.min(retry * 2, 10_000)));
    return;
  }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}${API}/ws`);
  socket = ws;
  let ping: number | undefined;
  ws.onopen = () => {
    const hello: ClientFrame = {
      type: "hello",
      version: { major: 1, minor: 0 },
      client: { kind: "web", name: "aidaemon-web" },
      resume_from: state.last_seq ?? undefined,
      token: grant!.access_token,
    };
    ws.send(JSON.stringify(hello));
    ping = setInterval(() => ws.send(JSON.stringify({ type: "ping" })), 25_000);
  };
  ws.onmessage = (message) => {
    const frame = JSON.parse(message.data) as ServerFrame;
    if (frame.type === "welcome") {
      retry = 500;
      status("Connected");
      send({ type: "list_sessions", limit: 200 });
    }
    if (frame.type === "error" && frame.error.code === "unauthorized") grant = null;
    apply(frame);
  };
  ws.onclose = () => {
    clearInterval(ping);
    status("Reconnecting…");
    setTimeout(connect, (retry = Math.min(retry * 2, 10_000)));
  };
}

// --- Screen ------------------------------------------------------------------

const KEY_LABEL: Record<ApprovalDecision, string> = {
  approve_once: "Allow once",
  approve_session: "Allow this session",
  approve_always: "Always allow",
  deny: "Deny",
};

function el(tag: string, props: Record<string, string> = {}, ...children: (Node | string)[]) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) node.setAttribute(k, v);
  node.append(...children);
  return node;
}

const title = (id: string) => titleOf(state, id);

function card(a: Approval): HTMLElement {
  const box = el("div", { class: `card ${a.risk}` }, el("strong", {}, a.summary));
  const impact = a.card?.impact;
  if (impact && impact.kind !== undefined) {
    const text = impact.kind === "deletes" ? `Deletes ${(impact.targets ?? []).join(", ") || "files"}`
      : impact.kind === "sends" ? "Sends something to another service"
      : impact.kind === "spends" ? "Spends money" : "Writes outside the work area";
    box.append(el("div", { class: "why" }, `⚠ ${text}`));
  }
  const reason = a.card?.reason;
  if (reason?.kind === "read_outside_content") box.append(el("div", { class: "why" }, reason.check_says ?? "It read outside content, so this step needs your OK."));
  if (reason?.kind === "after_a_decline") box.append(el("div", { class: "why" }, "You said no to an earlier step, so this one asks first."));
  for (const w of a.warnings ?? []) box.append(el("div", { class: "why" }, `• ${w}`));
  if (a.detail) box.append(el("code", {}, a.detail));
  if (a.channel) box.append(el("div", { class: "why" }, `Also shown in ${a.channel}. ${title(a.session_id)}`));
  const buttons = el("div");
  for (const decision of a.options) {
    const button = el("button", { class: decision === "deny" ? "" : "yes" }, KEY_LABEL[decision]);
    button.onclick = () => send({ type: "resolve_approval", approval_id: a.id, decision });
    buttons.append(button);
  }
  box.append(buttons);
  return box;
}

function render() {
  const list = $("sessionlist");
  list.replaceChildren(...state.order.map((id) => {
    const b = el("button", { class: id === selected ? "on" : "" }, `${state.sessions[id]?.live ? "● " : ""}${title(id)}`);
    b.onclick = () => {
      selected = id;
      if (!fetched.has(id)) {
        fetched.add(id);
        send({ type: "get_history", session_id: id, limit: 100 });
      }
      render();
    };
    return b;
  }));
  const first = nextApproval(state);
  $("cards").replaceChildren(...(first ? [card(first)] : []));
  const log = $("log");
  const v = selected ? state.sessions[selected] : undefined;
  const lines = (v?.lines ?? []).map((l) => {
    const who = l.kind === "user" ? (l.origin && l.origin !== "app" ? `You (${l.origin})` : "You")
      : l.kind === "assistant" ? "Assistant" : "";
    return el("p", { class: l.kind === "note" ? "msg live" : "msg" }, el("b", {}, who), l.text);
  });
  if (v?.live) lines.push(el("p", { class: "msg live" }, v.live.streamed, v.live.streamed ? "\n" : "", v.live.status ?? "Working..."));
  if (!v) lines.push(el("p", { class: "msg live" }, "Type a message to start a new session."));
  log.replaceChildren(...lines);
  log.scrollTop = log.scrollHeight;
  $("tasklist").replaceChildren(...running(state).map((t) => {
    const cancel = el("button", {}, "Cancel");
    cancel.onclick = () => send({ type: "cancel_task", task_id: t.id });
    return el("div", { class: "task" }, `#${t.number} ${t.description}`, el("br"), el("small", {}, title(t.session_id)), el("br"), cancel);
  }));
}

function showPairing() {
  $("app").hidden = true;
  $("unpair").hidden = true;
  $("pair").hidden = false;
  status("Not paired");
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get("code")) ($("code") as HTMLInputElement).value = hash.get("code")!;
  $("pairform").onsubmit = async (e) => {
    e.preventDefault();
    const offer = hash.get("offer");
    const code = ($("code") as HTMLInputElement).value.trim();
    const name = ($("devname") as HTMLInputElement).value.trim();
    try {
      if (!offer) throw new Error("Open the link from /pair_device (it names the offer)");
      await pair(offer, code, name, hash.get("fp") ?? undefined);
      history.replaceState(null, "", location.pathname);
      start();
    } catch (error) {
      $("pairnote").textContent = (error as Error).message;
    }
  };
}

function start() {
  $("pair").hidden = true;
  $("app").hidden = false;
  $("unpair").hidden = false;
  render();
  connect();
}

$("send").onsubmit = (e) => {
  e.preventDefault();
  const input = $("text") as HTMLInputElement;
  const text = input.value.trim();
  if (!text) return;
  if (text === "/stop" && selected) send({ type: "stop_turn", session_id: selected });
  else if (text.startsWith("/") && selected) {
    const [name, ...args] = text.slice(1).split(" ");
    send({ type: "run_slash", session_id: selected, command: name, args: args.join(" ") });
  } else {
    const writable = !selected || state.sessions[selected]?.summary?.writable !== false;
    if (!writable) return status("That session belongs to another chat: reply there, or start a new session.");
    awaitingNew = !selected;
    send({ type: "send_message", session_id: selected ?? undefined, text });
  }
  input.value = "";
};
document.querySelector<HTMLButtonElement>("[data-new]")!.onclick = () => {
  selected = null;
  render();
};
$("unpair").onclick = async () => {
  await dropKey();
  localStorage.removeItem(DEVICE_KEY);
  localStorage.removeItem(SEQ_KEY);
  socket?.close();
  location.reload();
};

if (localStorage.getItem(DEVICE_KEY)) start();
else showPairing();
