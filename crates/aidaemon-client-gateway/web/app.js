// The AIdaemon web client. Pairs this browser with a P-256 key that never
// leaves it (WebCrypto, non-extractable, kept in IndexedDB), signs in with a
// challenge for a short-lived token, then speaks the client protocol over a
// WebSocket: sessions, chat, approval cards and running work. Reconnects
// resume from the last event applied.
const API = "/client/v1";
const $ = (id) => document.getElementById(id);
// --- Device identity --------------------------------------------------------
const DB = "aidaemon-client";
const DEVICE_KEY = "aidaemon.device";
const SEQ_KEY = "aidaemon.lastSeq";
function idb(mode, run) {
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
const loadKey = () => idb("readonly", (s) => s.get("device"));
const saveKey = (key) => idb("readwrite", (s) => s.put(key, "device"));
const dropKey = () => idb("readwrite", (s) => s.delete("device"));
function b64url(bytes) {
    let text = "";
    for (const b of new Uint8Array(bytes))
        text += String.fromCharCode(b);
    return btoa(text).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}
async function post(path, body, token) {
    const headers = { "content-type": "application/json" };
    if (token)
        headers.authorization = `Bearer ${token}`;
    const response = await fetch(API + path, { method: "POST", headers, body: JSON.stringify(body) });
    const value = await response.json();
    if (!response.ok)
        throw new Error(value?.error?.message ?? response.statusText);
    return value;
}
async function signIn() {
    const deviceId = localStorage.getItem(DEVICE_KEY);
    const key = await loadKey();
    if (!deviceId || !key)
        throw new Error("not paired");
    const challenge = await post("/sessions/challenge", { device_id: deviceId });
    const message = new TextEncoder().encode(`aidaemon-client-v1:${challenge.challenge_id}:${challenge.nonce}`);
    const signature = await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, key.privateKey, message);
    return post("/sessions", { challenge_id: challenge.challenge_id, signature: b64url(signature) });
}
async function pair(offerId, code, name, fingerprint) {
    const key = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
    const publicKey = b64url(await crypto.subtle.exportKey("raw", key.publicKey));
    const note = $("pairnote");
    let enrollment = await post("/enrollments/redeem", {
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
        enrollment = (await response.json());
    }
    if (enrollment.state !== "approved" || !enrollment.device_id) {
        throw new Error(`Pairing ${enrollment.state}`);
    }
    await saveKey(key);
    localStorage.setItem(DEVICE_KEY, enrollment.device_id);
    localStorage.removeItem(SEQ_KEY);
}
const sessions = new Map();
const approvals = new Map();
const tasks = new Map();
let selected = null;
let awaitingNew = false;
let lastSeq = Number(localStorage.getItem(SEQ_KEY) ?? "") || null;
const view = (id) => {
    let v = sessions.get(id);
    if (!v)
        sessions.set(id, (v = { lines: [] }));
    return v;
};
function liveOf(turnId) {
    for (const v of sessions.values())
        if (v.live?.turnId === turnId)
            return v;
    return undefined;
}
function statusText(event) {
    const s = event.status;
    switch (s.kind) {
        case "thinking": return "Thinking…";
        case "tool_started": return `${s.name}${s.summary ? `: ${s.summary}` : ""}…`;
        case "tool_progress": return `${s.name}: ${s.chunk.slice(0, 80)}`;
        case "tool_completed": return `✓ ${s.name} ${s.summary ?? ""}`;
        case "plan_step": return `Step ${s.step + 1} of ${s.total}: ${s.description}`;
        case "checklist": return "Updated the plan";
        case "note": return s.text;
    }
    return "Working…";
}
function applyEvent(sessionId, event) {
    switch (event.type) {
        case "session_created":
        case "session_updated":
            view(event.session.id).summary = event.session;
            break;
        case "user_message":
            view(event.session_id).lines.push({ who: event.origin === "app" ? "You" : `You (${event.origin})`, text: event.text });
            break;
        case "assistant_message":
            view(event.session_id).lines.push({ who: "Assistant", text: event.text });
            break;
        case "turn_started":
            view(event.session_id).live = { turnId: event.turn_id, text: "", status: "Working…" };
            break;
        case "turn_delta": {
            const v = liveOf(event.turn_id);
            if (v?.live)
                v.live.text += event.text;
            break;
        }
        case "turn_status": {
            const v = liveOf(event.turn_id);
            if (v?.live)
                v.live.status = statusText(event);
            break;
        }
        case "user_action_required":
            if (sessionId)
                view(sessionId).lines.push({ who: "Action needed", text: event.message, note: true });
            break;
        case "turn_completed": {
            const v = liveOf(event.turn_id) ?? (sessionId ? view(sessionId) : undefined);
            if (!v)
                break;
            v.live = undefined;
            if (event.text)
                v.lines.push({ who: "Assistant", text: event.text });
            if (event.state === "cancelled")
                v.lines.push({ who: "", text: "Stopped.", note: true });
            if (event.state === "failed")
                v.lines.push({ who: "", text: `Failed: ${event.error ?? ""}`, note: true });
            break;
        }
        case "approval_requested":
            approvals.set(event.approval.id, event.approval);
            break;
        case "approval_resolved":
            approvals.delete(event.approval_id);
            break;
        case "task_updated":
            if (event.task.state === "running" || event.task.state === "cancelling")
                tasks.set(event.task.id, event.task);
            else
                tasks.delete(event.task.id);
            break;
    }
}
function apply(frame) {
    switch (frame.type) {
        case "snapshot":
            approvals.clear();
            tasks.clear();
            for (const s of frame.state.sessions)
                view(s.id).summary = s;
            for (const a of frame.state.pending_approvals)
                approvals.set(a.id, a);
            for (const t of frame.state.running_tasks)
                tasks.set(t.id, t);
            setSeq(frame.seq);
            break;
        case "event":
            if (frame.seq != null) {
                if (lastSeq != null && frame.seq <= lastSeq)
                    return;
                setSeq(frame.seq);
            }
            applyEvent(frame.session_id ?? undefined, frame.event);
            break;
        case "ack": {
            const r = frame.result;
            if (r.type === "message_accepted" && r.new_session && awaitingNew) {
                awaitingNew = false;
                selected = r.session_id;
            }
            else if (r.type === "sessions") {
                for (const s of r.sessions)
                    view(s.id).summary = s;
            }
            else if (r.type === "history") {
                const v = view(r.session_id);
                v.lines = r.messages.map((m) => ({ who: m.role === "user" ? "You" : "Assistant", text: m.text })).concat(v.lines);
            }
            else if (r.type === "stopped" || r.type === "reply") {
                status(r.type === "stopped" ? r.report : r.text);
            }
            break;
        }
        case "error":
            status(frame.error.message);
            break;
    }
    render();
}
function setSeq(seq) {
    lastSeq = seq;
    localStorage.setItem(SEQ_KEY, String(seq));
}
// --- Connection --------------------------------------------------------------
let socket = null;
let grant = null;
let retry = 500;
const fetched = new Set();
function status(text) {
    $("status").textContent = text;
}
function send(command) {
    const frame = { type: "command", request_id: crypto.randomUUID(), command };
    if (socket?.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify(frame));
    else
        status("Not connected; try again in a moment");
}
async function connect() {
    try {
        if (!grant || grant.expires_at_ms - Date.now() < 60_000)
            grant = await signIn();
    }
    catch (error) {
        if (String(error).includes("not paired")) {
            localStorage.removeItem(DEVICE_KEY);
            return showPairing();
        }
        status(`Sign-in failed: ${error.message}`);
        setTimeout(connect, (retry = Math.min(retry * 2, 10_000)));
        return;
    }
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}${API}/ws`);
    socket = ws;
    let ping;
    ws.onopen = () => {
        const hello = {
            type: "hello",
            version: { major: 1, minor: 0 },
            client: { kind: "web", name: "aidaemon-web" },
            resume_from: lastSeq ?? undefined,
            token: grant.access_token,
        };
        ws.send(JSON.stringify(hello));
        ping = setInterval(() => ws.send(JSON.stringify({ type: "ping" })), 25_000);
    };
    ws.onmessage = (message) => {
        const frame = JSON.parse(message.data);
        if (frame.type === "welcome") {
            retry = 500;
            status("Connected");
            send({ type: "list_sessions", limit: 200 });
        }
        if (frame.type === "error" && frame.error.code === "unauthorized")
            grant = null;
        apply(frame);
    };
    ws.onclose = () => {
        clearInterval(ping);
        status("Reconnecting…");
        setTimeout(connect, (retry = Math.min(retry * 2, 10_000)));
    };
}
// --- Screen ------------------------------------------------------------------
const KEY_LABEL = {
    approve_once: "Allow once",
    approve_session: "Allow this session",
    approve_always: "Always allow",
    deny: "Deny",
};
function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props))
        node.setAttribute(k, v);
    node.append(...children);
    return node;
}
function title(id) {
    const s = sessions.get(id)?.summary;
    return s?.title ?? (s ? `${id.slice(0, 22)} (${s.channel})` : id.slice(0, 22));
}
function card(a) {
    const box = el("div", { class: `card ${a.risk}` }, el("strong", {}, a.summary));
    const impact = a.card?.impact;
    if (impact && impact.kind !== undefined) {
        const text = impact.kind === "deletes" ? `Deletes ${(impact.targets ?? []).join(", ") || "files"}`
            : impact.kind === "sends" ? "Sends something to another service"
                : impact.kind === "spends" ? "Spends money" : "Writes outside the work area";
        box.append(el("div", { class: "why" }, `⚠ ${text}`));
    }
    const reason = a.card?.reason;
    if (reason?.kind === "read_outside_content")
        box.append(el("div", { class: "why" }, reason.check_says ?? "It read outside content, so this step needs your OK."));
    if (reason?.kind === "after_a_decline")
        box.append(el("div", { class: "why" }, "You said no to an earlier step, so this one asks first."));
    for (const w of a.warnings ?? [])
        box.append(el("div", { class: "why" }, `• ${w}`));
    if (a.detail)
        box.append(el("code", {}, a.detail));
    if (a.channel)
        box.append(el("div", { class: "why" }, `Also shown in ${a.channel}. ${title(a.session_id)}`));
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
    list.replaceChildren(...[...sessions.keys()].map((id) => {
        const b = el("button", { class: id === selected ? "on" : "" }, `${sessions.get(id)?.live ? "● " : ""}${title(id)}`);
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
    $("cards").replaceChildren(...[...approvals.values()].sort((a, b) => a.requested_at_ms - b.requested_at_ms).map(card));
    const log = $("log");
    const v = selected ? sessions.get(selected) : undefined;
    const lines = (v?.lines ?? []).map((l) => el("p", { class: l.note ? "msg live" : "msg" }, el("b", {}, l.who), l.text));
    if (v?.live)
        lines.push(el("p", { class: "msg live" }, v.live.text, v.live.text ? "\n" : "", v.live.status));
    if (!v)
        lines.push(el("p", { class: "msg live" }, "Type a message to start a new session."));
    log.replaceChildren(...lines);
    log.scrollTop = log.scrollHeight;
    $("tasklist").replaceChildren(...[...tasks.values()].map((t) => {
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
    if (hash.get("code"))
        $("code").value = hash.get("code");
    $("pairform").onsubmit = async (e) => {
        e.preventDefault();
        const offer = hash.get("offer");
        const code = $("code").value.trim();
        const name = $("devname").value.trim();
        try {
            if (!offer)
                throw new Error("Open the link from /pair_device (it names the offer)");
            await pair(offer, code, name, hash.get("fp") ?? undefined);
            history.replaceState(null, "", location.pathname);
            start();
        }
        catch (error) {
            $("pairnote").textContent = error.message;
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
    const input = $("text");
    const text = input.value.trim();
    if (!text)
        return;
    if (text === "/stop" && selected)
        send({ type: "stop_turn", session_id: selected });
    else if (text.startsWith("/") && selected) {
        const [name, ...args] = text.slice(1).split(" ");
        send({ type: "run_slash", session_id: selected, command: name, args: args.join(" ") });
    }
    else {
        const writable = !selected || sessions.get(selected)?.summary?.writable !== false;
        if (!writable)
            return status("That session belongs to another chat: reply there, or start a new session.");
        awaitingNew = !selected;
        send({ type: "send_message", session_id: selected ?? undefined, text });
    }
    input.value = "";
};
document.querySelector("[data-new]").onclick = () => {
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
if (localStorage.getItem(DEVICE_KEY))
    start();
else
    showPairing();
export {};
