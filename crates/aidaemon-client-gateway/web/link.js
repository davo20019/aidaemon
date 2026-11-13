// The browser's own connection to the gateway, the TypeScript twin of the
// Rust link in aidaemon-client-core: signs in with the device key, says
// hello with the last `seq` applied, pings, reconnects with backoff and
// resumes, and sends commands not yet answered again under their own
// `request_id` after a reconnect. It reports its state so the screen can
// always say where it stands.
export const API = "/client/v1";
export class SignInError extends Error {
    kind;
    constructor(kind, message) {
        super(message);
        this.kind = kind;
    }
}
export function b64url(bytes) {
    let text = "";
    for (const b of new Uint8Array(bytes))
        text += String.fromCharCode(b);
    return btoa(text).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}
/** POST to the gateway; a network failure is the daemon not running. */
export async function post(base, path, body) {
    let response;
    try {
        response = await fetch(base + API + path, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
        });
    }
    catch (error) {
        throw new SignInError("daemon_not_running", "the daemon did not answer");
    }
    const value = await response.json().catch(() => null);
    if (!response.ok) {
        const message = value?.error?.message ?? response.statusText;
        throw new SignInError(response.status >= 400 && response.status < 500 ? "unauthorized" : "dropped", message);
    }
    return value;
}
export async function signIn(base, device) {
    const deviceId = await device.deviceId();
    if (!deviceId)
        throw new SignInError("not_paired", "this browser is not paired");
    const challenge = await post(base, "/sessions/challenge", { device_id: deviceId });
    const message = new TextEncoder().encode(`aidaemon-client-v1:${challenge.challenge_id}:${challenge.nonce}`);
    const signature = await device.sign(message);
    return post(base, "/sessions", { challenge_id: challenge.challenge_id, signature: b64url(signature) });
}
const RETRY_MIN = 500;
const RETRY_MAX = 10_000;
const PING_EVERY = 25_000;
/** A grant this close to its end is renewed before it is used. */
const RENEW_BEFORE = 60_000;
/** The browser's WebSocket link. */
export class SocketLink {
    options;
    socket = null;
    grant = null;
    retry = RETRY_MIN;
    timer;
    stopped = false;
    pending = new Map();
    handlers = null;
    resumeFrom = () => null;
    constructor(options) {
        this.options = options;
    }
    start(handlers, resumeFrom) {
        this.handlers = handlers;
        this.resumeFrom = resumeFrom;
        this.stopped = false;
        void this.connect();
    }
    stop() {
        this.stopped = true;
        clearTimeout(this.timer);
        this.socket?.close();
    }
    send(command) {
        const requestId = crypto.randomUUID();
        this.pending.set(requestId, command);
        this.write({ type: "command", request_id: requestId, command });
        return requestId;
    }
    write(frame) {
        if (this.socket?.readyState === WebSocket.OPEN)
            this.socket.send(JSON.stringify(frame));
    }
    later(kind, reason) {
        if (this.stopped)
            return;
        this.handlers?.onState({ phase: "reconnecting", kind, reason, retryInMs: this.retry });
        this.timer = setTimeout(() => void this.connect(), this.retry);
        this.retry = Math.min(this.retry * 2, RETRY_MAX);
    }
    async connect() {
        if (this.stopped)
            return;
        this.handlers?.onState({ phase: "connecting" });
        try {
            if (!this.grant || this.grant.expires_at_ms - Date.now() < RENEW_BEFORE) {
                this.grant = await signIn(this.options.base, this.options.device);
            }
        }
        catch (error) {
            this.grant = null;
            const kind = error instanceof SignInError ? error.kind : "dropped";
            const reason = error instanceof Error ? error.message : String(error);
            if (kind === "not_paired" || kind === "unauthorized") {
                // Unknown or revoked device: only pairing again helps.
                this.handlers?.onState({ phase: "not_paired", reason });
                return;
            }
            return this.later(kind, reason);
        }
        const ws = new WebSocket(this.options.socketUrl);
        this.socket = ws;
        let ping;
        let refused = null;
        ws.onopen = () => {
            const hello = {
                type: "hello",
                version: { major: 1, minor: 1 },
                client: this.options.client,
                capabilities: this.options.capabilities,
                resume_from: this.resumeFrom() ?? undefined,
                token: this.grant.access_token,
            };
            ws.send(JSON.stringify(hello));
            for (const [requestId, command] of this.pending) {
                ws.send(JSON.stringify({ type: "command", request_id: requestId, command }));
            }
            ping = setInterval(() => ws.send(JSON.stringify({ type: "ping" })), PING_EVERY);
        };
        ws.onmessage = (message) => {
            const frame = JSON.parse(message.data);
            if (frame.type === "welcome") {
                this.retry = RETRY_MIN;
                this.handlers?.onState({ phase: "connected" });
            }
            if (frame.type === "ack" || (frame.type === "error" && frame.request_id)) {
                this.pending.delete(frame.request_id);
            }
            if (frame.type === "error" && !frame.request_id) {
                if (frame.error.code === "unauthorized") {
                    this.grant = null;
                    refused = frame.error.message;
                }
                else if (frame.error.code === "version_unsupported") {
                    refused = frame.error.message;
                }
            }
            this.handlers?.onFrame(frame);
        };
        ws.onclose = () => {
            clearInterval(ping);
            if (this.socket === ws)
                this.socket = null;
            this.later(refused ? "unauthorized" : "dropped", refused ?? "the connection to the daemon was lost");
        };
    }
}
