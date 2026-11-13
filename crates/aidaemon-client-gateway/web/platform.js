// The only file that knows where the client runs. In a browser tab the page
// holds its own connection and its device key (WebCrypto, non-extractable,
// in IndexedDB); inside the desktop shell, the shell holds the one
// connection and the key (OS keychain), and the page receives frames and
// sends commands through it. Everything else is the same code.
import { API, b64url, post, SocketLink } from "./link.js";
const SEQ_KEY = "aidaemon.lastSeq";
function loadSeq() {
    try {
        return Number(localStorage.getItem(SEQ_KEY) ?? "") || null;
    }
    catch {
        return null;
    }
}
function saveSeq(seq) {
    try {
        if (seq == null)
            localStorage.removeItem(SEQ_KEY);
        else
            localStorage.setItem(SEQ_KEY, String(seq));
    }
    catch {
        // Private windows may refuse storage: the next connection takes a snapshot.
    }
}
// --- Browser -----------------------------------------------------------------
const DB = "aidaemon-client";
const DEVICE_KEY = "aidaemon.device";
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
class BrowserPlatform {
    name = "browser";
    connection(capabilities) {
        const scheme = location.protocol === "https:" ? "wss" : "ws";
        return new SocketLink({
            base: "",
            socketUrl: `${scheme}://${location.host}${API}/ws`,
            client: { kind: "web", name: "aidaemon-web" },
            capabilities,
            device: {
                deviceId: async () => ((await loadKey()) ? localStorage.getItem(DEVICE_KEY) : null),
                sign: async (message) => {
                    const key = await loadKey();
                    if (!key)
                        throw new Error("not paired");
                    return crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, key.privateKey, message);
                },
            },
        });
    }
    async isPaired() {
        return !!localStorage.getItem(DEVICE_KEY) && !!(await loadKey().catch(() => undefined));
    }
    async pair(request, progress) {
        const key = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
        const publicKey = b64url(await crypto.subtle.exportKey("raw", key.publicKey));
        const kind = "web";
        let enrollment = await post("", "/enrollments/redeem", {
            offer_id: request.offerId,
            code: request.code,
            device_name: request.deviceName,
            device_kind: kind,
            public_key: publicKey,
        });
        if (request.fingerprint && enrollment.fingerprint !== request.fingerprint) {
            throw new Error("This is not the daemon the code came from");
        }
        progress("Confirm the pairing in the chat you asked from…");
        while (enrollment.state === "pending" || enrollment.state === "open") {
            await new Promise((r) => setTimeout(r, 1500));
            const response = await fetch(`${API}/enrollments/${encodeURIComponent(request.offerId)}`);
            enrollment = (await response.json());
        }
        if (enrollment.state !== "approved" || !enrollment.device_id) {
            throw new Error(`Pairing ${enrollment.state}`);
        }
        await saveKey(key);
        localStorage.setItem(DEVICE_KEY, enrollment.device_id);
        saveSeq(null);
    }
    async unpair() {
        await dropKey();
        localStorage.removeItem(DEVICE_KEY);
        saveSeq(null);
    }
    loadSeq = loadSeq;
    saveSeq = saveSeq;
    notify(title, body) {
        if (!document.hidden || typeof Notification === "undefined")
            return;
        if (Notification.permission === "granted")
            new Notification(title, { body });
    }
    pairingLink() {
        const hash = new URLSearchParams(location.hash.slice(1));
        return {
            offerId: hash.get("offer") ?? undefined,
            code: hash.get("code") ?? undefined,
            fingerprint: hash.get("fp") ?? undefined,
        };
    }
    forgetPairingLink() {
        history.replaceState(null, "", location.pathname);
    }
}
function tauri() {
    return globalThis.__TAURI__;
}
/** Frames and commands go through the shell, which holds the connection. */
class ShellConnection {
    bridge;
    capabilities;
    unlisten = [];
    constructor(bridge, capabilities) {
        this.bridge = bridge;
        this.capabilities = capabilities;
    }
    start(handlers, resumeFrom) {
        void (async () => {
            this.unlisten.push(await this.bridge.event.listen("aidaemon://frame", (e) => handlers.onFrame(e.payload)));
            this.unlisten.push(await this.bridge.event.listen("aidaemon://link", (e) => handlers.onState(e.payload)));
            await this.bridge.core.invoke("link_start", { capabilities: this.capabilities, resumeFrom: resumeFrom() });
        })();
    }
    send(command) {
        const requestId = crypto.randomUUID();
        void this.bridge.core.invoke("link_send", { requestId, command });
        return requestId;
    }
    stop() {
        for (const off of this.unlisten.splice(0))
            off();
    }
}
class DesktopPlatform {
    bridge;
    name = "desktop";
    constructor(bridge) {
        this.bridge = bridge;
    }
    connection(capabilities) {
        return new ShellConnection(this.bridge, capabilities);
    }
    isPaired() {
        return this.bridge.core.invoke("is_paired");
    }
    async pair(request, progress) {
        progress("Confirm the pairing in the chat you asked from…");
        await this.bridge.core.invoke("pair", { request });
    }
    unpair() {
        return this.bridge.core.invoke("unpair");
    }
    loadSeq = loadSeq;
    saveSeq = saveSeq;
    // The shell raises its own native notifications, window open or not.
    notify() { }
    pairingLink() {
        return {};
    }
    forgetPairingLink() { }
}
/** The platform this page runs on. */
export function detect() {
    const bridge = tauri();
    return bridge ? new DesktopPlatform(bridge) : new BrowserPlatform();
}
