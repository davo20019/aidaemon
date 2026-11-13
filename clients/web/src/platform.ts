// The only file that knows where the client runs. In a browser tab the page
// holds its own connection and its device key (WebCrypto, non-extractable,
// in IndexedDB); inside the desktop shell, the shell holds the one
// connection and the key (OS keychain), and the page receives frames and
// sends commands through it. Everything else is the same code.

import type { Capability, ClientKind, Command, Enrollment } from "./protocol";
import { API, b64url, post, SocketLink, type Connection, type LinkHandlers } from "./link.js";

/** What a pairing needs: the offer from `/pair_device` or `aidaemon pair`. */
export interface PairRequest {
  offerId: string;
  code: string;
  deviceName: string;
  fingerprint?: string;
  /** A remote daemon's address (desktop only; a tab pairs with its origin). */
  gateway?: string;
}

export interface Platform {
  readonly name: "browser" | "desktop";
  /** The connection this screen drives. */
  connection(capabilities: Capability[]): Connection;
  isPaired(): Promise<boolean>;
  pair(request: PairRequest, progress: (text: string) => void): Promise<void>;
  unpair(): Promise<void>;
  /** The last `seq` applied, kept across reloads. */
  loadSeq(): number | null;
  saveSeq(seq: number | null): void;
  /** Tell the owner something while they may be looking elsewhere. */
  notify(title: string, body: string): void;
  /** A pairing offer handed over by the link that opened this page. */
  pairingLink(): Partial<PairRequest>;
  forgetPairingLink(): void;
}

const SEQ_KEY = "aidaemon.lastSeq";

function loadSeq(): number | null {
  try {
    return Number(localStorage.getItem(SEQ_KEY) ?? "") || null;
  } catch {
    return null;
  }
}

function saveSeq(seq: number | null) {
  try {
    if (seq == null) localStorage.removeItem(SEQ_KEY);
    else localStorage.setItem(SEQ_KEY, String(seq));
  } catch {
    // Private windows may refuse storage: the next connection takes a snapshot.
  }
}

// --- Browser -----------------------------------------------------------------

const DB = "aidaemon-client";
const DEVICE_KEY = "aidaemon.device";

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

class BrowserPlatform implements Platform {
  readonly name = "browser" as const;

  connection(capabilities: Capability[]): Connection {
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
          if (!key) throw new Error("not paired");
          return crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, key.privateKey, message as BufferSource);
        },
      },
    });
  }

  async isPaired() {
    return !!localStorage.getItem(DEVICE_KEY) && !!(await loadKey().catch(() => undefined));
  }

  async pair(request: PairRequest, progress: (text: string) => void) {
    const key = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
    const publicKey = b64url(await crypto.subtle.exportKey("raw", key.publicKey));
    const kind: ClientKind = "web";
    let enrollment = await post<Enrollment>("", "/enrollments/redeem", {
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
      enrollment = (await response.json()) as Enrollment;
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

  notify(title: string, body: string) {
    if (!document.hidden || typeof Notification === "undefined") return;
    if (Notification.permission === "granted") new Notification(title, { body });
  }

  pairingLink(): Partial<PairRequest> {
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

// --- Desktop shell -------------------------------------------------------------

/** The shell's bridge, as Tauri 2 exposes it with `withGlobalTauri`. */
interface TauriBridge {
  core: { invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> };
  event: { listen<T>(event: string, handler: (event: { payload: T }) => void): Promise<() => void> };
}

function tauri(): TauriBridge | undefined {
  return (globalThis as { __TAURI__?: TauriBridge }).__TAURI__;
}

/** Frames and commands go through the shell, which holds the connection. */
class ShellConnection implements Connection {
  private unlisten: Array<() => void> = [];

  constructor(private bridge: TauriBridge, private capabilities: Capability[]) {}

  start(handlers: LinkHandlers, resumeFrom: () => number | null) {
    void (async () => {
      this.unlisten.push(await this.bridge.event.listen("aidaemon://frame", (e) => handlers.onFrame(e.payload as never)));
      this.unlisten.push(await this.bridge.event.listen("aidaemon://link", (e) => handlers.onState(e.payload as never)));
      await this.bridge.core.invoke("link_start", { capabilities: this.capabilities, resumeFrom: resumeFrom() });
    })();
  }

  send(command: Command): string {
    const requestId = crypto.randomUUID();
    void this.bridge.core.invoke("link_send", { requestId, command });
    return requestId;
  }

  stop() {
    for (const off of this.unlisten.splice(0)) off();
  }
}

class DesktopPlatform implements Platform {
  readonly name = "desktop" as const;

  constructor(private bridge: TauriBridge) {}

  connection(capabilities: Capability[]): Connection {
    return new ShellConnection(this.bridge, capabilities);
  }

  isPaired() {
    return this.bridge.core.invoke<boolean>("is_paired");
  }

  async pair(request: PairRequest, progress: (text: string) => void) {
    progress("Confirm the pairing in the chat you asked from…");
    await this.bridge.core.invoke("pair", { request });
  }

  unpair() {
    return this.bridge.core.invoke<void>("unpair");
  }

  loadSeq = loadSeq;
  saveSeq = saveSeq;

  // The shell raises its own native notifications, window open or not.
  notify() {}

  pairingLink(): Partial<PairRequest> {
    return {};
  }

  forgetPairingLink() {}
}

/** The platform this page runs on. */
export function detect(): Platform {
  const bridge = tauri();
  return bridge ? new DesktopPlatform(bridge) : new BrowserPlatform();
}
