// The browser's half of the app's link: the page's WebSocket, and the
// device key as the TypeScript web client keeps it (WebCrypto P-256,
// non-extractable, in IndexedDB "aidaemon-client"; the device id in
// localStorage), so a browser paired with either client is paired with both.

const DB = "aidaemon-client";
const DEVICE_KEY = "aidaemon.device";
let pending = null;

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

function b64url(buffer) {
  let text = "";
  for (const byte of new Uint8Array(buffer)) text += String.fromCharCode(byte);
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function origin() {
  return location.origin;
}

/** The paired device's id, or "" when this browser is not paired. */
export async function deviceId() {
  try {
    const id = localStorage.getItem(DEVICE_KEY);
    const key = await idb("readonly", (s) => s.get("device"));
    return id && key ? id : "";
  } catch {
    return "";
  }
}

/** A new key for a pairing, kept until the pairing is approved. */
export async function newKey() {
  pending = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, false, ["sign", "verify"]);
  return b64url(await crypto.subtle.exportKey("raw", pending.publicKey));
}

/** The pairing was approved: keep the key and the id. */
export async function keepKey(id) {
  if (!pending) throw new Error("no key waiting");
  await idb("readwrite", (s) => s.put(pending, "device"));
  localStorage.setItem(DEVICE_KEY, id);
  pending = null;
}

export async function forget() {
  await idb("readwrite", (s) => s.delete("device"));
  localStorage.removeItem(DEVICE_KEY);
}

/** The signature (r || s, base64url) of `message`. */
export async function sign(message) {
  const key = await idb("readonly", (s) => s.get("device"));
  if (!key) throw new Error("this browser is not paired");
  const signature = await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    key.privateKey,
    new TextEncoder().encode(message),
  );
  return b64url(signature);
}

/** The pairing offer in the link that opened this page: offer, code, fp. */
export function pairingLink() {
  const hash = new URLSearchParams(location.hash.slice(1));
  return [hash.get("offer") ?? "", hash.get("code") ?? "", hash.get("fp") ?? ""];
}

// A pairing link opened in a tab that already shows the app changes only
// the fragment, which loads nothing: start the page afresh so the link and
// its pairing run from the offer, as a first visit would.
window.addEventListener("hashchange", () => {
  if (new URLSearchParams(location.hash.slice(1)).get("offer")) location.reload();
});

export function forgetPairingLink() {
  history.replaceState(null, "", location.pathname + location.search);
}

/** A socket to `url`; `event(kind, text)` hears "open", "message", "close". */
/**
 * Call `wake` when the page is shown again or the network comes back: a
 * phone freezes a background page, and its socket may have died meanwhile.
 */
/** Whether the owner can see this page now. */
// An iPhone, iPad or Mac (an iPad says it is a Mac).
export function appleDevice() {
  return /Macintosh|iPhone|iPad|iPod/.test(navigator.userAgent);
}

export function pageVisible() {
  return document.visibilityState === "visible";
}

export function onWake(wake) {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") wake();
  });
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) wake();
  });
  window.addEventListener("online", () => wake());
  window.addEventListener("focus", () => wake());
}

export class Socket {
  constructor(url, event) {
    this.ws = new WebSocket(url);
    this.ws.onopen = () => event("open", "");
    this.ws.onmessage = (m) => event("message", typeof m.data === "string" ? m.data : "");
    this.ws.onclose = (c) => event("close", c.reason || `closed (${c.code})`);
    this.ws.onerror = () => {};
  }
  send(text) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(text);
  }
  close() {
    this.ws.close();
  }
}

function b64(bytes) {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    text += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(text);
}

/**
 * A live view's frame socket (protocol 1.30): sends the ticket first, turns
 * each binary frame into a data: URI for the app and acks it, and passes
 * the other headers on as JSON. Nothing typed is kept here.
 */
/**
 * A live view driven by a mouse and keyboard: the page's keys go to the
 * agent's browser while `event(down, code, key, mods)` says it took them
 * (the owner is in control). Keys the browser keeps never arrive here.
 */
export class KeyCapture {
  constructor(event) {
    this.handler = (e) => {
      if (e.isComposing) return;
      const mods = [];
      if (e.shiftKey) mods.push("shift");
      if (e.ctrlKey) mods.push("ctrl");
      if (e.altKey) mods.push("alt");
      if (e.metaKey) mods.push("meta");
      if (event(e.type === "keydown", e.code || "", e.key || "", mods.join(","))) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    window.addEventListener("keydown", this.handler, true);
    window.addEventListener("keyup", this.handler, true);
  }
  stop() {
    window.removeEventListener("keydown", this.handler, true);
    window.removeEventListener("keyup", this.handler, true);
  }
}

export class MediaSocket {
  constructor(url, ticket, event) {
    this.ws = new WebSocket(url);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => this.ws.send(JSON.stringify({ type: "auth", ticket }));
    this.ws.onmessage = (m) => {
      if (typeof m.data === "string") return;
      const bytes = new Uint8Array(m.data);
      if (bytes.length < 4) return;
      const length = new DataView(m.data).getUint32(0);
      let header;
      try {
        header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length)));
      } catch {
        return;
      }
      if (header.type === "frame") {
        const mime = header.codec === "png" ? "image/png" : "image/jpeg";
        // Acked first: whatever happens drawing it, the next frame comes.
        this.send(JSON.stringify({ type: "ack", seq: header.seq }));
        const uri = `data:${mime};base64,${b64(bytes.subarray(4 + length))}`;
        event("frame", JSON.stringify({ seq: header.seq, w: header.w, h: header.h, uri }));
      } else {
        event("header", JSON.stringify(header));
      }
    };
    this.ws.onclose = (c) => event("close", c.reason || "the live view closed");
    this.ws.onerror = () => {};
  }
  send(text) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(text);
  }
  close() {
    this.ws.close();
  }
}

/**
 * Take a picture with the device's camera (a phone's browser opens it for a
 * file input with `capture`): its JPEG bytes, scaled to fit `maxDimension`
 * and upright, or null when nothing was taken.
 */
export function takePicture(maxDimension) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.capture = "environment";
    input.style.display = "none";
    let answered = false;
    const answer = async (file) => {
      if (answered) return;
      answered = true;
      input.remove();
      if (!file) return resolve(null);
      try {
        const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
        const scale = Math.min(1, maxDimension / Math.max(bitmap.width, bitmap.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(bitmap.width * scale));
        canvas.height = Math.max(1, Math.round(bitmap.height * scale));
        canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close();
        const blob = await new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.9));
        resolve(blob ? new Uint8Array(await blob.arrayBuffer()) : null);
      } catch {
        resolve(null);
      }
    };
    input.addEventListener("change", () => answer(input.files?.[0]));
    input.addEventListener("cancel", () => answer(null));
    document.body.append(input);
    input.click();
  });
}

/**
 * Save a file the page can reach (same origin) under `name`: the browser's
 * own download, without leaving the page.
 */
export function downloadUrl(url, name) {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.style.display = "none";
  document.body.append(link);
  link.click();
  link.remove();
}

// Pictures fetched ahead for the share sheet: a browser shares only during
// the tap, so the file must be ready when it comes.
const sharing = new Map();

/** Whether the browser has a share sheet. */
export function canShare() {
  return typeof navigator.share === "function";
}

/** Fetch a picture ahead for `shareFile` (kept for the page's life, at most a few). */
export function prefetchShare(url) {
  if (!canShare() || sharing.has(url)) return;
  if (sharing.size > 8) sharing.delete(sharing.keys().next().value);
  sharing.set(
    url,
    fetch(url)
      .then((r) => (r.ok ? r.blob() : null))
      .catch(() => null),
  );
}

/**
 * Share a picture as a file through the system's share sheet; where files
 * cannot be shared, save it instead (its link is private to the daemon).
 */
export async function shareFile(url, name) {
  prefetchShare(url);
  const blob = await sharing.get(url);
  if (blob) {
    const file = new File([blob], name, { type: blob.type });
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: name });
      } catch {
        // Cancelled, or the sheet refused: nothing to do.
      }
      return;
    }
  }
  downloadUrl(url, name);
}

/**
 * The picture at `url` with `strokes` drawn over it (JSON: strokes of
 * [x, y] from 0 to 1), as JPEG bytes no larger than `maxDimension`; null
 * when the picture cannot be read.
 */
export async function markUp(url, strokes, width, color, maxDimension) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const bitmap = await createImageBitmap(await response.blob(), {
      imageOrientation: "from-image",
    });
    const scale = Math.min(1, maxDimension / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const pen = canvas.getContext("2d");
    pen.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    pen.strokeStyle = color;
    pen.lineWidth = Math.max(2, width * Math.max(canvas.width, canvas.height));
    pen.lineCap = "round";
    pen.lineJoin = "round";
    for (const stroke of JSON.parse(strokes)) {
      if (!stroke.length) continue;
      pen.beginPath();
      pen.moveTo(stroke[0][0] * canvas.width, stroke[0][1] * canvas.height);
      if (stroke.length === 1) pen.lineTo(stroke[0][0] * canvas.width + 0.01, stroke[0][1] * canvas.height);
      for (const [x, y] of stroke.slice(1)) pen.lineTo(x * canvas.width, y * canvas.height);
      pen.stroke();
    }
    const blob = await new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.9));
    return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

/** Whether the page can recognize speech (the Web Speech API). */
export function dictationAvailable() {
  return typeof (window.SpeechRecognition || window.webkitSpeechRecognition) === "function";
}

/**
 * Dictation on the page's Web Speech API, reported as NSH's recognizer
 * reports it: `event(kind, text)` hears "start", "result" (the whole
 * transcript so far), "final" (the last one) and once "end" (the API's
 * error name, or ""). After `abort` nothing more is reported.
 */
export class Dictation {
  constructor(lang, interim, event) {
    let ended = false;
    this.send = (kind, text) => {
      if (ended) return;
      if (kind === "end") ended = true;
      event(kind, text);
    };
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      setTimeout(() => this.send("end", "unavailable"));
      return;
    }
    const r = new Recognition();
    this.r = r;
    r.continuous = true;
    r.interimResults = interim;
    if (lang) r.lang = lang;
    let error = "";
    let text = "";
    r.onstart = () => this.send("start", "");
    r.onresult = (e) => {
      let heard = "";
      for (const result of e.results) heard += result[0].transcript;
      text = heard.replace(/\s+/g, " ").trim();
      this.send("result", text);
    };
    r.onerror = (e) => {
      error = e.error || "failed";
    };
    r.onend = () => {
      if (text) this.send("final", text);
      this.send("end", error);
    };
    try {
      r.start();
    } catch {
      setTimeout(() => this.send("end", "failed"));
    }
  }
  stop() {
    this.r?.stop();
  }
  abort() {
    // The app dropped the session: it hears nothing more.
    this.send("end", "aborted");
    this.r?.abort();
  }
}

/** Whether the page can speak text (the Web Speech API's synthesis). */
export function speechAvailable() {
  return typeof window.speechSynthesis === "object" &&
    typeof window.SpeechSynthesisUtterance === "function";
}

/**
 * Text read aloud with the page's voice. `ended()` is called once, when the
 * last piece was spoken, it failed, or `stop` cut it short. Long text goes
 * in pieces of whole sentences: some browsers stop a long utterance
 * partway (Chrome after about fifteen seconds).
 */
export class Speaker {
  constructor(text, lang, ended) {
    let done = false;
    this.finish = () => {
      if (done) return;
      done = true;
      ended();
    };
    const synth = window.speechSynthesis;
    if (!synth || typeof window.SpeechSynthesisUtterance !== "function") {
      setTimeout(this.finish);
      return;
    }
    // One reading at a time on the page.
    synth.cancel();
    const pieces = [];
    for (const paragraph of text.split(/\n+/)) {
      let piece = "";
      for (const sentence of paragraph.match(/[^.!?]+[.!?]*\s*/g) || []) {
        if (piece && (piece + sentence).length > 200) {
          pieces.push(piece);
          piece = "";
        }
        piece += sentence;
      }
      if (piece.trim()) pieces.push(piece);
    }
    if (!pieces.length) {
      setTimeout(this.finish);
      return;
    }
    pieces.forEach((piece, n) => {
      const u = new SpeechSynthesisUtterance(piece);
      if (lang) u.lang = lang;
      if (n === pieces.length - 1) u.onend = this.finish;
      u.onerror = this.finish;
      synth.speak(u);
    });
  }
  stop() {
    window.speechSynthesis?.cancel();
    this.finish();
  }
}
