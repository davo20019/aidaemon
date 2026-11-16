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
