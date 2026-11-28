/*
 * aidaemon bridge protocol v1, page side (see ../SPEC.md and ../vectors/v1.json).
 *
 * A classic script: defines the global `AidaemonBridgeProto`, or exports it
 * through `module.exports` under Node (`require('./bridge-proto.js')`).
 * WebCrypto only, no dependencies. Written for Chrome 80 (Kindle Scribe):
 * no top-level await, no optional chaining or `??`, no class fields, no
 * String.replaceAll, no Promise.any, no WebAssembly.
 *
 * Everything a peer sends is untrusted; every parser refuses what the spec
 * does not allow, with a BridgeError whose `code` matches the Rust crate's
 * error variant (e.g. 'BadMac', 'Counter', 'TrailingBytes').
 */
(function (root, factory) {
  var api = factory(root.crypto);
  if (typeof module === 'object' && module && module.exports) {
    module.exports = api;
  } else {
    root.AidaemonBridgeProto = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : this), function (webcrypto) {
  'use strict';

  var subtle = webcrypto.subtle;

  // ---------------------------------------------------------------- constants

  var VERSION = 1;
  var HEADER_LEN = 11;
  var TAG_LEN = 16;
  var FLAG_FINAL = 0x01;
  var TYPE = { HELLO: 0x01, HELLO_REPLY: 0x02, FINISH: 0x03, DATA: 0x10, ABORT: 0x7f };
  var MAX_HANDSHAKE_FRAME = 1024;
  var MAX_DATA_FRAME = 64 * 1024;
  var MAX_FRAGMENT = MAX_DATA_FRAME - HEADER_LEN - TAG_LEN;
  var MAX_REQUEST = 16 * 1024 * 1024;
  var MAX_RESPONSE = 4 * 1024 * 1024;
  var POINT_LEN = 65;
  var NONCE_LEN = 32;
  var PAIRING_ID_LEN = 16;
  var MAX_DEVICE_NAME = 60;
  var MODE_SESSION = 1;
  var MODE_PAIRING = 2;
  var U64_MAX = BigInt('18446744073709551615');
  var CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
  var NAMEPLATE_LEN = 4;
  var SECRET_LEN = 13;

  var enc = new TextEncoder();
  function ascii(s) { return enc.encode(s); }

  var PROTOCOL_LABEL = ascii('aidaemon-bridge-v1');
  var SESSION_SIG_PREFIX = ascii('aidaemon-bridge-hs-v1:');
  var PAIRING_SIG_PREFIX = ascii('aidaemon-bridge-pair-v1:');
  var INFO_CONFIRM_DAEMON = ascii('aidaemon-bridge-v1 confirm daemon');
  var INFO_CONFIRM_DEVICE = ascii('aidaemon-bridge-v1 confirm device');
  var INFO_KEY_D2P = ascii('aidaemon-bridge-v1 key daemon to page');
  var INFO_KEY_P2D = ascii('aidaemon-bridge-v1 key page to daemon');
  var ROLE_DAEMON = ascii('daemon');
  var ROLE_DEVICE = ascii('device');
  var LABEL_DAEMON_ID = ascii('aidaemon-daemon-id-v1');
  var LABEL_DEVICE_ID = ascii('aidaemon-device-id-v1');
  var LABEL_ROUTE_ID = ascii('aidaemon-bridge-route-v1');

  var READY_JSON = '{"type":"ready"}';
  var REQUEST_HEADERS = ['authorization', 'content-type', 'x-aidaemon-request-id'];
  var RESPONSE_HEADERS = ['content-type'];
  var MAX_METHOD = 16;
  var MAX_PATH = 8 * 1024;
  var MAX_HEADER_VALUE = 8 * 1024;

  // ------------------------------------------------------------------ errors

  class BridgeError extends Error {
    constructor(code, detail) {
      super(detail ? code + ': ' + detail : code);
      this.name = 'BridgeError';
      this.code = code;
      this.detail = detail || null;
    }
  }
  function fail(code, detail) { throw new BridgeError(code, detail); }

  // ------------------------------------------------------------------- bytes

  function bytes(x) {
    if (x instanceof Uint8Array) return x;
    if (x instanceof ArrayBuffer) return new Uint8Array(x);
    if (ArrayBuffer.isView(x)) return new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
    if (typeof x === 'string') return enc.encode(x);
    fail('BadArgument', 'expected bytes');
  }

  function concat(parts) {
    var total = 0;
    var i;
    for (i = 0; i < parts.length; i++) total += parts[i].length;
    var out = new Uint8Array(total);
    var off = 0;
    for (i = 0; i < parts.length; i++) { out.set(parts[i], off); off += parts[i].length; }
    return out;
  }

  function u16(n) { return new Uint8Array([(n >>> 8) & 0xff, n & 0xff]); }
  function u32(n) {
    return new Uint8Array([(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff]);
  }

  /** lp(a) ‖ lp(b) ‖ … */
  function lp() {
    var parts = [];
    for (var i = 0; i < arguments.length; i++) {
      var b = bytes(arguments[i]);
      parts.push(u32(b.length), b);
    }
    return concat(parts);
  }

  function toHex(b) {
    b = bytes(b);
    var s = '';
    for (var i = 0; i < b.length; i++) s += (b[i] < 16 ? '0' : '') + b[i].toString(16);
    return s;
  }

  function fromHex(s) {
    if (typeof s !== 'string' || s.length % 2 !== 0 || !/^[0-9a-f]*$/.test(s)) fail('BadArgument', 'hex');
    var out = new Uint8Array(s.length / 2);
    for (var i = 0; i < out.length; i++) out[i] = parseInt(s.substr(i * 2, 2), 16);
    return out;
  }

  function equalBytes(a, b) {
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  /** Constant-time comparison of equal-length secrets (MACs). */
  function ctEqual(a, b) {
    if (a.length !== b.length) return false;
    var diff = 0;
    for (var i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
    return diff === 0;
  }

  function utf8Decode(b, what) {
    try {
      return new TextDecoder('utf-8', { fatal: true }).decode(b);
    } catch (e) {
      fail('BadMessage', what);
    }
  }

  /** Bounds-checked reader over a peer's bytes. */
  class Reader {
    constructor(b) { this.b = bytes(b); this.off = 0; }
    take(n) {
      if (this.b.length - this.off < n) fail('Short');
      var out = this.b.subarray(this.off, this.off + n);
      this.off += n;
      return out;
    }
    u8() { return this.take(1)[0]; }
    u16() { var b = this.take(2); return (b[0] << 8) | b[1]; }
    u32() { var b = this.take(4); return ((b[0] << 24) >>> 0) + (b[1] << 16) + (b[2] << 8) + b[3]; }
    lp() { return this.take(this.u32()); }
    lpFixed(n, field) {
      var b = this.lp();
      if (b.length !== n) fail('FieldLength', field + ' has length ' + b.length);
      return b;
    }
    finish() { if (this.off !== this.b.length) fail('TrailingBytes'); }
  }

  function randomBytes(n) {
    var out = new Uint8Array(n);
    webcrypto.getRandomValues(out);
    return out;
  }

  // -------------------------------------------------------------- primitives

  function sha256(x) {
    return subtle.digest('SHA-256', bytes(x)).then(function (d) { return new Uint8Array(d); });
  }

  function tagged(label, key) { return sha256(concat([label, bytes(key)])); }
  function daemonId(identityPub) { return tagged(LABEL_DAEMON_ID, identityPub); }
  function deviceId(devicePub) { return tagged(LABEL_DEVICE_ID, devicePub); }
  function routeId(loginPub) { return tagged(LABEL_ROUTE_ID, loginPub); }

  function checkPointShape(b, field) {
    if (b.length !== POINT_LEN || b[0] !== 0x04) fail('BadPoint', field);
  }

  /** Import a 65-byte uncompressed point (on-curve checked by WebCrypto). */
  async function importPoint(b, algorithm, usages, field) {
    b = bytes(b);
    checkPointShape(b, field);
    try {
      return await subtle.importKey('raw', b, { name: algorithm, namedCurve: 'P-256' }, true, usages);
    } catch (e) {
      fail('BadPoint', field);
    }
  }

  async function exportPoint(publicKey) {
    var raw = new Uint8Array(await subtle.exportKey('raw', publicKey));
    checkPointShape(raw, 'public key');
    return raw;
  }

  /** A fresh non-extractable ECDSA P-256 device key pair. */
  function generateDeviceKey() {
    return subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
  }

  /** A fresh ephemeral ECDH P-256 key pair. */
  function generateEphemeral() {
    return subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, false, ['deriveBits']);
  }

  async function ecdh(ephemeral, peerPoint, field) {
    var peer = await importPoint(peerPoint, 'ECDH', [], field);
    return new Uint8Array(await subtle.deriveBits({ name: 'ECDH', public: peer }, ephemeral.privateKey, 256));
  }

  async function sign(keyPair, message) {
    var sig = await subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, keyPair.privateKey, bytes(message));
    return new Uint8Array(sig);
  }

  /** Verify a 64-byte r‖s signature by a 65-byte point; throws BadSignature. */
  async function verify(publicPoint, message, signature) {
    var key = await importPoint(publicPoint, 'ECDSA', ['verify'], 'signing key');
    signature = bytes(signature);
    if (signature.length !== 64) fail('BadSignature');
    var ok = await subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, signature, bytes(message));
    if (!ok) fail('BadSignature');
  }

  /** HKDF-SHA-256 (extract with salt, expand with info) to 32 bytes. */
  async function hkdf(salt, ikm, info) {
    var key = await subtle.importKey('raw', bytes(ikm), 'HKDF', false, ['deriveBits']);
    var out = await subtle.deriveBits({ name: 'HKDF', hash: 'SHA-256', salt: bytes(salt), info: bytes(info) }, key, 256);
    return new Uint8Array(out);
  }

  async function hmac(key, message) {
    var k = await subtle.importKey('raw', bytes(key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
    return new Uint8Array(await subtle.sign('HMAC', k, bytes(message)));
  }

  async function deriveDataKeys(salt, ikm) {
    return { d2p: await hkdf(salt, ikm, INFO_KEY_D2P), p2d: await hkdf(salt, ikm, INFO_KEY_P2D) };
  }

  async function deriveConfirmKeys(salt, ikm) {
    return { daemon: await hkdf(salt, ikm, INFO_CONFIRM_DAEMON), device: await hkdf(salt, ikm, INFO_CONFIRM_DEVICE) };
  }

  function sessionTranscript(identityPub, devicePub, eK, eD, nK, nD) {
    return lp(PROTOCOL_LABEL, new Uint8Array([MODE_SESSION]), identityPub, devicePub, eK, eD, nK, nD);
  }

  function pairingTranscript(devicePub, deviceName, eK, nK, identityPub, eD, nD, pairingId) {
    return lp(PROTOCOL_LABEL, new Uint8Array([MODE_PAIRING]), devicePub, deviceName, eK, nK, identityPub, eD, nD, pairingId);
  }

  /** prefix ‖ H(lp(role) ‖ T); role is 'daemon' or 'device'. */
  async function signedMessage(prefix, role, transcript) {
    var label = role === 'daemon' ? ROLE_DAEMON : (role === 'device' ? ROLE_DEVICE : fail('BadArgument', 'role'));
    return concat([prefix, await sha256(concat([lp(label), bytes(transcript)]))]);
  }

  // ------------------------------------------------------------------ codes

  function randomCrockford(n) {
    // 256 is a multiple of 32, so the low 5 bits of each byte are uniform.
    var r = randomBytes(n);
    var s = '';
    for (var i = 0; i < n; i++) s += CROCKFORD.charAt(r[i] & 31);
    return s;
  }

  function isCrockford(s, n) {
    if (typeof s !== 'string' || s.length !== n) return false;
    for (var i = 0; i < n; i++) if (CROCKFORD.indexOf(s.charAt(i)) < 0) return false;
    return true;
  }

  function generateNameplate() { return randomCrockford(NAMEPLATE_LEN); }
  function generateSecret() { return randomCrockford(SECRET_LEN); }
  function isValidNameplate(s) { return isCrockford(s, NAMEPLATE_LEN); }
  function isValidSecret(s) { return isCrockford(s, SECRET_LEN); }
  /** What the owner typed, as a code: upper-cased, spaces and dashes removed. */
  function normalizeCode(s) { return String(s).toUpperCase().split(/[\s-]+/).join(''); }

  // ------------------------------------------------------------------ frames

  function header(type, flags, counter) {
    var h = new Uint8Array(HEADER_LEN);
    h[0] = VERSION;
    h[1] = type;
    h[2] = flags;
    new DataView(h.buffer).setBigUint64(3, BigInt(counter), false);
    return h;
  }

  function encodeFrame(type, flags, counter, body) {
    return concat([header(type, flags, counter), bytes(body || new Uint8Array(0))]);
  }

  function handshakeFrame(type, body) { return encodeFrame(type, 0, 0, body); }
  function abortFrame() { return encodeFrame(TYPE.ABORT, 0, 0, new Uint8Array(0)); }

  /** Decode a frame (strict header rules); counter is a BigInt. */
  function decodeFrame(b, maxLen) {
    b = bytes(b);
    if (b.length > maxLen) fail('FrameTooLarge', b.length + ' > ' + maxLen);
    if (b.length < HEADER_LEN) fail('Truncated');
    if (b[0] !== VERSION) fail('Version', String(b[0]));
    var type = b[1];
    if (type !== TYPE.HELLO && type !== TYPE.HELLO_REPLY && type !== TYPE.FINISH &&
        type !== TYPE.DATA && type !== TYPE.ABORT) fail('UnknownType', String(type));
    var flags = b[2];
    var counter = new DataView(b.buffer, b.byteOffset, b.byteLength).getBigUint64(3, false);
    var body = b.subarray(HEADER_LEN);
    if (type === TYPE.DATA) {
      if ((flags & ~FLAG_FINAL) !== 0) fail('Flags', String(flags));
      if (body.length < TAG_LEN) fail('Short');
    } else {
      if (flags !== 0) fail('Flags', String(flags));
      if (counter !== BigInt(0)) fail('HandshakeCounter', String(counter));
      if (type === TYPE.ABORT && body.length !== 0) fail('TrailingBytes');
    }
    return { type: type, flags: flags, counter: counter, body: body, header: b.subarray(0, HEADER_LEN) };
  }

  /** Decode a handshake frame of the expected type; ABORT → 'Aborted'. */
  function decodeHandshake(b, expected) {
    var f = decodeFrame(b, MAX_HANDSHAKE_FRAME);
    if (f.type === TYPE.ABORT) fail('Aborted');
    if (f.type !== expected) fail('UnexpectedFrame', String(f.type));
    return f.body;
  }

  function gcmNonce(counter) {
    var n = new Uint8Array(12);
    new DataView(n.buffer).setBigUint64(4, counter, false);
    return n;
  }

  function importAes(key, usage) {
    key = bytes(key);
    if (key.length !== 32) fail('BadArgument', 'key length');
    return subtle.importKey('raw', key, { name: 'AES-GCM' }, false, [usage]);
  }

  /**
   * Seals outgoing DATA for one direction. Counters are assigned in call
   * order and encryption is serialized, so frames resolve in counter order;
   * send each frame in that order.
   */
  class Sealer {
    constructor(key) {
      this._key = importAes(key, 'encrypt');
      this._next = BigInt(0);
      this._queue = Promise.resolve();
    }
    get nextCounter() { return this._next; }
    /** Seal one fragment (≤ MAX_FRAGMENT bytes); resolves to the frame. */
    sealFragment(plaintext, isFinal) {
      plaintext = bytes(plaintext);
      if (plaintext.length > MAX_FRAGMENT) return Promise.reject(new BridgeError('FrameTooLarge'));
      if (this._next > U64_MAX) return Promise.reject(new BridgeError('CounterExhausted'));
      var counter = this._next;
      this._next = counter + BigInt(1);
      var keyPromise = this._key;
      var job = this._queue.then(async function () {
        var aad = header(TYPE.DATA, isFinal ? FLAG_FINAL : 0, counter);
        var key = await keyPromise;
        var sealed = await subtle.encrypt({ name: 'AES-GCM', iv: gcmNonce(counter), additionalData: aad, tagLength: 128 }, key, plaintext);
        return concat([aad, new Uint8Array(sealed)]);
      });
      this._queue = job.catch(function () {});
      return job;
    }
    /** Seal a whole message; resolves to its frames, the last one FINAL. */
    sealMessage(message) {
      message = bytes(message);
      var jobs = [];
      if (message.length === 0) {
        jobs.push(this.sealFragment(message, true));
      } else {
        for (var off = 0; off < message.length; off += MAX_FRAGMENT) {
          var end = Math.min(off + MAX_FRAGMENT, message.length);
          jobs.push(this.sealFragment(message.subarray(off, end), end === message.length));
        }
      }
      return Promise.all(jobs);
    }
  }

  /**
   * Opens incoming DATA for one direction, enforcing the next counter, and
   * reassembles messages up to maxMessage. Frames are processed in call
   * order. After any error everything is refused ('Closed').
   */
  class Opener {
    constructor(key, maxMessage) {
      this._key = importAes(key, 'decrypt');
      this._next = BigInt(0);
      this._max = maxMessage;
      this._parts = [];
      this._pendingLen = 0;
      this._failed = false;
      this._queue = Promise.resolve();
    }
    get nextCounter() { return this._next; }
    /** Resolves to the completed message (Uint8Array) or null. */
    open(frameBytes) {
      var self = this;
      var copy = new Uint8Array(bytes(frameBytes));
      var job = this._queue.then(function () {
        if (self._failed) fail('Closed');
        return self._openInner(copy).catch(function (e) {
          self._failed = true;
          self._parts = [];
          self._pendingLen = 0;
          throw e;
        });
      });
      this._queue = job.catch(function () {});
      return job;
    }
    async _openInner(b) {
      var f = decodeFrame(b, MAX_DATA_FRAME);
      if (f.type === TYPE.ABORT) fail('Aborted');
      if (f.type !== TYPE.DATA) fail('UnexpectedFrame', String(f.type));
      if (f.counter !== this._next) fail('Counter', 'expected ' + this._next + ', got ' + f.counter);
      var key = await this._key;
      var plain;
      try {
        plain = new Uint8Array(await subtle.decrypt({ name: 'AES-GCM', iv: gcmNonce(f.counter), additionalData: f.header, tagLength: 128 }, key, f.body));
      } catch (e) {
        fail('Decrypt');
      }
      this._next = this._next + BigInt(1);
      if (this._pendingLen + plain.length > this._max) fail('MessageTooLarge', String(this._max));
      if ((f.flags & FLAG_FINAL) === 0) {
        this._parts.push(plain);
        this._pendingLen += plain.length;
        return null;
      }
      this._parts.push(plain);
      var message = concat(this._parts);
      this._parts = [];
      this._pendingLen = 0;
      return message;
    }
  }

  function pageChannel(keys) {
    return { sealer: new Sealer(keys.p2d), opener: new Opener(keys.d2p, MAX_RESPONSE) };
  }

  // -------------------------------------------------------------- handshakes

  function checkNonce(n, len) {
    n = bytes(n);
    if (n.length !== len) fail('BadArgument', 'nonce length');
    return n;
  }

  function useOnce(state, expected) {
    if (state._state !== expected) fail('BadState', 'expected ' + expected + ', in ' + state._state);
  }

  /**
   * Page, session mode (SPEC §5).
   *   const s = new PageSession({ deviceKey, identityPub });
   *   send(await s.hello());
   *   send((await s.onReply(replyFrame)).finish);
   *   const { sealer, opener } = await s.onReady(firstDataFrame);
   * Options `ephemeral` (ECDH CryptoKeyPair) and `nonce` are for tests only.
   */
  class PageSession {
    constructor(opts) {
      this._deviceKey = opts.deviceKey;
      this._identityPub = bytes(opts.identityPub);
      checkPointShape(this._identityPub, 'identity_pub');
      this._ephemeral = opts.ephemeral || null;
      this._nK = opts.nonce ? checkNonce(opts.nonce, NONCE_LEN) : randomBytes(NONCE_LEN);
      this._state = 'new';
    }
    async hello() {
      useOnce(this, 'new');
      this._state = 'hello';
      if (!this._ephemeral) this._ephemeral = await generateEphemeral();
      this._eK = await exportPoint(this._ephemeral.publicKey);
      this._devicePub = await exportPoint(this._deviceKey.publicKey);
      var id = await deviceId(this._devicePub);
      this._state = 'await_reply';
      return handshakeFrame(TYPE.HELLO, lp(new Uint8Array([MODE_SESSION]), this._eK, this._nK, id));
    }
    async onReply(frameBytes) {
      useOnce(this, 'await_reply');
      this._state = 'done';
      var r = new Reader(decodeHandshake(frameBytes, TYPE.HELLO_REPLY));
      var eD = r.lpFixed(POINT_LEN, 'E_d');
      var nD = r.lpFixed(NONCE_LEN, 'N_d');
      var sigD = r.lpFixed(64, 'sig_d');
      r.finish();
      var shared = await ecdh(this._ephemeral, eD, 'E_d');
      var t = sessionTranscript(this._identityPub, this._devicePub, this._eK, eD, this._nK, nD);
      await verify(this._identityPub, await signedMessage(SESSION_SIG_PREFIX, 'daemon', t), sigD);
      var sigK = await sign(this._deviceKey, await signedMessage(SESSION_SIG_PREFIX, 'device', t));
      var th = await sha256(t);
      this._channel = pageChannel(await deriveDataKeys(th, shared));
      this._th = th;
      this._state = 'await_ready';
      return { finish: handshakeFrame(TYPE.FINISH, lp(sigK)) };
    }
    /**
     * Open the daemon's first DATA frame, which must be the complete
     * `ready` control (exactly {"type":"ready"}). Only then does the session
     * resolve; the page sends no DATA before it (SPEC §5).
     */
    async onReady(frameBytes) {
      useOnce(this, 'await_ready');
      this._state = 'done';
      var ch = this._channel;
      this._channel = null;
      var msg = await ch.opener.open(frameBytes);
      if (msg === null) fail('BadMessage', 'ready must be one frame');
      var m = decodeFromDaemon(msg);
      if (m.type !== 'control' || m.json !== READY_JSON) fail('BadMessage', 'expected ready control');
      return { sealer: ch.sealer, opener: ch.opener, transcriptHash: this._th };
    }
  }

  /**
   * Page, pairing mode (SPEC §6).
   *   const p = new PagePairing({ deviceKey, deviceName, secret });
   *   send(await p.hello());
   *   const r = await p.onReply(replyFrame);   // checks mac_d and sig_d
   *   send(r.finish);
   *   const msg = await r.opener.open(nextFrame);
   *   const paired = await parsePairedControl(msg, r.identityPub);
   *   // only now store r.identityPub and paired.routeId; send no DATA
   *   // before `paired` has been opened (the bridge promotes first)
   */
  class PagePairing {
    constructor(opts) {
      this._deviceKey = opts.deviceKey;
      this._name = enc.encode(opts.deviceName);
      if (this._name.length < 1 || this._name.length > MAX_DEVICE_NAME) fail('BadDeviceName');
      if (!isValidSecret(opts.secret)) fail('BadSecret');
      this._secret = ascii(opts.secret);
      this._ephemeral = opts.ephemeral || null;
      this._nK = opts.nonce ? checkNonce(opts.nonce, NONCE_LEN) : randomBytes(NONCE_LEN);
      this._state = 'new';
    }
    async hello() {
      useOnce(this, 'new');
      this._state = 'hello';
      if (!this._ephemeral) this._ephemeral = await generateEphemeral();
      this._eK = await exportPoint(this._ephemeral.publicKey);
      this._devicePub = await exportPoint(this._deviceKey.publicKey);
      this._state = 'await_reply';
      return handshakeFrame(TYPE.HELLO, lp(new Uint8Array([MODE_PAIRING]), this._eK, this._nK, this._devicePub, this._name));
    }
    async onReply(frameBytes) {
      useOnce(this, 'await_reply');
      this._state = 'done';
      var r = new Reader(decodeHandshake(frameBytes, TYPE.HELLO_REPLY));
      var eD = r.lpFixed(POINT_LEN, 'E_d');
      var nD = r.lpFixed(NONCE_LEN, 'N_d');
      var identityPub = r.lpFixed(POINT_LEN, 'identity_pub');
      var pairingId = r.lpFixed(PAIRING_ID_LEN, 'pairing_id');
      var macD = r.lpFixed(32, 'mac_d');
      var sigD = r.lpFixed(64, 'sig_d');
      r.finish();
      var shared = await ecdh(this._ephemeral, eD, 'E_d');
      await importPoint(identityPub, 'ECDSA', ['verify'], 'identity_pub');
      var t = pairingTranscript(this._devicePub, this._name, this._eK, this._nK, identityPub, eD, nD, pairingId);
      var th = await sha256(t);
      var ikm = concat([shared, this._secret]);
      var confirm = await deriveConfirmKeys(th, ikm);
      if (!ctEqual(await hmac(confirm.daemon, th), macD)) fail('BadMac');
      await verify(identityPub, await signedMessage(PAIRING_SIG_PREFIX, 'daemon', t), sigD);
      var macK = await hmac(confirm.device, th);
      var sigK = await sign(this._deviceKey, await signedMessage(PAIRING_SIG_PREFIX, 'device', t));
      var ch = pageChannel(await deriveDataKeys(th, ikm));
      return {
        finish: handshakeFrame(TYPE.FINISH, lp(macK, sigK)),
        sealer: ch.sealer,
        opener: ch.opener,
        identityPub: new Uint8Array(identityPub),
        daemonId: await daemonId(identityPub),
        pairingId: new Uint8Array(pairingId),
        transcriptHash: th
      };
    }
  }

  /**
   * Check the `paired` control message that ends a pairing: it must be a
   * control message whose daemon_id matches the identity that signed.
   * Resolves to { daemonId, routeId } as hex.
   */
  async function parsePairedControl(messageBytes, identityPub) {
    var m = decodeFromDaemon(messageBytes);
    if (m.type !== 'control') fail('BadMessage', 'expected paired control');
    var obj;
    try { obj = JSON.parse(m.json); } catch (e) { fail('BadMessage', 'control json'); }
    if (!obj || obj.type !== 'paired' || typeof obj.daemon_id !== 'string' || typeof obj.route_id !== 'string' ||
        !/^[0-9a-f]{64}$/.test(obj.daemon_id) || !/^[0-9a-f]{64}$/.test(obj.route_id)) {
      fail('BadMessage', 'paired control');
    }
    if (obj.daemon_id !== toHex(await daemonId(identityPub))) fail('BadMessage', 'paired daemon_id mismatch');
    return { daemonId: obj.daemon_id, routeId: obj.route_id };
  }

  // ---------------------------------------------------------------- messages

  function isUpperAscii(s) {
    for (var i = 0; i < s.length; i++) { var c = s.charCodeAt(i); if (c < 0x41 || c > 0x5a) return false; }
    return true;
  }

  function checkMethod(m) {
    if (typeof m !== 'string' || m.length < 1 || m.length > MAX_METHOD || !isUpperAscii(m)) fail('BadMessage', 'method');
  }

  function checkPath(p) {
    if (typeof p !== 'string' || p.length < 1 || p.length > MAX_PATH || p.charAt(0) !== '/' || p.substr(0, 2) === '//') {
      fail('BadMessage', 'path_and_query');
    }
    for (var i = 0; i < p.length; i++) {
      var c = p.charCodeAt(i);
      if (c < 0x21 || c > 0x7e) fail('BadMessage', 'path_and_query');
    }
  }

  function checkHeaders(headers, allowed) {
    var seen = [];
    for (var i = 0; i < headers.length; i++) {
      var name = headers[i][0];
      var value = headers[i][1];
      if (typeof name !== 'string' || !/^[a-z-]+$/.test(name)) fail('BadMessage', 'header name');
      if (allowed.indexOf(name) < 0) fail('HeaderNotAllowed', name);
      if (seen.indexOf(name) >= 0) fail('BadMessage', 'repeated header');
      seen.push(name);
      if (typeof value !== 'string' || value.length > MAX_HEADER_VALUE) fail('BadMessage', 'header value');
      for (var j = 0; j < value.length; j++) {
        var c = value.charCodeAt(j);
        if (!(c === 0x09 || (c >= 0x20 && c <= 0x7e))) fail('BadMessage', 'header value');
      }
    }
  }

  function checkStatus(s) {
    if (typeof s !== 'number' || s !== Math.floor(s) || s < 100 || s > 599) fail('BadMessage', 'status');
  }

  function encodeHeaders(headers) {
    var parts = [u16(headers.length)];
    for (var i = 0; i < headers.length; i++) parts.push(lp(headers[i][0], headers[i][1]));
    return concat(parts);
  }

  function readHeaders(r) {
    var n = r.u16();
    var out = [];
    for (var i = 0; i < n; i++) {
      var name = utf8Decode(r.lp(), 'header name');
      var value = utf8Decode(r.lp(), 'header value');
      out.push([name, value]);
    }
    return out;
  }

  function checkId(id) {
    if (typeof id !== 'number' || id !== Math.floor(id) || id < 0 || id > 0xffffffff) fail('BadArgument', 'request id');
  }

  /** { requestId, method, pathAndQuery, headers: [[name, value]], body } → bytes. */
  function encodeRequest(req) {
    checkId(req.requestId);
    checkMethod(req.method);
    checkPath(req.pathAndQuery);
    var headers = req.headers || [];
    checkHeaders(headers, REQUEST_HEADERS);
    return concat([new Uint8Array([1]), u32(req.requestId), lp(req.method), lp(req.pathAndQuery),
      encodeHeaders(headers), lp(bytes(req.body || new Uint8Array(0)))]);
  }

  function decodeRequest(b) {
    var r = new Reader(b);
    if (r.u8() !== 1) fail('BadMessage', 'not a request');
    var requestId = r.u32();
    var method = utf8Decode(r.lp(), 'method');
    var pathAndQuery = utf8Decode(r.lp(), 'path_and_query');
    var headers = readHeaders(r);
    var body = new Uint8Array(r.lp());
    r.finish();
    checkMethod(method);
    checkPath(pathAndQuery);
    checkHeaders(headers, REQUEST_HEADERS);
    return { requestId: requestId, method: method, pathAndQuery: pathAndQuery, headers: headers, body: body };
  }

  /** { requestId, status, headers, body } → bytes. */
  function encodeResponse(resp) {
    checkId(resp.requestId);
    checkStatus(resp.status);
    var headers = resp.headers || [];
    checkHeaders(headers, RESPONSE_HEADERS);
    return concat([new Uint8Array([2]), u32(resp.requestId), u16(resp.status), encodeHeaders(headers),
      lp(bytes(resp.body || new Uint8Array(0)))]);
  }

  /** u8 3 ‖ lp(json); json is a string or bytes. */
  function encodeControl(json) {
    return concat([new Uint8Array([3]), lp(bytes(json))]);
  }

  /**
   * Decode what the daemon sent: { type: 'response', requestId, status,
   * headers, body } or { type: 'control', json } (json as a string).
   */
  function decodeFromDaemon(b) {
    var r = new Reader(b);
    var kind = r.u8();
    var out;
    if (kind === 2) {
      var requestId = r.u32();
      var status = r.u16();
      var headers = readHeaders(r);
      var body = new Uint8Array(r.lp());
      checkStatus(status);
      checkHeaders(headers, RESPONSE_HEADERS);
      out = { type: 'response', requestId: requestId, status: status, headers: headers, body: body };
    } else if (kind === 3) {
      var json = r.lp();
      if (json.length === 0) fail('BadMessage', 'control json');
      out = { type: 'control', json: utf8Decode(json, 'control json') };
    } else {
      fail('BadMessage', 'not a response or control');
    }
    r.finish();
    return out;
  }

  /** The value of a header in a decoded message, or null. */
  function header_(message, name) {
    for (var i = 0; i < message.headers.length; i++) if (message.headers[i][0] === name) return message.headers[i][1];
    return null;
  }

  return {
    VERSION: VERSION,
    TYPE: TYPE,
    FLAG_FINAL: FLAG_FINAL,
    HEADER_LEN: HEADER_LEN,
    MAX_HANDSHAKE_FRAME: MAX_HANDSHAKE_FRAME,
    MAX_DATA_FRAME: MAX_DATA_FRAME,
    MAX_FRAGMENT: MAX_FRAGMENT,
    MAX_REQUEST: MAX_REQUEST,
    MAX_RESPONSE: MAX_RESPONSE,
    MODE_SESSION: MODE_SESSION,
    MODE_PAIRING: MODE_PAIRING,
    CROCKFORD: CROCKFORD,
    READY_JSON: READY_JSON,
    REQUEST_HEADERS: REQUEST_HEADERS,
    RESPONSE_HEADERS: RESPONSE_HEADERS,
    BridgeError: BridgeError,
    // bytes and ids
    toHex: toHex,
    fromHex: fromHex,
    daemonId: daemonId,
    deviceId: deviceId,
    routeId: routeId,
    exportPoint: exportPoint,
    generateDeviceKey: generateDeviceKey,
    generateNameplate: generateNameplate,
    generateSecret: generateSecret,
    isValidNameplate: isValidNameplate,
    isValidSecret: isValidSecret,
    normalizeCode: normalizeCode,
    // frames
    encodeFrame: encodeFrame,
    decodeFrame: decodeFrame,
    decodeHandshake: decodeHandshake,
    abortFrame: abortFrame,
    Sealer: Sealer,
    Opener: Opener,
    // handshakes
    PageSession: PageSession,
    PagePairing: PagePairing,
    parsePairedControl: parsePairedControl,
    // messages
    encodeRequest: encodeRequest,
    decodeRequest: decodeRequest,
    encodeResponse: encodeResponse,
    encodeControl: encodeControl,
    decodeFromDaemon: decodeFromDaemon,
    header: header_,
    // building blocks (tests, a daemon-side double)
    primitives: {
      lp: lp,
      concat: concat,
      Reader: Reader,
      equalBytes: equalBytes,
      ctEqual: ctEqual,
      sha256: sha256,
      hkdf: hkdf,
      hmac: hmac,
      ecdh: ecdh,
      sign: sign,
      verify: verify,
      importPoint: importPoint,
      generateEphemeral: generateEphemeral,
      deriveDataKeys: deriveDataKeys,
      deriveConfirmKeys: deriveConfirmKeys,
      sessionTranscript: sessionTranscript,
      pairingTranscript: pairingTranscript,
      signedMessage: signedMessage,
      handshakeFrame: handshakeFrame,
      SESSION_SIG_PREFIX: SESSION_SIG_PREFIX,
      PAIRING_SIG_PREFIX: PAIRING_SIG_PREFIX
    }
  };
});
