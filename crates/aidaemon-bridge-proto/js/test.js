// Tests for bridge-proto.js: `node js/test.js` (Node 22+, WebCrypto global).
// Checks every vector in ../vectors/v1.golden the page can reproduce, verifies
// the Rust signatures, and runs full pairing + session handshakes against a
// daemon-side test double built from the same primitives.
'use strict';

var fs = require('fs');
var path = require('path');
var B = require('./bridge-proto.js');
var P = B.primitives;
var subtle = globalThis.crypto.subtle;

var V = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'vectors', 'v1.golden'), 'utf8'));
var hx = B.fromHex;
var toHex = B.toHex;

var passed = 0;
var failed = 0;
var tests = [];
function test(name, fn) { tests.push({ name: name, fn: fn }); }

function assert(cond, msg) { if (!cond) throw new Error('assertion failed: ' + msg); }
function eqHex(a, b, msg) {
  var x = typeof a === 'string' ? a : toHex(a);
  var y = typeof b === 'string' ? b : toHex(b);
  if (x !== y) throw new Error(msg + ': ' + x + ' != ' + y);
}
async function rejects(promiseOrFn, code, msg) {
  try {
    await (typeof promiseOrFn === 'function' ? promiseOrFn() : promiseOrFn);
  } catch (e) {
    if (code && e.code !== code) throw new Error(msg + ': expected ' + code + ', got ' + (e.code || e.message));
    return;
  }
  throw new Error(msg + ': expected rejection ' + (code || ''));
}

function b64u(b) { return Buffer.from(b).toString('base64').replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_'); }

// Import a fixed scalar as a CryptoKeyPair (tests only; real keys are generated).
async function keyPair(dHex, pubHex, alg) {
  var pub = hx(pubHex);
  var jwk = { kty: 'EC', crv: 'P-256', d: b64u(hx(dHex)), x: b64u(pub.subarray(1, 33)), y: b64u(pub.subarray(33)), ext: true };
  var usages = alg === 'ECDSA' ? ['sign'] : ['deriveBits'];
  var privateKey = await subtle.importKey('jwk', jwk, { name: alg, namedCurve: 'P-256' }, false, usages);
  var publicKey = await subtle.importKey('raw', pub, { name: alg, namedCurve: 'P-256' }, true, alg === 'ECDSA' ? ['verify'] : []);
  return { privateKey: privateKey, publicKey: publicKey };
}

function lpFields(frame) {
  var r = new P.Reader(frame.subarray(B.HEADER_LEN));
  var out = [];
  while (r.off < r.b.length) out.push(r.lp());
  return out;
}

// ------------------------------------------------------------------ vectors

test('ids', async function () {
  eqHex(await B.daemonId(hx(V.keys.identity_pub)), V.ids.daemon_id, 'daemon_id');
  eqHex(await B.deviceId(hx(V.keys.device_pub)), V.ids.device_id, 'device_id');
  eqHex(await B.routeId(hx(V.keys.login_pub)), V.ids.route_id, 'route_id');
  eqHex(B.abortFrame(), V.abort, 'abort');
});

test('login signature verifies', async function () {
  eqHex(P.concat([new TextEncoder().encode('aidaemon-bridge-auth-v1:'), hx(V.login.nonce)]), V.login.message, 'message');
  await P.verify(hx(V.keys.login_pub), hx(V.login.message), hx(V.login.sig));
});

async function vectorSession() {
  var S = V.session;
  var device = await keyPair(V.keys.device_private, V.keys.device_pub, 'ECDSA');
  var eph = await keyPair(S.page_ephemeral_private, S.E_k, 'ECDH');
  return new B.PageSession({ deviceKey: device, identityPub: hx(V.keys.identity_pub), ephemeral: eph, nonce: hx(S.N_k) });
}

test('session vectors', async function () {
  var S = V.session;
  var idp = hx(V.keys.identity_pub);
  var dp = hx(V.keys.device_pub);
  var t = P.sessionTranscript(idp, dp, hx(S.E_k), hx(S.E_d), hx(S.N_k), hx(S.N_d));
  eqHex(t, S.transcript, 'transcript');
  var th = await P.sha256(t);
  eqHex(th, S.transcript_hash, 'H(T)');
  var mD = await P.signedMessage(P.SESSION_SIG_PREFIX, 'daemon', t);
  var mK = await P.signedMessage(P.SESSION_SIG_PREFIX, 'device', t);
  eqHex(mD, S.daemon_signed_message, 'daemon signed message');
  eqHex(mK, S.device_signed_message, 'device signed message');
  await P.verify(idp, mD, hx(S.sig_d));
  await P.verify(idp, mD, hx(S.sig_d_high_s));
  await P.verify(dp, mK, hx(S.sig_k));
  await rejects(P.verify(dp, mD, hx(S.sig_k)), 'BadSignature', 'role swap');
  var pageEph = await keyPair(S.page_ephemeral_private, S.E_k, 'ECDH');
  var daemonEph = await keyPair(S.daemon_ephemeral_private, S.E_d, 'ECDH');
  eqHex(await P.ecdh(pageEph, hx(S.E_d), 'E_d'), S.ecdh, 'ecdh page');
  eqHex(await P.ecdh(daemonEph, hx(S.E_k), 'E_k'), S.ecdh, 'ecdh daemon');
  var keys = await P.deriveDataKeys(th, hx(S.ecdh));
  eqHex(keys.d2p, S.key_d2p, 'k_d2p');
  eqHex(keys.p2d, S.key_p2d, 'k_p2d');

  var page = await vectorSession();
  eqHex(await page.hello(), S.hello, 'HELLO');
  var res = await page.onReply(hx(S.hello_reply));
  var f = lpFields(res.finish);
  assert(f.length === 1, 'finish fields');
  await P.verify(dp, mK, f[0]);
  var data = S.data;
  assert(data[0].plaintext === toHex(B.encodeControl(S.ready_json)), 'ready plaintext');
  var ch = await page.onReady(hx(data[0].frame));
  eqHex(ch.transcriptHash, S.transcript_hash, 'channel H(T)');
  // Page → daemon request: AES-GCM is deterministic for a key and nonce.
  eqHex((await ch.sealer.sealFragment(hx(data[1].plaintext), true)), data[1].frame, 'p2d frame');
  eqHex(await ch.opener.open(hx(data[2].frame)), data[2].plaintext, 'd2p response');
  assert((await ch.opener.open(hx(data[3].frame))) === null, 'non-final fragment');
  eqHex(await ch.opener.open(hx(data[4].frame)), data[2].plaintext, 'reassembled');
});

test('pairing vectors', async function () {
  var S = V.pairing;
  var idp = hx(V.keys.identity_pub);
  var dp = hx(V.keys.device_pub);
  var name = new TextEncoder().encode(S.device_name);
  eqHex(name, S.device_name_utf8, 'device name utf8');
  var t = P.pairingTranscript(dp, name, hx(S.E_k), hx(S.N_k), idp, hx(S.E_d), hx(S.N_d), hx(S.pairing_id));
  eqHex(t, S.transcript, 'transcript');
  var th = await P.sha256(t);
  eqHex(th, S.transcript_hash, 'H(T)');
  var pageEph = await keyPair(S.page_ephemeral_private, S.E_k, 'ECDH');
  eqHex(await P.ecdh(pageEph, hx(S.E_d), 'E_d'), S.ecdh, 'ecdh');
  var ikm = P.concat([hx(S.ecdh), new TextEncoder().encode(S.secret)]);
  eqHex(ikm, S.ikm, 'ikm');
  var confirm = await P.deriveConfirmKeys(th, ikm);
  eqHex(confirm.daemon, S.confirm_daemon, 'confirm_d');
  eqHex(confirm.device, S.confirm_device, 'confirm_k');
  eqHex(await P.hmac(confirm.daemon, th), S.mac_d, 'mac_d');
  eqHex(await P.hmac(confirm.device, th), S.mac_k, 'mac_k');
  var keys = await P.deriveDataKeys(th, ikm);
  eqHex(keys.d2p, S.key_d2p, 'k_d2p');
  eqHex(keys.p2d, S.key_p2d, 'k_p2d');
  var mD = await P.signedMessage(P.PAIRING_SIG_PREFIX, 'daemon', t);
  var mK = await P.signedMessage(P.PAIRING_SIG_PREFIX, 'device', t);
  eqHex(mD, S.daemon_signed_message, 'daemon signed message');
  eqHex(mK, S.device_signed_message, 'device signed message');
  await P.verify(idp, mD, hx(S.sig_d));
  await P.verify(dp, mK, hx(S.sig_k));
  assert(toHex(await B.deviceId(dp)).indexOf(S.device_key_hash) === 0, 'device_key_hash prefix');
  assert(B.isValidSecret(S.secret) && B.isValidNameplate(S.nameplate), 'codes');
  assert(!B.isValidNameplate(S.nameplate.toLowerCase()) && B.isValidNameplate(B.normalizeCode(' 4f-7q ')), 'nameplate case');

  var device = await keyPair(V.keys.device_private, V.keys.device_pub, 'ECDSA');
  var page = new B.PagePairing({ deviceKey: device, deviceName: S.device_name, secret: S.secret, ephemeral: pageEph, nonce: hx(S.N_k) });
  eqHex(await page.hello(), S.hello, 'HELLO');
  var r = await page.onReply(hx(S.hello_reply));
  var f = lpFields(r.finish);
  eqHex(f[0], S.mac_k, 'FINISH mac_k');
  await P.verify(dp, mK, f[1]);
  eqHex(r.identityPub, V.keys.identity_pub, 'identity_pub');
  eqHex(r.daemonId, V.ids.daemon_id, 'daemon id');
  eqHex(r.pairingId, S.pairing_id, 'pairing id');
  var msg = await r.opener.open(hx(S.data[0].frame));
  eqHex(msg, S.data[0].plaintext, 'paired control');
  var paired = await B.parsePairedControl(msg, r.identityPub);
  assert(paired.routeId === V.ids.route_id && paired.daemonId === V.ids.daemon_id && paired.device === 'dv-0123456789abcdef0123', 'paired ids');
  eqHex(await r.sealer.sealFragment(hx(S.data[1].plaintext), true), S.data[1].frame, 'p2d frame');
});

test('message codec vectors', async function () {
  var M = V.messages;
  var req = { requestId: M.request.request_id, method: M.request.method, pathAndQuery: M.request.path_and_query, headers: M.request.headers, body: hx(M.request.body) };
  eqHex(B.encodeRequest(req), M.request.encoded, 'request');
  var back = B.decodeRequest(hx(M.request.encoded));
  assert(back.requestId === req.requestId && back.pathAndQuery === req.pathAndQuery && back.headers.length === 3, 'request decode');
  assert(B.header(back, 'x-aidaemon-request-id') === 'req-1', 'header lookup');
  var resp = { requestId: M.response.request_id, status: M.response.status, headers: M.response.headers, body: hx(M.response.body) };
  eqHex(B.encodeResponse(resp), M.response.encoded, 'response');
  var d = B.decodeFromDaemon(hx(M.response.encoded));
  assert(d.type === 'response' && d.status === 201 && toHex(d.body) === M.response.body, 'response decode');
  eqHex(B.encodeControl(M.control.json), M.control.encoded, 'control');
  var c = B.decodeFromDaemon(hx(M.control.encoded));
  assert(c.type === 'control' && c.json === M.control.json, 'control decode');
});

// ---------------------------------------------------------- daemon double

async function daemonKeys() {
  var identity = await subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify']);
  return { key: identity, pub: await B.exportPoint(identity.publicKey) };
}

// Session mode, daemon side, as the Rust crate does it.
async function daemonSession(identity, helloFrame, devicePub) {
  var r = new P.Reader(B.decodeHandshake(helloFrame, B.TYPE.HELLO));
  var mode = r.lpFixed(1, 'mode')[0];
  if (mode !== 1) throw new Error('mode');
  var eK = r.lpFixed(65, 'E_k');
  var nK = r.lpFixed(32, 'N_k');
  var id = r.lpFixed(32, 'device_id');
  r.finish();
  if (toHex(await B.deviceId(devicePub)) !== toHex(id)) throw new Error('unknown device');
  var eph = await P.generateEphemeral();
  var eD = await B.exportPoint(eph.publicKey);
  var nD = globalThis.crypto.getRandomValues(new Uint8Array(32));
  var t = P.sessionTranscript(identity.pub, devicePub, eK, eD, nK, nD);
  var sigD = await P.sign(identity.key, await P.signedMessage(P.SESSION_SIG_PREFIX, 'daemon', t));
  var shared = await P.ecdh(eph, eK, 'E_k');
  return {
    reply: P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(eD, nD, sigD)),
    finish: async function (finishFrame) {
      var fr = new P.Reader(B.decodeHandshake(finishFrame, B.TYPE.FINISH));
      var sigK = fr.lpFixed(64, 'sig_k');
      fr.finish();
      await P.verify(devicePub, await P.signedMessage(P.SESSION_SIG_PREFIX, 'device', t), sigK);
      var keys = await P.deriveDataKeys(await P.sha256(t), shared);
      return { sealer: new B.Sealer(keys.d2p), opener: new B.Opener(keys.p2d, B.MAX_REQUEST) };
    }
  };
}

async function daemonPairing(identity, helloFrame, secret, keyHash) {
  var r = new P.Reader(B.decodeHandshake(helloFrame, B.TYPE.HELLO));
  if (r.lpFixed(1, 'mode')[0] !== 2) throw new Error('mode');
  var eK = r.lpFixed(65, 'E_k');
  var nK = r.lpFixed(32, 'N_k');
  var devicePub = r.lpFixed(65, 'device_pub');
  var name = r.lp();
  r.finish();
  var id = await B.deviceId(devicePub);
  if (!P.ctEqual(id.subarray(0, keyHash.length), keyHash)) throw new Error('device key hash');
  var eph = await P.generateEphemeral();
  var eD = await B.exportPoint(eph.publicKey);
  var nD = globalThis.crypto.getRandomValues(new Uint8Array(32));
  var pid = globalThis.crypto.getRandomValues(new Uint8Array(16));
  var t = P.pairingTranscript(devicePub, name, eK, nK, identity.pub, eD, nD, pid);
  var th = await P.sha256(t);
  var ikm = P.concat([await P.ecdh(eph, eK, 'E_k'), new TextEncoder().encode(secret)]);
  var confirm = await P.deriveConfirmKeys(th, ikm);
  var macD = await P.hmac(confirm.daemon, th);
  var sigD = await P.sign(identity.key, await P.signedMessage(P.PAIRING_SIG_PREFIX, 'daemon', t));
  return {
    devicePub: devicePub,
    reply: P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(eD, nD, identity.pub, pid, macD, sigD)),
    finish: async function (finishFrame) {
      var fr = new P.Reader(B.decodeHandshake(finishFrame, B.TYPE.FINISH));
      var macK = fr.lpFixed(32, 'mac_k');
      var sigK = fr.lpFixed(64, 'sig_k');
      fr.finish();
      if (!P.ctEqual(await P.hmac(confirm.device, th), macK)) throw new B.BridgeError('BadMac');
      await P.verify(devicePub, await P.signedMessage(P.PAIRING_SIG_PREFIX, 'device', t), sigK);
      var keys = await P.deriveDataKeys(th, ikm);
      return { sealer: new B.Sealer(keys.d2p), opener: new B.Opener(keys.p2d, B.MAX_REQUEST) };
    }
  };
}

test('full pairing then session, live', async function () {
  var identity = await daemonKeys();
  var deviceKey = await B.generateDeviceKey();
  var devicePub = await B.exportPoint(deviceKey.publicKey);
  var secret = B.generateSecret();
  assert(B.isValidSecret(secret) && B.isValidNameplate(B.generateNameplate()), 'generated codes');
  var routeId = await B.routeId((await daemonKeys()).pub);

  // Pairing.
  var page = new B.PagePairing({ deviceKey: deviceKey, deviceName: 'Synthetic Reader', secret: secret });
  var hello = await page.hello();
  var d = await daemonPairing(identity, hello, secret, (await B.deviceId(devicePub)).subarray(0, 8));
  var r = await page.onReply(d.reply);
  var dch = await d.finish(r.finish);
  var json = JSON.stringify({ type: 'paired', daemon_id: toHex(await B.daemonId(identity.pub)), route_id: toHex(routeId), device: 'dv-0123456789abcdef0123' });
  var frames = await dch.sealer.sealMessage(B.encodeControl(json));
  var paired = await B.parsePairedControl(await r.opener.open(frames[0]), r.identityPub);
  assert(paired.routeId === toHex(routeId), 'route id');

  // Session with the pinned identity.
  var s = new B.PageSession({ deviceKey: deviceKey, identityPub: r.identityPub });
  var ds = await daemonSession(identity, await s.hello(), devicePub);
  var fin = await s.onReply(ds.reply);
  var dsch = await ds.finish(fin.finish);
  var ready = (await dsch.sealer.sealMessage(B.encodeControl(B.READY_JSON)))[0];
  var ch = await s.onReady(ready);

  var body = new Uint8Array(200000);
  for (var i = 0; i < body.length; i++) body[i] = i & 0xff;
  var req = { requestId: 9, method: 'POST', pathAndQuery: '/client/v1/upload', headers: [['content-type', 'application/octet-stream']], body: body };
  var reqFrames = await ch.sealer.sealMessage(B.encodeRequest(req));
  assert(reqFrames.length === 4 && reqFrames[0].length === B.MAX_DATA_FRAME, 'fragmented');
  var got = null;
  for (i = 0; i < reqFrames.length; i++) got = await dsch.opener.open(reqFrames[i]);
  var back = B.decodeRequest(got);
  assert(back.body.length === body.length && back.body[199999] === body[199999], 'request round trip');
  var respFrames = await dsch.sealer.sealMessage(B.encodeResponse({ requestId: 9, status: 200, headers: [], body: new Uint8Array(0) }));
  var resp = B.decodeFromDaemon(await ch.opener.open(respFrames[0]));
  assert(resp.type === 'response' && resp.requestId === 9, 'response');
  // Parallel opens still apply in call order.
  var many = [];
  for (i = 0; i < 5; i++) many.push(dsch.sealer.sealFragment(new Uint8Array([i]), true));
  many = await Promise.all(many);
  var outs = await Promise.all(many.map(function (f) { return ch.opener.open(f); }));
  for (i = 0; i < 5; i++) assert(outs[i][0] === i, 'ordered ' + i);
});

// ---------------------------------------------------------------- refusals

test('frame refusals', async function () {
  var key = new Uint8Array(32).fill(7);
  var s = new B.Sealer(key);
  var f0 = await s.sealFragment(new Uint8Array([1]), true);
  var f1 = await s.sealFragment(new Uint8Array([2]), true);
  await rejects(new B.Opener(key, 100).open(f1), 'Counter', 'gap');
  var o = new B.Opener(key, 100);
  await o.open(f0);
  await rejects(o.open(f0), 'Counter', 'repeat');
  await rejects(o.open(f1), 'Closed', 'poisoned');
  for (var i = 0; i < f0.length; i++) {
    var bad = new Uint8Array(f0);
    bad[i] ^= 1;
    await rejects(new B.Opener(key, 100).open(bad), null, 'flip ' + i);
  }
  var bad2 = new Uint8Array(f0); bad2[2] = 0;
  await rejects(new B.Opener(key, 100).open(bad2), 'Decrypt', 'FINAL stripped');
  var v = new Uint8Array(f0); v[0] = 2;
  await rejects(new B.Opener(key, 100).open(v), 'Version', 'version');
  var t = new Uint8Array(f0); t[1] = 0x11;
  await rejects(new B.Opener(key, 100).open(t), 'UnknownType', 'type');
  var fl = new Uint8Array(f0); fl[2] = 3;
  await rejects(new B.Opener(key, 100).open(fl), 'Flags', 'flags');
  await rejects(new B.Opener(key, 100).open(B.encodeFrame(B.TYPE.HELLO, 0, 0, new Uint8Array(20))), 'UnexpectedFrame', 'handshake after handshake');
  await rejects(new B.Opener(key, 100).open(B.abortFrame()), 'Aborted', 'abort');
  await rejects(new B.Opener(key, 100).open(B.encodeFrame(B.TYPE.DATA, 1, 0, new Uint8Array(B.MAX_DATA_FRAME))), 'FrameTooLarge', 'oversize frame');
  var big = new B.Sealer(key);
  var a = await big.sealFragment(new Uint8Array(60), false);
  var b = await big.sealFragment(new Uint8Array(60), true);
  var lim = new B.Opener(key, 100);
  assert((await lim.open(a)) === null, 'first');
  await rejects(lim.open(b), 'MessageTooLarge', 'reassembly limit');
  await rejects(new B.Sealer(key).sealFragment(new Uint8Array(B.MAX_FRAGMENT + 1), true), 'FrameTooLarge', 'oversize seal');
  await rejects(function () { return B.decodeFrame(B.encodeFrame(B.TYPE.FINISH, 0, 1, new Uint8Array(0)), 2048); }, 'HandshakeCounter', 'hs counter');
  await rejects(function () { return B.decodeFrame(B.encodeFrame(B.TYPE.ABORT, 0, 0, new Uint8Array(1)), 2048); }, 'TrailingBytes', 'abort body');
  await rejects(function () { return B.decodeHandshake(B.encodeFrame(B.TYPE.HELLO, 0, 0, new Uint8Array(1100)), B.TYPE.HELLO); }, 'FrameTooLarge', 'hs oversize');
});

test('session handshake refusals', async function () {
  var S = V.session;
  // Tampered reply bytes.
  var reply = hx(S.hello_reply);
  for (var i = B.HEADER_LEN; i < reply.length; i += 7) {
    var page = await vectorSession();
    await page.hello();
    var bad = new Uint8Array(reply); bad[i] ^= 0x80;
    await rejects(page.onReply(bad), null, 'flip ' + i);
  }
  // Wrong pinned identity.
  var device = await keyPair(V.keys.device_private, V.keys.device_pub, 'ECDSA');
  var other = await daemonKeys();
  var wrong = new B.PageSession({ deviceKey: device, identityPub: other.pub, ephemeral: await keyPair(S.page_ephemeral_private, S.E_k, 'ECDH'), nonce: hx(S.N_k) });
  await wrong.hello();
  await rejects(wrong.onReply(reply), 'BadSignature', 'wrong identity');
  // Trailing bytes.
  page = await vectorSession(); await page.hello();
  await rejects(page.onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.concat([reply.subarray(B.HEADER_LEN), new Uint8Array(4)]))), 'TrailingBytes', 'trailing');
  // Compressed E_d.
  page = await vectorSession(); await page.hello();
  var f = lpFields(reply);
  var comp = new Uint8Array(33); comp[0] = 2; comp.set(f[0].subarray(1, 33), 1);
  await rejects(page.onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(comp, f[1], f[2]))), 'FieldLength', 'compressed point');
  page = await vectorSession(); await page.hello();
  var off = new Uint8Array(f[0]); off[64] ^= 1;
  await rejects(page.onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(off, f[1], f[2]))), 'BadPoint', 'off-curve point');
  // Pairing prefix instead of session prefix.
  page = await vectorSession(); await page.hello();
  var identity = await keyPair(V.keys.identity_private, V.keys.identity_pub, 'ECDSA');
  var badSig = await P.sign(identity, await P.signedMessage(P.PAIRING_SIG_PREFIX, 'daemon', hx(S.transcript)));
  await rejects(page.onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(f[0], f[1], badSig))), 'BadSignature', 'wrong prefix');
  // Not twice.
  page = await vectorSession(); await page.hello();
  await page.onReply(reply);
  await rejects(page.onReply(reply), 'BadState', 'twice');
  // Ready refusals: the response frame (counter 1) and a bad control.
  page = await vectorSession(); await page.hello(); await page.onReply(reply);
  await rejects(page.onReady(hx(S.data[2].frame)), 'Counter', 'ready at wrong counter');
  var k = await P.deriveDataKeys(hx(S.transcript_hash), hx(S.ecdh));
  page = await vectorSession(); await page.hello(); await page.onReply(reply);
  var notReady = await new B.Sealer(k.d2p).sealFragment(B.encodeControl('{"type":"ready","x":1}'), true);
  await rejects(page.onReady(notReady), 'BadMessage', 'wrong ready json');
  page = await vectorSession(); await page.hello(); await page.onReply(reply);
  var partial = await new B.Sealer(k.d2p).sealFragment(B.encodeControl(B.READY_JSON), false);
  await rejects(page.onReady(partial), 'BadMessage', 'ready not final');
});

test('pairing handshake refusals', async function () {
  var S = V.pairing;
  var device = await keyPair(V.keys.device_private, V.keys.device_pub, 'ECDSA');
  async function fresh(secret) {
    var p = new B.PagePairing({ deviceKey: device, deviceName: S.device_name, secret: secret || S.secret, ephemeral: await keyPair(S.page_ephemeral_private, S.E_k, 'ECDH'), nonce: hx(S.N_k) });
    await p.hello();
    return p;
  }
  var reply = hx(S.hello_reply);
  await rejects((await fresh('7K3M9QXR2TVW5')).onReply(reply), 'BadMac', 'wrong secret');
  var f = lpFields(reply);
  var mac = new Uint8Array(f[4]); mac[0] ^= 1;
  await rejects((await fresh()).onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(f[0], f[1], f[2], f[3], mac, f[5]))), 'BadMac', 'tampered mac');
  var sig = new Uint8Array(f[5]); sig[63] ^= 1;
  await rejects((await fresh()).onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(f[0], f[1], f[2], f[3], f[4], sig))), 'BadSignature', 'tampered sig');
  await rejects((await fresh()).onReply(P.handshakeFrame(B.TYPE.HELLO_REPLY, P.lp(f[0], f[1], f[2], f[3], f[4]))), 'Short', 'missing sig');
  await rejects((await fresh()).onReply(B.abortFrame()), 'Aborted', 'abort');
  await rejects((await fresh()).onReply(hx(S.finish)), 'UnexpectedFrame', 'wrong type');
  await rejects(function () { return new B.PagePairing({ deviceKey: device, deviceName: '', secret: S.secret }); }, 'BadDeviceName', 'empty name');
  await rejects(function () { return new B.PagePairing({ deviceKey: device, deviceName: new Array(62).join('a'), secret: S.secret }); }, 'BadDeviceName', 'long name');
  await rejects(function () { return new B.PagePairing({ deviceKey: device, deviceName: 'x', secret: '7k3m9qxr2tvw4' }); }, 'BadSecret', 'lowercase secret');
  // A paired control for another daemon.
  var r = await (await fresh()).onReply(reply);
  var msg = await r.opener.open(hx(S.data[0].frame));
  await rejects(B.parsePairedControl(msg, (await daemonKeys()).pub), 'BadMessage', 'paired for another daemon');
});

test('message refusals', async function () {
  var base = { requestId: 1, method: 'GET', pathAndQuery: '/x', headers: [], body: new Uint8Array(0) };
  function req(over) { var r = Object.assign({}, base); return Object.assign(r, over); }
  await rejects(function () { return B.encodeRequest(req({ headers: [['cookie', 'a']] })); }, 'HeaderNotAllowed', 'cookie');
  await rejects(function () { return B.encodeRequest(req({ headers: [['Content-Type', 'a']] })); }, 'BadMessage', 'uppercase header');
  await rejects(function () { return B.encodeRequest(req({ headers: [['content-type', 'a'], ['content-type', 'b']] })); }, 'BadMessage', 'repeat');
  await rejects(function () { return B.encodeRequest(req({ headers: [['content-type', 'a\r\nb']] })); }, 'BadMessage', 'crlf');
  await rejects(function () { return B.encodeRequest(req({ method: 'get' })); }, 'BadMessage', 'method');
  await rejects(function () { return B.encodeRequest(req({ pathAndQuery: '//evil.example/' })); }, 'BadMessage', 'path');
  await rejects(function () { return B.encodeResponse({ requestId: 1, status: 200, headers: [['authorization', 'x']], body: new Uint8Array(0) }); }, 'HeaderNotAllowed', 'response header');
  await rejects(function () { return B.encodeResponse({ requestId: 1, status: 600, headers: [], body: new Uint8Array(0) }); }, 'BadMessage', 'status');
  var enc = hx(V.messages.request.encoded);
  await rejects(function () { return B.decodeRequest(P.concat([enc, new Uint8Array([0])])); }, 'TrailingBytes', 'trailing');
  await rejects(function () { return B.decodeRequest(enc.subarray(0, enc.length - 1)); }, 'Short', 'short');
  await rejects(function () { return B.decodeFromDaemon(enc); }, 'BadMessage', 'request from daemon');
  await rejects(function () { return B.decodeFromDaemon(B.encodeControl(new Uint8Array([0xff]))); }, 'BadMessage', 'non-utf8 control');
  await rejects(function () { return B.decodeFromDaemon(new Uint8Array([2, 0, 0, 0, 1, 0, 200, 0xff, 0xff])); }, 'Short', 'header count');
});

(async function main() {
  for (var i = 0; i < tests.length; i++) {
    try {
      await tests[i].fn();
      passed++;
      console.log('ok   ' + tests[i].name);
    } catch (e) {
      failed++;
      console.log('FAIL ' + tests[i].name + '\n     ' + (e && e.stack ? e.stack : e));
    }
  }
  console.log(passed + ' passed, ' + failed + ' failed');
  process.exit(failed ? 1 : 0);
})();
