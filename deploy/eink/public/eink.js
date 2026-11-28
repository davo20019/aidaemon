/* The aidaemon e-ink page: pairs with the owner's daemon through a bridge,
   then shows chats and approvals over an end-to-end encrypted channel
   (crates/aidaemon-bridge-proto/SPEC.md). Chrome 80 (Kindle Scribe): no
   optional chaining, no top-level await, no WebAssembly.

   Everything from the daemon is shown with textContent only: chats, cards
   and diffs carry text from emails and websites, never trusted as HTML. */
(function () {
  'use strict';

  if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' &&
      window.location.hostname !== '127.0.0.1') {
    window.location.replace('https://' + window.location.host + window.location.pathname);
    return;
  }

  var B = window.AidaemonBridgeProto;
  var CONFIG = window.AIDAEMON_EINK_CONFIG || {};
  var BRIDGE = String(CONFIG.bridge || '').replace(/\/+$/, '');
  // Where the owner's app opens a scanned pairing code (self-hosters: their
  // own web app's /pair-eink).
  var PAIR_URL = String(CONFIG.pair_url || 'https://app.aidaemon.ai/pair-eink');
  var enc = new TextEncoder();
  var dec = new TextDecoder();

  // ---------------------------------------------------------------- storage
  var DB_NAME = 'aidaemon-eink';
  var STORE = 'state';

  function openDb() {
    return new Promise(function (resolve, reject) {
      var req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = function () { req.result.createObjectStore(STORE); };
      req.onerror = function () { reject(req.error); };
      req.onsuccess = function () { resolve(req.result); };
    });
  }
  function dbGet(key) {
    return openDb().then(function (db) {
      return new Promise(function (resolve, reject) {
        var req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
        req.onsuccess = function () { resolve(req.result); };
        req.onerror = function () { reject(req.error); };
      });
    });
  }
  function dbPut(key, value) {
    return openDb().then(function (db) {
      return new Promise(function (resolve, reject) {
        var req = db.transaction(STORE, 'readwrite').objectStore(STORE).put(value, key);
        req.onsuccess = function () { resolve(); };
        req.onerror = function () { reject(req.error); };
      });
    });
  }

  // What the last visit showed, so a reopened page has something on screen
  // at once (marked with its time) while it refreshes. Per device only;
  // a private window or cleared storage just starts empty.
  var CACHE_KEY = 'aidaemon-eink-last';
  function cacheLoad() {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY) || 'null'); } catch (e) { return null; }
  }
  function cacheSave(snapshot) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), snapshot: snapshot })); } catch (e) { /* full or blocked */ }
  }

  // -------------------------------------------------------------------- dom
  function $(id) { return document.getElementById(id); }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  // Symbols the Kindle's fonts lack (they draw as empty boxes): emoji
  // outside the basic plane, media-control symbols, emoji presentation
  // selectors. Machine format: Unicode ranges.
  function drawable(text) {
    return String(text)
      .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, '')
      .replace(/[\u23E9-\u23FA\uFE0F\u200D]/g, '');
  }

  // A little Markdown, built from DOM nodes and text only (never HTML):
  // headings lose their #, **bold** and `code` show as such, lines stay.
  function richText(container, text) {
    // Links show as their text (no links leave this page); bullets as •.
    var lines = drawable(text)
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .split('\n')
      .map(function (line) { return line.replace(/^(\s*)[*-]\s+/, '$1\u2022 '); });
    lines.forEach(function (line, i) {
      if (i) container.appendChild(document.createElement('br'));
      var heading = /^\s{0,3}#{1,6}\s+/.exec(line);
      if (heading) line = line.slice(heading[0].length);
      var parts = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/);
      parts.forEach(function (part) {
        if (!part) return;
        var node;
        if (/^\*\*[^*]+\*\*$/.test(part)) node = el('b', null, part.slice(2, -2));
        else if (/^`[^`]+`$/.test(part)) node = el('code', null, part.slice(1, -1));
        else node = document.createTextNode(part);
        if (heading && node.nodeType === 3) { var b = el('b'); b.appendChild(node); node = b; }
        container.appendChild(node);
      });
    });
  }

  function note(text) { $('global-note').textContent = text || ''; }
  function setLink(text) { $('link-state').textContent = text; }
  function timeOf(ms) {
    if (!ms) return '';
    var d = new Date(ms), h = d.getHours(), m = d.getMinutes();
    return ((h % 12) || 12) + ':' + (m < 10 ? '0' : '') + m + (h < 12 ? ' AM' : ' PM');
  }

  var TAB_OF = { pair: null, reader: null, chat: 'chat', convo: 'chat', code: 'code', newcode: 'code', run: 'code',
    detail: 'code', approvals: 'approvals', approval: 'approvals', confirm: 'approvals' };
  var current = { view: 'chat', session: null, approval: null, run: null };
  // Bumped by each load of a screen, so an older load that finishes late
  // never adds to what a newer one drew.
  var loads = { history: 0, chats: 0, code: 0, run: 0 };

  // A conversation (or a run inside one) belongs to the tab it was opened from.
  function tabFor(view) {
    if (view === 'convo' || view === 'reader') return current.from || 'chat';
    return TAB_OF[view];
  }

  function show(view) {
    if (view !== 'convo' && view !== 'reader') stopWatching();
    current.view = view;
    var views = document.querySelectorAll('.view');
    for (var i = 0; i < views.length; i++) views[i].className = 'view';
    $('view-' + view).className = 'view active';
    $('tabs').style.display = view === 'pair' ? 'none' : 'flex';
    var tabs = document.querySelectorAll('.tab');
    for (var j = 0; j < tabs.length; j++) {
      tabs[j].className = tabs[j].getAttribute('data-view') === tabFor(view) ? 'tab active' : 'tab';
    }
    window.scrollTo(0, 0);
    remember(view);
  }

  // ---------------------------------------------------------------- address
  // Each screen has its own address after '#' (it never leaves the Kindle),
  // so a reload comes back to the same tab or conversation and the
  // browser's back button steps back: #chat, #code, #approvals,
  // #chat/<conversation>, #code/<conversation>.
  function routeOf(view) {
    if (view === 'convo' && current.session && current.session.id) {
      return '#' + (current.from || 'chat') + '/' + encodeURIComponent(current.session.id);
    }
    if (view === 'chat' || view === 'code' || view === 'approvals') return '#' + view;
    return null;
  }

  function remember(view) {
    var route = routeOf(view);
    if (!route || route === window.location.hash || !window.history || !history.pushState) return;
    // Opening a conversation is a step back can undo; switching tabs is not.
    if (view === 'convo') { history.pushState(null, '', route); current.pushed = true; }
    else history.replaceState(null, '', route);
  }

  // Go where the address says (on load, and when the back button changes it).
  function followRoute() {
    var m = /^#(chat|code|approvals)(?:\/(.+))?$/.exec(window.location.hash || '');
    if (!m) { show('chat'); return false; }
    if (m[2] && m[1] !== 'approvals') {
      var id = decodeURIComponent(m[2]);
      var known = null;
      var last = cacheLoad();
      ((last && last.snapshot && last.snapshot.sessions) || []).forEach(function (s) { if (s.id === id) known = s; });
      openChat(known || { id: id, title: '' }, m[1]);
      return true;
    }
    show(m[1]);
    return false;
  }

  // ------------------------------------------------------------- the channel
  // One WebSocket to the bridge, one request at a time (SPEC §8).
  var link = null;

  function socketTo(url) {
    var ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    var inbox = [];
    var waiters = [];
    var closed = null;
    ws.onmessage = function (ev) {
      var item = typeof ev.data === 'string' ? { text: ev.data } : { bytes: new Uint8Array(ev.data) };
      if (waiters.length) waiters.shift().resolve(item); else inbox.push(item);
    };
    ws.onclose = function (ev) {
      closed = 'closed (' + ev.code + ')';
      while (waiters.length) waiters.shift().reject(new Error(closed));
    };
    return {
      ws: ws,
      opened: new Promise(function (resolve, reject) {
        ws.onopen = function () { resolve(); };
        ws.onerror = function () { reject(new Error('Could not reach the bridge')); };
      }),
      next: function (ms) {
        if (inbox.length) return Promise.resolve(inbox.shift());
        if (closed) return Promise.reject(new Error(closed));
        return new Promise(function (resolve, reject) {
          var w = { resolve: resolve, reject: reject };
          waiters.push(w);
          if (ms) {
            setTimeout(function () {
              var i = waiters.indexOf(w);
              if (i >= 0) { waiters.splice(i, 1); reject(new Error('timed out')); }
            }, ms);
          }
        });
      },
      send: function (bytes) { ws.send(bytes); },
      close: function () { try { ws.close(); } catch (e) { /* gone */ } }
    };
  }

  async function nextBytes(sock, ms) {
    var item = await sock.next(ms);
    if (!item.bytes) throw new Error('unexpected message from the bridge');
    return item.bytes;
  }

  function makeLink(sock, channel, pinned) {
    var nextId = 1;
    var busy = Promise.resolve();
    var token = null;
    var tokenExpires = 0;
    async function raw(method, path, body, auth) {
      var requestId = nextId++;
      var headers = [['content-type', 'application/json']];
      if (auth) headers.push(['authorization', 'Bearer ' + auth]);
      var bytes = B.encodeRequest({
        requestId: requestId, method: method, pathAndQuery: path, headers: headers,
        body: method === 'GET' ? new Uint8Array(0) : enc.encode(JSON.stringify(body))
      });
      var frames = await channel.sealer.sealMessage(bytes);
      for (var i = 0; i < frames.length; i++) sock.send(frames[i]);
      for (;;) {
        var msg = await channel.opener.open(await nextBytes(sock, 60000));
        if (!msg) continue;
        var m = B.decodeFromDaemon(msg);
        if (m.type !== 'response' || m.requestId !== requestId) continue;
        var json = null;
        var type = B.header(m, 'content-type') || '';
        if (type.indexOf('json') >= 0) {
          try { json = JSON.parse(dec.decode(m.body)); } catch (e) { json = null; }
        }
        return { status: m.status, json: json, body: m.body, type: type };
      }
    }
    async function signIn() {
      var deviceHex = B.toHex(await B.deviceId(await B.exportPoint(pinned.key.publicKey)));
      var ch = await raw('POST', '/client/v1/sessions/challenge', { device_id: pinned.device, endpoint: deviceHex });
      if (ch.status !== 200 || !ch.json) throw new Error('Sign-in refused');
      var audience = 'bridge:' + pinned.daemonId;
      var message = 'aidaemon-client-v2:' + ch.json.challenge_id + ':' + ch.json.nonce + ':' + audience + ':' + deviceHex;
      var sig = new Uint8Array(await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, pinned.key.privateKey, enc.encode(message)));
      var grant = await raw('POST', '/client/v1/sessions', {
        challenge_id: ch.json.challenge_id, signature: base64url(sig), audience: audience
      });
      if (grant.status !== 200 || !grant.json || !grant.json.access_token) throw new Error('Sign-in refused');
      token = grant.json.access_token;
      tokenExpires = grant.json.expires_at_ms || 0;
    }
    function run(job) {
      var result = busy.then(job);
      busy = result.catch(function () {});
      return result;
    }
    return {
      // A command: its result, or throws the daemon's error message.
      command: function (command) {
        return run(async function () {
          if (!token || Date.now() > tokenExpires - 60000) await signIn();
          var requestId = 'k' + Date.now() + '-' + Math.floor(Math.random() * 1e9);
          var reply = await raw('POST', '/client/v1/commands',
            { type: 'command', request_id: requestId, command: command }, token);
          var frame = reply.json || {};
          if (frame.type === 'error' || reply.status !== 200) {
            var error = frame.error || {};
            throw new Error(error.message || 'Refused (' + reply.status + ')');
          }
          return frame.result;
        });
      },
      // A message's file (its signed link), as bytes, through the channel.
      file: function (path) {
        return run(async function () {
          if (!token || Date.now() > tokenExpires - 60000) await signIn();
          var reply = await raw('GET', path, {}, token);
          if (reply.status !== 200) throw new Error('Could not load a file');
          return { body: reply.body, type: reply.type };
        });
      },
      close: function () { sock.close(); }
    };
  }

  function base64url(bytes) {
    var s = '';
    for (var i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  async function connect(pinned) {
    setLink('connecting');
    var sock = socketTo(BRIDGE + '/v1/connect?route=' + pinned.routeId);
    await sock.opened;
    var session = new B.PageSession({ deviceKey: pinned.key, identityPub: B.fromHex(pinned.identityPub) });
    sock.send(await session.hello());
    var reply = await session.onReply(await nextBytes(sock, 30000));
    sock.send(reply.finish);
    var channel = await session.onReady(await nextBytes(sock, 30000));
    sock.ws.addEventListener('close', function () { link = null; setLink('offline'); });
    setLink('connected');
    return makeLink(sock, channel, pinned);
  }

  async function ensureLink() {
    if (link) return link;
    var pinned = await dbGet('pinned');
    if (!pinned) { show('pair'); throw new Error('Not paired'); }
    link = await connect(pinned);
    return link;
  }

  // ---------------------------------------------------------------- pairing
  async function pair() {
    $('btn-pair').disabled = true;
    $('pair-note').textContent = '';
    try {
      var key = await B.generateDeviceKey();
      var nameplate = B.generateNameplate();
      var secret = B.generateSecret();
      var deviceId = await B.deviceId(await B.exportPoint(key.publicKey));
      var code = nameplate + secret + base32First5(deviceId);
      $('pair-code').textContent = code.match(/.{1,5}/g).join('-');
      $('pair-check').textContent = await checkCode(nameplate, secret);
      var link = PAIR_URL + '#n=' + nameplate + '&s=' + secret + '&k=' + B.toHex(deviceId);
      if (window.AidaemonQr) window.AidaemonQr.drawToCanvas($('pair-qr'), link, 5);
      $('pair-code-box').style.display = 'block';
      var sock = socketTo(BRIDGE + '/v1/wait?nameplate=' + nameplate);
      await sock.opened;
      var joined = await sock.next(600000);
      if (!joined.text || JSON.parse(joined.text).type !== 'joined') throw new Error('unexpected message');
      $('pair-status').textContent = 'Your daemon answered. Pairing…';
      var pairing = new B.PagePairing({ deviceKey: key, deviceName: 'Kindle', secret: secret });
      sock.send(await pairing.hello());
      var r = await pairing.onReply(await nextBytes(sock, 30000));
      sock.send(r.finish);
      var msg = null;
      while (!msg) msg = await r.opener.open(await nextBytes(sock, 30000));
      var paired = await B.parsePairedControl(msg, r.identityPub);
      var pinned = {
        key: key, identityPub: B.toHex(r.identityPub), routeId: paired.routeId,
        daemonId: paired.daemonId, device: paired.device
      };
      await dbPut('pinned', pinned);
      sock.close();
      $('pair-status').textContent = 'Paired.';
      link = await connect(pinned);
      show('chat');
      refreshAll();
    } catch (e) {
      $('pair-note').textContent = 'Pairing did not finish (' + e.message + '). Tap Pair to start again.';
      $('pair-code-box').style.display = 'none';
    } finally {
      $('btn-pair').disabled = false;
    }
  }

  // The check code the owner's app asks for after a scan: the first 4
  // Crockford characters of SHA-256("aidaemon-eink-check-v1" ‖ nameplate ‖
  // secret) (aidaemon-client-core computes the same).
  async function checkCode(nameplate, secret) {
    var digest = new Uint8Array(await crypto.subtle.digest('SHA-256',
      enc.encode('aidaemon-eink-check-v1' + nameplate + secret)));
    return base32First5(digest).slice(0, 4);
  }

  // The first 5 bytes of the device id as 8 Crockford Base32 characters.
  function base32First5(bytes) {
    var bits = 0, value = 0, out = '';
    for (var i = 0; i < 5; i++) {
      value = (value << 8) | bytes[i];
      bits += 8;
      while (bits >= 5) {
        out += B.CROCKFORD[(value >>> (bits - 5)) & 31];
        bits -= 5;
      }
      value &= (1 << bits) - 1;
    }
    return out;
  }

  // ------------------------------------------------------------------- chat
  var sessions = [];
  // Project ids whose project has a folder (read once per connection).
  var folderProjects = null;

  // The app's rule (aidaemon-client-core's Client::is_coding): a
  // conversation works on code when it is set to a coding agent, is in a
  // project with a folder, or a coding agent ran in it.
  function isCoding(s) {
    if (s.agent) return true;
    return !!(s.project_id && folderProjects && folderProjects[s.project_id]);
  }

  function sortedSessions(list) {
    return (list || []).filter(function (s) { return !s.archived; }).sort(function (a, b) {
      return (b.last_activity_ms || 0) - (a.last_activity_ms || 0);
    });
  }

  function renderChats() {
    var all = ofKind('chat');
    sessions = all.slice(0, shown.chat);
    moreButton('chat', all.length);
    var ul = $('chat-list');
    clear(ul);
    sessions.forEach(function (s) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', s.title || 'Untitled chat'));
      b.appendChild(el('span', 'meta', s.channel + ' · ' + timeOf(s.last_activity_ms)));
      b.addEventListener('click', function () { openChat(s); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    if (!sessions.length) ul.appendChild(el('li', 'empty muted', 'No chats yet.'));
  }

  var blobUrls = [];

  // Pictures in messages (handwriting, screenshots), one at a time through
  // the channel, shown as blob: URLs (the content policy allows those).
  async function loadImages(images, mine) {
    for (var i = 0; i < images.length; i++) {
      if (mine !== loads.history) return;
      try {
        var path = images[i].url.replace(/^https?:\/\/[^\/]+/, '');
        var f = await (await ensureLink()).file(path);
        var url = URL.createObjectURL(new Blob([f.body], { type: f.type || 'image/png' }));
        blobUrls.push(url);
        images[i].img.src = url;
      } catch (e) {
        images[i].img.alt = 'Could not load the picture';
      }
    }
  }

  async function openChat(s, from) {
    current.session = s;
    current.historyLimit = 8;
    current.historySig = null;
    current.from = from || 'chat';
    $('convo-title').textContent = (s && s.title) || (s && s.id ? 'Conversation' : 'New chat');
    $('convo-note').textContent = '';
    show('convo');
    current.toEnd = true;
    await loadHistory();
    var busyNow = ((overview && overview.running_tasks) || []).some(function (t) {
      return s && t.session_id === s.id;
    });
    if (s && s.id && busyNow) watchConversation(s.id, true);
  }



  // How much of a message shows in the conversation; the rest is a tap
  // away in the reader, where it pages a screen at a time.
  var PREVIEW_LINES = 6;
  var PREVIEW_CHARS = 420;

  function preview(text) {
    var lines = String(text || '').split('\n').filter(function (l, i, all) {
      return l.trim() || (i > 0 && all[i - 1].trim());
    });
    var cut = lines.slice(0, PREVIEW_LINES).join('\n');
    if (cut.length > PREVIEW_CHARS) cut = cut.slice(0, cut.lastIndexOf(' ', PREVIEW_CHARS) > 200 ? cut.lastIndexOf(' ', PREVIEW_CHARS) : PREVIEW_CHARS);
    var whole = lines.join('\n');
    return { text: cut.length < whole.length ? cut.replace(/\s+$/, '') + ' \u2026' : whole, more: cut.length < whole.length };
  }

  // A message body: a short one whole, a long one previewed with Read all.
  function messageBody(container, text, from) {
    var p = preview(text);
    var body = el('div', 'msg-text');
    richText(body, p.text);
    container.appendChild(body);
    if (p.more) {
      var more = el('button', 'btn btn-small', 'Read all \u25b8');
      more.type = 'button';
      more.addEventListener('click', function () { openReader(text, from); });
      container.appendChild(more);
    }
  }

  var readerReturn = 0;
  function openReader(text, from) {
    readerReturn = window.pageYOffset;
    $('reader-from').textContent = from || '';
    var body = $('reader-body');
    clear(body);
    richText(body, text);
    show('reader');
  }

  function pageBy(direction) {
    // A screen at a time, keeping two lines from the last one.
    window.scrollBy(0, direction * Math.max(200, window.innerHeight - 140));
  }

  // While work runs in the open conversation, a small glance every few
  // seconds; when it ends, the conversation reloads once and shows the new
  // messages at the end. Nothing redraws while nothing changed (e-ink).
  var WATCH_MS = 8000;
  var watching = null;

  function watchConversation(id, expectWork) {
    stopWatching();
    var sawWork = !!expectWork;
    var quietChecks = 0;
    watching = setInterval(async function () {
      if (current.view !== 'convo' || !current.session || current.session.id !== id) { stopWatching(); return; }
      try {
        var glance = (await (await ensureLink()).command({ type: 'get_glance' })).glance || {};
        var busy = (glance.running_tasks || []).some(function (t) { return t.session_id === id; });
        if (busy) { sawWork = true; quietChecks = 0; return; }
        // Just sent: give the daemon a few checks to start before giving up.
        if (!sawWork && ++quietChecks < 4) return;
        // The turn is done; a coding agent may still be running in the
        // background. Reload (redrawn only if something changed) and keep
        // watching while one is.
        current.toEnd = true;
        var agentsRunning = await loadHistory();
        if (!agentsRunning) stopWatching();
        else sawWork = true;
      } catch (e) {
        stopWatching();
      }
    }, WATCH_MS);
  }

  function stopWatching() {
    if (watching) { clearInterval(watching); watching = null; }
  }

  async function loadHistory() {
    var mine = ++loads.history;
    var box = $('convo-messages');
    if (!current.session || !current.session.id) { clear(box); return; }
    var l = await ensureLink();
    var result = await l.command({ type: 'get_history', session_id: current.session.id, limit: current.historyLimit || 8 });
    if (mine !== loads.history) return;
    var running = (result.runs || []).some(function (r) { return r.phase === 'running'; });
    // Same messages and runs as on screen: leave the page alone (e-ink).
    var sig = (result.messages || []).map(function (m) { return m.id; }).join(',') + '|' +
      (result.runs || []).map(function (r) { return r.id + ':' + r.phase; }).join(',') + '|' + (current.historyLimit || 8);
    if (sig === current.historySig && current.historySession === current.session.id) {
      current.toEnd = false;
      return running;
    }
    current.historySig = sig;
    current.historySession = current.session.id;
    clear(box);
    blobUrls.forEach(function (u) { URL.revokeObjectURL(u); });
    blobUrls = [];
    var images = [];
    // A coding agent's answer lives on its run, not in a message: put each
    // ended run in time order among the messages, its answer a tap away.
    var items = (result.messages || []).map(function (m) { return { at: m.at_ms || 0, message: m }; });
    // Only the runs within the messages shown (the history lists them all).
    var oldest = items.length ? items[0].at : 0;
    (result.runs || []).forEach(function (r) {
      var at = r.ended_at_ms || r.started_at_ms || 0;
      if (!result.more || at >= oldest) items.push({ at: at, run: r });
    });
    items.sort(function (a, b) { return a.at - b.at; });
    // Older messages are fetched only when asked for.
    if (result.more) {
      var earlier = el('button', 'btn', 'Show earlier');
      earlier.type = 'button';
      earlier.addEventListener('click', function () {
        current.historyLimit = (current.historyLimit || 8) + 10;
        loadHistory();
      });
      box.appendChild(earlier);
    }
    items.forEach(function (item) {
      if (item.run) {
        var r = item.run;
        var card = el('div', 'msg');
        card.appendChild(el('div', 'msg-from', (r.agent || 'Agent') + ' · ' + (PHASE_WORD[r.phase] || r.phase) +
          (r.took ? ' · ' + r.took : '')));
        // A run's result is { text, report, mismatches }: its answer is text.
        var said = r.result && typeof r.result.text === 'string' ? r.result.text : '';
        messageBody(card, said || (r.phase === 'running' ? 'Working…' : 'No answer recorded.'), r.agent || 'Agent');
        var steps = el('button', 'btn btn-small', 'Steps \u25b8');
        steps.type = 'button';
        steps.addEventListener('click', function () { openRun(r); });
        card.appendChild(steps);
        box.appendChild(card);
        return;
      }
      var m = item.message;
      var d = el('div', m.role === 'user' ? 'msg msg-you' : 'msg');
      d.appendChild(el('div', 'msg-from', m.role === 'user' ? 'You' : 'aidaemon'));
      messageBody(d, m.text, m.role === 'user' ? 'You' : 'aidaemon');
      (m.attachments || []).forEach(function (a) {
        if (!a.url || !/^image\//.test(a.mime || '')) {
          d.appendChild(el('div', 'meta', 'File: ' + a.name));
          return;
        }
        var img = document.createElement('img');
        img.className = 'msg-image';
        img.alt = a.name;
        d.appendChild(img);
        images.push({ img: img, url: a.url });
      });
      box.appendChild(d);
    });
    loadImages(images, mine);
    if (current.toEnd) {
      current.toEnd = false;
      window.scrollTo(0, document.body.scrollHeight);
    }
    // An agent still running here: watch for its answer.
    if (running && !watching) watchConversation(current.session.id, true);
    return running;
  }

  // ---------------------------------------------------------- writing strip
  // Handwriting goes as pictures the model reads. Lines are written on a
  // short strip with "fast ink" (pure black squares, no smoothing: what the
  // Scribe redraws fastest), kept as canvases, and combined at send time
  // into a few sheets within the daemon's upload limits.
  var LINES_PER_SHEET = 8;
  var MAX_SHEETS = 4;
  var writing = { mode: 'type', lines: [], inked: false, drawing: false, last: null };

  function strip() { return $('strip'); }
  function stripCtx() { return strip().getContext('2d'); }

  function sizeStrip() {
    var c = strip();
    if (!c.clientWidth) return;
    if (c.width === c.clientWidth && c.height === c.clientHeight) return;
    c.width = c.clientWidth;
    c.height = c.clientHeight;
    clearStrip();
  }

  function clearStrip() {
    var ctx = stripCtx();
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, strip().width, strip().height);
    writing.inked = false;
  }

  function plot(a, b) {
    var ctx = stripCtx();
    ctx.fillStyle = '#000';
    var dx = b.x - a.x, dy = b.y - a.y;
    var n = Math.max(1, Math.ceil(Math.max(Math.abs(dx), Math.abs(dy))));
    for (var i = 0; i <= n; i++) {
      ctx.fillRect(Math.round(a.x + dx * i / n) - 1, Math.round(a.y + dy * i / n) - 1, 3, 3);
    }
    writing.inked = true;
  }

  function stripPoint(x, y) {
    var r = strip().getBoundingClientRect();
    return { x: x - r.left, y: y - r.top };
  }

  function setupStrip() {
    var c = strip();
    c.addEventListener('pointerdown', function (e) {
      sizeStrip();
      writing.drawing = true;
      writing.last = stripPoint(e.clientX, e.clientY);
      try { c.setPointerCapture(e.pointerId); } catch (err) { /* old browser */ }
      e.preventDefault();
    });
    c.addEventListener('pointermove', function (e) {
      if (!writing.drawing) return;
      // The browser batches pen samples; take them all for smooth curves.
      var list = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [];
      if (!list || !list.length) list = [e];
      for (var i = 0; i < list.length; i++) {
        var p = stripPoint(list[i].clientX, list[i].clientY);
        plot(writing.last, p);
        writing.last = p;
      }
      e.preventDefault();
    });
    function end() { writing.drawing = false; writing.last = null; }
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('touchstart', function (e) { e.preventDefault(); }, { passive: false });
    c.addEventListener('touchmove', function (e) { e.preventDefault(); }, { passive: false });
  }

  function keepLine() {
    if (!writing.inked) return true;
    if (writing.lines.length >= LINES_PER_SHEET * MAX_SHEETS) {
      $('convo-note').textContent = 'That is as many lines as one message takes. Send it, then write more.';
      return false;
    }
    var copy = document.createElement('canvas');
    copy.width = strip().width;
    copy.height = strip().height;
    copy.getContext('2d').drawImage(strip(), 0, 0);
    writing.lines.push(copy);
    var img = document.createElement('canvas');
    img.width = copy.width;
    img.height = copy.height;
    img.getContext('2d').drawImage(copy, 0, 0);
    $('write-lines').appendChild(img);
    clearStrip();
    return true;
  }

  function clearAllLines() {
    writing.lines = [];
    clear($('write-lines'));
    clearStrip();
  }

  function setMode(mode) {
    writing.mode = mode;
    // One input at a time: the strip, or the text box.
    $('write-box').className = mode === 'write' ? '' : 'hidden';
    $('convo-text').className = mode === 'type' ? '' : 'hidden';
    $('mode-type').className = mode === 'type' ? 'btn mode active' : 'btn mode';
    $('mode-write').className = mode === 'write' ? 'btn mode active' : 'btn mode';
    if (mode === 'write') sizeStrip();
  }

  // The kept lines as PNG sheets (base64), at most MAX_SHEETS.
  async function sheets() {
    var out = [];
    for (var i = 0; i < writing.lines.length; i += LINES_PER_SHEET) {
      var group = writing.lines.slice(i, i + LINES_PER_SHEET);
      var sheet = document.createElement('canvas');
      sheet.width = group[0].width;
      sheet.height = group.reduce(function (h, c) { return h + c.height; }, 0);
      var ctx = sheet.getContext('2d');
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, sheet.width, sheet.height);
      var y = 0;
      group.forEach(function (c) { ctx.drawImage(c, 0, y); y += c.height; });
      out.push(sheet.toDataURL('image/png').split(',')[1]);
    }
    return out;
  }

  async function send() {
    var text = $('convo-text').value.trim();
    if (writing.mode === 'write' && !keepLine()) return;
    var handwriting = writing.mode === 'write' && writing.lines.length > 0;
    if (!text && !handwriting) {
      $('convo-note').textContent = writing.mode === 'write' ? 'Write something first.' : 'Type something first.';
      return;
    }
    $('btn-send').disabled = true;
    try {
      var l = await ensureLink();
      var attachments = [];
      if (handwriting) {
        $('convo-note').textContent = 'Sending your handwriting…';
        var pngs = await sheets();
        for (var i = 0; i < pngs.length; i++) {
          var up = await l.command({
            type: 'upload_file', name: 'handwriting-' + (i + 1) + '.png', mime: 'image/png', data: pngs[i]
          });
          attachments.push(up.attachment);
        }
      }
      var command = { type: 'send_message', text: text };
      if (attachments.length) command.attachments = attachments;
      if (current.session && current.session.id) command.session_id = current.session.id;
      var result = await l.command(command);
      // Strokes are kept until the daemon took the message.
      if (handwriting) clearAllLines();
      $('convo-text').value = '';
      if (!current.session || !current.session.id) current.session = { id: result.session_id, title: text.slice(0, 40) };
      $('convo-note').textContent = 'Sent. The reply appears here when it is ready.';
      current.toEnd = true;
      await loadHistory();
      watchConversation(current.session.id, false);
    } catch (e) {
      $('convo-note').textContent = e.message;
    } finally {
      $('btn-send').disabled = false;
    }
  }

  // ------------------------------------------------------------------- code
  var PHASE_WORD = { running: 'running', succeeded: 'done', failed: 'failed', stopped: 'stopped' };

  // What the lists show, loaded a page at a time: the daemon's full
  // snapshot lists every conversation (hundreds, ~70 KB) on each refresh,
  // too much for the Kindle's link. A page of conversations is ~2.5 KB.
  var PAGE = 10;
  var overview = { sessions: [], cursor: null, done: false, running_tasks: [], pending_approvals: [] };
  var shown = { chat: PAGE, code: PAGE };

  async function loadSnapshot() {
    var l = await ensureLink();
    var mine = ++loads.code;
    if (!folderProjects) {
      var projects = (await l.command({ type: 'list_projects' })).projects || [];
      folderProjects = {};
      projects.forEach(function (p) { if (p.path) folderProjects[p.id] = true; });
    }
    var glance = (await l.command({ type: 'get_glance' })).glance || {};
    var page = await l.command({ type: 'list_sessions', limit: PAGE * 2 });
    if (mine !== loads.code) return;
    overview = {
      folder_projects: folderProjects,
      sessions: page.sessions || [], cursor: page.next_cursor || null, done: !page.next_cursor,
      running_tasks: glance.running_tasks || [], pending_approvals: glance.pending_approvals || []
    };
    shown = { chat: PAGE, code: PAGE };
    // Fill the open tab's first screen if the first page was mostly the other kind.
    await fillTo(current.view === 'code' ? 'code' : 'chat', PAGE, 3);
    renderSnapshot(overview);
    cacheSave(overview);
    updated(Date.now());
  }

  function ofKind(kind) {
    return sortedSessions(overview.sessions).filter(function (s) {
      return kind === 'code' ? isCoding(s) : !isCoding(s);
    });
  }

  // Fetch further pages until `kind` has `want` conversations, or the end.
  async function fillTo(kind, want, maxPages) {
    var l = await ensureLink();
    for (var i = 0; i < maxPages && !overview.done && ofKind(kind).length < want; i++) {
      var page = await l.command({ type: 'list_sessions', limit: PAGE * 2, cursor: overview.cursor });
      overview.sessions = overview.sessions.concat(page.sessions || []);
      overview.cursor = page.next_cursor || null;
      overview.done = !page.next_cursor;
    }
  }

  async function loadMore(kind) {
    var button = $('btn-more-' + kind);
    button.disabled = true;
    try {
      shown[kind] += PAGE;
      await fillTo(kind, shown[kind], 5);
      renderSnapshot(overview);
    } catch (e) {
      note('Could not load more: ' + e.message);
    } finally {
      button.disabled = false;
    }
  }

  function moreButton(kind, total) {
    var more = total > shown[kind] || !overview.done;
    $('btn-more-' + kind).className = more ? 'btn' : 'btn hidden';
  }

  function renderSnapshot(state) {
    if (state.folder_projects) folderProjects = state.folder_projects;
    if (state !== overview) overview = state;
    renderChats();
    renderApprovals(state);
    renderCode();
  }

  // Code: the coding conversations, those with an agent at work first.
  function renderCode() {
    var coding = ofKind('code');
    var working = {};
    (overview.running_tasks || []).forEach(function (t) { working[t.session_id] = t; });
    var busy = coding.filter(function (s) { return working[s.id]; });
    var rest = coding.filter(function (s) { return !working[s.id]; });
    $('code-running').textContent = busy.length ? String(busy.length) : '';
    fillSessions($('running-list'), busy, working);
    fillSessions($('done-list'), rest.slice(0, shown.code), working);
    moreButton('code', rest.length);
  }

  function fillSessions(ul, list, working) {
    clear(ul);
    list.forEach(function (s) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', s.title || 'Untitled'));
      var task = working[s.id];
      b.appendChild(el('span', 'meta', (s.agent || 'aidaemon') + ' · ' + timeOf(s.last_activity_ms)));
      if (task && task.description) b.appendChild(el('span', 'meta', 'Working: ' + task.description));
      b.addEventListener('click', function () { openChat(s, 'code'); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    if (!list.length) ul.appendChild(el('li', 'empty muted', 'Nothing here.'));
  }

  function runRow(r) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', r.label));
      b.appendChild(el('span', 'meta', (r.agent || 'aidaemon') + ' · ' + (PHASE_WORD[r.phase] || r.phase) +
        ' · ' + (r.took || timeOf(r.started_at_ms))));
      if (r.phase === 'running' && r.step) b.appendChild(el('span', 'meta', 'Now: ' + r.step));
      b.addEventListener('click', function () { openRun(r); });
      li.appendChild(b);
      return li;
  }

  async function openRun(r) {
    current.run = r;
    var back = document.querySelector('#view-run [data-back]');
    back.setAttribute('data-back', current.view === 'convo' ? 'convo' : 'code');
    $('run-title').textContent = r.label;
    $('run-meta').textContent = (r.agent || 'aidaemon') + ' · ' + (PHASE_WORD[r.phase] || r.phase);
    $('run-note').textContent = '';
    $('steer-text').value = '';
    clear($('run-steps'));
    clear($('run-result'));
    show('run');
    var actions = $('run-actions');
    clear(actions);
    if (r.phase === 'running') {
      var stop = el('button', 'btn', 'Stop this run');
      stop.type = 'button';
      stop.addEventListener('click', function () { stopRun(r); });
      actions.appendChild(stop);
    }
    await loadRun();
  }

  // Finished steps fold together; the step still going and failures stay
  // on their own lines (the app's rule, simplified).
  async function loadRun() {
    var r = current.run;
    if (!r) return;
    var l = await ensureLink();
    var mine = ++loads.run;
    var result = await l.command({ type: 'get_run', run_id: r.id });
    if (mine !== loads.run) return;
    var run = result.run || {};
    var steps = run.steps || [];
    var ul = $('run-steps');
    clear(ul);
    var group = [];
    function flush() {
      if (!group.length) return;
      var g = group;
      group = [];
      if (g.length === 1) { ul.appendChild(stepItem(g[0], '\u2713 ')); return; }
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', '\u2713 Finished ' + g.length + ' steps  \u25b8'));
      b.appendChild(el('span', 'meta', g[0].words + ' \u2026 ' + g[g.length - 1].words));
      b.addEventListener('click', function () {
        var next = li.nextSibling;
        ul.removeChild(li);
        g.forEach(function (st) { ul.insertBefore(stepItem(st, '\u2713 '), next); });
      });
      li.appendChild(b);
      ul.appendChild(li);
    }
    var lastDone = -1;
    steps.forEach(function (st, i) { if (st.state === 'done') lastDone = i; });
    steps.forEach(function (st, i) {
      // A failure something finished after was recovered from: it folds.
      if (st.state === 'done' || (st.state === 'failed' && i < lastDone)) { group.push(st); return; }
      flush();
      ul.appendChild(stepItem(st, st.state === 'failed' ? '\u2717 ' : '\u25b6 '));
    });
    flush();
    if (!steps.length) ul.appendChild(el('li', 'empty muted', run.task ? 'Task: ' + run.task : 'No steps recorded.'));
    var res = $('run-result');
    clear(res);
    if (run.result) {
      res.appendChild(el('h3', null, 'Result'));
      res.appendChild(el('pre', 'exact', run.result));
    }
  }

  function stepItem(st, mark) {
    var li = el('li');
    var b = el('button', 'item step');
    b.type = 'button';
    b.appendChild(el('span', 'title', mark + st.words));
    var detail = st.detail || st.output || st.command || '';
    if (detail) b.appendChild(el('span', 'meta', 'tap for details'));
    b.addEventListener('click', function () {
      if (!detail) return;
      $('detail-title').textContent = st.words;
      $('detail-body').textContent = typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
      show('detail');
    });
    li.appendChild(b);
    return li;
  }

  async function stopRun(r) {
    try {
      var l = await ensureLink();
      await l.command({ type: 'stop_turn', session_id: r.session_id });
      $('run-note').textContent = 'Asked it to stop.';
      await loadSnapshot();
    } catch (e) { $('run-note').textContent = e.message; }
  }

  async function steer() {
    var r = current.run;
    var text = $('steer-text').value.trim();
    if (!r || !text) { $('run-note').textContent = 'Type a message first.'; return; }
    try {
      var l = await ensureLink();
      await l.command({ type: 'send_message', session_id: r.session_id, text: text, in_run: r.id, mid_run: 'steer' });
      $('steer-text').value = '';
      $('run-note').textContent = 'Sent to the agent.';
    } catch (e) { $('run-note').textContent = e.message; }
  }

  async function openNewCode() {
    $('newcode-note').textContent = '';
    show('newcode');
    try {
      var l = await ensureLink();
      var projects = (await l.command({ type: 'list_projects' })).projects || [];
      var options = (await l.command({ type: 'list_models' })).options || {};
      var p = $('project');
      clear(p);
      projects.filter(function (x) { return x.path; }).forEach(function (x) {
        var o = el('option', null, x.name);
        o.value = x.id;
        p.appendChild(o);
      });
      var a = $('agent');
      clear(a);
      (options.agents || []).forEach(function (x) {
        var o = el('option', null, x.name);
        o.value = x.id;
        a.appendChild(o);
      });
      if (!p.options.length) $('newcode-note').textContent = 'No project with a folder yet.';
    } catch (e) { $('newcode-note').textContent = e.message; }
  }

  async function startCode() {
    var text = $('newcode-text').value.trim();
    var project = $('project').value;
    var agent = $('agent').value;
    if (!text || !project || !agent) { $('newcode-note').textContent = 'Pick a project and an agent, and say what to do.'; return; }
    $('btn-start-code').disabled = true;
    try {
      var l = await ensureLink();
      await l.command({ type: 'send_message', text: text, project_id: project, model: { mode: 'agent', agent: agent } });
      $('newcode-text').value = '';
      show('code');
      setTimeout(function () { loadSnapshot().catch(function () {}); }, 3000);
    } catch (e) {
      $('newcode-note').textContent = e.message;
    } finally {
      $('btn-start-code').disabled = false;
    }
  }

  // -------------------------------------------------------------- approvals
  var approvals = [];

  function renderApprovals(state) {
    approvals = state.pending_approvals || [];
    $('approval-count').textContent = approvals.length ? String(approvals.length) : '';
    var running = (state.running_tasks || []).length;
    $('chat-working').textContent = running ? String(running) : '';
    var ul = $('approval-list');
    clear(ul);
    approvals.forEach(function (a) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', a.summary));
      b.appendChild(el('span', 'meta', 'Risk: ' + a.risk));
      b.addEventListener('click', function () { openApproval(a); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    if (!approvals.length) ul.appendChild(el('li', 'empty muted', 'Nothing is waiting on you.'));
  }

  function openApproval(a) {
    current.approval = a;
    $('ap-title').textContent = a.summary;
    var t = $('ap-details');
    clear(t);
    [['Risk', a.risk], ['Kind', a.kind], ['Asked', timeOf(a.requested_at_ms)]].forEach(function (row) {
      var tr = el('tr');
      tr.appendChild(el('th', null, row[0]));
      tr.appendChild(el('td', null, row[1]));
      t.appendChild(tr);
    });
    $('ap-exact').textContent = a.detail || a.summary;
    var mayApprove = (a.options || []).indexOf('approve_once') >= 0 && (a.risk === 'low' || a.risk === 'medium');
    $('btn-approve').style.display = mayApprove ? '' : 'none';
    $('ap-note').textContent = mayApprove ? '' : 'Approve this on your phone or computer. You can deny it from here.';
    show('approval');
  }

  async function resolve(decision) {
    var a = current.approval;
    try {
      var l = await ensureLink();
      await l.command({ type: 'resolve_approval', approval_id: a.id, decision: decision });
      note((decision === 'deny' ? 'Denied: ' : 'Approved once: ') + a.summary);
    } catch (e) {
      note(e.message);
    }
    show('approvals');
    loadSnapshot().catch(function (e) { note(e.message); });
  }

  // ------------------------------------------------------------------ setup
  function updated(at) {
    var labels = document.querySelectorAll('.updated');
    for (var i = 0; i < labels.length; i++) labels[i].textContent = 'Updated ' + timeOf(at);
  }

  async function refreshAll() {
    note('');
    try {
      if (current.view === 'convo') await loadHistory();
      else if (current.view === 'run') await loadRun();
      else await loadSnapshot();
    } catch (e) {
      if (e.message !== 'Not paired') { link = null; note('Could not refresh: ' + e.message); setLink('offline'); }
    }
  }

  function init() {
    var tabs = document.querySelectorAll('.tab');
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].addEventListener('click', function () {
        show(this.getAttribute('data-view'));
        refreshAll();
      });
    }
    var backs = document.querySelectorAll('[data-back]');
    for (var j = 0; j < backs.length; j++) {
      backs[j].addEventListener('click', function () { show(this.getAttribute('data-back')); });
    }
    var refreshes = document.querySelectorAll('[data-refresh]');
    for (var k = 0; k < refreshes.length; k++) refreshes[k].addEventListener('click', refreshAll);
    $('btn-pair').addEventListener('click', pair);
    $('btn-new-chat').addEventListener('click', function () { openChat(null, 'chat'); });
    $('btn-convo-back').addEventListener('click', function () {
      // Undo the step that opened it, so the back button and this agree.
      if (current.pushed) { current.pushed = false; history.back(); return; }
      show(current.from || 'chat');
      refreshAll();
    });
    $('btn-send').addEventListener('click', send);
    $('mode-type').addEventListener('click', function () { setMode('type'); });
    $('mode-write').addEventListener('click', function () { setMode('write'); });
    $('btn-next-line').addEventListener('click', keepLine);
    $('btn-clear-line').addEventListener('click', clearStrip);
    $('btn-clear-all').addEventListener('click', clearAllLines);
    setupStrip();
    $('btn-page-down').addEventListener('click', function () { pageBy(1); });
    $('btn-page-up').addEventListener('click', function () { pageBy(-1); });
    $('btn-reader-back').addEventListener('click', function () {
      show('convo');
      window.scrollTo(0, readerReturn);
    });
    setMode('type');
    $('btn-new-code').addEventListener('click', openNewCode);
    $('btn-more-chat').addEventListener('click', function () { loadMore('chat'); });
    $('btn-more-code').addEventListener('click', function () { loadMore('code'); });
    $('btn-start-code').addEventListener('click', startCode);
    $('btn-steer').addEventListener('click', steer);
    $('btn-approve').addEventListener('click', function () {
      $('cf-text').textContent = current.approval.summary;
      $('cf-exact').textContent = current.approval.detail || current.approval.summary;
      show('confirm');
    });
    $('btn-confirm').addEventListener('click', function () { resolve('approve_once'); });
    $('btn-deny').addEventListener('click', function () { resolve('deny'); });

    if (!BRIDGE || !B) { note('This page is missing its configuration.'); return; }
    // Paint the last visit's screen at once, then refresh it.
    var last = cacheLoad();
    if (last && last.snapshot) {
      renderSnapshot(last.snapshot);
      updated(last.at);
    }
    window.addEventListener('popstate', function () { followRoute(); refreshAll(); });
    // An address typed or changed by hand.
    window.addEventListener('hashchange', function () {
      var view = current.view;
      if (routeOf(view) !== window.location.hash) { followRoute(); refreshAll(); }
    });
    dbGet('pinned').then(function (pinned) {
      if (!pinned) { show('pair'); setLink('not paired'); return; }
      // A conversation loads itself; a list refreshes.
      if (!followRoute()) refreshAll();
    }).catch(function () { show('pair'); });
  }

  init();
})();
