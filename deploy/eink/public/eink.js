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
  function busyButton(idOrEl, label) {
    var button = typeof idOrEl === 'string' ? $(idOrEl) : idOrEl;
    if (!button) return function () {};
    var original = button.textContent;
    button.disabled = true;
    button.textContent = label;
    return function () {
      button.textContent = original;
      button.disabled = false;
    };
  }
  // The link's state in plain words (connect() and the rest pass a key).
  var LINK_WORD = { connecting: 'Connecting…', connected: 'Connected', offline: 'Offline – tap Refresh',
    'not paired': 'Not paired' };
  function setLink(text) {
    var node = $('link-state');
    setText(node, LINK_WORD[text] || text);
    var offline = text === 'offline';
    if (node.disabled !== !offline) node.disabled = !offline;
    var cls = offline ? 'linktag linktag-offline' : 'linktag';
    if (node.className !== cls) node.className = cls;
  }
  // Set a text only when it changes (each change redraws on e-ink).
  function setText(node, text) { if (node.textContent !== text) node.textContent = text; }
  function timeOf(ms) {
    if (!ms) return '';
    var d = new Date(ms), h = d.getHours(), m = d.getMinutes();
    return ((h % 12) || 12) + ':' + (m < 10 ? '0' : '') + m + (h < 12 ? ' AM' : ' PM');
  }
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  // When something happened: "3:15 PM" today, "Yesterday 3:15 PM",
  // "Nov 21" this year, "Nov 21, 2025" before. Days are the Kindle's own
  // midnights (built from the date, so a daylight-saving change is right).
  function whenOf(ms) {
    if (!ms) return '';
    var d = new Date(ms), now = new Date();
    var today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    var yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).getTime();
    if (ms >= today) return timeOf(ms);
    if (ms >= yesterday) return 'Yesterday ' + timeOf(ms);
    var day = MONTHS[d.getMonth()] + ' ' + d.getDate();
    return d.getFullYear() === now.getFullYear() ? day : day + ', ' + d.getFullYear();
  }

  // ------------------------------------------------------------ plain words
  // The daemon's typed values (snake_case) as the owner reads them.
  function plainWord(value) {
    var s = String(value || '').replace(/_/g, ' ');
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
  var RISK_WORD = { low: 'Low risk', medium: 'Medium risk', high: 'High risk', critical: 'Critical risk' };
  function riskWord(risk) { return RISK_WORD[risk] || plainWord(risk); }
  // ApprovalKind (client-protocol), plus tool-style names a card may carry.
  var KIND_WORD = {
    command: 'Allow an action', goal_confirmation: 'Confirm a scheduled goal',
    autopilot_confirmation: 'Confirm an Autopilot setting', handoff: 'Something for you to do',
    write_file: 'Write a file', edit_file: 'Change a file', read_file: 'Read a file', delete_file: 'Delete a file',
    run_command: 'Run a command', terminal: 'Run a command', web_fetch: 'Fetch a web page', send_email: 'Send an email'
  };
  function kindWord(kind) { return KIND_WORD[kind] || plainWord(kind); }
  var CHANNEL_WORD = {
    app: 'App', telegram: 'Telegram', kindle: 'Kindle', eink: 'Kindle', slack: 'Slack', discord: 'Discord',
    whatsapp: 'WhatsApp', signal: 'Signal', imessage: 'iMessage', email: 'Email', sms: 'SMS', matrix: 'Matrix',
    web: 'Web', cli: 'Terminal', watch: 'Watch'
  };
  function channelWord(channel) { return CHANNEL_WORD[String(channel || '').toLowerCase()] || plainWord(channel); }
  // An error as a sentence: the channel's own technical ones (socketTo,
  // makeLink) in plain words; the daemon's messages, already plain, as
  // they are, after `what` ("Could not refresh") when given.
  function plainError(e, what) {
    var m = String((e && e.message) || e || '');
    if (/^closed \(\d+\)$/.test(m) || m === 'unexpected message from the bridge') {
      return 'Lost the connection to your computer. Tap Refresh.';
    }
    if (m === 'Could not reach the bridge') return 'Could not connect. Check the Kindle’s Wi-Fi, then tap Refresh.';
    if (m === 'timed out') return 'Your computer did not answer in time. Tap Refresh to try again.';
    if (m === 'Sign-in refused') return 'Your computer did not let this Kindle sign in. If this keeps happening, pair it again.';
    if (/^Refused \(\d+\)$/.test(m)) return 'Your computer turned that down. Try again.';
    if (m === 'Could not load a file') return 'Could not load that file.';
    return what ? what + ': ' + m : m;
  }

  var TAB_OF = { pair: null, reader: null, chat: 'chat', convo: 'chat', code: 'code', newcode: 'code', run: 'code',
    approvals: 'approvals', approval: 'approvals', confirm: 'approvals' };
  var current = { view: 'chat', session: null, approval: null, run: null, runStack: [], runBack: 'code',
    historyRuns: {}, readerFrom: 'convo' };
  // Bumped by each load of a screen, so an older load that finishes late
  // never adds to what a newer one drew.
  var loads = { history: 0, chats: 0, code: 0, run: 0 };

  // A conversation (or a run/reader opened from one) belongs to its tab.
  function tabFor(view) {
    if (view === 'convo') return current.from || 'chat';
    if (view === 'run') return current.runBack === 'convo' ? (current.from || 'chat') : 'code';
    if (view === 'reader') return tabFor(current.readerFrom || 'convo');
    return TAB_OF[view];
  }

  function show(view, replaceRoute) {
    if (current.view === 'convo' && view !== 'convo') {
      current.convoScroll = window.scrollY;
      $('convo-note').textContent = '';
    }
    if (view !== current.view) note('');
    if (view !== 'convo') stopWatching();
    if (view === 'approvals') startApprovalPolling();
    else stopApprovalPolling();
    current.view = view;
    var views = document.querySelectorAll('.view');
    for (var i = 0; i < views.length; i++) views[i].className = 'view';
    $('view-' + view).className = 'view active';
    $('tabs').style.display = view === 'pair' ? 'none' : 'flex';
    var tabs = document.querySelectorAll('.tab');
    for (var j = 0; j < tabs.length; j++) {
      tabs[j].className = tabs[j].getAttribute('data-view') === tabFor(view) ? 'tab active' : 'tab';
      tabs[j].setAttribute('aria-selected', tabs[j].getAttribute('data-view') === tabFor(view) ? 'true' : 'false');
    }
    window.scrollTo(0, 0);
    remember(view, replaceRoute);
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
    if (view === 'convo') return '#chat/new';
    if (view === 'newcode') return '#code/new';
    if (view === 'chat' || view === 'code' || view === 'approvals') return '#' + view;
    return null;
  }

  function remember(view, replaceRoute) {
    var route = routeOf(view);
    if (!route || !window.history || !history.pushState) return;
    if (route !== window.location.hash) {
      if (!replaceRoute && (view === 'convo' || view === 'newcode')) {
        history.pushState({ einkBack: true }, '', route);
      } else {
        history.replaceState(replaceRoute ? history.state : null, '', route);
      }
    }
    current.pushed = !!(history.state && history.state.einkBack);
  }

  // Go where the address says (on load, and when the back button changes it).
  function followRoute() {
    var m = /^#(chat|code|approvals)(?:\/(.+))?$/.exec(window.location.hash || '');
    if (!m) { show('chat'); return false; }
    // A form to start something new (no conversation id is "new").
    if (m[2] === 'new' && m[1] === 'code') { openNewCode(); return true; }
    if (m[2] === 'new' && m[1] === 'chat') { openChat(null, 'chat'); return true; }
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
  // What the last sign-in said this device may approve up to: a Risk
  // ('medium' for a new e-ink device, more if the owner gave it more),
  // 'none' for no limit, null before any sign-in.
  var riskCeiling = null;
  // What scopes the last sign-in granted ('observe', 'chat', 'approve');
  // null before any sign-in.
  var deviceScopes = null;

  function hasScope(scope) {
    return !deviceScopes || deviceScopes.indexOf(scope) >= 0;
  }

  function applyScopes() {
    var canChat = hasScope('chat');
    var newBtnCls = canChat ? 'btn btn-strong' : 'btn btn-strong hidden';
    if ($('btn-new-chat').className !== newBtnCls) $('btn-new-chat').className = newBtnCls;
    if ($('btn-new-code').className !== newBtnCls) $('btn-new-code').className = newBtnCls;
    var roCls = canChat ? 'muted hidden' : 'muted';
    if ($('convo-readonly').className !== roCls) $('convo-readonly').className = roCls;
    var modesCls = canChat ? 'modes' : 'modes hidden';
    if ($('convo-modes').className !== modesCls) $('convo-modes').className = modesCls;
    if (!canChat) {
      replyTo(null);
      $('convo-text').className = 'hidden';
      if (writing.form === 'convo') $('write-box').className = 'hidden';
    } else if (writing.form === 'convo') {
      $('convo-text').className = writing.mode === 'type' ? '' : 'hidden';
      $('write-box').className = writing.mode === 'write' ? '' : 'hidden';
    }
    inlineSig = null;
    renderInlineApprovals();
  }

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
      // The highest risk this device may approve (AccessGrant.risk_ceiling,
      // from the device's record; absent: no limit) and its scopes.
      riskCeiling = grant.json.risk_ceiling || 'none';
      deviceScopes = Array.isArray(grant.json.scopes) ? grant.json.scopes : null;
      applyScopes();
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
    var ready = busyButton('btn-pair', 'Pairing…');
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
      $('pair-note').textContent = 'Pairing did not finish (' + plainError(e) + '). Tap Pair to start again.';
      $('pair-code-box').style.display = 'none';
    } finally {
      ready();
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

  // What each list last drew, so a refresh that brings the same rows (a
  // tab switch, mostly) leaves it alone: e-ink redraws every change.
  var listSigs = { chat: null, running: null, done: null, approvals: null };
  function listSig(rows) { return JSON.stringify(rows); }

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
    // The same rows as on screen: leave the list alone (e-ink).
    var sig = listSig(sessions.map(function (s) { return [s.id, s.title, s.channel, whenOf(s.last_activity_ms)]; }));
    if (sig === listSigs.chat) return;
    listSigs.chat = sig;
    clear(ul);
    sessions.forEach(function (s) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', s.title || 'Untitled chat'));
      b.appendChild(el('span', 'meta', channelWord(s.channel) + ' · ' + whenOf(s.last_activity_ms)));
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

  // The agent the composer addresses (send_message's in_run), until
  // cancelled or another conversation opens.
  function replyTo(r) {
    current.replyThread = r ? (r.thread || r.id) : null;
    $('reply-to').className = r ? 'reply-to' : 'reply-to hidden';
    setText($('reply-to-label'), r ? 'Replying to ' + (r.agent || 'the agent') : '');
    if (r) {
      $('convo-note').textContent = '';
      if (writing.mode !== 'write') $('convo-text').focus();
      window.scrollTo(0, document.body.scrollHeight);
    }
  }

  async function openChat(s, from, started) {
    replyTo(null);
    stopWatching();
    if (writing.form !== 'convo') setMode('type', 'convo');
    current.session = s;
    current.historyLimit = 8;
    current.historySig = null;
    current.from = from || 'chat';
    $('convo-title').textContent = (s && s.title) || (s && s.id ? 'Conversation' : 'New chat');
    $('convo-note').textContent = '';
    show('convo', started);
    inlineSig = null;
    renderInlineApprovals();
    current.toEnd = true;
    if (s && s.id && (started || conversationActive(s.id))) watchConversation(s.id, !started);
    await loadHistory();
  }

  function conversationActive(id) {
    return overview.running_tasks.some(function (t) { return t.session_id === id; }) ||
      overview.pending_approvals.some(function (a) { return a.session_id === id; });
  }

  async function resumeConversation(expectWork) {
    var session = current.session;
    var scroll = current.convoScroll;
    current.toEnd = false;
    show('convo');
    renderInlineApprovals();
    window.scrollTo(0, scroll);
    var restored = window.scrollY;
    // Check for work that may have started while the detail screen was open.
    if (session && session.id) watchConversation(session.id, !expectWork && conversationActive(session.id), true);
    try {
      await loadHistory();
      if (current.view === 'convo' && current.session === session && window.scrollY === restored) {
        window.scrollTo(0, scroll);
      }
    } catch (e) {
      if (current.view === 'convo' && current.session === session && !$('convo-note').textContent) {
        $('convo-note').textContent = plainError(e);
      }
    }
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
    current.readerFrom = current.view === 'run' ? 'run' : 'convo';
    readerReturn = window.pageYOffset;
    $('reader-from').textContent = from || '';
    var body = $('reader-body');
    clear(body);
    richText(body, text);
    show('reader');
    readerPage();
  }

  // A screen at a time, keeping two lines from the last one.
  function pageStep() { return Math.max(200, window.innerHeight - 140); }

  function pageBy(direction) {
    window.scrollBy(0, direction * pageStep());
    readerPage();
  }

  // "Page N of M" from where the reader is scrolled to, and no Previous on
  // the first page or Next on the last.
  function readerPage() {
    if (current.view !== 'reader') return;
    var step = pageStep();
    var last = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
    var at = window.pageYOffset;
    var pages = 1 + Math.ceil(last / step);
    // At the end is the last page even when the last step was a short one.
    var page = at >= last - 2 ? pages : Math.min(pages, 1 + Math.round(at / step));
    setText($('reader-page'), pages > 1 ? 'Page ' + page + ' of ' + pages : '');
    var first = page <= 1, end = page >= pages;
    if ($('btn-page-up').disabled !== first) $('btn-page-up').disabled = first;
    if ($('btn-page-down').disabled !== end) $('btn-page-down').disabled = end;
  }
  // A swipe scrolls too: count the page once it settles, not on every
  // scroll event (each change redraws).
  var readerSettle = null;
  window.addEventListener('scroll', function () {
    if (current.view !== 'reader') return;
    if (readerSettle) clearTimeout(readerSettle);
    readerSettle = setTimeout(function () { readerSettle = null; readerPage(); }, 250);
  });

  // While work runs in the open conversation, a small glance every few
  // seconds; when it ends, the conversation reloads once and shows the new
  // messages at the end. Nothing redraws while nothing changed (e-ink).
  var WATCH_MS = 6000;
  var FIRST_LOOK_MS = 4000;
  var watching = null;

  function watchConversation(id, expectWork, preserveScroll) {
    stopWatching();
    var sawWork = !!expectWork;
    var quietChecks = 0;
    var checking = false;
    // This watch's own token: a check that ends after another watch began
    // (or after watching stopped) changes nothing.
    var token = {};
    watchToken = token;
    var tick = async function () {
      if (checking || watchToken !== token) return;
      if (current.view !== 'convo' || !current.session || current.session.id !== id) { stopWatching(); return; }
      checking = true;
      try {
        var glance = (await (await ensureLink()).command({ type: 'get_glance' })).glance || {};
        if (watchToken !== token || current.view !== 'convo') return;
        applyGlance(glance);
        renderInlineApprovals();
        var busy = (glance.running_tasks || []).some(function (t) { return t.session_id === id; });
        var waiting = (glance.pending_approvals || []).some(function (a) { return a.session_id === id; });
        if (busy || waiting) { sawWork = true; quietChecks = 0; return; }
        // aidaemon's own turn is over (handing work to a coding agent ends it
        // at once): look at the conversation now, redrawn only if something
        // changed, and keep watching while an agent runs. Just after sending,
        // give the daemon a few checks to start before stopping.
        current.toEnd = !preserveScroll;
        var agentsRunning = await loadHistory();
        if (watchToken !== token) return;
        if (agentsRunning) { sawWork = true; quietChecks = 0; return; }
        if (!sawWork && ++quietChecks < 4) return;
        stopWatching();
      } catch (e) {
        if (watchToken === token) stopWatching();
      } finally {
        checking = false;
      }
    };
    watching = setInterval(tick, WATCH_MS);
    firstLook = setTimeout(tick, FIRST_LOOK_MS);
  }
  var firstLook = null;
  var watchToken = null;

  // A waiting approval for the open conversation shows on top of the
  // message box, so it is never missed while reading the conversation.
  var inlineSig = '';
  function renderInlineApprovals() {
    var box = $('convo-approvals');
    var id = current.session && current.session.id;
    var mine = (overview.pending_approvals || []).filter(function (a) { return a.session_id === id; });
    var canApprove = hasScope('approve');
    var sig = (canApprove ? '1:' : '0:') + mine.map(function (a) { return a.id; }).join(',');
    if (sig === inlineSig && box.childNodes.length === (mine.length ? mine.length : 0)) return;
    inlineSig = sig;
    clear(box);
    mine.forEach(function (a) {
      var card = el('div', 'inline-approval');
      card.appendChild(el('div', 'msg-from', 'Needs your OK · ' + riskWord(a.risk)));
      var text = el('div', 'msg-text');
      richText(text, preview(a.detail || a.summary).text);
      card.appendChild(text);
      var actions = el('div', 'actions');
      var review = el('button', 'btn btn-strong btn-small', 'Review \u25b8');
      review.type = 'button';
      review.addEventListener('click', function () { current.approvalFrom = 'convo'; openApproval(a); });
      actions.appendChild(review);
      if (canApprove) {
        var deny = el('button', 'btn btn-small', 'Deny');
        deny.type = 'button';
        deny.addEventListener('click', function () {
          current.approval = a;
          current.approvalFrom = 'convo';
          resolve('deny', deny);
        });
        actions.appendChild(deny);
      }
      card.appendChild(actions);
      box.appendChild(card);
    });
  }

  function stopWatching() {
    if (firstLook) { clearTimeout(firstLook); firstLook = null; }
    if (watching) { clearInterval(watching); watching = null; }
    watchToken = null;
  }

  async function loadHistory() {
    var mine = ++loads.history;
    var box = $('convo-messages');
    if (!current.session || !current.session.id) { clear(box); return; }
    var id = current.session.id;
    var l = await ensureLink();
    var result = await l.command({ type: 'get_history', session_id: id, limit: current.historyLimit || 8 });
    if (mine !== loads.history || current.view !== 'convo' || !current.session || current.session.id !== id) return;
    var running = (result.runs || []).some(function (r) { return r.phase === 'running'; });
    current.historyRuns = {};
    (result.runs || []).forEach(function (r) { current.historyRuns[r.id] = r; });
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
        // Reply goes to this agent's own session (RunSummary.thread, or the
        // run itself when it started the thread), as in the app.
        if (r.session_id && hasScope('chat')) {
          var reply = el('button', 'btn btn-small', 'Reply');
          reply.type = 'button';
          reply.addEventListener('click', function () { replyTo(r); });
          card.appendChild(reply);
        }
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

  // The Kindle reports the pen as a "touch", like a finger or a resting
  // palm. So: one pointer draws a stroke at a time; a large contact (a
  // palm) never draws, and a smaller one (the pen) takes over from it; and
  // while the strip is out, two touches at once (pen plus palm) never become
  // a page scroll or zoom, which would cancel the pen's stroke.
  var PALM_PX = 24;
  var strokePointer = null;
  var strokeIsPalm = false;

  function isPalm(e) {
    // Width and height are the contact's size in CSS pixels (0 or 1 when
    // the browser does not know).
    return (e.width || 0) > PALM_PX || (e.height || 0) > PALM_PX;
  }

  function writingOpen() {
    return writing.mode === 'write' && $('write-box').className !== 'hidden';
  }

  function setupStrip() {
    var c = strip();
    c.addEventListener('pointerdown', function (e) {
      e.preventDefault();
      var palm = isPalm(e);
      if (strokePointer !== null && strokePointer !== e.pointerId) {
        // Another contact is already stroking: a large one never takes over;
        // a small one (the pen) takes over from a large one (a palm).
        if (palm || !strokeIsPalm) return;
        try { c.releasePointerCapture(strokePointer); } catch (err) { /* gone */ }
      }
      sizeStrip();
      strokePointer = e.pointerId;
      // A large contact alone still draws (in case the pen reports a large
      // size); any smaller contact replaces it.
      strokeIsPalm = palm;
      writing.drawing = true;
      writing.last = stripPoint(e.clientX, e.clientY);
      try { c.setPointerCapture(e.pointerId); } catch (err) { /* old browser */ }
    });
    c.addEventListener('pointermove', function (e) {
      if (!writing.drawing || e.pointerId !== strokePointer) return;
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
    function end(e) {
      if (e.pointerId !== strokePointer) return;
      strokePointer = null;
      writing.drawing = false;
      writing.last = null;
    }
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('touchstart', function (e) { e.preventDefault(); }, { passive: false });
    c.addEventListener('touchmove', function (e) { e.preventDefault(); }, { passive: false });
    // Page-wide: while writing, a resting palm plus the pen must not scroll
    // or zoom the page (one finger alone still scrolls).
    function guard(e) {
      if (!writingOpen()) return;
      if (writing.drawing || (e.touches && e.touches.length > 1)) e.preventDefault();
    }
    document.addEventListener('touchstart', guard, { passive: false });
    document.addEventListener('touchmove', guard, { passive: false });
  }

  function renderWriteLines() {
    var box = $('write-lines');
    clear(box);
    var count = writing.lines.length;
    $('btn-undo-line').disabled = !count;
    if (!count) return;
    box.appendChild(el('div', null, count + (count === 1 ? ' line kept' : ' lines kept')));
    var last = writing.lines[count - 1];
    var previewCanvas = document.createElement('canvas');
    previewCanvas.className = 'lines-preview';
    previewCanvas.width = last.width;
    previewCanvas.height = last.height;
    previewCanvas.getContext('2d').drawImage(last, 0, 0);
    box.appendChild(previewCanvas);
  }

  function keepLine() {
    if (!writing.inked) return true;
    if (writing.lines.length >= LINES_PER_SHEET * MAX_SHEETS) {
      $(writing.form === 'newcode' ? 'newcode-note' : 'convo-note').textContent =
        'That is as many lines as one message takes. Send it, then write more.';
      return false;
    }
    var copy = document.createElement('canvas');
    copy.width = strip().width;
    copy.height = strip().height;
    copy.getContext('2d').drawImage(strip(), 0, 0);
    writing.lines.push(copy);
    renderWriteLines();
    clearStrip();
    return true;
  }

  function undoLine() {
    if (!writing.lines.length) return;
    writing.lines.pop();
    renderWriteLines();
  }

  function clearAllLines() {
    writing.lines = [];
    renderWriteLines();
    clearStrip();
  }

  // One writing strip, moved into whichever form is open: a conversation's
  // message box, or a new coding session.
  var FORMS = {
    convo: { slot: 'convo-write-slot', text: 'convo-text', type: 'mode-type', write: 'mode-write' },
    newcode: { slot: 'newcode-write-slot', text: 'newcode-text', type: 'nc-mode-type', write: 'nc-mode-write' }
  };
  writing.form = 'convo';

  function setMode(mode, form) {
    form = form || writing.form;
    if (form !== writing.form) clearAllLines();
    writing.form = form;
    writing.mode = mode;
    var f = FORMS[form];
    $(f.slot).appendChild($('write-box'));
    var blocked = form === 'convo' && !hasScope('chat');
    // One input at a time: the strip, or the text box (hidden when view-only).
    $('write-box').className = !blocked && mode === 'write' ? '' : 'hidden';
    $(f.text).className = !blocked && mode === 'type' ? '' : 'hidden';
    $(f.type).className = mode === 'type' ? 'btn mode active' : 'btn mode';
    $(f.write).className = mode === 'write' ? 'btn mode active' : 'btn mode';
    if (!blocked && mode === 'write') { writing.inked = false; strip().width = 0; sizeStrip(); }
  }

  // The written lines, uploaded as PNG sheets: their attachments.
  async function uploadHandwriting(l, noteEl) {
    noteEl.textContent = 'Sending your handwriting…';
    var pngs = await sheets();
    var attachments = [];
    for (var i = 0; i < pngs.length; i++) {
      var up = await l.command({
        type: 'upload_file', name: 'handwriting-' + (i + 1) + '.png', mime: 'image/png', data: pngs[i]
      });
      attachments.push(up.attachment);
    }
    return attachments;
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
    var session = current.session;
    var text = writing.mode === 'write' ? '' : $('convo-text').value.trim();
    if (writing.mode === 'write' && !keepLine()) return;
    var handwriting = writing.mode === 'write' && writing.lines.length > 0;
    if (!text && !handwriting) {
      $('convo-note').textContent = writing.mode === 'write' ? 'Write something first.' : 'Type something first.';
      return;
    }
    var ready = busyButton('btn-send', 'Sending…');
    try {
      var l = await ensureLink();
      var attachments = [];
      if (handwriting) attachments = await uploadHandwriting(l, $('convo-note'));
      var command = { type: 'send_message', text: text };
      if (attachments.length) command.attachments = attachments;
      if (session && session.id) command.session_id = session.id;
      var thread = current.replyThread;
      if (thread && session && session.id) command.in_run = thread;
      var result = await l.command(command);
      if (current.view !== 'convo' || current.session !== session) return;
      // Strokes are kept until the daemon took the message.
      if (handwriting) clearAllLines();
      else $('convo-text').value = '';
      if (!session || !session.id) {
        current.session = { id: result.session_id, title: text.slice(0, 40) || 'Handwritten chat' };
        $('convo-title').textContent = current.session.title;
        remember('convo', true);
      }
      $('convo-note').textContent = command.in_run ? 'Sent to ' + $('reply-to-label').textContent.replace(/^Replying to /, '') + '. Its answer appears here.' : 'Sent. The reply appears here when it is ready.';
      current.toEnd = true;
      watchConversation(current.session.id, false);
      await loadHistory();
    } catch (e) {
      $('convo-note').textContent = plainError(e);
    } finally {
      ready();
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
    var ready = busyButton('btn-more-' + kind, 'Loading…');
    try {
      shown[kind] += PAGE;
      await fillTo(kind, shown[kind], 5);
      renderSnapshot(overview);
    } catch (e) {
      note(plainError(e, 'Could not load more'));
    } finally {
      ready();
    }
  }

  function moreButton(kind, total) {
    var more = total > shown[kind] || !overview.done;
    var cls = more ? 'btn' : 'btn hidden';
    if ($('btn-more-' + kind).className !== cls) $('btn-more-' + kind).className = cls;
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
    setText($('code-running'), busy.length ? String(busy.length) : '');
    fillSessions($('running-list'), busy, working, 'running');
    fillSessions($('done-list'), rest.slice(0, shown.code), working, 'done');
    moreButton('code', rest.length);
  }

  function fillSessions(ul, list, working, which) {
    // The same rows as on screen: leave the list alone (e-ink).
    var sig = listSig(list.map(function (s) {
      var task = working[s.id];
      return [s.id, s.title, s.agent, whenOf(s.last_activity_ms), task ? task.description : null];
    }));
    if (sig === listSigs[which]) return;
    listSigs[which] = sig;
    clear(ul);
    list.forEach(function (s) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', s.title || 'Untitled'));
      var task = working[s.id];
      b.appendChild(el('span', 'meta', (s.agent || 'aidaemon') + ' · ' + whenOf(s.last_activity_ms)));
      if (task && task.description) b.appendChild(el('span', 'meta', 'Working: ' + task.description));
      b.addEventListener('click', function () { openChat(s, 'code'); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    if (!list.length) ul.appendChild(el('li', 'empty muted', 'Nothing here.'));
  }

  // The run screen's line under its title: who, where it is, when, and
  // from its record (`detail`, once loaded) the model and the folder.
  function runMeta(r, detail) {
    var parts = [r.agent || 'aidaemon'];
    // A hand-off's run opened from its step has no phase until it is known.
    if (r.phase && r.phase !== 'unknown') parts.push(PHASE_WORD[r.phase] || plainWord(r.phase));
    if (r.took) parts.push(r.took);
    else if (r.started_at_ms) parts.push(whenOf(r.started_at_ms));
    if (detail && detail.model) parts.push(detail.model);
    if (detail && detail.working_dir) parts.push(detail.working_dir);
    return parts.join(' · ');
  }

  function runStatus(r) {
    if (r.out_of_usage) {
      var reset = r.out_of_usage.resets_at_ms ? ' · resets ' + whenOf(r.out_of_usage.resets_at_ms) : '';
      return 'Paused: out of agent usage' + reset;
    }
    if (r.reason) return r.reason;
    return '';
  }

  async function openRun(r, isBack) {
    var fromRun = current.view === 'run';
    if (!isBack) {
      if (fromRun && current.run) current.runStack.push(current.run);
      else {
        current.runStack = [];
        current.runBack = current.view === 'convo' ? 'convo' : 'code';
      }
    }
    current.run = r;
    var backBtn = $('btn-run-back');
    if (current.runStack.length) backBtn.textContent = '\u2190 Previous run';
    else backBtn.textContent = current.runBack === 'convo' ? '\u2190 Conversation' : '\u2190 Code';
    $('run-title').textContent = r.label || ((r.agent || 'Agent') + ' run');
    $('run-meta').textContent = runMeta(r, null);
    $('run-status').textContent = runStatus(r);
    $('run-note').textContent = '';
    $('steer-text').value = '';
    clear($('run-steps'));
    clear($('run-extra'));
    clear($('run-result'));
    // The message box shows once the run's record says what it takes.
    $('steer-box').className = 'hidden';
    show('run');
    var actions = $('run-actions');
    clear(actions);
    if (r.phase === 'running' && hasScope('chat')) {
      var stop = el('button', 'btn', 'Stop this run');
      stop.type = 'button';
      stop.addEventListener('click', function () { stopRun(r, stop); });
      actions.appendChild(stop);
    }
    try {
      await loadRun();
    } catch (e) {
      $('run-note').textContent = plainError(e, 'Could not load the run');
    }
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
      if (g.length === 1) { ul.appendChild(stepItem(g[0], markOf(g[0]), running)); return; }
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', '\u2713 Finished ' + g.length + ' steps  \u25b8'));
      b.appendChild(el('span', 'meta', g[0].words + ' \u2026 ' + g[g.length - 1].words));
      b.addEventListener('click', function () {
        var next = li.nextSibling;
        ul.removeChild(li);
        g.forEach(function (st) { ul.insertBefore(stepItem(st, markOf(st), running), next); });
      });
      li.appendChild(b);
      ul.appendChild(li);
    }
    // RunStepState is snake_case: 'started' (no end on record yet), 'done',
    // 'failed', or 'unknown' from a newer daemon. A started step is going
    // only while the run runs; in an ended run it just has no end recorded.
    var running = r.phase === 'running';
    function markOf(st) {
      if (st.state === 'done') return '\u2713 ';
      if (st.state === 'failed') return '\u2717 ';
      if (st.state === 'started' && running) return '\u25b6 ';
      return '\u2022 ';
    }
    var lastDone = -1;
    steps.forEach(function (st, i) { if (st.state === 'done') lastDone = i; });
    steps.forEach(function (st, i) {
      // A hand-off opens its run: it never folds out of reach.
      if (st.run) { flush(); ul.appendChild(stepItem(st, markOf(st), running)); return; }
      // A failure something finished after was recovered from: it folds.
      if (st.state === 'done' || (st.state === 'failed' && i < lastDone)) { group.push(st); return; }
      flush();
      ul.appendChild(stepItem(st, markOf(st), running));
    });
    flush();
    if (!steps.length) ul.appendChild(el('li', 'empty muted', run.task ? 'Task: ' + run.task : 'No steps recorded.'));
    $('run-meta').textContent = runMeta(r, run);
    $('run-status').textContent = runStatus(r);
    runExtra(run);
    var res = $('run-result');
    clear(res);
    // RunDetail.result is a String (what it reported at its end).
    if (typeof run.result === 'string' && run.result) {
      res.appendChild(el('h3', null, 'Result'));
      var card = el('div', 'msg');
      messageBody(card, run.result, r.agent || 'Agent');
      res.appendChild(card);
    }
    steerBox(r, run);
  }

  // A run's plan (as it last wrote it) and what it asked permission for,
  // one short line each.
  var PLAN_MARK = { done: '\u2713 ', in_progress: '\u25b6 ', pending: '\u25cb ' };
  var PERMISSION_WORD = { allowed: 'Allowed', refused: 'Refused', declined: 'Declined', unanswered: 'Not answered' };
  function runExtra(run) {
    var box = $('run-extra');
    clear(box);
    var plan = run.plan || [];
    if (plan.length) {
      box.appendChild(el('h3', null, 'Plan'));
      var ul = el('ul', 'steps plain');
      plan.forEach(function (p) { ul.appendChild(el('li', null, (PLAN_MARK[p.status] || '\u2022 ') + p.text)); });
      box.appendChild(ul);
    }
    var asked = run.permissions || [];
    if (asked.length) {
      box.appendChild(el('h3', null, 'Permissions it asked for'));
      var pl = el('ul', 'steps plain');
      asked.forEach(function (p) {
        var said = PERMISSION_WORD[p.outcome] || plainWord(p.outcome);
        pl.appendChild(el('li', null, p.label + ' \u2014 ' + said + (p.reason ? ' (' + p.reason + ')' : '')));
      });
      box.appendChild(pl);
    }
  }

  // A step: what it did and the file it names. Only a hand-off (it names
  // the run it started) opens something; the rest are plain lines.
  function stepItem(st, mark, running) {
    var li = el('li');
    var cls = 'item step' + (st.state === 'failed' ? ' step-failed' : '') +
      (st.state === 'started' && running ? ' step-running' : '') + (st.run ? ' step-handoff' : '');
    var b = el(st.run ? 'button' : 'div', cls);
    b.appendChild(el('span', 'title', mark + st.words));
    if (st.path) b.appendChild(el('span', 'meta', st.path));
    if (st.run) {
      b.type = 'button';
      b.appendChild(el('span', 'meta', 'Open ' + (st.agent ? st.agent + '\u2019s' : 'its') + ' run \u25b8'));
      var parent = current.run;
      b.addEventListener('click', function () {
        // Use the conversation's RunSummary when loaded (has real phase,
        // reason, took); else infer phase from the hand-off step itself.
        var known = current.historyRuns && current.historyRuns[st.run];
        if (known) { openRun(known); return; }
        var phase = st.state === 'started' && running ? 'running' :
          (st.state === 'done' ? 'succeeded' : (st.state === 'failed' ? 'failed' : null));
        openRun({ id: st.run, label: (st.agent || 'Agent') + ' run', agent: st.agent || null,
          session_id: parent && parent.session_id, phase: phase });
      });
    }
    li.appendChild(b);
    return li;
  }

  // The message box under a run: steer (or queue after) a running one;
  // continue an ended one whose agent kept its session
  // (RunDetail.can_continue: send_message with in_run, no mid_run); else none.
  function steerBox(r, run) {
    var mode = null;
    if (hasScope('chat')) {
      if (r.phase === 'running') mode = r.accepts_mid_run === false ? 'queue' : 'steer';
      else if (run.can_continue) mode = 'continue';
    }
    current.runMode = mode;
    $('steer-box').className = mode ? '' : 'hidden';
    setText($('steer-label'), mode === 'continue' ? 'Continue it' :
      (mode === 'queue' ? 'Queue a follow-up' : 'Steer it'));
  }

  async function stopRun(r, button) {
    var ready = busyButton(button, 'Stopping…');
    try {
      var l = await ensureLink();
      await l.command({ type: 'stop_turn', session_id: r.session_id });
      r.phase = 'stopped';
      $('run-note').textContent = 'Asked it to stop.';
      clear($('run-actions'));
      await loadRun();
    } catch (e) {
      $('run-note').textContent = plainError(e);
      ready();
    }
  }

  async function steer() {
    var r = current.run;
    var mode = current.runMode;
    var text = $('steer-text').value.trim();
    if (!r || !mode) return;
    // Without its conversation the message would start a new one.
    if (!r.session_id) { $('run-note').textContent = 'Open this run from its conversation to message it.'; return; }
    if (!text) { $('run-note').textContent = 'Type a message first.'; return; }
    var ready = busyButton('btn-steer', 'Sending…');
    try {
      var l = await ensureLink();
      var command = { type: 'send_message', session_id: r.session_id, text: text, in_run: r.id };
      // Into the work running now; a continued run starts from its end.
      if (mode === 'steer') command.mid_run = 'steer';
      else if (mode === 'queue') command.mid_run = 'queue';
      await l.command(command);
      $('steer-text').value = '';
      $('run-note').textContent = mode === 'steer' ? 'Sent to the agent.' :
        (mode === 'queue' ? 'Queued for when this run finishes.' : 'Sent. It goes on from where it ended.');
    } catch (e) { $('run-note').textContent = plainError(e); }
    finally { ready(); }
  }

  async function loadNewCodeOptions() {
    try {
      var l = await ensureLink();
      var projects = (await l.command({ type: 'list_projects' })).projects || [];
      var options = (await l.command({ type: 'list_models' })).options || {};
      var p = $('project');
      var prevProject = p.value;
      clear(p);
      projects.filter(function (x) { return x.path; }).forEach(function (x) {
        var o = el('option', null, x.name);
        o.value = x.id;
        p.appendChild(o);
      });
      if (prevProject) p.value = prevProject;
      var a = $('agent');
      var prevAgent = a.value;
      clear(a);
      (options.agents || []).forEach(function (x) {
        var o = el('option', null, x.name);
        o.value = x.id;
        a.appendChild(o);
      });
      if (prevAgent) a.value = prevAgent;
      if (!p.options.length) $('newcode-note').textContent = 'No project with a folder yet.';
    } catch (e) { $('newcode-note').textContent = plainError(e); }
  }

  async function openNewCode() {
    $('newcode-note').textContent = '';
    setMode('type', 'newcode');
    show('newcode');
    await loadNewCodeOptions();
  }

  async function startCode() {
    var writingHere = writing.form === 'newcode' && writing.mode === 'write';
    if (writingHere && !keepLine()) return;
    var text = writingHere ? '' : $('newcode-text').value.trim();
    var handwriting = writingHere && writing.lines.length > 0;
    var project = $('project').value;
    var agent = $('agent').value;
    if ((!text && !handwriting) || !project || !agent) {
      $('newcode-note').textContent = 'Pick a project and an agent, and say what to do.';
      return;
    }
    var ready = busyButton('btn-start-code', 'Starting…');
    try {
      var l = await ensureLink();
      var command = { type: 'send_message', text: text, project_id: project, model: { mode: 'agent', agent: agent } };
      if (handwriting) command.attachments = await uploadHandwriting(l, $('newcode-note'));
      var result = await l.command(command);
      if (current.view !== 'newcode') return;
      if (handwriting) clearAllLines();
      else $('newcode-text').value = '';
      await openChat({ id: result.session_id, title: text.slice(0, 40) || 'Handwritten session' }, 'code', true);
    } catch (e) {
      $(current.view === 'convo' ? 'convo-note' : 'newcode-note').textContent = plainError(e);
    } finally {
      ready();
    }
  }

  // -------------------------------------------------------------- approvals
  var approvals = [];
  var approvalPoll = null;

  function applyGlance(glance) {
    overview.pending_approvals = glance.pending_approvals || [];
    overview.running_tasks = glance.running_tasks || [];
    renderApprovals(overview);
    renderCode();
  }

  function startApprovalPolling() {
    if (approvalPoll !== null) return;
    var checking = false;
    var timer = setInterval(async function () {
      if (checking || approvalPoll !== timer) return;
      checking = true;
      try {
        var glance = (await (await ensureLink()).command({ type: 'get_glance' })).glance || {};
        if (approvalPoll === timer) applyGlance(glance);
      } catch (e) {
        // Retry on the next poll.
      } finally {
        checking = false;
      }
    }, 45000);
    approvalPoll = timer;
  }

  function stopApprovalPolling() {
    if (approvalPoll === null) return;
    clearInterval(approvalPoll);
    approvalPoll = null;
  }

  function renderApprovals(state) {
    approvals = state.pending_approvals || [];
    setText($('approval-count'), approvals.length ? String(approvals.length) : '');
    // Chat's count: the conversations with work running that are not
    // coding ones (Code counts those). A task whose conversation is not
    // among the pages loaded counts here: the busy conversation is nearly
    // always a recent one, so it is loaded and its kind known; and work
    // whose kind is unknown should still show somewhere rather than nowhere.
    var known = {};
    (state.sessions || []).forEach(function (s) { known[s.id] = s; });
    var chatBusy = {};
    (state.running_tasks || []).forEach(function (t) {
      var s = known[t.session_id];
      if (!s || !isCoding(s)) chatBusy[t.session_id] = true;
    });
    var running = Object.keys(chatBusy).length;
    setText($('chat-working'), running ? String(running) : '');
    var ul = $('approval-list');
    // The same cards as on screen: leave the list alone (e-ink).
    var sig = listSig(approvals.map(function (a) { return [a.id, a.summary, a.risk, whenOf(a.requested_at_ms)]; }));
    if (sig === listSigs.approvals) return;
    listSigs.approvals = sig;
    clear(ul);
    approvals.forEach(function (a) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', a.summary));
      b.appendChild(el('span', 'meta', riskWord(a.risk) + ' · Asked ' + whenOf(a.requested_at_ms)));
      b.addEventListener('click', function () { current.approvalFrom = null; openApproval(a); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    if (!approvals.length) ul.appendChild(el('li', 'empty muted', 'Nothing is waiting on you.'));
  }

  function renderWarnings(ulId, warnings) {
    var ul = $(ulId);
    clear(ul);
    (warnings || []).forEach(function (w) {
      if (w) ul.appendChild(el('li', null, w));
    });
  }

  function openApproval(a) {
    current.approval = a;
    var back = document.querySelector('#view-approval [data-back]');
    var fromConvo = current.approvalFrom === 'convo';
    back.setAttribute('data-back', fromConvo ? 'convo' : 'approvals');
    back.textContent = fromConvo ? '\u2190 Conversation' : '\u2190 Approvals';
    $('ap-title').textContent = a.summary;
    var t = $('ap-details');
    clear(t);
    var rows = [['Risk', riskWord(a.risk)], ['Kind', kindWord(a.kind)], ['Asked', whenOf(a.requested_at_ms)]];
    if (a.expires_at_ms) rows.push(['Expires', whenOf(a.expires_at_ms)]);
    if (a.channel) rows.push(['Also in', channelWord(a.channel)]);
    rows.forEach(function (row) {
      var tr = el('tr');
      tr.appendChild(el('th', null, row[0]));
      tr.appendChild(el('td', null, row[1]));
      t.appendChild(tr);
    });
    renderWarnings('ap-warnings', a.warnings);
    $('ap-exact').textContent = a.detail || a.summary;
    // Whether to offer Approve. The daemon (approvals.rs `resolve`) lets an
    // e-ink device allow only an agent's call ('command'), once, at or
    // below the device's risk ceiling; the sign-in's grant says that
    // ceiling (AccessGrant.risk_ceiling: Medium for a new e-ink device,
    // higher if the owner raised it) and scopes. The card's `options` are
    // not filtered per device, so the page judges with scopes + ceiling,
    // and the daemon still decides: a refusal comes back as its own message.
    // Critical is never approved from here. Before any sign-in (a page
    // drawn from the cache) the ceiling is taken as Medium.
    var canApproveScope = hasScope('approve');
    var RANK = { low: 1, medium: 2, high: 3, critical: 4 };
    var ceiling = riskCeiling === 'none' ? 3 : (RANK[riskCeiling || 'medium'] || 2);
    var rank = RANK[a.risk] || 4;
    var mayApprove = canApproveScope && (a.options || []).indexOf('approve_once') >= 0 && a.kind === 'command' &&
      rank < 4 && rank <= ceiling;
    $('btn-approve').style.display = mayApprove ? '' : 'none';
    $('btn-deny').style.display = canApproveScope ? '' : 'none';
    $('ap-note').textContent = !canApproveScope
      ? 'This Kindle is set to view only. Answer this on your phone or computer, or raise access in the app.'
      : (mayApprove ? '' : 'Approve this on your phone or computer. You can deny it from here.');
    show('approval');
  }

  var resolving = false;
  async function resolve(decision, triggerEl) {
    if (resolving) return;
    var a = current.approval;
    if (!a) return;
    resolving = true;
    var back = current.approvalFrom || 'approvals';
    var view = current.view;
    var session = current.session;
    var ready = busyButton(triggerEl, decision === 'deny' ? 'Denying…' : 'Approving…');
    var other = triggerEl === $('btn-deny') ? $('btn-approve') : null;
    if (other) other.disabled = true;
    var feedback;
    var failed = false;
    try {
      var l = await ensureLink();
      await l.command({ type: 'resolve_approval', approval_id: a.id, decision: decision });
      feedback = (decision === 'deny' ? 'Denied: ' : 'Approved once: ') + a.summary;
      overview.pending_approvals = overview.pending_approvals.filter(function (x) { return x.id !== a.id; });
      renderApprovals(overview);
    } catch (e) {
      failed = true;
      feedback = plainError(e);
    } finally {
      ready();
      if (other) other.disabled = false;
      resolving = false;
    }
    if (current.view !== view || current.session !== session || current.approval !== a) return;
    if (failed && (view === 'approval' || view === 'confirm')) {
      if (view === 'confirm') show('approval');
      $('ap-note').textContent = feedback;
      return;
    }
    current.approvalFrom = null;
    if (back === 'convo') {
      if (view !== 'convo') resumeConversation(true);
      else {
        renderInlineApprovals();
        watchConversation(session.id, false, true);
      }
      $('convo-note').textContent = feedback;
      return;
    }
    show('approvals');
    note(feedback);
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
      else if (current.view === 'newcode') await loadNewCodeOptions();
      else await loadSnapshot();
    } catch (e) {
      if (e.message !== 'Not paired') { link = null; note(plainError(e, 'Could not refresh')); setLink('offline'); }
    }
  }

  function init() {
    $('link-state').addEventListener('click', function () {
      if (!this.disabled) refreshAll();
    });
    var tabs = document.querySelectorAll('.tab');
    for (var i = 0; i < tabs.length; i++) {
      tabs[i].addEventListener('click', function () {
        show(this.getAttribute('data-view'));
        refreshAll();
      });
    }
    var backs = document.querySelectorAll('[data-back]');
    for (var j = 0; j < backs.length; j++) {
      backs[j].addEventListener('click', function () {
        var back = this.getAttribute('data-back');
        if (back === 'convo') resumeConversation(false);
        else show(back);
      });
    }
    $('btn-run-back').addEventListener('click', function () {
      if (current.runStack.length) { openRun(current.runStack.pop(), true); return; }
      if (current.runBack === 'convo') resumeConversation(false);
      else show('code');
    });
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
    $('btn-reply-cancel').addEventListener('click', function () { replyTo(null); });
    $('mode-type').addEventListener('click', function () { setMode('type', 'convo'); });
    $('mode-write').addEventListener('click', function () { setMode('write', 'convo'); });
    $('nc-mode-type').addEventListener('click', function () { setMode('type', 'newcode'); });
    $('nc-mode-write').addEventListener('click', function () { setMode('write', 'newcode'); });
    $('btn-next-line').addEventListener('click', keepLine);
    $('btn-clear-line').addEventListener('click', clearStrip);
    $('btn-undo-line').addEventListener('click', undoLine);
    $('btn-clear-all').addEventListener('click', clearAllLines);
    setupStrip();
    $('btn-page-down').addEventListener('click', function () { pageBy(1); });
    $('btn-page-up').addEventListener('click', function () { pageBy(-1); });
    $('btn-reader-back').addEventListener('click', function () {
      if (current.readerFrom === 'convo') { resumeConversation(false); return; }
      show(current.readerFrom || 'code');
      window.scrollTo(0, readerReturn);
    });
    setMode('type', 'convo');
    $('btn-new-code').addEventListener('click', openNewCode);
    $('btn-more-chat').addEventListener('click', function () { loadMore('chat'); });
    $('btn-more-code').addEventListener('click', function () { loadMore('code'); });
    $('btn-start-code').addEventListener('click', startCode);
    $('btn-steer').addEventListener('click', steer);
    $('btn-approve').addEventListener('click', function () {
      $('cf-text').textContent = current.approval.summary;
      renderWarnings('cf-warnings', current.approval.warnings);
      $('cf-exact').textContent = current.approval.detail || current.approval.summary;
      show('confirm');
    });
    $('btn-confirm').addEventListener('click', function () { resolve('approve_once', $('btn-confirm')); });
    $('btn-deny').addEventListener('click', function () { resolve('deny', $('btn-deny')); });

    if (!BRIDGE || !B) { note('This page is missing its configuration.'); return; }
    // Paint the last visit's screen at once, then refresh it.
    var last = cacheLoad();
    if (last && last.snapshot) {
      renderSnapshot(last.snapshot);
      updated(last.at);
    }
    window.addEventListener('popstate', function () { if (!followRoute()) refreshAll(); });
    // An address typed or changed by hand.
    window.addEventListener('hashchange', function () {
      var view = current.view;
      if (routeOf(view) !== window.location.hash) { if (!followRoute()) refreshAll(); }
    });
    dbGet('pinned').then(function (pinned) {
      if (!pinned) { show('pair'); setLink('not paired'); return; }
      // A conversation loads itself; a list refreshes.
      if (!followRoute()) refreshAll();
    }).catch(function () { show('pair'); });
  }

  init();
})();
