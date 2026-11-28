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
  function note(text) { $('global-note').textContent = text || ''; }
  function setLink(text) { $('link-state').textContent = text; }
  function timeOf(ms) {
    if (!ms) return '';
    var d = new Date(ms), h = d.getHours(), m = d.getMinutes();
    return ((h % 12) || 12) + ':' + (m < 10 ? '0' : '') + m + (h < 12 ? ' AM' : ' PM');
  }

  var TAB_OF = { pair: null, chat: 'chat', convo: 'chat', code: 'code', newcode: 'code', run: 'code',
    detail: 'code', approvals: 'approvals', approval: 'approvals', confirm: 'approvals' };
  var current = { view: 'chat', session: null, approval: null, run: null };
  // Bumped by each load of a screen, so an older load that finishes late
  // never adds to what a newer one drew.
  var loads = { history: 0, chats: 0, code: 0, run: 0 };

  function show(view) {
    current.view = view;
    var views = document.querySelectorAll('.view');
    for (var i = 0; i < views.length; i++) views[i].className = 'view';
    $('view-' + view).className = 'view active';
    $('tabs').style.display = view === 'pair' ? 'none' : 'flex';
    var tabs = document.querySelectorAll('.tab');
    for (var j = 0; j < tabs.length; j++) {
      tabs[j].className = tabs[j].getAttribute('data-view') === TAB_OF[view] ? 'tab active' : 'tab';
    }
    window.scrollTo(0, 0);
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
        body: enc.encode(JSON.stringify(body))
      });
      var frames = await channel.sealer.sealMessage(bytes);
      for (var i = 0; i < frames.length; i++) sock.send(frames[i]);
      for (;;) {
        var msg = await channel.opener.open(await nextBytes(sock, 60000));
        if (!msg) continue;
        var m = B.decodeFromDaemon(msg);
        if (m.type !== 'response' || m.requestId !== requestId) continue;
        var text = dec.decode(m.body);
        var json = null;
        try { json = text ? JSON.parse(text) : null; } catch (e) { json = null; }
        return { status: m.status, json: json };
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

  function renderChats(list) {
    sessions = (list || []).slice().sort(function (a, b) {
      return (b.last_activity_ms || 0) - (a.last_activity_ms || 0);
    }).slice(0, 30);
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

  async function openChat(s) {
    current.session = s;
    $('convo-title').textContent = (s && s.title) || 'New chat';
    $('convo-note').textContent = '';
    show('convo');
    await loadHistory();
  }

  async function loadHistory() {
    var mine = ++loads.history;
    var box = $('convo-messages');
    if (!current.session || !current.session.id) { clear(box); return; }
    var l = await ensureLink();
    var result = await l.command({ type: 'get_history', session_id: current.session.id, limit: 20 });
    if (mine !== loads.history) return;
    clear(box);
    (result.messages || []).forEach(function (m) {
      var d = el('div', m.role === 'user' ? 'msg msg-you' : 'msg');
      d.appendChild(el('div', 'msg-from', m.role === 'user' ? 'You' : 'aidaemon'));
      d.appendChild(el('div', 'msg-text', m.text));
      box.appendChild(d);
    });
  }

  async function send() {
    var text = $('convo-text').value.trim();
    if (!text) { $('convo-note').textContent = 'Type something first.'; return; }
    $('btn-send').disabled = true;
    try {
      var l = await ensureLink();
      var command = { type: 'send_message', text: text };
      if (current.session && current.session.id) command.session_id = current.session.id;
      var result = await l.command(command);
      $('convo-text').value = '';
      if (!current.session || !current.session.id) current.session = { id: result.session_id, title: text.slice(0, 40) };
      $('convo-note').textContent = 'Sent. Tap Refresh for the reply.';
      await loadHistory();
    } catch (e) {
      $('convo-note').textContent = e.message;
    } finally {
      $('btn-send').disabled = false;
    }
  }

  // ------------------------------------------------------------------- code
  var PHASE_WORD = { running: 'running', succeeded: 'done', failed: 'failed', stopped: 'stopped' };

  // One request for chats, approvals and runs together.
  async function loadSnapshot() {
    var l = await ensureLink();
    var mine = ++loads.code;
    var result = await l.command({ type: 'get_snapshot' });
    if (mine !== loads.code) return;
    var state = result.state || {};
    var keep = {
      sessions: state.sessions || [], pending_approvals: state.pending_approvals || [],
      running_tasks: state.running_tasks || [], runs: state.runs || []
    };
    renderSnapshot(keep);
    cacheSave(keep);
    updated(Date.now());
  }

  function renderSnapshot(state) {
    renderChats(state.sessions);
    renderApprovals(state);
    renderCode(state.runs);
  }

  function renderCode(runs) {
    runs = runs || [];
    var running = runs.filter(function (r) { return r.phase === 'running'; });
    var done = runs.filter(function (r) { return r.phase !== 'running'; })
      .sort(function (a, b) { return (b.ended_at_ms || 0) - (a.ended_at_ms || 0); }).slice(0, 15);
    $('code-running').textContent = running.length ? String(running.length) : '';
    fillRuns($('running-list'), running);
    fillRuns($('done-list'), done);
  }

  function fillRuns(ul, runs) {
    clear(ul);
    runs.forEach(function (r) {
      var li = el('li');
      var b = el('button', 'item');
      b.type = 'button';
      b.appendChild(el('span', 'title', r.label));
      b.appendChild(el('span', 'meta', (r.agent || 'aidaemon') + ' · ' + (PHASE_WORD[r.phase] || r.phase) +
        ' · ' + (r.took || timeOf(r.started_at_ms))));
      if (r.phase === 'running' && r.step) b.appendChild(el('span', 'meta', 'Now: ' + r.step));
      b.addEventListener('click', function () { openRun(r); });
      li.appendChild(b);
      ul.appendChild(li);
    });
    if (!runs.length) ul.appendChild(el('li', 'empty muted', 'Nothing here.'));
  }

  async function openRun(r) {
    current.run = r;
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
    $('btn-new-chat').addEventListener('click', function () { openChat(null); });
    $('btn-send').addEventListener('click', send);
    $('btn-new-code').addEventListener('click', openNewCode);
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
      show('chat');
      renderSnapshot(last.snapshot);
      updated(last.at);
    }
    dbGet('pinned').then(function (pinned) {
      if (!pinned) { show('pair'); setLink('not paired'); return; }
      if (!last) show('chat');
      refreshAll();
    }).catch(function () { show('pair'); });
  }

  init();
})();
