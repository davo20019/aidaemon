//! The page a pairing link opens (`/client/v1/pair#gateway=…&offer=…`):
//! it hands the offer in its fragment to the phone app (an Android intent
//! with the store as fallback, iOS's `aidaemon-pair:`), or pairs this
//! browser through the web app (`__WEB_APP__`, filled in when served).

pub(crate) const PAIR_PAGE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="robots" content="noindex">
<meta name="referrer" content="no-referrer">
<title>Pair your phone · AIDaemon</title>
<style>
:root{--bg:#0a0a0c;--surface:#0f1012;--border:#2a2a32;--text:#e8e8ed;--muted:#8a8a96;--dim:#5a5a66;--green:#00ff9d;--amber:#ffb800}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px 16px}
main{width:100%;max-width:420px;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:28px 24px}
.brand{font-family:'JetBrains Mono',ui-monospace,monospace;font-size:13px;color:var(--muted);margin-bottom:18px}
.brand b{color:var(--green);font-weight:600}
h1{font-size:22px;font-weight:600;margin-bottom:10px}
p{color:var(--muted);font-size:15px;margin-bottom:14px}
.open{display:block;width:100%;text-align:center;padding:15px 18px;margin:20px 0 14px;background:var(--green);color:var(--bg);font-weight:600;font-size:17px;border-radius:10px;text-decoration:none}
.small{font-size:13px;color:var(--dim)}
.small a{color:var(--muted)}
.code{font-family:ui-monospace,monospace;color:var(--text);letter-spacing:.04em}
.warn{color:var(--amber)}
[hidden]{display:none!important}
</style>
</head>
<body>
<main>
  <div class="brand"><b>AI</b>Daemon</div>

  <section id="phone" hidden>
    <h1>Pair this phone</h1>
    <p>Tap the button to open the AIDaemon app. Then go back to the chat you came from and approve this phone.</p>
    <a id="open" class="open" href="#">Open AIDaemon</a>
    <p class="small" id="code-line">Code <span id="code" class="code"></span> · valid for 10 minutes</p>
    <p class="small" id="get-app">Don't have the app yet? <a id="store" href="https://play.google.com/store/apps/details?id=ai.aidaemon.app">Get AIDaemon</a>, then come back here and tap the button again.</p>
    <p class="small"><a class="web" href="#">Or use AIDaemon in this browser</a></p>
  </section>

  <section id="computer" hidden>
    <h1>Open this on your phone</h1>
    <p>This link pairs the AIDaemon phone app. Open it on the phone you want to pair, or scan the QR code in the AIDaemon desktop app (Settings → Paired devices → Pair a phone).</p>
    <a class="open web" href="#">Use AIDaemon in this browser</a>
  </section>

  <section id="broken" hidden>
    <h1>This pairing link is incomplete</h1>
    <p>Send <span class="code">/pair_device</span> to your agent again and open the new link it replies with.</p>
  </section>

  <p class="small">This page comes from your own AIDaemon. The pairing details after the # stay on this device.</p>
</main>
<script>
(function () {
  var params = new URLSearchParams(location.hash.slice(1));
  var gateway = params.get('gateway') || '';
  var offer = params.get('offer') || '';
  var code = params.get('code') || '';
  var fp = params.get('fp') || '';
  var token = /^[A-Za-z0-9_-]{1,128}$/;
  var url = null;
  try { url = new URL(gateway); } catch (e) {}
  var valid = url && (url.protocol === 'https:' || url.protocol === 'http:') &&
    token.test(offer) && token.test(code) && (fp === '' || token.test(fp));
  if (!valid) { document.getElementById('broken').hidden = false; return; }

  // Rebuilt from the checked values only, so nothing else reaches the app.
  var query = 'gateway=' + encodeURIComponent(gateway) + '&offer=' + offer + '&code=' + code +
    (fp ? '&fp=' + fp : '');
  var ua = navigator.userAgent;
  var android = /Android/i.test(ua);
  var ios = /iPhone|iPad|iPod/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  // The web app pairs this browser with the same offer.
  var web = '__WEB_APP__#offer=' + offer + '&code=' + code + (fp ? '&fp=' + fp : '');
  document.querySelectorAll('a.web').forEach(function (a) { a.href = web; });
  if (!android && !ios) { document.getElementById('computer').hidden = false; return; }

  var open = document.getElementById('open');
  if (android) {
    // Chrome opens the app, or its store page when it is missing.
    open.href = 'intent:?' + query + '#Intent;scheme=aidaemon-pair;package=ai.aidaemon.app;' +
      'S.browser_fallback_url=' + encodeURIComponent('https://play.google.com/store/apps/details?id=ai.aidaemon.app') + ';end';
  } else {
    open.href = 'aidaemon-pair:?' + query;
    // No App Store listing yet: the browser link below is the way in.
    document.getElementById('get-app').hidden = true;
  }
  document.getElementById('code').textContent = code;
  document.getElementById('phone').hidden = false;
})();
</script>
</body>
</html>"##;
