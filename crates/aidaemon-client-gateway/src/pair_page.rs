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

  <section id="foreign" hidden>
    <h1>This pairing link names another daemon</h1>
    <p>It came from this AIDaemon's address but points your phone somewhere else. Don't use it: send <span class="code">/pair_device</span> to your agent again and open the new link it replies with.</p>
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
  // The daemon's peer-to-peer address: iroh://<key>[@<relay host>[:port]].
  var remote = params.get('remote') || '';
  var token = /^[A-Za-z0-9_-]{1,128}$/;
  var peer = /^iroh:\/\/[0-9a-f]{64}(@[A-Za-z0-9.-]{1,253}(:[0-9]{1,5})?)?$/;
  var url = null;
  try { url = new URL(gateway); } catch (e) {}
  var valid = url && (url.protocol === 'https:' || url.protocol === 'http:') &&
    token.test(offer) && token.test(code) && (fp === '' || token.test(fp)) &&
    (remote === '' || peer.test(remote));
  if (!valid) { document.getElementById('broken').hidden = false; return; }
  // Filled in when served: the addresses and the peer-to-peer key of the
  // daemon serving this page. A link here may only pair with that daemon,
  // so a link edited to name another one is refused, not handed on.
  var ownOrigins = __OWN_ORIGINS__;
  var ownKey = __OWN_KEY__;
  var ours = url.origin === location.origin || ownOrigins.indexOf(url.origin) >= 0;
  var remoteKey = remote ? remote.slice('iroh://'.length, 'iroh://'.length + 64) : '';
  if (!ours || (remote && remoteKey !== ownKey)) {
    document.getElementById('foreign').hidden = false;
    return;
  }

  // Rebuilt from the checked values only, so nothing else reaches the app.
  var query = 'gateway=' + encodeURIComponent(gateway) + '&offer=' + offer + '&code=' + code +
    (fp ? '&fp=' + fp : '') + (remote ? '&remote=' + encodeURIComponent(remote) : '');
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

/// [`PAIR_PAGE`] as `daemon` serves it: its web app's path, the origins it
/// is reached at (`origins`) and its peer-to-peer key (`key`, 64 lowercase
/// hex digits) filled in as script literals.
pub(crate) fn pair_page(web_app: &str, origins: &[String], key: Option<&str>) -> String {
    // JSON is a script literal; `<` escaped so no value can close the tag.
    let literal = |value: serde_json::Value| value.to_string().replace('<', "\\u003c");
    let origins: Vec<String> = origins
        .iter()
        .filter_map(|origin| aidaemon_client_protocol::sign_in_audience(origin))
        .filter(|origin| origin.starts_with("http"))
        .collect();
    PAIR_PAGE
        .replace("__WEB_APP__", web_app)
        .replace("__OWN_ORIGINS__", &literal(serde_json::json!(origins)))
        .replace(
            "__OWN_KEY__",
            &literal(serde_json::json!(key.unwrap_or(""))),
        )
}

#[cfg(test)]
mod tests {
    use super::{PAIR_PAGE, pair_page};

    /// A link on the daemon's page pairs only with that daemon: its
    /// addresses and key are on the page, and the check is made before the
    /// link reaches the app.
    #[test]
    fn the_page_knows_its_daemon_and_refuses_another() {
        let page = pair_page(
            "/client/v1/next/",
            &[
                "https://app.example.com/".into(),
                "http://127.0.0.1:8788".into(),
            ],
            Some(&"ab".repeat(32)),
        );
        assert!(
            page.contains(
                r#"var ownOrigins = ["https://app.example.com","http://127.0.0.1:8788"];"#
            )
        );
        assert!(page.contains(&format!("var ownKey = \"{}\";", "ab".repeat(32))));
        assert!(page.contains("if (!ours || (remote && remoteKey !== ownKey))"));
        assert!(!page.contains("__OWN_"));
        let odd = pair_page("/", &["https://x.example.com/</script>".into()], None);
        assert!(!odd.contains("</script>\""), "{odd}");
    }

    /// The page rebuilds the app's link from checked values: the daemon's
    /// iroh address must be one of them, or the app pairs over the tunnel.
    #[test]
    fn the_app_link_keeps_the_checked_remote_address() {
        assert!(PAIR_PAGE.contains("var peer = /^iroh:"));
        assert!(PAIR_PAGE.contains("(remote === '' || peer.test(remote))"));
        assert!(PAIR_PAGE.contains("'&remote=' + encodeURIComponent(remote)"));
    }
}
