#!/usr/bin/env bash
# Assemble deploy/eink/dist: the e-ink page plus the bridge protocol's page
# side (one source: crates/aidaemon-bridge-proto/js/bridge-proto.js).
#
# Built for a slow e-ink browser on a slow link:
# - scripts and styles get content-hashed names under assets/ and are cached
#   for good (immutable); a reopened page downloads only index.html, which
#   is always revalidated;
# - whole-line comments and indentation are stripped (gzip does the rest);
# - _headers carries the strict content policy and the cache rules for a
#   static host that reads it (Cloudflare); set the same headers elsewhere.
#
# AIDAEMON_EINK_BRIDGE sets the bridge (default wss://bridge.aidaemon.ai).
# WebCrypto needs a secure context: serve over HTTPS (http://localhost also
# counts for development).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
bridge="${AIDAEMON_EINK_BRIDGE:-wss://bridge.aidaemon.ai}"
rm -rf "$here/dist"
mkdir -p "$here/dist/assets"

python3 - "$here/public" "$root/crates/aidaemon-bridge-proto/js/bridge-proto.js" "$here/dist" "$bridge" <<'PY'
import hashlib, pathlib, re, sys
public, proto, dist = (pathlib.Path(a) for a in sys.argv[1:4])
bridge = sys.argv[4]

def strip_js(text):
    out, in_block = [], False
    for line in text.splitlines():
        s = line.strip()
        if in_block:
            if '*/' in s:
                in_block = False
            continue
        if s.startswith('/*'):
            if '*/' not in s:
                in_block = True
            continue
        if not s or s.startswith('//'):
            continue
        out.append(s)
    return '\n'.join(out) + '\n'

def strip_css(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return '\n'.join(l.strip() for l in text.splitlines() if l.strip()) + '\n'

config = "window.AIDAEMON_EINK_CONFIG = { bridge: '%s' };\n" % bridge
assets = {
    'config.js': config,
    'bridge-proto.js': strip_js(proto.read_text()),
    'eink.js': strip_js((public / 'eink.js').read_text()),
    'eink.css': strip_css((public / 'eink.css').read_text()),
}
html = (public / 'index.html').read_text()
for name, text in assets.items():
    stem, ext = name.rsplit('.', 1)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    hashed = f'assets/{stem}.{digest}.{ext}'
    (dist / hashed).write_text(text)
    html = html.replace(f'./{name}', f'./{hashed}')
(dist / 'index.html').write_text(html)

origin = re.match(r'^(wss?://[^/]+)', bridge).group(1)
policy = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
          f"connect-src {origin}; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
(dist / '_headers').write_text(f"""/*
  Content-Security-Policy: {policy}
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  X-Frame-Options: DENY
  Strict-Transport-Security: max-age=31536000

/
  Cache-Control: no-cache

/index.html
  Cache-Control: no-cache

/assets/*
  Cache-Control: public, max-age=31536000, immutable
""")
PY
echo "built $here/dist (bridge $bridge)"
