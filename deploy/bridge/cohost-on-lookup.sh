#!/bin/bash
# Run as root on the address-lookup server (dns.aidaemon.ai) to host the
# e-ink bridge there too: Caddy takes ports 80/443 and terminates TLS for
# both names; iroh-dns-server and aidaemon-bridge listen on loopback only.
#
#   dns.aidaemon.ai    -> 127.0.0.1:8081 (iroh-dns-server, plain HTTP)
#   bridge.aidaemon.ai -> 127.0.0.1:8787 (aidaemon-bridge)
#
# The pkarr rate limit becomes "smart" (it reads X-Forwarded-For, which
# Caddy sets to the real client address and never takes from a client), so
# clients are not all counted as Caddy. Nothing writes access logs.
#
# Needs the bridge binary (deploy.sh's build) at $BRIDGE_BIN, staged where
# only you can write (not a shared /tmp path), with BRIDGE_SHA256 its
# sha256, and the bridge.aidaemon.ai A/AAAA records pointing here. Keeps a
# copy of the old dns config.
set -euxo pipefail

CONTACT=relay@aidaemon.ai
BRIDGE_BIN="${BRIDGE_BIN:?set BRIDGE_BIN to the staged bridge binary}"
BRIDGE_SHA256="${BRIDGE_SHA256:?set BRIDGE_SHA256 to its sha256}"
echo "$BRIDGE_SHA256  $BRIDGE_BIN" | sha256sum -c -

export DEBIAN_FRONTEND=noninteractive
if ! command -v caddy >/dev/null; then
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https gnupg curl
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update
  apt-get install -y caddy
fi
systemctl stop caddy || true

# The bridge, on loopback behind Caddy.
install -m 0755 "$BRIDGE_BIN" /usr/local/bin/aidaemon-bridge
rm -f "$BRIDGE_BIN"
id aidaemon-bridge >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin aidaemon-bridge
cat > /etc/systemd/system/aidaemon-bridge.service <<'UNIT'
[Unit]
Description=aidaemon bridge (e-ink access relay)
After=network-online.target
Wants=network-online.target

[Service]
User=aidaemon-bridge
Group=aidaemon-bridge
ExecStart=/usr/local/bin/aidaemon-bridge --listen 127.0.0.1:8787 --trusted-proxy-header X-Forwarded-For
Restart=always
RestartSec=5
Environment=RUST_LOG=info
Environment=NO_COLOR=1
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
MemoryMax=256M
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT

# iroh-dns-server: plain HTTP on loopback; Caddy does TLS now.
cp -n /etc/iroh-dns-server.toml /etc/iroh-dns-server.toml.before-caddy
python3 - <<'PY'
import re
p = '/etc/iroh-dns-server.toml'
s = open(p).read()
s = re.sub(r'(?m)^pkarr_put_rate_limit = .*$', 'pkarr_put_rate_limit = "smart"', s)
s = re.sub(r'\[http\]\nport = \d+\nbind_addr = "[^"]*"', '[http]\nport = 8081\nbind_addr = "127.0.0.1"', s)
s = re.sub(r'\n\[https\]\n(?:[^\[\n][^\n]*\n)*', '\n', s)
open(p, 'w').write(s)
PY
grep -q 'port = 8081' /etc/iroh-dns-server.toml
! grep -q '^\[https\]' /etc/iroh-dns-server.toml

cat > /etc/caddy/Caddyfile <<CADDY
{
	email ${CONTACT}
}

dns.aidaemon.ai {
	header -Server
	reverse_proxy 127.0.0.1:8081
}

bridge.aidaemon.ai {
	header -Server
	reverse_proxy 127.0.0.1:8787
}
CADDY
caddy validate --config /etc/caddy/Caddyfile

systemctl daemon-reload
systemctl enable aidaemon-bridge.service caddy.service
systemctl restart aidaemon-bridge
# The switch: free 80/443, then Caddy takes them (seconds of downtime).
systemctl restart iroh-dns-server
systemctl start caddy
sleep 3
systemctl is-active aidaemon-bridge iroh-dns-server caddy
