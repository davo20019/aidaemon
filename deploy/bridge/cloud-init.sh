#!/bin/bash
# aidaemon bridge (bridge.aidaemon.ai): aidaemon-bridge behind Caddy, which
# terminates TLS (https/wss only). The bridge only forwards end-to-end
# encrypted frames; Caddy keeps no access log. Installs and enables both;
# Caddy is started once DNS points here (Let's Encrypt), and deploy.sh
# installs the bridge binary.
set -euxo pipefail

HOSTNAME_FQDN=bridge.aidaemon.ai
CONTACT=relay@aidaemon.ai   # Let's Encrypt account contact

# Oracle's Ubuntu images reject everything but SSH in the host firewall.
for t in iptables ip6tables; do
  $t -C INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT 2>/dev/null \
    || $t -I INPUT 1 -p tcp -m multiport --dports 80,443 -j ACCEPT
done
netfilter-persistent save

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y unattended-upgrades curl debian-keyring debian-archive-keyring apt-transport-https gnupg

# Caddy from its official repository.
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
  > /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y caddy
systemctl stop caddy || true

cat > /etc/caddy/Caddyfile <<CADDY
{
	email ${CONTACT}
}

${HOSTNAME_FQDN} {
	# No access log: the bridge's privacy rule extends to the proxy.
	header -Server
	reverse_proxy 127.0.0.1:8787
}
CADDY

id aidaemon-bridge >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin aidaemon-bridge

cat > /etc/systemd/system/aidaemon-bridge.service <<'UNIT'
[Unit]
Description=aidaemon bridge (e-ink access relay)
After=network-online.target
Wants=network-online.target
ConditionPathExists=/usr/local/bin/aidaemon-bridge

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
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT

mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nMaxRetentionSec=7day\n' > /etc/systemd/journald.conf.d/retention.conf
systemctl restart systemd-journald
systemctl daemon-reload
systemctl enable aidaemon-bridge.service caddy.service
touch /var/lib/.aidaemon-bridge-cloud-init-done
