#!/bin/bash
# aidaemon relay (relay.aidaemon.ai): iroh-relay on an Always Free E2.1.Micro.
# Installs and enables the service; it is started once DNS points here, so
# Let's Encrypt validation does not fail and burn its rate limit.
set -euxo pipefail

IROH_RELAY_VERSION=v1.5.0
HOSTNAME_FQDN=relay.aidaemon.ai
CONTACT=relay@aidaemon.ai   # Let's Encrypt account contact

# 1 GB RAM: add swap so apt upgrades do not run out of memory.
if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# Oracle's Ubuntu images reject everything but SSH in the host firewall.
iptables -C INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT 2>/dev/null \
  || iptables -I INPUT 1 -p tcp -m multiport --dports 80,443 -j ACCEPT
iptables -C INPUT -p udp --dport 7842 -j ACCEPT 2>/dev/null \
  || iptables -I INPUT 1 -p udp --dport 7842 -j ACCEPT
ip6tables -C INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT 2>/dev/null \
  || ip6tables -I INPUT 1 -p tcp -m multiport --dports 80,443 -j ACCEPT
ip6tables -C INPUT -p udp --dport 7842 -j ACCEPT 2>/dev/null \
  || ip6tables -I INPUT 1 -p udp --dport 7842 -j ACCEPT
netfilter-persistent save

# Unattended security updates.
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y unattended-upgrades curl

# iroh-relay, the official static build for this CPU (build-relay.sh then
# swaps in the build with the AIDaemon page).
ARCH=$(uname -m)   # x86_64 or aarch64
tmp=$(mktemp -d)
curl -fsSL -o "$tmp/relay.tgz" \
  "https://github.com/n0-computer/iroh/releases/download/${IROH_RELAY_VERSION}/iroh-relay-${IROH_RELAY_VERSION}-${ARCH}-unknown-linux-musl.tar.gz"
# The archive's sha256 for IROH_RELAY_VERSION (change both together): a
# swapped release asset must not become a root-installed binary.
case "$ARCH" in
  x86_64) RELAY_SHA256=935892e4ad570d694f7ad2780f800077b99fd72d9a56ac982792fdd04ef631b4 ;;
  aarch64) RELAY_SHA256=99a14d2ff541711f1b175a15e0c1571a45f46c171d655cda29849d91d90a5c9c ;;
  *) echo "no pinned checksum for $ARCH" >&2; exit 1 ;;
esac
echo "${RELAY_SHA256}  $tmp/relay.tgz" | sha256sum -c -
tar -xzf "$tmp/relay.tgz" -C "$tmp"
install -m 0755 "$(find "$tmp" -name iroh-relay -type f | head -1)" /usr/local/bin/iroh-relay
rm -rf "$tmp"

id iroh-relay >/dev/null 2>&1 || useradd --system --home /var/lib/iroh-relay --create-home --shell /usr/sbin/nologin iroh-relay
install -d -o iroh-relay -g iroh-relay -m 0700 /var/lib/iroh-relay/certs

cat > /etc/iroh-relay.toml <<EOF
enable_relay = true
http_bind_addr = "[::]:80"
enable_quic_addr_discovery = true
enable_metrics = true
metrics_bind_addr = "127.0.0.1:9090"

[tls]
cert_mode = "LetsEncrypt"
hostname = "${HOSTNAME_FQDN}"
contact = "${CONTACT}"
cert_dir = "/var/lib/iroh-relay/certs"
prod_tls = true
https_bind_addr = "[::]:443"
quic_bind_addr = "[::]:7842"

[limits]
accept_conn_limit = 20.0
accept_conn_burst = 100

[limits.client.rx]
bytes_per_second = 4194304
max_burst_bytes = 8388608
EOF

cat > /etc/systemd/system/iroh-relay.service <<'EOF'
[Unit]
Description=aidaemon relay (iroh-relay)
After=network-online.target
Wants=network-online.target

[Service]
User=iroh-relay
Group=iroh-relay
ExecStart=/usr/local/bin/iroh-relay --config-path /etc/iroh-relay.toml
Restart=always
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/iroh-relay
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# Privacy: warnings only (no per-connection lines), logs kept 7 days.
mkdir -p /etc/systemd/system/iroh-relay.service.d /etc/systemd/journald.conf.d
printf '[Service]\nEnvironment=RUST_LOG=warn\nEnvironment=NO_COLOR=1\n' > /etc/systemd/system/iroh-relay.service.d/privacy.conf
printf '[Journal]\nMaxRetentionSec=7day\n' > /etc/systemd/journald.conf.d/retention.conf
systemctl restart systemd-journald

systemctl daemon-reload
systemctl enable iroh-relay.service
touch /var/lib/iroh-relay/.cloud-init-done
