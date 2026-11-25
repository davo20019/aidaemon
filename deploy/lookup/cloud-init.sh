#!/bin/bash
# aidaemon address lookup (dns.aidaemon.ai): iroh-dns-server as a pkarr relay
# on an Always Free E2.1.Micro. Daemons publish "key -> current relay URL"
# (signed with their key, relay only, never IP addresses); apps look a key up
# before dialing, so a daemon that moves relays is still found.
# Installs and enables the service; it is started once DNS points here.
set -euxo pipefail

VERSION=v1.5.0
HOSTNAME_FQDN=dns.aidaemon.ai
CONTACT=relay@aidaemon.ai   # Let's Encrypt account contact

if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# Oracle's Ubuntu images reject everything but SSH in the host firewall.
iptables -C INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT 2>/dev/null \
  || iptables -I INPUT 1 -p tcp -m multiport --dports 80,443 -j ACCEPT
ip6tables -C INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT 2>/dev/null \
  || ip6tables -I INPUT 1 -p tcp -m multiport --dports 80,443 -j ACCEPT
netfilter-persistent save

DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y unattended-upgrades curl

tmp=$(mktemp -d)
curl -fsSL -o "$tmp/dns.tgz" \
  "https://github.com/n0-computer/iroh/releases/download/${VERSION}/iroh-dns-server-${VERSION}-x86_64-unknown-linux-musl.tar.gz"
# The archive's sha256 for VERSION (change both together).
DNS_SHA256=e49b72cfa4dbb30d25b40211db4d4b2ffc081797687fe4fcd51be11fba578a97
echo "${DNS_SHA256}  $tmp/dns.tgz" | sha256sum -c -
tar -xzf "$tmp/dns.tgz" -C "$tmp"
install -m 0755 "$(find "$tmp" -name iroh-dns-server -type f | head -1)" /usr/local/bin/iroh-dns-server
rm -rf "$tmp"

id iroh-dns >/dev/null 2>&1 || useradd --system --home /var/lib/iroh-dns --create-home --shell /usr/sbin/nologin iroh-dns

cat > /etc/iroh-dns-server.toml <<EOF
# HTTPS only: the pkarr relay (PUT/GET /pkarr/<key>). DNS answers stay on
# loopback; nothing here resolves names for the public.
pkarr_put_rate_limit = "simple"
data_dir = "/var/lib/iroh-dns"

[http]
port = 80
bind_addr = "::"

[https]
port = 443
bind_addr = "::"
domains = ["${HOSTNAME_FQDN}"]
cert_mode = "lets_encrypt"
letsencrypt_prod = true
letsencrypt_contact = "${CONTACT}"

[dns]
port = 5353
bind_addr = "127.0.0.1"
default_soa = "${HOSTNAME_FQDN} hostmaster.aidaemon.ai 0 10800 3600 604800 3600"
default_ttl = 30
origins = ["${HOSTNAME_FQDN}", "."]

[metrics]
disabled = false
bind_addr = "127.0.0.1:9117"

[mainline]
enabled = false
EOF

cat > /etc/systemd/system/iroh-dns-server.service <<'EOF'
[Unit]
Description=aidaemon address lookup (iroh-dns-server)
After=network-online.target
Wants=network-online.target

[Service]
User=iroh-dns
Group=iroh-dns
WorkingDirectory=/var/lib/iroh-dns
Environment=RUST_LOG=warn
ExecStart=/usr/local/bin/iroh-dns-server --config /etc/iroh-dns-server.toml
Restart=always
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/iroh-dns

[Install]
WantedBy=multi-user.target
EOF

# Short log retention: no visitor history kept around.
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nMaxRetentionSec=7day\n' > /etc/systemd/journald.conf.d/retention.conf
systemctl restart systemd-journald

systemctl daemon-reload
systemctl enable iroh-dns-server.service
touch /var/lib/iroh-dns/.cloud-init-done
