# deploy

How the project's two public servers are set up. Both run n0's open-source
iroh servers at the iroh version the code uses (1.5.0), on any Ubuntu 24.04
machine with a public address.

| Server | What it does | Ports |
|---|---|---|
| `relay.aidaemon.ai` (`relay/`) | Remote access's default relay: daemons keep a connection here so apps find them by key, and it forwards end-to-end encrypted traffic when two devices can't connect directly. | 443/tcp, 80/tcp (certificates), 7842/udp (address discovery) |
| `app.aidaemon.ai` (`web/`) | The web app on a site of its own: static files on a Cloudflare Worker. A phone with no app pairs there from a QR code and reaches its daemon by key through the relay; this site never sees that traffic. | 443/tcp (Cloudflare) |
| `dns.aidaemon.ai` (`lookup/`) | Address lookup: a daemon on the project's relays publishes which relay it is on (signed with its key, the relay URL only), so an app whose saved relay is stale still finds it. | 443/tcp, 80/tcp |

Each `cloud-init.sh` installs the server, its firewall rules (IPv4 and IPv6),
swap, unattended upgrades and a locked-down systemd unit, then enables the
service without starting it. Pass it as a new machine's user data, or run it
as root on one.

## Set one up

1. Create the machine and open its ports in the provider's firewall too
   (`oracle-ingress.json` holds the rules used on Oracle Cloud).
2. Run `cloud-init.sh` (or pass it at creation).
3. Point the hostname's A and AAAA records at the machine, DNS only (no
   proxy: the relay needs UDP).
4. Once the name resolves, `sudo systemctl start iroh-relay` (or
   `iroh-dns-server`). Starting earlier fails the Let's Encrypt check and
   spends its rate limit.
5. Relay only: `relay/build-relay.sh` builds `iroh-relay` with the AIDaemon
   landing page (`relay/index.html`), cross-compiled with `cargo zigbuild`,
   and deploys it (`HOST=`, `TARGET=aarch64-unknown-linux-musl` for ARM).

## The web app

`web/build.sh` builds the app's web package with its iroh module and service
worker into `web/dist`; `cd web && npx wrangler deploy` publishes it. To host
your own copy elsewhere, serve `web/dist` as static files (with the headers
in `web/_headers`) and point your daemon at it:

```toml
[client_gateway.remote]
web_app = "https://app.example.com"
```

## Your own relay

Change the hostname and contact address at the top of `relay/cloud-init.sh`,
set it up as above, then point your daemon at it:

```toml
[client_gateway.remote]
relays = ["https://relay.example.com"]
```

Your devices then use only that relay. The project's address lookup goes
with the project's relays only, so a self-hosted relay publishes nowhere.

## Usage counts

`lookup/stats/` counts, once a day, how many daemons published to the
address lookup in the last day and the last 7 days (each daemon republishes
every 5 minutes under a key that stays the same across restarts). It reads a
copy of the lookup server's store and appends counts only, never keys or
addresses, to `/var/lib/aidaemon-stats/daily.csv` on that server. It sees only
daemons on the project's relays with remote access on (0.22.1 and later).
`lookup/stats/deploy.sh` builds and installs it with its systemd timer and
leaves the maintainers' own daemons out of the count (the public key of each
file in `OWN_KEYS`, default `~/.aidaemon/remote-key`, goes to
`/etc/aidaemon-stats/exclude`). Read the counts with
`ssh ubuntu@dns.aidaemon.ai cat /var/lib/aidaemon-stats/daily.csv`.

## Upgrading iroh

The servers must match the iroh version in `Cargo.toml`. When it changes,
bump `IROH_RELAY_VERSION` in `relay/cloud-init.sh`, `VERSION` in
`relay/build-relay.sh` and `VERSION` in `lookup/cloud-init.sh`, rerun
`build-relay.sh`, and replace the lookup server's binary.

`.github/workflows/relay-health.yml` checks both servers from outside every
15 minutes. Resource ids (`ids.env`) stay out of git.
