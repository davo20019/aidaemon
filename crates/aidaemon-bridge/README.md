# aidaemon-bridge

A small public WebSocket relay for aidaemon's bridge protocol
(`crates/aidaemon-bridge-proto/SPEC.md`). It lets a browser that cannot run
iroh (an e-ink reader's old Chrome) reach its owner's daemon: the daemon
keeps one WebSocket open to the bridge, and the bridge joins browsers to
it and forwards their frames.

## What it can and cannot see

The page and the daemon encrypt end to end (SPEC §5–7). The bridge never
holds a key and never decrypts anything.

It **can** see: that a daemon with a given login key (its `route_id`) is
online; when browsers connect to it, from which address, and for how long;
pairing nameplates (4 characters, single use, never the pairing secret);
the number, size and timing of frames.

It **cannot** see: requests, responses, chat, tokens or the pairing secret;
which device a browser is (the device id travels inside the handshake);
anything that would let it impersonate the daemon (the handshake is signed
by the daemon's identity key, which the bridge never sees) or a device.

A malicious bridge can drop, delay or refuse traffic, and learn traffic
patterns. It cannot read or forge it.

The bridge logs events and refusals only ("daemon logged in", "connection
refused: limit"), never frame contents, sizes, nameplates, route ids or
client addresses.

## Routes

| Route | Who | What |
|---|---|---|
| `GET /v1/daemon` | daemon | challenge (32 random bytes), signed login, `ready` with `route_id`; then the multiplexed session. A new login with the same key replaces the old one. |
| `GET /v1/connect?route=<64 hex>` | browser | joins a logged-in daemon; `404` if none |
| `GET /v1/wait?nameplate=<4 chars>` | browser | waits up to 10 minutes for the daemon to `join`; `409` if that nameplate is already waiting |
| `GET /healthz` | anyone | `ok` |

Refusals before the upgrade are HTTP statuses (`400` malformed query,
`404`, `409`, `429` rate limit, `503` full). Refusals after it are WebSocket
close codes: `1003` text frame from a browser, `1008` protocol violation or
handshake timeout, `1009` frame over the limit, `1013` a cap or the day's
byte budget, `1001` daemon replaced, `1000` normal close.

## Running it

```sh
cargo build --release -p aidaemon-bridge
./target/release/aidaemon-bridge --listen 127.0.0.1:8787 \
    --trusted-proxy-header X-Forwarded-For
```

Every option is a flag (`--name value` or `--name=value`) or an
environment variable `AIDAEMON_BRIDGE_<NAME>` (uppercase, `-` → `_`); a
flag wins. `aidaemon-bridge --help` lists them.

The bridge speaks plain HTTP and must sit behind a reverse proxy that
terminates TLS and serves **only https/wss**: never expose the bridge's
port itself, and never serve the proxy over plain http (frames stay
encrypted, but plain `ws://` exposes the control messages and nameplates to
the network and lets anyone on the path redirect or drop them). Bind the bridge to loopback, and set
`--trusted-proxy-header` only when that proxy is the sole way in; otherwise
any client could pick its own address for the rate limits.

Caddy (it gets and renews the certificate itself):

```caddyfile
{
	auto_https disable_redirects
}

bridge.example.com {
	# Caddy appends the client address it saw to X-Forwarded-For; the
	# bridge reads the last entry.
	reverse_proxy 127.0.0.1:8787
}
```

`disable_redirects` stops Caddy's http-to-https redirects (port 80 then
only answers ACME challenges); drop it if you prefer the redirect. Either
way nothing is proxied over plain http, so no WebSocket is served on `ws://`.

## Limits

| Option | Default | Meaning |
|---|---|---|
| `handshake-frame-max` | 1024 | largest browser frame before the daemon sends `promote` for it (SPEC §7) |
| `data-frame-max` | 65536 | largest browser frame after `promote`; daemon frames may be 4 bytes more (the conn id) |
| `max-conns-per-daemon` | 8 | browsers one daemon carries |
| `max-handshaking-per-daemon` | 4 | of those, not yet promoted; one more is refused so a flood cannot starve real ones |
| `handshake-timeout-secs` | 30 | a connection not promoted in time is closed |
| `wait-timeout-secs` | 600 | how long a pairing browser waits |
| `login-timeout-secs` | 10 | a daemon must answer the challenge in time |
| `idle-timeout-secs` | 600 | a browser connection with no frame either way is closed |
| `keepalive-secs` | 30 | the bridge pings each daemon; three silent intervals close it |
| `send-timeout-secs` | 10 | a peer that does not take a frame in time is closed, never buffered for |
| `browser-queue` / `daemon-queue` | 64 / 256 | frames queued toward one browser / one daemon |
| `connects-per-ip-per-minute` | 30 | new `/v1/connect` per client address (IPv6: per /64) |
| `logins-per-ip-per-minute` | 10 | daemon logins per client address |
| `waits-per-ip-per-minute` | 10 | pairing waits per client address |
| `daily-bytes-per-route` | 2 GiB | bytes forwarded per daemon (both ways) in a day; over it, the connection closes and new ones are refused until the day rolls over |
| `max-daemons` / `max-waiting` | 10000 / 10000 | daemons logged in / browsers waiting, bridge-wide |
