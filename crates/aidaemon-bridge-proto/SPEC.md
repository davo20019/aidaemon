# aidaemon bridge protocol, v1

How a browser that cannot run iroh (the Kindle Scribe's Chrome 80) reaches an
owner's daemon through a public, untrusted relay (`aidaemon-bridge`). The
bridge forwards bytes; the page and the daemon encrypt end to end. Design
and threat model: `reports/2026-11-27 Kindle e-ink access plan.md` (local)
and, once published, `crates/aidaemon-bridge/SECURITY.md`.

Every implementation (Rust here, JavaScript in the page) must match this
document and `vectors/v1.json` exactly. Anything not described is refused.

## 1. Primitives and encodings

| Name | Meaning |
|---|---|
| `H(x)` | SHA-256 |
| `lp(x)` | 4-byte big-endian length of `x`, then `x` |
| `u8`, `u16`, `u32`, `u64` | big-endian unsigned integers |
| point | P-256 public key, **65-byte uncompressed SEC1 only** (`0x04 ‖ X ‖ Y`). Compressed or other lengths are refused. |
| ECDH | P-256; the shared secret is the 32-byte X coordinate (WebCrypto `deriveBits(…, 256)`) |
| signature | ECDSA P-256 with SHA-256 over the message bytes, encoded as 64 bytes `r ‖ s` (IEEE P1363, WebCrypto's format). High-S signatures are accepted (WebCrypto does not normalize). |
| HKDF | HKDF-SHA-256 (RFC 5869) |
| HMAC | HMAC-SHA-256 |
| AEAD | AES-256-GCM, 12-byte nonce, 16-byte tag |
| hex | lowercase |

String constants are ASCII bytes with no terminator.

## 2. Identifiers

```
daemon_id = H("aidaemon-daemon-id-v1"     ‖ identity_pub)   // daemon's handshake key
device_id = H("aidaemon-device-id-v1"     ‖ device_pub)     // page's device key
route_id  = H("aidaemon-bridge-route-v1"  ‖ login_pub)      // daemon's bridge login key
```

All 32 bytes; 64 hex characters when written as text. `device_id` is also
the `BridgePeer` endpoint id the gateway records (it requires 64 hex).

The daemon holds two separate P-256 keys: the **login key** (only for the
bridge login, §3) and the **identity key** (only for handshakes, §5–6).

## 3. Bridge transport (plaintext control, opaque data)

All connections are `wss://`. Text frames are JSON control messages; binary
frames carry protocol frames (§4) the bridge never reads.

### Daemon side: `wss://<bridge>/v1/daemon`

1. Bridge → daemon: `{"type":"challenge","nonce":"<64 hex>"}` (32 random bytes).
2. Daemon → bridge: `{"type":"login","login_pub":"<130 hex>","sig":"<128 hex>"}`,
   `sig = Sign(login_key, "aidaemon-bridge-auth-v1:" ‖ nonce_bytes)`.
3. Bridge verifies, derives `route_id`, replies `{"type":"ready","route_id":"<64 hex>"}`.
   A second daemon logging in with the same `route_id` replaces the first.
4. The daemon's one WebSocket carries every browser connection routed to it:
   - Bridge → daemon `{"type":"open","conn":<u32>}` when a browser joins.
   - Binary frames both ways are `u32 conn ‖ protocol frame`.
   - Daemon → bridge `{"type":"promote","conn":n}` after a handshake
     completes: the bridge raises that connection's frame limit from the
     handshake limit to the data limit (§7).
   - Either side `{"type":"close","conn":n}`; the bridge closes the browser.
   - Daemon → bridge `{"type":"join","nameplate":"<4 chars>"}` to join a
     waiting browser (pairing, §6); the bridge answers with `open` for that
     browser or `{"type":"join_failed","nameplate":…}`.

### Browser side

- After pairing: `wss://<bridge>/v1/connect?route=<route_id hex>`. Refused
  (closed) if no daemon with that route is logged in.
- Pairing: `wss://<bridge>/v1/wait?nameplate=<4 chars>`. The browser waits
  until a daemon joins that nameplate, up to 10 minutes. A nameplate already
  waiting is refused (the page picks a new random one). Nameplates are 4
  characters from the Crockford Base32 alphabet `0123456789ABCDEFGHJKMNPQRSTVWXYZ`,
  uppercase only (the page uppercases what the owner types).
- Binary frames are protocol frames (§4), unprefixed.

Refusals before a WebSocket upgrade are HTTP statuses (400 malformed, 404
unknown route, 409 nameplate already waiting, 429 rate limit, 503 full);
after it, close codes (1003 text frame from a browser, 1008 protocol
violation or timeout, 1009 oversize, 1013 cap or budget spent, 1001 daemon
replaced). `promote`, `close` and binary frames for an unknown `conn` are
ignored (they can race a close). See `crates/aidaemon-bridge/README.md`.

## 4. Frames

```
frame = version:u8 (=1) ‖ type:u8 ‖ flags:u8 ‖ counter:u64 ‖ body
```

| type | name | body |
|---|---|---|
| `0x01` | HELLO | plaintext, page → daemon |
| `0x02` | HELLO_REPLY | plaintext, daemon → page |
| `0x03` | FINISH | plaintext, page → daemon |
| `0x10` | DATA | ciphertext ‖ tag |
| `0x7F` | ABORT | empty; either side; the receiver closes |

Handshake frames have `flags = 0`, `counter = 0`. Their bodies are
concatenations of `lp` fields, exactly as listed; trailing bytes are refused.

DATA: `flags` bit 0 = FINAL (last fragment of a message); other bits must be
0. `counter` starts at 0 in each direction after the handshake and increases
by exactly 1 per DATA frame. AEAD with that direction's key,
`nonce = 0x00000000 ‖ u64(counter)`, associated data = the 11 header bytes.

A wrong version, unknown type, nonzero reserved flags, a counter that is not
the next one, a failed decryption, or any handshake frame after the
handshake: close the connection. Never retry or skip.

## 5. Session handshake (mode 1, after pairing)

The page knows `identity_pub` (pinned at pairing). The daemon knows
`device_pub` (on record, not revoked).

```
HELLO       = lp(u8 mode=1) ‖ lp(E_k) ‖ lp(N_k) ‖ lp(device_id)
HELLO_REPLY = lp(E_d) ‖ lp(N_d) ‖ lp(sig_d)
FINISH      = lp(sig_k)

T = lp("aidaemon-bridge-v1") ‖ lp(u8 1) ‖ lp(identity_pub) ‖ lp(device_pub)
    ‖ lp(E_k) ‖ lp(E_d) ‖ lp(N_k) ‖ lp(N_d)

sig_d = Sign(identity_key, "aidaemon-bridge-hs-v1:" ‖ H(lp("daemon") ‖ T))
sig_k = Sign(device_key,   "aidaemon-bridge-hs-v1:" ‖ H(lp("device") ‖ T))
```

`E_k`, `E_d`: fresh ECDH points per connection. `N_k`, `N_d`: 32 random bytes.

Order: the daemon looks up `device_id`, refuses unknown or revoked devices,
sends HELLO_REPLY. The page verifies `sig_d` with the pinned
`identity_pub` (a mismatch is a hard failure, never re-pinned), sends FINISH.
The daemon verifies `sig_k`. Then keys (§7) with
`ikm = ECDH(E_k, E_d)`, `salt = H(T)`.

The daemon then sends `promote` to the bridge and, after it, a DATA control
message (§8) `{"type":"ready"}`. **The page sends no DATA frame until it
has received and opened that `ready`**: before the bridge's `promote`, a
page frame over the handshake limit would close the connection.

## 6. Pairing handshake (mode 2)

The page generated `device_key`, a `secret` (13 Crockford Base32 characters,
65 bits; its ASCII bytes are used below), and a `nameplate`, waits on the
bridge (§3), and shows them to the owner. The owner's trusted app sends
`PairBridgeDevice { nameplate, secret, device_key_hash }` to the daemon,
which joins the nameplate.

```
HELLO       = lp(u8 mode=2) ‖ lp(E_k) ‖ lp(N_k) ‖ lp(device_pub) ‖ lp(device_name)
HELLO_REPLY = lp(E_d) ‖ lp(N_d) ‖ lp(identity_pub) ‖ lp(pairing_id) ‖ lp(mac_d) ‖ lp(sig_d)
FINISH      = lp(mac_k) ‖ lp(sig_k)

T = lp("aidaemon-bridge-v1") ‖ lp(u8 2) ‖ lp(device_pub) ‖ lp(device_name)
    ‖ lp(E_k) ‖ lp(N_k) ‖ lp(identity_pub) ‖ lp(E_d) ‖ lp(N_d) ‖ lp(pairing_id)

prk        = HKDF-Extract(salt = H(T), ikm = ECDH(E_k, E_d) ‖ secret)
confirm_d  = HKDF-Expand(prk, "aidaemon-bridge-v1 confirm daemon", 32)
confirm_k  = HKDF-Expand(prk, "aidaemon-bridge-v1 confirm device", 32)
mac_d      = HMAC(confirm_d, H(T))
mac_k      = HMAC(confirm_k, H(T))
sig_d      = Sign(identity_key, "aidaemon-bridge-pair-v1:" ‖ H(lp("daemon") ‖ T))
sig_k      = Sign(device_key,   "aidaemon-bridge-pair-v1:" ‖ H(lp("device") ‖ T))
```

`device_name`: UTF-8, 1–60 bytes. `pairing_id`: 16 random bytes from the
daemon.

Order: the daemon checks `H(device_pub)` (as `device_id`) starts with the
`device_key_hash` the app sent (the full 32 bytes from a QR scan, or the
leading bytes from a typed code), sends HELLO_REPLY. The page checks
`mac_d` (constant-time) and `sig_d`; on failure it shows "pairing failed"
and starts over. The page sends FINISH; the daemon checks `mac_k`
(constant-time) and `sig_k`, creates the device, and sends a DATA control
message (§8) `{"type":"paired","daemon_id":…,"route_id":…}`. Only then does
the page store `identity_pub` and `route_id`. Any failure ends this
pairing; the page starts a new one.

Data keys (§7) use `ikm = ECDH(E_k, E_d) ‖ secret`, `salt = H(T)`.

## 7. Data keys and limits

```
prk     = HKDF-Extract(salt, ikm)              // as given per mode
k_d2p   = HKDF-Expand(prk, "aidaemon-bridge-v1 key daemon to page", 32)
k_p2d   = HKDF-Expand(prk, "aidaemon-bridge-v1 key page to daemon", 32)
```

| Limit | Value |
|---|---|
| handshake frame, page → bridge, before `promote` | 1 KiB |
| DATA frame | 64 KiB |
| reassembled request | 16 MiB |
| reassembled response | 4 MiB |
| handshake time | 30 s |

## 8. Messages (DATA plaintext, reassembled up to the FINAL fragment)

```
request  = u8 1 ‖ u32 request_id ‖ lp(method) ‖ lp(path_and_query)
           ‖ u16 n ‖ n × (lp(name) ‖ lp(value)) ‖ lp(body)
response = u8 2 ‖ u32 request_id ‖ u16 status
           ‖ u16 n ‖ n × (lp(name) ‖ lp(value)) ‖ lp(body)
control  = u8 3 ‖ lp(json)
```

Header names are lowercase ASCII. A request may carry only `authorization`,
`content-type` and `x-aidaemon-request-id`; others are refused. Responses
carry only `content-type`. The page sends one request at a time and waits
for its response.

## 9. Test vectors

`vectors/v1.json` fixes private scalars, nonces, secret and names, and gives
the expected ids, transcripts, transcript hashes, HKDF outputs, MACs, one
encrypted DATA frame per direction and encoded request/response messages.
ECDSA signatures are randomized in WebCrypto, so vectors give Rust's
deterministic (RFC 6979) signatures for the JavaScript side to **verify**,
not reproduce.
