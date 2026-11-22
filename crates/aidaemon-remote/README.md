# aidaemon-remote

Remote access for [aidaemon](https://aidaemon.ai/) over
[iroh](https://iroh.computer/): the owner's apps reach the daemon's loopback
gateway from anywhere by the daemon's endpoint key, with no tunnel, account
or domain.

- Connections go directly between the devices when the network allows, and
  through a relay otherwise. Traffic is end-to-end encrypted either way
  (QUIC with TLS 1.3 and raw public keys), so a relay sees only ciphertext.
- Each QUIC stream carries one HTTP/1.1 connection, with upgrades, served by
  the gateway's own router, so WebSockets and every route work unchanged.
- The transport authenticates the peer's key but never authorizes: the
  gateway's tokens and signed challenges still decide what a caller may do.
  `Access` only narrows which keys may open a connection, and handlers see
  the peer's key as a `RemotePeer` request extension.

Relays default to n0's public relays; `Relays::Custom` points at your own
`iroh-relay`.

Implementation crate of the aidaemon SDK; no stability promise.
