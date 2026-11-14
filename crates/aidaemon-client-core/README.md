# aidaemon-client-core

The shared core of AIdaemon's own clients, with no rendering:

- `link`: the connection to the Client Gateway. Signs in, says `hello` with the last `seq` applied, pings, reconnects with backoff and resumes; commands not yet answered are resent under their own `request_id`. It reports its state (`Connecting`, `Connected`, `Disconnected` with a `DropKind`: daemon not running, unauthorized, version unsupported, dropped) so a client can always say where it stands.
- `state`: the event reducer. The client's view (sessions, transcripts, live turns, approvals, running work, the last `seq`) built only from server frames. Its JSON form is what the shared reducer fixtures in `tests/fixtures/reducer/` compare; the web client runs the same files against its TypeScript reducer.
- `signin`: the owner's local token, or a paired device key. `pair` redeems an offer with a new P-256 key and waits for the owner's confirmation; `sign_in` trades a signed challenge for a short-lived grant, which `GrantCache` renews before it expires.
- `keystore`: where a device keeps its identity. `FileKeyStore` (one owner-only file) and `MemoryKeyStore`.

Used by the nsh app (`clients/app`). Part of [aidaemon](https://github.com/davo20019/aidaemon). MIT licensed.
