# aidaemon-terminal-bridge

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

The terminal bridge runs CLI coding agents (codex, claude, gemini, opencode) in local PTYs and relays them, end-to-end encrypted (P-256 ECDH, HKDF, AES-GCM), over a daemon websocket to the owner's chat sessions. It also runs read-only code reviews with those agents (auto-approve flags are always removed), takes image uploads, and serves the local commands that attach a terminal to a running session, share it with a one-time code, or start a new one.

```rust,ignore
let started = aidaemon_terminal_bridge::spawn_if_configured(settings, host, secrets);
```

`BridgeSettings` carries what the host's config says (owner, websocket URL, bot tokens or a static connect token, review agents, upload limits). `BridgeHost` gives the bots added at runtime and stores the relay handoff codes and session bindings the host's chat commands share. The bridge key lives in an `aidaemon_secrets::SecretStore`, with an owner-only file as the fallback.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed. Docs: https://docs.rs/aidaemon-terminal-bridge
