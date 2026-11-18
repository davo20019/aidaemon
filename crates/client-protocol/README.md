# aidaemon-client-protocol

**Tier: internal.** No stability promise yet; the protocol itself is versioned (`ProtocolVersion`, v1).

The wire types of the AIdaemon Client Protocol v1: the JSON frames a terminal UI, web page, desktop or phone app, or watch exchanges with the daemon's Client Gateway at `/client/v1` (WebSocket at `/client/v1/ws`, or plain HTTP: `POST /client/v1/commands` and the long-poll `GET /client/v1/events?since=<seq>&wait=<s>`). Depends only on serde and schemars, so any Rust client can use it directly.

The Rust types are the source of truth. The JSON Schema non-Rust clients generate from is checked in at `schemas/client-protocol/v1.schema.json`; regenerate it with

```sh
cargo run -p aidaemon-client-protocol --bin export-client-schema
```

A test fails when the checked-in file is stale, and golden fixtures (`tests/fixtures/*.golden`) cover every frame, command, result and event.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed.
