# aidaemon-client-gateway

**Tier: internal.** Internal to aidaemon, no stability promise. Published only because the daemon depends on it.

The Client Gateway: serves the AIdaemon Client Protocol v1 (`aidaemon-client-protocol`) so the owner's own apps drive and watch the daemon. An app's message runs on the same `TurnRunner` as the chat bots; every chat's turns are reported (`TurnObserver`); every approval card the hub routes can be answered from an app, first answer wins (`ApprovalTap`); durable events go to a resume log (`SqliteClientEventLog`) so a reconnecting client misses nothing and sees nothing twice; commands are idempotent by `request_id` for 24 hours.

Routes: `GET /client/v1/ws` (WebSocket), `POST /client/v1/commands` and `GET /client/v1/events?since=<seq>&wait=<s>` (HTTP mode; `X-Aidaemon-Client`, `X-Aidaemon-Device` and `X-Aidaemon-Capabilities` headers name the caller). Loopback only.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed.
