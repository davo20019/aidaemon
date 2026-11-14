# aidaemon-client-gateway

**Tier: internal.** Internal to aidaemon, no stability promise. Published only because the daemon depends on it.

The Client Gateway: serves the AIdaemon Client Protocol v1 (`aidaemon-client-protocol`) so the owner's own apps drive and watch the daemon. An app's message runs on the same `TurnRunner` as the chat bots; every chat's turns are reported (`TurnObserver`); every approval card the hub routes can be answered from an app, first answer wins (`ApprovalTap`); durable events go to a resume log (`SqliteClientEventLog`) so a reconnecting client misses nothing and sees nothing twice; commands are idempotent by `request_id` for 24 hours.

Routes: `GET /client/v1/ws` (WebSocket), `POST /client/v1/commands` and `GET /client/v1/events?since=<seq>&wait=<s>` (HTTP mode; `X-Aidaemon-Client`, `X-Aidaemon-Device` and `X-Aidaemon-Capabilities` headers name the caller). Pairing: the owner makes a single-use offer (`POST /client/v1/pairing/offers` with the local token, `/pair_device` in a chat, or `aidaemon pair`), a device redeems it with its P-256 key (`POST /client/v1/enrollments/redeem`) and is confirmed in the chat (or by `POST /client/v1/pairing/offers/{id}/decision`); each connection signs a challenge (`POST /client/v1/sessions/challenge`, then `POST /client/v1/sessions`) for a 15-minute token with the device's scopes and risk ceiling. `GET /client/v1/devices` and `POST /client/v1/devices/{id}/revoke` manage them. The app's web build (`clients/app`, `nsh package --target web`) is served from `app_dir` at `/client/v1/next/`; `/client/v1/app` redirects there, or says the web app is not installed. Loopback only.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed.
