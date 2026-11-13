# aidaemon-tui

A terminal client for the AIdaemon Client Gateway, built on `aidaemon-client-protocol` alone.

```sh
aidaemon tui    # with [client_gateway] enabled; mints and saves the local token on first use
```

The standalone binary takes the address and token explicitly:
`aidaemon-tui [--url ws://127.0.0.1:8788/client/v1/ws] [--token-file FILE] [--device NAME]`
(or `AIDAEMON_CLIENT_TOKEN`).

Sessions on the left (every chat's, newest first; ● marks one that is working), the chat in the middle with live status, running work on the right, the oldest waiting approval as a card above the message box, and a status row that always says what state the connection is in (connecting, connected, reconnecting and why, daemon not running, signed out) and how many approvals wait.

The message box owns the keyboard:

| Key | Does |
|---|---|
| Enter | send (a message with no session open starts one) |
| Esc | stop the session's work (closes the palette or leaves a pane first) |
| y / s / a / n | answer the card, with the box empty, once the card has been on screen for 600 ms |
| Ctrl-P | palette: new session, stop, the waiting approval, devices, every session by title |
| Ctrl-N | new session |
| Alt-Up / Alt-Down | previous / next session |
| Tab | complete a `/` command, else move between panes (`c` cancels in the work pane) |
| Ctrl-C | clear the box, then quit |

`/new`, `/stop`, `/sessions` and `/help` are handled here; any other `/command` runs as in the chat. The connection (from `aidaemon-client-core`) reconnects by itself and resumes from the last event it received, so nothing is missed or shown twice.

Part of [aidaemon](https://github.com/davo20019/aidaemon). MIT licensed.
