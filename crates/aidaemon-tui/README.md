# aidaemon-tui

A terminal client for the AIdaemon Client Gateway, built on `aidaemon-client-protocol` alone.

```sh
aidaemon client-token > ~/.aidaemon/tui-token   # once; enable [client_gateway] first
cargo run -p aidaemon-tui -- --token-file ~/.aidaemon/tui-token
```

Sessions on the left (every chat's, newest first; ● marks one that is working), the chat in the middle with live status, the oldest waiting approval as a card under it, running work on the right. Tab moves between panes; in the card, `y` allows once, `s`/`a` allows for the session or always (as the card offers), `n` denies; in the work pane `c` cancels the selected entry. Type `/stop` to stop the session's work, any other `/command` runs as in the chat; Ctrl-N starts a new session. The connection reconnects by itself and resumes from the last event it received, so nothing is missed or shown twice.

Part of [aidaemon](https://github.com/davo20019/aidaemon). MIT licensed.
