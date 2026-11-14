# AIdaemon app

One Rust app for chat, approvals and the agent's running work, on the desktops, the web and a terminal, built on [Native Shell](https://nsh.dev) (nsh 0.5). It replaces the TUI, the TypeScript web page and the Tauri shell over time.

- `src/ui/`: the design system (tokens, motion, styled controls). Features never style `nsh_base` controls directly. See `DESIGN.md`.
- `src/store/`: the shared reducer from `aidaemon-client-core` in signals, plus selection, drafts and each turn's activity.
- `src/link/`: on desktops and in a terminal, `aidaemon-client-core`'s tokio link on NSH's background threads; the browser link drives the same `LinkMachine` (in progress).
- `src/features/`: sidebar, conversation, approval cards, activity inspector. `src/shell.rs` lays them out and holds the command palette (Cmd/Ctrl+K).

This directory is its own Cargo workspace; the root workspace excludes it.

## Run

Needs the `nsh` CLI 0.5.0 (`brew install longbridge/tap/nsh`) and the repository's Rust toolchain.

```sh
nsh run                 # macOS/Linux/Windows window, against the local daemon
nsh run --target tui    # the same app in this terminal
cargo test              # headless tests on the real engine
nsh test                # cargo test, then App.test.tsx
```

On the daemon's machine the app signs in with the owner's local token (`~/.aidaemon/client-token`, from `aidaemon client-token`). `AIDAEMON_GATEWAY_URL` points it elsewhere (default `ws://127.0.0.1:8788/client/v1/ws`).
