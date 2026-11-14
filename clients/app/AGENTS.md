# clients/app: agent guide

The AIdaemon app: one Rust crate on Native Shell (nsh 0.5.0, pinned with
`=`) for desktop, web and terminal. `CLAUDE.md` is a symlink to this file.
It is its own Cargo workspace; the root workspace excludes it, so the
root's `cargo test`/clippy never build it. Run its checks here.

## Build, run, test

```sh
cargo check --all-targets && cargo clippy --all-targets -- -D warnings
cargo test                 # headless, on the real engine (tests/app.rs)
nsh run                    # desktop window against the local daemon
nsh test --target tui      # the terminal build
CC_wasm32_unknown_unknown=/opt/homebrew/opt/llvm/bin/clang \
AR_wasm32_unknown_unknown=/opt/homebrew/opt/llvm/bin/llvm-ar \
nsh package --target web   # .nsh/out/AIdaemon; needs wasm-bindgen-cli 0.2.130
```

A plain `cargo check --target wasm32-unknown-unknown` needs
`RUSTFLAGS='--cfg getrandom_backend="custom"'` (nsh's web build sets it;
`src/link/random.rs` supplies the backend).

The web build is served by the gateway from `[client_gateway] app_dir`
at `/client/v2/app/` (same origin, so sign-in needs no CORS). The demo
gateway serves it with
`AIDAEMON_APP_DIR=<path to .nsh/out/AIdaemon> cargo run -p aidaemon-client-gateway --example demo_gateway -- <port>`;
its pairing link approves itself (use `/client/v2/app/#offer=…`).

## Invariants

- Read the nsh guides before writing UI (nsh repo `skills/nsh`, tag
  v0.5.0): components run once, only bindings rerun; never invent an API,
  look it up in the resolved crate source.
- Features build only from `src/ui/` (tokens, motion, styled controls). No
  raw colours, radii or font sizes in `src/features/`; a missing look is a
  new variant in `src/ui/` first. `DESIGN.md` holds the standards and the
  motion inventory; update it with each screen.
- One link per platform, one protocol brain: `aidaemon-client-core`'s
  `LinkMachine` (resume, resend, ping, backoff) and `State` reducer. Desktop
  and terminal run the core's tokio link on nsh's `background` threads
  (`src/link/native.rs`); the browser drives `LinkMachine` over the page's
  WebSocket (`src/link/web.rs`, `js/browser.js`). Do not add protocol logic
  to the app.
- The browser's device key is WebCrypto, non-extractable, in IndexedDB
  `aidaemon-client` (shared with `clients/web`). Never store a key a script
  can read.
- Model output renders as Markdown with HTML off (`html: false`).
- The store (`src/store/`) is the only writer of app state; screens call
  its methods.
- New daemon data (goals, mandates, coding receipts, the dashboard's
  status/health/usage) arrives as typed gateway protocol additions first:
  `crates/client-protocol`, the schema export and its test, golden
  fixtures, the gateway's `dispatch` arm and `ClientHost`, then the
  reducer and its fixtures. Then the screen.
