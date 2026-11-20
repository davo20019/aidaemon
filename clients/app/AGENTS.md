# clients/app: agent guide

The AIDaemon app: one Rust crate on Native Shell (nsh 0.6.0 from
crates.io; CI and releases pin the same nsh CLI version) for desktop, web,
terminal, iOS and Android. `CLAUDE.md` is a symlink to this file.
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

The phones build the library alone (`src/main.rs` is the desktops' binary):

```sh
nsh package --target android --abi arm64-v8a   # .apk; needs ANDROID_HOME, a JDK and the NDK
nsh package --target ios-simulator --run       # needs Xcode
nsh run --target android                       # or ios: hot reload on a device or simulator
```

The app shows its own version on the System page (`link::app_version`, from
`build.rs`: `AIDAEMON_APP_VERSION`/`AIDAEMON_APP_BUILD` when packaging sets
them, else `nsh.json`); pass the same values as `--app-version`/`--build`.

A phone never runs the daemon (`link::RUNS_DAEMON` is false, `link::MOBILE`
true): it pairs from a `/pair_device` link or an opened `aidaemon-pair:`
link (`nsh.json` `schemes`; the desktop's Settings → Paired devices → Pair a
phone shows one as a QR code, `features::pairing::pair_phone`) as `ClientKind::Mobile`, keeps its key through
`aidaemon_client_core::local::device_store` (the iOS keychain; on Android a
file in the app's data directory) and offers "Pair again" when the daemon
refuses it. Android's TLS trusts the bundled Mozilla roots: its platform
verifier needs a Java component nsh's APK does not ship.

A plain `cargo check --target wasm32-unknown-unknown` needs
`RUSTFLAGS='--cfg getrandom_backend="custom"'` (nsh's web build sets it;
`src/link/random.rs` supplies the backend).

The web build is served by the gateway from `[client_gateway] app_dir`
at `/client/v1/next/` (same origin, so sign-in needs no CORS). The demo
gateway serves it with
`AIDAEMON_APP_DIR=<path to .nsh/out/AIdaemon> cargo run -p aidaemon-client-gateway --example demo_gateway -- <port>`;
its pairing link approves itself (use `/client/v1/next/#offer=…`). With
`AIDAEMON_DEMO_SETUP=1` it starts in first-run setup with a model found
(`=none`: with none), so the setup screens can be tried with no real key.

## Invariants

- Read the nsh guides before writing UI (nsh repo `skills/nsh`, tag
  v0.6.0): components run once, only bindings rerun; never invent an API,
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
  `aidaemon-client` (the name the retired TypeScript client used; keep it).
  Never store a key a script can read.
- Model output renders as Markdown with HTML off (`html: false`).
- Nothing but drawing lives in the app. Connection state, the reducer,
  reads and their request ids, turns' activity, uploads, unread and
  stopping, coding detection, platform naming and sign-in are
  `aidaemon-client-core` (`client::Client`, `state`, `machine`, `signin`),
  so another UI toolkit reuses them unchanged. The store (`src/store/`)
  holds one `Client` in a signal (pages read it through memos, so a page
  rebuilds only when its own read changes) plus screen-only state (page,
  panes, selection, drafts, toasts, filters, this device's pins). It is
  the only writer of app state; screens call its methods. A new daemon read
  or command goes into `Client` first, with a unit test there.
- New daemon data (goals, mandates, coding receipts, the dashboard's
  status/health/usage) arrives as typed gateway protocol additions first:
  `crates/client-protocol`, the schema export and its test, golden
  fixtures, the gateway's `dispatch` arm and `ClientHost`, then the
  reducer and its fixtures. Then the screen.
