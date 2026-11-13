# AIdaemon desktop app

A Tauri 2 shell around the web client (`clients/web`). It is not a second UI: the window shows the same page the gateway serves at `/client/v1/app`, from the app's own copy, so it opens instantly and has a useful screen when the daemon is down.

What the shell adds:

- **The one connection.** The Rust side links `aidaemon-client-core` as `ClientKind::Desktop` and holds the only socket to the daemon. The page receives frames (`aidaemon://frame`) and the link's state (`aidaemon://link`) and sends commands (`link_send`) through the shell, so the gateway needs no cross-origin allowance. Only the bundled page in the main window may call the shell (`src-tauri/capabilities/default.json` lists no remote origin).
- **The device key in the OS keychain** (Keychain, Credential Manager, Secret Service), never in the web view. On the daemon's own machine the first run pairs by itself with the local token (`~/.aidaemon/client-token`, as `aidaemon tui` uses); a remote daemon pairs with the normal code (`/pair_device`), with its address in the pairing form.
- **Tray and notifications.** The tray shows the connection state and how many approvals wait; each new approval raises a native notification. Closing the window only hides it, so notifications keep coming; the tray's Open brings it back. Quit closes the app and leaves the daemon running.
- **Starting the daemon.** When nothing answers at the local address, the tray and the palette (Ctrl/Cmd+K) offer to start it through the service manager that already runs it (`launchctl kickstart gui/<uid>/ai.aidaemon` on macOS, `systemctl start aidaemon` on Linux). The app never spawns the daemon as its own child and never stops it. Not available on Windows yet.
- The app and the daemon are installed separately and may differ in version: the protocol version and the capability lists in `hello`/`welcome` cover that, and the page hides what the daemon does not offer.

The local daemon's address: `AIDAEMON_GATEWAY_URL`, else `[client_gateway] bind`/`port` in `~/.aidaemon/config.toml` (or `AIDAEMON_CONFIG`), else `http://127.0.0.1:8788`.

## Build

Needs Rust, Node 22 and the Tauri 2 prerequisites for the platform (on Linux: `libwebkit2gtk-4.1-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`).

```sh
npm install
npm run build:web      # build clients/web and copy it into dist/
npm run dev            # run the shell against the local daemon
npm run build          # release bundles (.app/.dmg, AppImage/.deb, NSIS)
```

`src-tauri` is its own Cargo workspace; the root workspace excludes `clients/desktop`, so the daemon's `cargo build`, `cargo test`, clippy and `cargo publish` never see Tauri. The icons in `src-tauri/icons` are plain placeholders; `npx tauri icon <source.png>` regenerates the full set (including `.icns` and `.ico`) from a real mark.

Releases (not wired yet): macOS signed with the project's Developer ID and notarized in `release.yml`, with a Homebrew cask; Linux as an unsigned AppImage and `.deb` with checksums; Windows as an unsigned preview.
