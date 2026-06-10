# Computer Use on macOS

`computer_use` lets the agent operate **native macOS apps** — it reads an app's
accessibility tree, takes a screenshot, and clicks / types / scrolls to complete
a task, looping until done. It complements the `browser` tool (web pages) and
`terminal` (shell).

It is **macOS-only**, **off by default**, and **feature-gated**. You opt in by
building with the feature and enabling it in config.

---

## Why there's a setup step (read this first)

macOS requires two permissions for desktop automation — **Accessibility**
(to read the UI and send input) and **Screen Recording** (to capture the window
screenshot). Apple only lets a *user* grant these in System Settings; no app can
self-grant. That one-time manual toggle is unavoidable.

The catch that makes naive setups painful: macOS keys those grants to a program's
**code signature**. A bare binary built by `cargo` is *ad-hoc signed*, and that
signature changes on **every rebuild** — so the grant you made yesterday no
longer matches today's binary, and you're asked to re-approve endlessly. Running
as a launchd background agent makes it worse (the system won't reliably show the
grant dialog).

The mitigation is to package the daemon as a small signed **`.app` bundle** with
a fixed **bundle identifier** (`ai.aidaemon`), signed by a stable local identity
(`aidaemon-dev`). macOS keys the grant to the bundle's *designated requirement*
(identifier + signing certificate), which is the same across rebuilds — so in
principle re-signing a new build keeps the grant valid. The scripts below
automate this.

> **Honest caveat (self-signed dev setup):** with a *self-signed* identity on
> recent macOS, this is not bulletproof. The grant can still need re-approving
> after a rebuild, and has been observed to degrade at runtime into a
> "trusted-but-no-access" state — `AXIsProcessTrusted()` returns true while the
> actual accessibility/screen queries return an empty stub (you'll see a blank
> screenshot and an empty tree). When that happens, re-grant: System Settings →
> Privacy & Security → Accessibility (and Screen Recording), remove the
> `aidaemon` entry, re-add `~/Applications/aidaemon.app`, toggle it on, and
> restart the daemon. **The durable fix is a real Apple Developer ID
> certificate + notarization** (instead of the self-signed `aidaemon-dev`),
> which makes the grant stable across rebuilds and avoids the runtime
> degradation; sign releases with it for end users.

> One-time manual grant: unavoidable (Apple's rule).
> Re-granting after a rebuild: usually avoided with the signed bundle, but a
> self-signed dev identity may still require it (and a Developer ID fixes it).

---

## Quick start

```bash
# 1. Build with the feature.
cargo build --release --features computer_use-macos

# 2. Create a stable local signing identity (one-time, ~10s).
scripts/create-signing-identity.sh

# 3. Package as a signed .app bundle and install/start the launchd agent.
scripts/package-macos-app.sh --build      # --build does the release build for you

# 4. Enable the tool in config.toml:
#    [computer_use]
#    enabled = true

# 5. Grant permissions in System Settings → Privacy & Security:
#      • Accessibility    → enable "aidaemon"
#      • Screen Recording → enable "aidaemon"
#    The first computer_use action will prompt for / register each one.

# 6. Screen Recording only applies after a restart, so refresh once more:
scripts/package-macos-app.sh
```

After a rebuild, re-run `scripts/package-macos-app.sh`. The bundle id and signing
identity don't change, so the grant usually carries over — but on a self-signed
dev setup you may need to re-grant if access degrades (see the caveat above). A
Developer ID identity removes that uncertainty.

---

## How it fits together

- **`scripts/create-signing-identity.sh`** — creates a self-signed
  `aidaemon-dev` code-signing certificate in your login keychain and trusts it.
  One time. (For distributing notarized releases, an Apple Developer ID is the
  production alternative — same idea, stronger trust.)
- **`scripts/package-macos-app.sh`** — copies the built binary into
  `~/Applications/aidaemon.app`, writes its `Info.plist` (bundle id `ai.aidaemon`),
  signs it with `aidaemon-dev` (or ad-hoc if the identity is missing), writes the
  launchd plist pointing at the binary **inside** the bundle, and (re)loads the
  agent. Idempotent — run it after every rebuild.
- **`aidaemon install-service`** — the built-in installer does the same bundle +
  sign + plist steps in one shot (handy for a Homebrew / release install). It
  also prefers the `aidaemon-dev` identity if present.
- **Keep-awake** — the launchd plist launches the binary *directly* (not wrapped
  in `caffeinate`, which would blur the code identity). The daemon instead spawns
  its own `caffeinate -i -w <pid>` at startup, so the Mac still won't idle-sleep
  while it runs.

## Configuration

```toml
[computer_use]
enabled = true
# Screenshot downscaling (smaller = cheaper/faster, larger = better click accuracy)
screenshot_max_width = 1280
screenshot_max_height = 800
screenshot_max_bytes = 2000000
# Accessibility-tree traversal limits
ax_max_depth = 12
ax_max_nodes = 500
action_timeout_secs = 10
# Budgets for a single GUI task
max_mutating_actions = 15
max_consecutive_observations = 3
max_wall_clock_secs = 600
# Exact bundle ids allowed to be inspected without a per-session prompt (empty by
# default — screenshots can expose private content).
always_allowed_apps = []
# Also send each screenshot to the chat channel (noisy; off by default).
mirror_screenshots_to_channel = false
```

`computer_use` also requires a **vision-capable model** that supports tool
calling over the OpenAI-compatible wire format. The agent pins such a model for
the duration of a GUI task. If no model in your chain qualifies, the loop aborts
at start with an actionable error rather than silently dropping screenshots.

## Safety model

Permission is layered and enforced in code, not just prompts:

1. **OS permissions** — Accessibility + Screen Recording (above).
2. **Per-session approval** — enabling computer use for a chat session.
3. **Per-app approval** — separate *inspect* vs *control* scopes, keyed to the
   exact bundle id.
4. **Point-of-action confirmation** — consequential actions (send, delete,
   purchase, publish…) require a one-use approval that can't be replayed.
5. **Hard blocks** — secure text fields, login/authorization windows, the
   Terminal, and aidaemon itself never receive input.

Actions are bound to the snapshot generation that produced them, so a stale
element index is rejected rather than clicking the wrong control after the UI
changed.

## Troubleshooting

- **"Accessibility permission is required" even though it's on** — the granted
  entry is for an older signature. Re-run `scripts/package-macos-app.sh` (re-signs
  with the stable identity); if you never created the identity, run
  `scripts/create-signing-identity.sh` first, then remove the stale "aidaemon"
  row in System Settings → Accessibility and let it be recreated.
- **"Screen Recording permission is required"** — grant it, then restart the
  daemon (`scripts/package-macos-app.sh` or
  `launchctl kickstart -k gui/$(id -u)/ai.aidaemon`); Screen Recording grants only
  take effect for a freshly started process.
- **"No capturable window" / "No running app matching X"** — the target app
  isn't open or isn't frontmost. Ask the agent to open/activate it first
  (`activate_app` is an explicit, approved step — `get_app_state` never steals
  focus).
- **Clicks abort with "Foreground target mismatch"** — the target app lost focus
  (you or another app grabbed it) mid-action. This is the safety guard working;
  re-issue the request without competing for the cursor.

## Operational expectations

GUI steps are slow (seconds each: vision prefill + a tool call) and best suited
to short flows (5–15 steps, one app at a time). Small local models can drive
simple UIs (Calculator, basic dialogs) but may stumble on dense professional
apps; stronger vision+tools models complete longer sequences more reliably with
no harness changes. Only the latest screenshot is sent to the model on each turn,
so context stays bounded across a multi-step task.
