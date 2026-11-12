# Publishing the SDK

Status: **decided** (PLAN.md, "Decisions (2026-11-11)", decisions 2 and 3).
Nothing has been published. Every SDK crate keeps `publish = false` until the
owner approves the first release; that flag is removed only then.

Publishing to crates.io is public and permanent: a version can be yanked but
never deleted, and a crate name cannot be released once taken.

## Shape

- The crates are published, not folded into the daemon: the daemon is built
  on them, and `cargo publish` of the daemon needs every path dependency on
  crates.io.
- **One shared version** for every SDK crate (`crates/aidaemon-*`), set in
  `[workspace.package]` of the root `Cargo.toml`, starting at `0.1.0` and
  separate from the daemon's version. Crates use `version.workspace = true`.
  Split the versions later only if outside users need it.
- `aidaemon-node-protocol` and `aidaemon-node-sdk` are already published and
  keep their own versions.
- **The daemon pins them exactly** (`=0.1.x`, alongside `path`), so a daemon
  release always builds against the crates it was tested with. Path
  dependencies between SDK crates carry `version = "0.1.0"` so
  `cargo package` works; dev-dependencies stay path-only.
- MSRV: `rust-version = "1.88"` in `[workspace.package]`, the floor of the
  SDK crates' dependency tree with all features (`image`, `ignore`); the CI
  `msrv` job checks it.

## Names

Checked on 2026-10-31 against the crates.io API: `aidaemon` and
`aidaemon-node-sdk` (and the node protocol) are ours already; all 17 SDK
names (`aidaemon-core`, `-sdk`, `-providers`, `-tools`, `-store-sqlite`,
`-mcp`, `-agents`, `-policies`, `-eval`, `-memory`, `-channels`,
`-channel-telegram`, `-channel-slack`, `-channel-discord`, `-events`,
`-work`, `-decisions`) were free (the last three were checked only in the
sense that nothing in them existed then; re-check all of them). Names can be
taken at any time: re-check right before the first release. No placeholder
0.0.x versions.

## Stability tiers

| Tier | Crates | Promise |
|---|---|---|
| Public contract | `aidaemon-core`, `aidaemon-sdk`, the `channel` module of `aidaemon-channels`, the traits in `aidaemon-memory` | semver; breaking changes only in a new minor version while 0.x, with a migration note; checked by `cargo-semver-checks` in CI |
| Implementation | providers, tools, store-sqlite, mcp, agents, policies, eval, channel adapters, the rest of channels and memory | no promise while 0.x; description says so |
| Internal | `aidaemon-events`, `aidaemon-work`, `aidaemon-decisions` | published only because the daemon depends on them; description says "internal to aidaemon, no stability promise" |

## Before the first release (checklist)

- [x] `[workspace.package]`, per-crate `readme`, `keywords`, `categories`,
      `documentation`, descriptions.
- [x] A `README.md` per crate; `crates/CHANGELOG.md`.
- [x] Getting-started guide source in `crates/sdk-guide/` (to be moved to the
      docs.aidaemon.ai site, which is built outside this repository).
- [x] CI: `cargo doc` per crate with `-D warnings`, MSRV, `cargo-semver-checks`.
- [x] `cargo package --list` reviewed for every crate (see below).
- [x] `release.yml` publishes the SDK crates in order, then the daemon.
- [ ] Re-check the names are free; the owner approves; remove
      `publish = false` from all SDK crates in one commit.
- [ ] Settle Phase 9 steps 0 and 1 first (PLAN.md, Milestone 2), so the
      contracts are proven by the daemon before they become permanent.
- [ ] The outsider check from the published crates.

## Release order

Dependency order, computed from the non-dev dependency edges:

1. `aidaemon-core`
2. `aidaemon-agents`, `-channels`, `-decisions`, `-eval`, `-mcp`,
   `-policies`, `-providers`, `-store-sqlite`, `-tools`
3. `aidaemon-channel-discord`, `-channel-slack`, `-channel-telegram`,
   `-events`, `-memory`
4. `aidaemon-work`
5. `aidaemon-sdk`
6. `aidaemon` (the daemon)

`release.yml` holds the same list (`SDK_CRATES`) with a comment on how to
regenerate it.

## Automation

`.github/workflows/release.yml` runs on the daemon's `v*` tag (no separate
tag scheme). After the binaries and the GitHub release, the
`publish-crates` job:

1. publishes the node crates as before;
2. stops with a notice while any SDK crate still has `publish = false`
   (GitHub binaries and Homebrew are unaffected);
3. otherwise checks that the daemon's exact pin matches the workspace
   version, publishes each SDK crate in order, skipping versions already on
   crates.io, waits for crates.io to report each, then publishes the daemon.

`ci.yml` has `sdk-docs`, `msrv` and `semver` jobs. `semver` skips a crate
that is not on crates.io yet (no baseline) and starts checking on its own
once 0.1.0 is out.

## Changelog

`crates/CHANGELOG.md` covers all SDK crates (Keep a Changelog), one section
per shared release, listing changes per crate or tier.
