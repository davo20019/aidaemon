# Publishing the SDK (draft plan)

Status: **draft for the owner's decision**. Nothing has been published.
Every SDK crate has `publish = false` until this plan is approved.

Publishing to crates.io is public and permanent: a version can be yanked
but never deleted, and a crate name cannot be released once taken. Each
step below that publishes is the owner's call.

## Names

Checked on 2026-10-31 against the crates.io API:

| Name | Status |
|---|---|
| `aidaemon` | taken (the existing daemon release) |
| `aidaemon-node-sdk` | taken (existing) |
| All 14 SDK crate names (`aidaemon-core`, `-sdk`, `-providers`, `-tools`, `-store-sqlite`, `-mcp`, `-agents`, `-policies`, `-eval`, `-memory`, `-channels`, `-channel-telegram`, `-channel-slack`, `-channel-discord`) | free |

Names can be taken by anyone at any time; re-check right before the first
release.

Option to consider: publish placeholder 0.0.x versions early only to
reserve the names. That is still a public, permanent action.

## Stability tiers

| Tier | Crates | Promise |
|---|---|---|
| Stable contracts | `aidaemon-core`, `aidaemon-channels` (the `channel` module), the traits in `aidaemon-memory` | semver; breaking changes only in a new minor version while 0.x, with a migration note |
| Implementations | providers, tools, store-sqlite, mcp, agents, policies, eval, memory implementations, channel runner and adapters | may change in any minor version while 0.x |
| Bundle | `aidaemon-sdk` | re-exports; its version moves with the stable tier |

Third-party crates depend on the stable tier, so it should change rarely.

## Versioning

- Start every SDK crate at `0.1.0`. Leave 1.0 for after the daemon itself
  runs on the SDK (PLAN.md, Phase 7) and the contracts have held for a few
  releases.
- Version crates independently. `aidaemon-sdk` pins compatible minor
  ranges of the others.
- Add `cargo-semver-checks` to CI for the stable tier before the first
  release, so accidental breaking changes fail the build.
- Declare an MSRV (`rust-version`) in each crate and test it in CI.

## Before the first release

1. Fill in package metadata for every crate: `readme`, `keywords`,
   `categories`, `documentation`, `rust-version`.
2. Add a short `README.md` per crate (crates.io shows it).
3. Check `cargo package --list -p <crate>` for each crate: no test
   fixtures with secrets, no large files, nothing untracked.
4. Make sure docs build: `cargo doc -p <crate> --no-deps --all-features`.
5. Decide the order of the daemon migration (Phase 7) relative to the
   first release. Publishing first lets outside users start, but the
   contracts are less proven.

## Release order

Crates must be published in dependency order:

1. `aidaemon-core`
2. `aidaemon-channels`, `aidaemon-providers`, `aidaemon-tools`,
   `aidaemon-store-sqlite`, `aidaemon-mcp`, `aidaemon-agents`,
   `aidaemon-policies`, `aidaemon-eval`
3. `aidaemon-memory` (depends on policies), `aidaemon-channel-telegram`,
   `aidaemon-channel-slack`, `aidaemon-channel-discord`
4. `aidaemon-sdk`

## Automation

The existing `.github/workflows/release.yml` already publishes the daemon
with `cargo publish`. Extend it (or add a separate `sdk-release.yml`) to:

- trigger on tags like `sdk-v0.1.0`,
- run the SDK checks (fmt, clippy, tests, semver checks),
- publish in the order above, skipping crates whose version is already on
  crates.io,
- create a GitHub release with the changelog section.

Keep SDK and daemon tags separate so a daemon release never publishes SDK
crates by accident.

## Changelog

Add a `crates/CHANGELOG.md` (Keep a Changelog format, like the root one)
covering all SDK crates, with one section per release listing changes per
crate.
