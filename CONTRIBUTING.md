# Contributing to aidaemon

Thanks for helping. This repository holds two things:

- **The daemon** (`src/`): the personal agent that runs on Telegram, Slack
  and Discord.
- **The SDK** (`crates/aidaemon-*`): the kernel and crates that let anyone
  build their own agents. Start with the SDK guide at
  <https://docs.aidaemon.ai/sdk>.

Most new functionality (a channel, a provider, a tool, a memory backend, a
policy) does not need to live in this repository at all. Every extension
point is a public trait, so you can publish your own crate. See the
SDK guide at <https://docs.aidaemon.ai/sdk>.

## Before you start

- Read [`CLAUDE.md`](CLAUDE.md) (also available as `AGENTS.md`). It is the
  shared guide for humans and coding agents: build commands, safety rules,
  and design rules such as "prefer structural fixes over phrase patches".
- For anything larger than a bug fix, open an issue first describing the
  problem and the approach, so we can agree on the shape before you build it.

## Ground rules

1. **Synthetic data only.** Never put real names, emails, phone numbers,
   chat or channel IDs, tokens, or other personal data in code, tests,
   fixtures or docs. Use placeholders such as `Alice Rivera`,
   `Acme Corp`, `telegram:synthetic-user-1`.
2. **Decisions come from typed state, not from matching words.** Do not fix
   agent behaviour by adding phrases to a list. Use receipts, statuses,
   annotations, access manifests and message origins.
3. **No network in tests.** Use local mock servers and the scripted
   providers in `aidaemon_core::testing`.
4. **Keep the kernel small.** If a feature cannot be built on the public
   hooks, propose a new hook point; do not add special cases to
   `aidaemon-core`.
5. **Measure behaviour changes.** Changes meant to make agents more capable
   or reliable should come with eval scenarios (`aidaemon-eval`) and pass
   rates over repeated runs, not a single run.

## Checks to run before opening a PR

```bash
cargo fmt
cargo clippy --all-features -- -D warnings          # the daemon
cargo test                                           # the daemon

# SDK crates (list them with `-p`, as CI does in .github/workflows/ci.yml)
cargo clippy -p aidaemon-core -p aidaemon-channels ... --all-targets --all-features -- -D warnings
cargo test   -p aidaemon-core -p aidaemon-channels ... --all-features
```

If you implement an SDK trait, also run its conformance kit (the crate
docs name it) and include the result in the PR description.

If a local test run aborts with "has overflowed its stack", run it with
`RUST_MIN_STACK=8388608`, as CI does.

## Pull requests

- One topic per PR, with a description of the problem, the change, and how
  you tested it.
- Add tests for new behaviour, including at least one negative case.
- Update the crate docs when you change a public API.
- Public SDK contracts (`aidaemon-core`, the `Channel` types, the memory
  traits) follow semver; breaking them needs a clear reason and a note in
  the PR.

## License

By contributing you agree that your contributions are licensed under the
MIT license, the same as the project.
