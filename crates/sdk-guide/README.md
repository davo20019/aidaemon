# aidaemon SDK: build agents in Rust

A general agent SDK for Rust: a small loop you can read in an afternoon, tools
as plain trait impls, behaviour (policy, memory, tracing) as hooks, and runs
you can pause, persist and resume. It also ships the things production agents
need and rarely get right: approvals, a receipt for every tool call, and
handling of untrusted tool output. They are features you turn on, not a
framework you buy into.

Status: version 0.1.0, not yet published to crates.io. Until it is, use a git
dependency (see Install).

> These pages are the source for the guide on docs.aidaemon.ai (that site is
> built outside this repository). Every identifier below exists in the crates
> at this commit.

1. [Install](install.md)
2. [Quickstart](quickstart.md)
3. [A custom tool](custom-tool.md)
4. [A hook](hook.md)
5. [Swapping memory](memory.md)
6. [A channel](channel.md)
7. [Coming from the Claude Agent SDK, Vercel AI SDK or LangGraph](concepts.md)

Two tiers, so you know what to lean on: the public contract
(`aidaemon-core`, `aidaemon-sdk`, the `channel` module of `aidaemon-channels`,
the `aidaemon-memory` traits) follows semver and is checked in CI. Everything
else may change in any 0.x minor version.
