# aidaemon-runtime-core

**Tier: internal.** Internal to aidaemon, no stability promise. Published only because the daemon depends on it; do not build on it.

The core of aidaemon's turn runtime, everything but the built-in tools and the agent (those are `aidaemon-runtime`, which re-exports this crate's modules): model routing and the side-call gateway, provider error classification, context budgets, and the turn on top of the `aidaemon-core` kernel. Used by the aidaemon daemon.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.22.0). MIT licensed. Docs: https://docs.rs/aidaemon-runtime-core
