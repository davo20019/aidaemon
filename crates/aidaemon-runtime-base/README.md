# aidaemon-runtime-base

**Tier: internal.** Internal to aidaemon, no stability promise. Published only because the daemon depends on it; do not build on it.

The lower half of aidaemon's turn runtime, with no reference to the turn or the engine (those are `aidaemon-runtime-core`, which re-exports this crate's modules): the model runtime and side calls, the host traits a turn runs on, tool authority, sub-agent coordination and runs, the untrusted-content rule, memory, history and the SQLite state store. Used by the aidaemon daemon.
