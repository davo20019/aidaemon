# aidaemon-store-sqlite

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

A SQLite `RunStore`: an append-only event journal plus the latest run-state snapshot, so runs survive restarts and paused runs can resume.

```rust,ignore
let store = std::sync::Arc::new(aidaemon_store_sqlite::SqliteStore::open("runs.db").await?);
for state in store.in_flight_runs().await? { /* agent.run_from(state, token) */ }
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-store-sqlite
