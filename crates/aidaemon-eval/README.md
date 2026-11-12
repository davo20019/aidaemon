# aidaemon-eval

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

An eval kit: YAML scenarios (scripted or live), N-of-M pass rates, trajectory assertions and report comparison.

```rust,ignore
// `Suite::new(scenarios).run(&factory, n_runs)` returns a `Report` (Markdown or JSON);
// `factory` implements `AgentFactory` and builds the agent under test.
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed. Docs: https://docs.rs/aidaemon-eval
