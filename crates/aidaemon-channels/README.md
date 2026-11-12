# aidaemon-channels

**Tier: public contract.** Semver-checked in CI; breaking changes only in a new minor version while 0.x, with a migration note.

The `Channel` trait (the contract for chat platforms) and `ChannelRunner`, which turns channel events into agent runs: one session per conversation, one run at a time per session, and approvals as buttons that resume paused runs.

```rust,ignore
use std::sync::Arc;
use aidaemon_channels::ChannelRunner;
use tokio_util::sync::CancellationToken;

let runner = ChannelRunner::builder(agent)
    .channel(Arc::new(my_channel)) // anything implementing `Channel`
    .build()?;
runner.run(CancellationToken::new()).await;
```

Adapters: `aidaemon-channel-telegram`, `-slack`, `-discord`. Feature `conformance` exposes a test kit for adapter authors.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed. Docs: https://docs.rs/aidaemon-channels
