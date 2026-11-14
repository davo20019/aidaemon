# aidaemon-channel-slack

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Slack adapter for `aidaemon-channels`. Transport only; sessions, authorization and approvals live in the channel runner.

```rust,ignore
use aidaemon_channel_slack::SlackChannel;
let channel = std::sync::Arc::new(SlackChannel::new("xoxb-...", "xapp-..."));
// runner_builder.channel(channel)
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed. Docs: https://docs.rs/aidaemon-channel-slack
