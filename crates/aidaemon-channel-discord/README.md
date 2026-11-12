# aidaemon-channel-discord

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Discord adapter for `aidaemon-channels`. Transport only; sessions, authorization and approvals live in the channel runner.

```rust,ignore
use aidaemon_channel_discord::DiscordChannel;
let channel = std::sync::Arc::new(DiscordChannel::new("bot-token"));
// runner_builder.channel(channel)
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-channel-discord
