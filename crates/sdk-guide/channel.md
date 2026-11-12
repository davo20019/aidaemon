# A channel

A channel connects a chat platform to an agent. The contract is the `Channel`
trait in `aidaemon-channels` (SDK feature `channels`). Built-in adapters exist
for Telegram, Slack and Discord (features `telegram`, `slack`, `discord`).

Use a built-in adapter:

```rust,ignore
use std::sync::Arc;
use aidaemon_channels::ChannelRunner;
use aidaemon_channel_telegram::TelegramChannel;
use tokio_util::sync::CancellationToken;

let channel = TelegramChannel::new(bot_token);
let runner = ChannelRunner::builder(agent)
    .channel(Arc::new(channel))
    .build()?;
runner.run(CancellationToken::new()).await; // until cancelled
```

The runner gives you one session per conversation, one run at a time per
session, streamed progress, and approvals as buttons that resume interrupted
runs (see "A hook").

Write your own by implementing four required methods:

```rust,ignore
use aidaemon_channels::{
    Channel, ChannelCapabilities, ChannelError, ChannelEvent, Conversation,
    MessageRef, OutboundMessage,
};
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

struct MyChat { /* client handles */ }

#[async_trait]
impl Channel for MyChat {
    fn name(&self) -> &str { "mychat" }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities { max_message_len: 4000, ..Default::default() }
    }

    // Forward platform updates as `ChannelEvent::Message(InboundMessage)`
    // until `shutdown` is cancelled.
    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        todo!()
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        todo!()
    }
}
```

`edit`, `ask` (buttons), `typing`, files and commands are optional; declare
what you support in `ChannelCapabilities` and the runner adapts. Enable the
crate's `conformance` feature to run the adapter conformance kit
(`aidaemon_channels::conformance`) against your channel.
