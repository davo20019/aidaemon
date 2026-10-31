# Extending the aidaemon SDK

Every capability in the SDK sits behind a public trait. You can add a
channel, a model provider, a tool, a policy, a memory backend, a sandbox,
run storage or telemetry **in your own crate**, without changing this
repository. The built-in implementations (Telegram, Anthropic, the SQLite
store, ...) use exactly the same traits, so they double as templates.

## Extension points

| To add | Implement | Crate | Template to copy |
|---|---|---|---|
| A chat channel | `Channel` | `aidaemon-channels` | `aidaemon-channel-telegram` |
| A model provider | `Provider` | `aidaemon-core` | `aidaemon-providers` (`OpenAiCompatible`) |
| A tool | `Tool` | `aidaemon-core` | `aidaemon-tools` (`ReadFile`) |
| A policy or guardrail | `Hook` | `aidaemon-core` | `aidaemon-policies` (`ScopeGuard`) |
| A memory store | `MemoryStore` | `aidaemon-memory` | `InMemoryMemoryStore` |
| An embedding model | `Embedder` | `aidaemon-memory` | `HashEmbedder`, `FastEmbedder` |
| Memory ranking | `Scorer` | `aidaemon-memory` | `HybridScorer` |
| A whole memory system | `Recall` | `aidaemon-policies` | `aidaemon_memory::MemoryRecall` |
| A sandbox | `ExecutionBackend` | `aidaemon-tools` | `BubblewrapBackend` |
| Run storage | `RunStore` | `aidaemon-core` | `aidaemon-store-sqlite` |
| Telemetry | `EventSink` | `aidaemon-core` | `aidaemon_policies::TracingSink` |

## A channel in ~50 lines of structure

A channel only moves messages. Access control, sessions, queueing,
streaming and approvals are handled by `ChannelRunner`, so every channel
behaves the same way.

```rust
use aidaemon_channels::{
    Channel, ChannelCapabilities, ChannelError, ChannelEvent, Conversation, MessageRef,
    OutboundMessage,
};
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct MatrixChannel { /* HTTP client, homeserver URL, token */ }

#[async_trait]
impl Channel for MatrixChannel {
    fn name(&self) -> &str { "matrix" }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities { edit_messages: true, max_message_len: 32_000, ..Default::default() }
    }

    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        // Sync loop: map each platform message to ChannelEvent::Message,
        // set `addressed` from structural mention data, reconnect on
        // transient errors, return Err(ChannelError::Auth(..)) on bad tokens,
        // and return Ok(()) when `shutdown` is cancelled.
        todo!()
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        // Convert Markdown, split to max_message_len, post, return the last ref.
        todo!()
    }

    // Optional: edit (live status), ask (approval buttons), typing.
}
```

Rules every channel follows:

- `addressed` comes from structured data (mention entities, reply targets,
  DMs), never from searching message text for the bot's name.
- `Sender::id` is the platform's stable user id, never a display name.
- `send` splits long messages so no chunk exceeds `max_message_len`.
- `edit` and `ask` return `ChannelError::Unsupported` when the matching
  capability is false. Without buttons, the runner falls back to
  `/approve <id>` and `/deny <id>` commands automatically.

## Other extension points in brief

- **Provider**: implement `complete`; implement `stream` too if the API
  streams (yield `StreamEvent::TextDelta`s, then exactly one
  `StreamEvent::Done`). Map HTTP errors to `ProviderError` (429 →
  `RateLimited`, 5xx → `Transient`, 401/403 → `Auth`). Retries belong in
  the `Retry` wrapper, not in your provider.
- **Tool**: `spec` (JSON schema with `additionalProperties: false`) and
  `call`. Declare honest `annotations` (read-only tools run in parallel;
  unannotated tools are treated as destructive) and an `access` manifest
  so scope policies and sandboxes can reason about the call. Return
  failures as `ToolOutput::error`, not panics.
- **Hook**: implement only the methods you need (`before_model`,
  `before_tool`, `after_tool`, `on_stop`, `on_stuck`). Base decisions on
  typed state in `HookContext`, `ToolInfo` and receipts.
- **Memory**: implement `MemoryStore` to plug in a database (override
  `search_facts`/`search_episodes` if it ranks natively), `Embedder` for an
  embedding API, or `Scorer` for different ranking. To replace the whole
  memory system, implement `aidaemon_policies::Recall` and use it with
  `MemoryInjection`.

## Proving compatibility: conformance kits

Each contract has a conformance kit that checks the behaviour the SDK
relies on and reports every failure at once. Run the kit for what you
implemented in your own test suite and mention the result in your PR or
README.

| Contract | Kit | Feature |
|---|---|---|
| `Tool` | `aidaemon_core::conformance::check_tool` | `aidaemon-core/testing` |
| `Provider` | `aidaemon_core::conformance::check_provider` | `aidaemon-core/testing` |
| `Channel` | `aidaemon_channels::conformance::check_channel` (you implement a small harness for your platform mock) | `aidaemon-channels/conformance` |
| `MemoryStore` | `aidaemon_memory::conformance::check_store` | `aidaemon-memory/conformance` |

Each built-in implementation in this repository passes its kit in CI.

```rust
// In your crate's tests:
aidaemon_core::conformance::check_tool(&MyTool::new()).assert_all_passed();
aidaemon_core::conformance::check_provider(&my_provider_against_mock).await.assert_all_passed();
aidaemon_channels::conformance::check_channel(channel, &my_platform_harness).await.assert_all_passed();
aidaemon_memory::conformance::check_store(|| async {
    std::sync::Arc::new(MyStore::new()) as std::sync::Arc<dyn aidaemon_memory::MemoryStore>
})
.await
.assert_all_passed();
```

The provider kit needs a deterministic backend (your mock server), and the
channel kit needs a small `ChannelHarness` that injects messages and button
presses into your platform mock; see the Telegram, Slack and Discord test
suites for examples. If a tool's schema alone cannot produce valid
arguments, use `check_tool_with(..., ToolKitOptions::default().valid_arguments(..))`.

## Naming and publishing your crate

- Use the prefix for the kind of extension: `aidaemon-channel-<platform>`,
  `aidaemon-provider-<name>`, `aidaemon-tools-<domain>`,
  `aidaemon-memory-<backend>`, `aidaemon-policy-<name>`.
- Depend on the narrowest crate that defines the trait (for a channel,
  `aidaemon-channels`, not `aidaemon-sdk`).
- State which SDK version range you support. The stable contracts
  (`aidaemon-core`, the `Channel` types, the memory traits) follow semver;
  other crates may change faster while the SDK is 0.x.

## Contributing it upstream

If your extension is broadly useful, it can live in this repository like
the built-in adapters. Open an issue first. Upstream extensions need tests
against local mocks (no network), a passing conformance kit, synthetic test
data only, and docs. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
