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
| A sandbox | `Sandbox` | `aidaemon-tools` | `BubblewrapSandbox` |
| Run storage | `RunStore` | `aidaemon-core` | `aidaemon-store-sqlite` |
| Telemetry | `EventSink` | `aidaemon-core` | `aidaemon_policies::TracingSink` |
| A decision backend | `Decider` | `aidaemon-decisions` | `ProviderDecider` |

## A channel in ~50 lines of structure

A channel only moves messages. Access control, sessions, queueing,
streaming and approvals are handled by `ChannelRunner`, so every channel
behaves the same way.

Build inbound values with their constructors (`InboundMessage::new`,
`Attachment::new`, `ApprovalAnswer::new`): those structs are `#[non_exhaustive]`
so a new field does not break your adapter. `edit` has one rule for text too
long for the platform: stay one message, cut at the end, finish with
`split::TRUNCATION_MARK`. Override `download_limited` when your platform
lets you stop reading part way through; the `http` feature has a bounded
downloader (`aidaemon_channels::download`) that keeps a token on your
platform's own hosts.

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
  unannotated tools are treated as destructive; set `egress` on a call
  that sends owner data off the machine, and declare the host it contacts
  as an `Egress` effect, which is how a network read is covered), an `access` manifest
  so scope policies and sandboxes can reason about the call, and
  `effects` (see below). Never prompt the owner or refuse on policy
  inside `call`; that is a hook's job. Return failures as
  `ToolOutput::error`, not panics.
- **Hook**: implement only the methods you need (`before_model`,
  `before_tool`, `after_tool`, `on_stop`, and the others on the `Hook`
  trait). Base decisions on typed state in `HookContext`, `ToolInfo` and
  receipts. Compile hooks in; never load them from a repository or config.
- **Memory**: implement `MemoryStore` to plug in a database (override
  `search_facts`/`search_episodes` if it ranks natively), `Embedder` for an
  embedding API, or `Scorer` for different ranking. To replace the whole
  memory system, implement `aidaemon_policies::Recall` and use it with
  `MemoryInjection`.

## The tool contract

A tool is described in two layers.

- **The kernel `Tool`** (`aidaemon-core`) is what every agent runs: `spec`,
  `call`, `annotations`, `access`, `effects`, `source_trust` and
  `timeout`. Its `ToolContext` carries `Extensions`, typed application
  values keyed by type that the application sets once with
  `AgentBuilder::tool_extensions`.
- **`aidaemon_tools::ToolSemantics`** is the richer description the
  daemon's dispatcher reads before it runs a call: capabilities, typed call
  semantics, access manifests, receipt kinds, durable replay, output trust,
  `dispatch_facts` (how the dispatcher treats the tool: always offered,
  same-tool allowance, file delivery, child spawning and so on, never keyed
  on its name) and `owner_approval_request`. Every method has a
  conservative default. The daemon's own tool trait adds only execution on
  top (`Tool: ToolSemantics`); a bridged tool reaches the kernel as a
  kernel `Tool`.

Facts about a call (who is calling, the session, the channel's
visibility, the task, the mandate fence, approval scope) reach a daemon
tool as a typed `ToolCallContext`, filled by the dispatcher. A tool never
reads them from its arguments: the dispatcher strips any `_`-prefixed key
a model sends, and the context is never serialized.

`owner_approval_request(call, facts)` is how a tool states the owner
approval a call needs: `Ask(request)`, `NotNeeded`, or `Unjudged` (the
policy's mode decides). The daemon's approval rules hold only the turn's
facts and ask the tool; they never match a tool by name or type.

A tool ends a run with a typed finish: `ToolOutput::finish(Finish)`,
with a `FinishOutcome` of `Succeeded`, `Partial` or `Blocked` and an
optional final reply. The run ends `RunStatus::Finished`; a `Succeeded`
finish in a run where the owner declined something is reported as
`Partial`. Nothing in a tool's output text ends a run.

Tools get their dependencies through constructors: store traits from
`aidaemon_tools::services` (facts, episodes, people, settings, OAuth) and
an `ExecutionBackend`, the workspace a tool's commands and files live in
(local, Docker, SSH). A `Sandbox` is narrower: the runner that confines
one command inside that workspace.

## Effects, safeguards and owner approvals

A tool's `effects(arguments)` returns `CallEffects`: the `EffectKey`s the
call touches and whether that list is `complete`. Build keys with
`EffectKey::egress(scheme, host, port)` for a host it contacts,
`EffectKey::path(EffectClass::Delete | Write | Configuration, abs_path)`
for a path, and `EffectKey::named(EffectClass::Credential | Process, name)`.
Set `complete: false` when the call may do more than it lists (an opaque
script). The default is no effects for a read-only call that is not
egress, and unknown otherwise.

`AgentBuilder::build` refuses an agent with tools unless hooks (or
`AgentBuilder::attest`) cover every `Safeguard::REQUIRED`: `Authority`,
`Scope`, `UntrustedOutput`, `Approval` and `Egress`. From
`aidaemon-policies`:

- `ApprovalPolicy` (`Approval`) applies hard blocks, deny rules, your
  `ApprovalRules`, allow rules and an `ApprovalMode`, and turns an ask
  into an `InterruptKind::OwnerApproval` interruption carrying the call's
  effects, unless your `ApprovalGrants` (or `SessionGrants`) cover it.
  After a decline in the run, a call with incomplete effects is asked
  again even where a rule or grant would allow it: an egress or
  destructive call after any decline, and any call that is not
  read-only after a declined delete, write, configuration or call. A
  tool whose only effect is the agent's own state declares
  `CallEffects::none()` and is never asked again.
- `EgressGuard` (`Egress`) refuses a call annotated `egress` or declaring
  an `Egress` effect whose arguments carry a secret, with
  `[EGRESS BLOCKED]` (`aidaemon_core::EGRESS_BLOCKED`).

The kernel enforces the decline invariant itself: when an owner approval
is resolved `Deny`, its effects (or its subject, if it declared none) are
remembered for the run, and any later call overlapping them is refused
with `[OWNER DECLINED]` before any hook runs, whichever tool makes it.
Declare effects honestly: a key you leave out cannot be bound.

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
