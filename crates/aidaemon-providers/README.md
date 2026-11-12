# aidaemon-providers

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Model providers implementing `aidaemon_core::Provider`: Anthropic, OpenAI-compatible endpoints (plus Google, xAI, ChatGPT), with streaming.

```rust,ignore
let provider = aidaemon_providers::Anthropic::builder().api_key("<key>").build()?;
let agent = aidaemon_core::Agent::builder("a").model("claude-sonnet-5-5").provider(provider)/* ... */;
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.2.0). MIT licensed. Docs: https://docs.rs/aidaemon-providers
