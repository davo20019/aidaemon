# aidaemon-sdk

**Tier: public contract.** Semver-checked in CI; breaking changes only in a new minor version while 0.x, with a migration note.

One dependency for building agents in Rust. It re-exports `aidaemon-core` and adds optional modules, one feature each: `providers`, `tools`, `policies`, `agents`, `sqlite` (all default), plus `mcp`, `eval`, `memory`, `channels`, `telegram`, `slack`, `discord`, `testing`.

```rust,ignore
use aidaemon_sdk::prelude::*;

let agent = Agent::builder("helper")
    .allow_unguarded()
    .model("claude-sonnet-5-5")
    .instructions("You are a helpful assistant.")
    .provider(providers::Anthropic::builder().api_key("<key>").build()?)
    .build()?;
let outcome = agent.run("cli:me", "Hello").await?;
println!("{}", outcome.final_text.unwrap_or_default());
```

See the `coding_agent` and `support_triage` examples.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-sdk
