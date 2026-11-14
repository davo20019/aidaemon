# aidaemon-sdk

**Tier: public contract.** Semver-checked in CI; breaking changes only in a new minor version while 0.x, with a migration note.

One dependency for building agents in Rust. It re-exports `aidaemon-core` and adds optional modules, one feature each: `providers`, `tools`, `policies`, `agents`, `sqlite`, `macros`, `memory` (all default), plus `mcp`, `eval`, `channels`, `telegram`, `slack`, `discord`, `conformance`, `testing`.

```rust,ignore
use aidaemon_sdk::prelude::*;

/// Today's weather for a city.
#[tool(read_only)]
async fn forecast(city: String) -> String {
    format!("{city}: 21°C and sunny")
}

let agent = preset("weather")
    .provider_from_env() // ANTHROPIC_API_KEY or OPENAI_API_KEY
    .instructions("Answer in one sentence.")
    .tool(Forecast)
    .build()
    .await?;
let outcome = agent.run("cli:me", "Weather in Lisbon?").await?;
println!("{}", outcome.final_text.unwrap_or_default());
```

`preset` turns on the safeguards and a run store; the model is offered only the tools you name. In tests, `testing::ScriptedProvider` stands in for the model. Background work runs on the kernel's `ServiceHost`. See the examples (`quickstart`, `sub_agent`, `scheduled_worker`, `own_channel`, `coding_agent`).

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon); every SDK crate shares the workspace version. Not published to crates.io: depend on the repository. MIT licensed.
