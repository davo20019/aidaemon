# aidaemon-macros

The `#[tool]` attribute macro for [aidaemon](https://aidaemon.ai) agents.
Implementation crate of the aidaemon SDK, no stability promise: use it
through `aidaemon-sdk` (feature `macros`, on by default), which re-exports it
as `aidaemon_sdk::tool`.

```rust
use aidaemon_sdk::tool;

/// Today's weather for a city.
#[tool(read_only)]
async fn forecast(city: String) -> String {
    format!("{city}: 21°C and sunny")
}
// `Forecast` now implements `aidaemon_core::Tool`; pass it to an agent with `.tool(Forecast)`.
```

The JSON schema is derived with `schemars`; doc comments become the tool and
parameter descriptions. Effects are explicit (`read_only`, `write`,
`destructive`); with none given a tool is treated as destructive, the same as
a hand-written one.
