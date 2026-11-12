# Quickstart

An agent with one tool, runnable with no API key thanks to the scripted
provider (`aidaemon-sdk` with the `testing` feature):

```rust
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::FnTool;
use aidaemon_sdk::testing::{call, spec, text, tool_calls, ScriptedProvider};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The "model": ask for the tool once, then answer.
    let model = ScriptedProvider::new([
        tool_calls([call("c1", "shout", json!({"text": "hello"}))]),
        text("It said HELLO."),
    ]);

    let shout = FnTool::new(
        spec("shout", "Upper-case some text", &["text"]),
        ToolAnnotations::READ_ONLY,
        |args| async move {
            let t = args["text"].as_str().unwrap_or_default().to_uppercase();
            Ok(ToolOutput::text(t))
        },
    );

    let agent = Agent::builder("quickstart")
        .allow_unguarded() // no hooks yet; see "A hook"
        .model("scripted-model")
        .instructions("Use tools when useful.")
        .provider(model)
        .tool(shout)
        .build()?;

    let outcome = agent.run("cli:me", "Shout hello").await?;
    println!("{:?}: {}", outcome.status, outcome.final_text.unwrap_or_default());
    Ok(())
}
```

Swap in a real model by replacing `.provider(..)`:

```rust,ignore
.provider(providers::Anthropic::builder().api_key(key).build()?)
// or providers::OpenAiCompatible::builder().base_url(url).api_key(key).build()?
```

What you get back: `outcome.status` (`RunStatus::Completed`, `Interrupted`,
`LimitReached`, `Stopped`, `Stuck`, ...), `outcome.final_text`, a receipt per
tool call (`outcome.receipts()`), usage (`outcome.usage()`) and
`outcome.state`, a serializable `RunState`.

`Agent::run(session_id, message)` takes a session id you choose; the same id
continues the same conversation when a store is attached with
`.store(Arc<dyn RunStore>)` (`aidaemon-store-sqlite` provides one).

The runnable version of a longer tour, with approvals and resume, is
`cargo run -p aidaemon-core --example quickstart --features testing`.

<!-- TODO(dx): a builder preset (one call for model + provider + common hooks) is being built; document it here when it lands. -->
