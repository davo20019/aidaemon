# aidaemon-core

**Tier: public contract.** Semver-checked in CI; breaking changes only in a new minor version while 0.x, with a migration note.

The kernel of the aidaemon agent SDK: a small agent loop driven by hooks. It owns the loop, tools, providers, typed outcomes, receipts for every tool call, limits, events and serializable run state that can be paused and resumed.

```rust,ignore
use aidaemon_core::testing::{spec, text, ScriptedProvider};
use aidaemon_core::{Agent, FnTool, ToolAnnotations, ToolOutput};

let echo = FnTool::new(
    spec("echo", "Echo text back", &["text"]),
    ToolAnnotations::READ_ONLY,
    |args| async move { Ok(ToolOutput::text(args["text"].as_str().unwrap_or_default())) },
);
let agent = Agent::builder("demo")
    .allow_unguarded()
    .model("scripted-model")
    .instructions("Be brief.")
    .provider(ScriptedProvider::new([text("hello")])) // `testing` feature; use a real provider in production
    .tool(echo)
    .build()?;
let outcome = agent.run("cli:me", "say hello").await?;
println!("{:?}: {}", outcome.status, outcome.final_text.unwrap_or_default());
```

Run the full tour (approvals, resume) with `cargo run -p aidaemon-core --example quickstart --features testing`.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-core
