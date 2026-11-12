# Install

Requires Rust 1.88 or newer (edition 2021).

Once 0.1.0 is on crates.io:

```toml
[dependencies]
aidaemon-sdk = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
async-trait = "0.1"
```

Before that, depend on the repository:

```toml
aidaemon-sdk = { git = "https://github.com/davo20019/aidaemon" }
```

`aidaemon-sdk` re-exports the kernel (`aidaemon-core`) and adds modules behind
features of the same name:

| Feature | Module | Adds |
|---|---|---|
| `providers` (default) | `providers` | Anthropic and OpenAI-compatible models, streaming |
| `tools` (default) | `tools` | workspace file tools, search, sandboxed shell, web fetch |
| `policies` (default) | `policies` | approvals, scope guard, redaction, memory injection, tracing |
| `agents` (default) | `agents` | agents as tools |
| `sqlite` (default) | `store` | durable runs on SQLite |
| `mcp` | `mcp` | MCP servers as tools |
| `memory` | `memory` | long-term memory |
| `channels`, `telegram`, `slack`, `discord` | `channels` | chat channels |
| `eval` | `eval` | YAML scenario evals |
| `testing` | `testing` | a scripted provider for tests |

Take only the kernel with `aidaemon-core` if you want nothing else.

Provider keys are passed to builders (`providers::Anthropic::builder().api_key(..)`);
read them from your environment yourself.
