# A custom tool

For one-off tools, `FnTool` (see the quickstart) is enough. For anything with
state or configuration, implement the `Tool` trait. Only `spec` and `call` are
required.

<!-- TODO(dx): switch to #[tool] -->

```rust
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::{ToolContext, ToolError, ToolResult, ToolSpec};
use async_trait::async_trait;
use serde_json::{json, Value};

struct OrderStatus {
    api_base: String,
}

#[async_trait]
impl Tool for OrderStatus {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "order_status".into(),
            description: "Look up the status of an order by id".into(),
            // JSON Schema for the arguments object.
            parameters: json!({
                "type": "object",
                "properties": { "order_id": { "type": "string" } },
                "required": ["order_id"]
            }),
        }
    }

    // Tell the kernel what this tool does. It drives parallelism, approvals
    // and receipts. READ_ONLY tools may run concurrently.
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let id = arguments["order_id"].as_str().ok_or("order_id must be a string")?;
        // `ctx.cancel` is cancelled when the run is; long tools should watch it.
        let _ = &ctx.cancel;
        Ok(ToolOutput::text(format!("order {id} at {}: shipped", self.api_base)))
    }
}
```

Register it with `.tool(OrderStatus { .. })`, or `.tool_arc(Arc<dyn Tool>)`.

Things worth knowing:

- Errors: return `Err(ToolError)` (a string wrapper; `&str` and `String`
  convert). The model sees the message and the receipt is marked failed.
- `Tool::validate` runs before any policy or I/O; the default checks the
  schema's `required` keys.
- Output from tools is treated as untrusted by default (`Tool::source_trust`),
  so policies can screen it. Return `SourceTrust::Trusted` only when the output
  is your own data.
- `ToolAnnotations::{READ_ONLY, WRITE, DESTRUCTIVE}` are presets;
  `annotations_for(&arguments)` can relax them per call.
- A tool that touches files should declare them (`Tool::access`, or
  `access = ..` on `#[tool]`) so the scope guard can check them. Under the
  preset, a tool that declares nothing is refused when a workspace is set
  and, without one, asked about before it runs if it may change things. A
  tool that touches no files (an order lookup, a refund, a ticket) says so
  with `preset(..).path_free_tool("name")`.
- Test tools without a model using the `testing` feature and the tool
  conformance kit in `aidaemon_core::conformance`.
