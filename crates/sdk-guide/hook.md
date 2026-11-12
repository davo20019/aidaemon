# A hook

Hooks are how behaviour is added to the loop: the kernel stays small and
policy, memory, logging and limits are all hooks. `Hook` has default no-op
methods; implement the ones you need: `on_run_start`, `edit_transcript`,
`before_model`, `after_model`, `before_tool`, `after_tool`, `on_stop`,
`on_run_end`, among others.

A hook that blocks tool calls named in a deny list:

```rust
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::{Hook, HookContext, ToolCall, ToolDecision, ToolInfo};
use async_trait::async_trait;

struct DenyTools(Vec<String>);

#[async_trait]
impl Hook for DenyTools {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        _tool: &ToolInfo,
    ) -> ToolDecision {
        if self.0.contains(&call.name) {
            // The reason goes back to the model as the tool result.
            ToolDecision::Deny { reason: format!("`{}` is not allowed here", call.name) }
        } else {
            ToolDecision::Allow
        }
    }
}

// builder.hook(DenyTools(vec!["delete_file".into()]))
```

`ToolDecision` also has `Rewrite { arguments }` (change the arguments) and
`Interrupt { .. }` (pause the run, for example to ask a person). An interrupted
run returns `RunStatus::Interrupted`; continue it, in this or another process,
with `agent.resume(outcome.state, Resolution::Approve)` (or
`Resolution::Deny { reason }`). The state is serde-serializable, so it can sit
in a database until someone answers; `examples/quickstart.rs` in
`aidaemon-core` shows the full round trip.

Ready-made hooks live in `aidaemon_sdk::policies` (feature `policies`):
`ApprovalPolicy` with `ApprovalMode`, `ScopeGuard`, `EgressGuard`,
`UntrustedOutput`, `MemoryInjection` and more. See the `coding_agent` example
in `aidaemon-sdk`.
