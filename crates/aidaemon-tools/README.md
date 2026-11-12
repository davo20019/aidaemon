# aidaemon-tools

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Ready-made `aidaemon_core::Tool`s: workspace file tools, search, a sandboxed shell, web fetch and a checklist.

```rust,ignore
let workspace = aidaemon_tools::Workspace::new(".")?;
let mut builder = aidaemon_core::Agent::builder("helper"); // ...model, provider
for tool in aidaemon_tools::default_tools(workspace) {
    builder = builder.tool_arc(tool);
}
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-tools
