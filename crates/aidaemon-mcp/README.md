# aidaemon-mcp

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

A Model Context Protocol client (stdio) that exposes a server's tools as `aidaemon_core::Tool`s.

```rust,ignore
use aidaemon_mcp::McpServer;
let server = McpServer::spawn("fs", "npx",
    ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    Vec::<(String, String)>::new()).await?;
for tool in server.tools() { builder = builder.tool_arc(tool); }
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-mcp
