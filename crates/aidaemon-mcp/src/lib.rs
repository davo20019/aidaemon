//! # aidaemon-mcp
//!
//! A Model Context Protocol client (stdio transport, newline-delimited
//! JSON-RPC 2.0) that exposes a server's tools as [`aidaemon_core::Tool`]s.
//!
//! ```no_run
//! # async fn demo() -> Result<(), aidaemon_mcp::McpError> {
//! use aidaemon_mcp::McpServer;
//!
//! let server = McpServer::spawn(
//!     "fs",
//!     "npx",
//!     ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
//!     Vec::<(String, String)>::new(),
//! )
//! .await?;
//! let mut builder = aidaemon_core::Agent::builder("assistant").allow_unguarded();
//! for tool in server.tools() {
//!     builder = builder.tool_arc(tool); // named `fs__read_file`, ...
//! }
//! # Ok(()) }
//! ```
//!
//! Tool names are `{server}__{tool}`, sanitized to `[a-zA-Z0-9_-]` and cut
//! to 64 characters. MCP annotations map onto [`aidaemon_core::ToolAnnotations`]
//! with MCP's conservative defaults: a tool without annotations is treated
//! as mutating and destructive, and [`McpOptions::annotations`] replaces the
//! hints altogether for callers that do not trust them. A server that
//! crashes or times out never panics the agent; the call fails with a tool
//! error carrying the server's recent stderr.
//!
//! The child gets a scrubbed environment ([`SAFE_ENV_KEYS`] plus the `env`
//! you pass) and is killed when the [`McpServer`] and every tool taken from
//! it are dropped.

mod client;
mod tool;

pub use client::{
    McpError, McpOptions, McpServer, McpServerSpec, ServerInfo, StderrCallback,
    MAX_STDERR_LINE_BYTES, PROTOCOL_VERSION, SAFE_ENV_KEYS, SUPPORTED_PROTOCOL_VERSIONS,
};
pub use tool::{map_annotations, normalize_schema, render_result, tool_name, McpTool};
