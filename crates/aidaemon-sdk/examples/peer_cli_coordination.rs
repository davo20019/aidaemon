//! Multi-CLI peer coordination on `aidaemon-sdk` — adaptable to any Rust
//! agentic framework with zero orchestrating LLM calls.
//!
//! Demonstrates how any host framework can:
//! 1. Implement [`PeerLauncher`] to plug in its own CLI process runner,
//!    sandboxing, and git worktrees.
//! 2. Start a [`LocalMcpServer`] + [`PeerMcpContext`] to mint per-CLI scoped
//!    MCP endpoints (with ephemeral 256-bit tokens).
//! 3. Let one CLI delegate to another CLI, share a coordination board, and
//!    resume finished peers over standard MCP JSON-RPC 2.0.

use std::sync::{Arc, Mutex};

use aidaemon_sdk::coding_agents::peer::{
    PeerAgentInfo, PeerLaunchRequest, PeerLaunchResult, PeerLauncher, PeerLimits, PeerMcpContext,
    PeerMcpHandler, PeerRunStatus, PeerState, PeerTree,
};
use aidaemon_sdk::mcp::{LocalMcpServer, ServerMeta, run_io_bridge};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Minimal custom host launcher — replace with your own framework's process
/// runner, container manager, or worktree orchestrator.
struct CustomFrameworkLauncher {
    log: Mutex<Vec<String>>,
}

#[async_trait]
impl PeerLauncher for CustomFrameworkLauncher {
    async fn available_agents(&self) -> Vec<PeerAgentInfo> {
        vec![
            PeerAgentInfo {
                name: "claude".into(),
                description: "Claude Code".into(),
            },
            PeerAgentInfo {
                name: "codex".into(),
                description: "Codex CLI".into(),
            },
            PeerAgentInfo {
                name: "agy".into(),
                description: "Antigravity CLI".into(),
            },
        ]
    }

    async fn launch(&self, request: PeerLaunchRequest) -> Result<PeerLaunchResult, String> {
        self.log.lock().unwrap().push(format!(
            "{} delegated `{}` to {} ({:?})",
            request.parent_peer_id, request.title, request.agent, request.workspace
        ));
        Ok(PeerLaunchResult {
            run_id: format!("custom-run-{}", request.peer_id),
            working_dir: Some(format!("/synthetic/worktrees/{}", request.peer_id)),
            branch: Some(format!("peer/{}", request.peer_id)),
            finished: request.wait,
            success: request.wait.then_some(true),
            output: request
                .wait
                .then(|| format!("{} finished: {}", request.agent, request.title)),
            resumable: true,
        })
    }

    async fn check(&self, run_id: &str, _wait: bool) -> Result<PeerRunStatus, String> {
        Ok(PeerRunStatus {
            run_id: run_id.to_string(),
            state: PeerState::Completed,
            output: Some("ready".into()),
            working_dir: None,
            branch: None,
            resumable: true,
            out_of_usage: None,
        })
    }

    async fn cancel(&self, run_id: &str) -> Result<String, String> {
        Ok(format!("cancelled {run_id}"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Bind one local MCP server for the host process.
    let server = LocalMcpServer::bind(ServerMeta::new("my-framework-peers", "0.1.0")).await?;
    let mcp_ctx = PeerMcpContext::new(server, "my-framework", vec!["mcp-bridge".into()]);

    // 2. Create a coordination tree for a top-level CLI run.
    let launcher = Arc::new(CustomFrameworkLauncher {
        log: Mutex::new(Vec::new()),
    });
    let tree = PeerTree::new(
        "claude",
        "Feature X",
        Some("/synthetic/repo".into()),
        PeerLimits::default(),
    );
    let root_handler = PeerMcpHandler::new(tree, "peer-root", launcher.clone(), mcp_ctx.clone());
    let (endpoint, scoped) = mcp_ctx.endpoint_for(root_handler);

    // 3. Simulate a CLI agent connecting through the stdio MCP bridge.
    let (mut cli_w, bridge_r) = tokio::io::duplex(16 * 1024);
    let (bridge_w, cli_r) = tokio::io::duplex(16 * 1024);
    let addr = scoped.addr().to_string();
    let token = scoped.token().to_string();
    let bridge =
        tokio::spawn(async move { run_io_bridge(&addr, &token, bridge_r, bridge_w).await });
    let mut responses = BufReader::new(cli_r).lines();

    let send = |id: u64, method: &str, params: Value| {
        format!(
            "{}\n",
            json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
        )
    };

    cli_w
        .write_all(send(1, "initialize", json!({ "protocolVersion": "2025-06-18" })).as_bytes())
        .await?;
    let _ = responses.next_line().await?;

    cli_w
        .write_all(
            send(
                2,
                "tools/call",
                json!({
                    "name": "post_board",
                    "arguments": { "topic": "contract", "message": "GET /health -> 200 OK" }
                }),
            )
            .as_bytes(),
        )
        .await?;
    let _ = responses.next_line().await?;

    cli_w
        .write_all(
            send(
                3,
                "tools/call",
                json!({
                    "name": "delegate_cli",
                    "arguments": {
                        "agent": "codex",
                        "title": "Implement /health",
                        "prompt": "Add GET /health endpoint matching the board contract",
                        "workspace": "worktree",
                        "wait": true
                    }
                }),
            )
            .as_bytes(),
        )
        .await?;
    let reply = responses.next_line().await?.unwrap_or_default();
    println!("Injected MCP server `{}` replied:\n{reply}", endpoint.name);
    println!("Host log: {:?}", launcher.log.lock().unwrap());

    drop(cli_w);
    bridge.await??;
    Ok(())
}
