//! `manage_mcp` lives in `aidaemon_mcp::registry::manage`; its tests on the
//! daemon's assembled store stay here.

pub use aidaemon_mcp::registry::manage::ManageMcpTool;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[tokio::test]
    async fn restart_asks_with_the_command_it_will_run() {
        use crate::traits::OwnerApprovalNeed;
        let db = tempfile::NamedTempFile::new().unwrap();
        let embeddings =
            std::sync::Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        let state = std::sync::Arc::new(
            crate::state::SqliteStateStore::new(db.path().to_str().unwrap(), 100, None, embeddings)
                .await
                .unwrap(),
        );
        let registry = crate::testing::fake_mcp_registry(state).await;
        let tool = ManageMcpTool::new(registry);

        let OwnerApprovalNeed::Ask(request) = tool.respawn_approval("restart", "fake").await else {
            panic!("restarting a running server must ask");
        };
        let subject: Value = serde_json::from_str(&request.subject).unwrap();
        assert_eq!(subject["action"], "restart");
        assert_eq!(subject["name"], "fake");
        assert!(subject["command"]
            .as_str()
            .unwrap()
            .contains("fake_mcp_server"));
        assert!(request.summary.starts_with("Restart MCP server 'fake'"));

        // Nothing would start: no card.
        assert!(matches!(
            tool.respawn_approval("restart", "missing").await,
            OwnerApprovalNeed::NotNeeded
        ));
        assert!(matches!(
            tool.respawn_approval("enable", "fake").await,
            OwnerApprovalNeed::NotNeeded
        ));
    }
}
