//! What one request's tree of sub-agents knows about each other lives in
//! `aidaemon-runtime`; the old paths stay.

#[cfg(test)]
pub(crate) use aidaemon_runtime::coordination::*;

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_runtime::host::TurnHost as _;
    use aidaemon_tools::semantics::FileFence;

    async fn fence(
        c: &Coordination,
        member: &str,
        call: &str,
    ) -> (CallFence, crate::testing::TestHarness) {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        (
            CallFence {
                coordination: c.clone(),
                member: member.to_string(),
                call_id: call.to_string(),
                backend: crate::execution::active_execution_backend(),
                escalator: Escalator::new(
                    harness.agent.side_calls().clone(),
                    harness.state.clone(),
                    harness.state.clone(),
                ),
                untrusted_seen: false,
            },
            harness,
        )
    }

    /// Two writers both passed the hook against the same content: the one
    /// that reaches the file second is told it changed.
    #[tokio::test]
    async fn the_second_of_two_racing_writers_is_stopped_at_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "base\n").unwrap();
        let raw = file.to_str().unwrap();
        let c = Coordination::default();
        let judged = file_state(Some(b"base\n"));
        c.note_judged("a", "call-1", judged, 0);
        c.note_judged("b", "call-2", judged, 0);

        let (first_fence, _a) = fence(&c, "a", "call-1").await;
        let (second_fence, _b) = fence(&c, "b", "call-2").await;
        let first = first_fence.begin_write(raw).await.unwrap();
        std::fs::write(&file, "from a\n").unwrap();
        first.committed().await;

        let second = second_fence.begin_write(raw).await;
        assert!(second.is_err());
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from a\n");
    }
}
