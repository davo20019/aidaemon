//! The channel conformance kit against the in-memory `TestChannel`, with
//! and without the optional capabilities.

use std::sync::Arc;

use aidaemon_channels::conformance::{
    channel_checks, check_channel, ChannelHarness, TestChannelHarness,
};
use aidaemon_channels::testing::{self, TestChannel};
use aidaemon_channels::{
    ApprovalPrompt, Channel, ChannelCapabilities, ChannelError, ChannelEvent, Conversation,
    MessageRef, OutboundMessage,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_channel_with_every_capability_conforms() {
    let channel = Arc::new(
        TestChannel::new("test")
            .with_edits(true)
            .with_buttons(true)
            .with_typing(true)
            .with_files(true)
            .with_commands(true)
            .with_max_len(500),
    );
    let harness = TestChannelHarness::new(channel.clone());
    let report = check_channel(channel.clone(), &harness).await;
    report.assert_all_passed();
    assert!(report.notes.is_empty(), "{report}");
    // At least three parts for the 3x message, plus the edit target.
    assert!(channel.sent().len() > 3);
    assert_eq!(channel.run_count(), 2);
}

#[tokio::test]
async fn test_channel_without_optional_capabilities_conforms() {
    let channel = Arc::new(TestChannel::new("test-plain"));
    let harness = TestChannelHarness::new(channel.clone());
    let report = check_channel(channel, &harness).await;
    report.assert_all_passed();
    assert!(report.passed_check(channel_checks::BUTTON_PRESS_ROUND_TRIP));
    // Buttons (2), long edits, download limits and the menu have nothing
    // to test.
    assert_eq!(report.notes.len(), 5, "{report}");
}

/// A channel that lies about its capabilities and ignores shutdown: the
/// kit reports each problem instead of hanging or panicking.
struct Stubborn(TestChannel);

#[async_trait::async_trait]
impl Channel for Stubborn {
    fn name(&self) -> &str {
        ""
    }
    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            edit_messages: true,
            ..self.0.capabilities()
        }
    }
    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        _shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        self.0.run(events, CancellationToken::new()).await
    }
    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        self.0.send(conversation, message).await
    }
}

struct StubbornHarness(Arc<Stubborn>);

#[async_trait::async_trait]
impl ChannelHarness for StubbornHarness {
    fn direct_conversation(&self) -> Conversation {
        testing::direct("dm:synthetic-user-1")
    }
    async fn inject_direct_message(&self, text: &str) {
        let m = testing::message(self.direct_conversation(), "test:synthetic-user-1", text);
        self.0 .0.inject_message(m);
    }
    async fn inject_group_message(&self, text: &str, addressed: bool) {
        let mut m = testing::message(testing::group("group:acme"), "test:synthetic-user-1", text);
        m.addressed = addressed;
        self.0 .0.inject_message(m);
    }
    async fn press_button(&self, _prompt: &ApprovalPrompt, _choice_id: &str) {}
    async fn sent_messages(&self) -> Vec<String> {
        self.0
             .0
            .sent()
            .into_iter()
            .map(|m| m.message.text)
            .collect()
    }
}

#[tokio::test]
async fn misbehaving_channel_fails_with_every_reason() {
    let channel = Arc::new(Stubborn(TestChannel::new("stubborn")));
    let harness = StubbornHarness(channel.clone());
    let report = check_channel(channel, &harness).await;
    for check in [
        channel_checks::NAME_NON_EMPTY,
        channel_checks::EDIT_MATCHES_CAPABILITY,
        channel_checks::EDIT_TRUNCATES_LONG_TEXT,
        channel_checks::SHUTDOWN_STOPS_RUN,
    ] {
        assert!(report.failure(check).is_some(), "{check}: {report}");
    }
    // Everything else still ran and passed.
    assert!(report.passed_check(channel_checks::DIRECT_MESSAGE_FORWARDED));
    assert!(report.passed_check(channel_checks::RECEIVER_DROP_STOPS_RUN));
    assert_eq!(report.failed.len(), 4, "{report}");
}
