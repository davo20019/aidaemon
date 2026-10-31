//! Channel conformance kit (feature `conformance`).
//!
//! Proves that a [`Channel`] adapter behaves the way the runner expects:
//! inbound events are mapped and forwarded, long messages are split within
//! the advertised limit, optional capabilities are honest, approvals round
//! trip, and `run` stops promptly.
//!
//! Adapter authors implement [`ChannelHarness`] over their platform mock
//! (the same mock server their own tests use) and call [`check_channel`]:
//!
//! ```
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! use std::sync::Arc;
//! use aidaemon_channels::conformance::{check_channel, TestChannelHarness};
//! use aidaemon_channels::testing::TestChannel;
//!
//! let channel = Arc::new(TestChannel::new("test").with_buttons(true));
//! let harness = TestChannelHarness::new(channel.clone());
//! check_channel(channel, &harness).await.assert_all_passed();
//! # }
//! ```
//!
//! The kit runs every check and returns an
//! [`aidaemon_core::conformance::ConformanceReport`]; check names are the
//! constants in [`channel_checks`], each documenting its rule.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::conformance::ConformanceReport;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::channel::{
    ApprovalAnswer, ApprovalChoice, ApprovalPrompt, Channel, ChannelError, ChannelEvent,
    Conversation, ConversationKind, InboundMessage, OutboundMessage,
};
use crate::testing::{sender, TestChannel};

/// How long the kit waits for an injected event to arrive, and for `run` to
/// return after shutdown or after the receiver is dropped.
pub const EVENT_TIMEOUT: Duration = Duration::from_secs(5);

/// Stable names of the checks run by [`check_channel`].
pub mod channel_checks {
    /// `name()` is not empty (it prefixes every session id).
    pub const NAME_NON_EMPTY: &str = "channel.name_non_empty";
    /// `capabilities().max_message_len > 0`.
    pub const MAX_MESSAGE_LEN_POSITIVE: &str = "channel.max_message_len_positive";
    /// `run` forwards an injected direct message as `ChannelEvent::Message`
    /// with `ConversationKind::Direct`, `addressed == true`, a non-empty
    /// sender id and a non-empty message id.
    pub const DIRECT_MESSAGE_FORWARDED: &str = "channel.direct_message_forwarded";
    /// Two direct messages from the same platform user carry the same
    /// sender id and conversation id, and distinct message ids.
    pub const SENDER_ID_STABLE: &str = "channel.sender_id_stable";
    /// A group message that does not address the bot arrives with
    /// `addressed == false` and a non-direct conversation kind.
    pub const UNADDRESSED_GROUP_MESSAGE: &str = "channel.unaddressed_group_message";
    /// A group message that addresses the bot (mention, reply or command)
    /// arrives with `addressed == true` and a non-direct conversation kind.
    pub const ADDRESSED_GROUP_MESSAGE: &str = "channel.addressed_group_message";
    /// `send` of a message three times `max_message_len` long produces only
    /// platform messages of at most `max_message_len` characters, and
    /// preserves all non-whitespace content in order.
    pub const SEND_SPLITS_LONG_MESSAGES: &str = "channel.send_splits_long_messages";
    /// `edit` returns `ChannelError::Unsupported` exactly when
    /// `capabilities().edit_messages` is false; otherwise it succeeds.
    pub const EDIT_MATCHES_CAPABILITY: &str = "channel.edit_matches_capability";
    /// `ask` returns `ChannelError::Unsupported` exactly when
    /// `capabilities().buttons` is false; otherwise it succeeds.
    pub const ASK_MATCHES_CAPABILITY: &str = "channel.ask_matches_capability";
    /// With buttons, pressing a button on an `ask` prompt yields
    /// `ChannelEvent::Approval` with the same prompt id and choice id.
    /// Vacuous (passes with a note) without buttons.
    pub const BUTTON_PRESS_ROUND_TRIP: &str = "channel.button_press_round_trip";
    /// `run` returns `Ok(())` within [`super::EVENT_TIMEOUT`] after its
    /// shutdown token is cancelled.
    pub const SHUTDOWN_STOPS_RUN: &str = "channel.shutdown_stops_run";
    /// `run` returns `Ok(())` within [`super::EVENT_TIMEOUT`] once the event
    /// receiver is dropped and another platform event arrives (adapters may
    /// notice earlier, but must notice by the next forward).
    pub const RECEIVER_DROP_STOPS_RUN: &str = "channel.receiver_drop_stops_run";
}

/// Drives the platform side of an adapter under test: injects inbound
/// updates into the platform mock and reports what the adapter sent.
///
/// Every inject method should return once the update is queued or delivered
/// by the mock; the kit itself waits up to [`EVENT_TIMEOUT`] for the
/// resulting event. Updates injected while `run` is not active should be
/// delivered when it next runs (as a real platform would).
#[async_trait]
pub trait ChannelHarness: Send + Sync {
    /// A direct (one-to-one) conversation that `send`, `edit` and `ask` can
    /// target. Usually the conversation of [`Self::inject_direct_message`].
    fn direct_conversation(&self) -> Conversation;

    /// Deliver a direct message with this text, always from the same
    /// synthetic platform user, with a fresh platform message id.
    async fn inject_direct_message(&self, text: &str);

    /// Deliver a group message with this text from the same synthetic user.
    /// When `addressed`, address the bot the platform's way (a mention
    /// entity, `<@BOT>` prefix, ...); otherwise do not mention it.
    async fn inject_group_message(&self, text: &str, addressed: bool);

    /// Press the button for `choice_id` on a prompt the adapter showed with
    /// [`Channel::ask`], using whatever the adapter put on the button
    /// (callback data, button value, custom id).
    async fn press_button(&self, prompt: &ApprovalPrompt, choice_id: &str);

    /// Text of every message the adapter sent to the platform, in order,
    /// as raw platform payload (after Markdown conversion and splitting).
    async fn sent_messages(&self) -> Vec<String>;
}

/// Run every channel check (see [`channel_checks`]). `run` is started twice
/// (once stopped by shutdown, once by dropping the receiver), so the channel
/// must support being run again after `run` returns.
pub async fn check_channel(
    channel: Arc<dyn Channel>,
    harness: &dyn ChannelHarness,
) -> ConformanceReport {
    use channel_checks::*;

    let name = channel.name().to_string();
    let mut report = ConformanceReport::new(if name.is_empty() {
        "<unnamed channel>".to_string()
    } else {
        name.clone()
    });
    let caps = channel.capabilities();
    report.record(
        NAME_NON_EMPTY,
        if name.trim().is_empty() {
            Err("name() is empty".into())
        } else {
            Ok(())
        },
    );
    report.record(
        MAX_MESSAGE_LEN_POSITIVE,
        if caps.max_message_len > 0 {
            Ok(())
        } else {
            Err("max_message_len is 0".into())
        },
    );

    // Session A: inbound mapping, outbound calls, then shutdown.
    let mut session = Session::start(channel.clone());

    harness
        .inject_direct_message("conformance probe direct-one")
        .await;
    let first = session.next_message("direct-one").await;
    report.record(
        DIRECT_MESSAGE_FORWARDED,
        first.as_ref().map_err(Clone::clone).and_then(|m| {
            let mut problems = Vec::new();
            if m.conversation.kind != ConversationKind::Direct {
                problems.push(format!("kind is {:?}, not Direct", m.conversation.kind));
            }
            if !m.addressed {
                problems.push("addressed is false".to_string());
            }
            if m.sender.id.trim().is_empty() {
                problems.push("sender id is empty".to_string());
            }
            if m.message_id.trim().is_empty() {
                problems.push("message id is empty".to_string());
            }
            joined(problems)
        }),
    );

    harness
        .inject_direct_message("conformance probe direct-two")
        .await;
    let second = session.next_message("direct-two").await;
    report.record(
        SENDER_ID_STABLE,
        match (&first, &second) {
            (Ok(a), Ok(b)) => {
                let mut problems = Vec::new();
                if a.sender.id != b.sender.id {
                    problems.push(format!(
                        "sender id changed from {:?} to {:?}",
                        a.sender.id, b.sender.id
                    ));
                }
                if a.conversation.id != b.conversation.id {
                    problems.push(format!(
                        "conversation id changed from {:?} to {:?}",
                        a.conversation.id, b.conversation.id
                    ));
                }
                if b.message_id.trim().is_empty() || a.message_id == b.message_id {
                    problems.push(format!(
                        "message ids must be distinct and non-empty: {:?}, {:?}",
                        a.message_id, b.message_id
                    ));
                }
                joined(problems)
            }
            (Err(e), _) | (_, Err(e)) => Err(e.clone()),
        },
    );

    for (check, addressed, marker) in [
        (UNADDRESSED_GROUP_MESSAGE, false, "group-plain"),
        (ADDRESSED_GROUP_MESSAGE, true, "group-addressed"),
    ] {
        harness
            .inject_group_message(&format!("conformance probe {marker}"), addressed)
            .await;
        let result = session.next_message(marker).await.and_then(|m| {
            let mut problems = Vec::new();
            if m.addressed != addressed {
                problems.push(format!(
                    "addressed is {}, expected {addressed}",
                    m.addressed
                ));
            }
            if m.conversation.kind == ConversationKind::Direct {
                problems.push("group message reported as Direct".to_string());
            }
            joined(problems)
        });
        report.record(check, result);
    }

    let conversation = harness.direct_conversation();
    let max = caps.max_message_len.max(1);
    let long = long_text(max * 3);
    let before = harness.sent_messages().await.len();
    let sent = channel
        .send(&conversation, &OutboundMessage::text(long.clone()))
        .await;
    let after = harness.sent_messages().await;
    report.record(
        SEND_SPLITS_LONG_MESSAGES,
        match sent {
            Err(e) => Err(format!("send failed: {e}")),
            Ok(_) => {
                let parts = after.get(before..).unwrap_or_default();
                let mut problems = Vec::new();
                if parts.is_empty() {
                    problems.push("the harness saw no sent messages".to_string());
                }
                for (i, part) in parts.iter().enumerate() {
                    let len = part.chars().count();
                    if len > caps.max_message_len {
                        problems.push(format!(
                            "part {i} has {len} chars > max_message_len {}",
                            caps.max_message_len
                        ));
                    }
                }
                let sent_content: String = parts.concat().split_whitespace().collect();
                let expected: String = long.split_whitespace().collect();
                if sent_content != expected {
                    problems.push(format!(
                        "non-whitespace content differs: expected {} chars, platform got {} \
                         (first difference at char {})",
                        expected.chars().count(),
                        sent_content.chars().count(),
                        first_difference(&expected, &sent_content)
                    ));
                }
                joined(problems)
            }
        },
    );

    let edit = match channel
        .send(
            &conversation,
            &OutboundMessage::text("conformance probe edit target"),
        )
        .await
    {
        Err(e) => Err(format!("send before edit failed: {e}")),
        Ok(message_ref) => {
            let result = channel
                .edit(&message_ref, "conformance probe edit target (edited)")
                .await;
            capability_result("edit", caps.edit_messages, result.map(|_| ()))
        }
    };
    report.record(EDIT_MATCHES_CAPABILITY, edit);

    let prompt = ApprovalPrompt {
        id: "conformance-prompt-1".to_string(),
        conversation: conversation.clone(),
        text: "Allow the conformance probe to continue?".to_string(),
        choices: vec![
            ApprovalChoice {
                id: "approve".to_string(),
                label: "Approve".to_string(),
            },
            ApprovalChoice {
                id: "deny".to_string(),
                label: "Deny".to_string(),
            },
        ],
    };
    let asked = channel.ask(&prompt).await;
    let ask_ok = asked.is_ok();
    report.record(
        ASK_MATCHES_CAPABILITY,
        capability_result("ask", caps.buttons, asked.map(|_| ())),
    );
    if !caps.buttons {
        report.pass(BUTTON_PRESS_ROUND_TRIP);
        report.note(format!(
            "{BUTTON_PRESS_ROUND_TRIP}: channel has no buttons; check is vacuous"
        ));
    } else if !ask_ok {
        report.fail(BUTTON_PRESS_ROUND_TRIP, "skipped: ask failed");
    } else {
        harness.press_button(&prompt, "deny").await;
        let result = session.next_approval(&prompt.id).await.and_then(|a| {
            if a.choice_id == "deny" {
                Ok(())
            } else {
                Err(format!("choice id is {:?}, expected \"deny\"", a.choice_id))
            }
        });
        report.record(BUTTON_PRESS_ROUND_TRIP, result);
    }

    session.shutdown.cancel();
    report.record(SHUTDOWN_STOPS_RUN, session.join().await);

    // Session B: dropping the receiver stops `run`.
    let mut session = Session::start(channel);
    harness
        .inject_direct_message("conformance probe direct-three")
        .await;
    let result = match session.next_message("direct-three").await {
        Err(e) => Err(format!("could not reach a running state: {e}")),
        Ok(_) => {
            drop(session.events.take());
            harness
                .inject_direct_message("conformance probe direct-four")
                .await;
            session.join().await
        }
    };
    session.shutdown.cancel();
    report.record(RECEIVER_DROP_STOPS_RUN, result);

    report
}

struct Session {
    events: Option<mpsc::Receiver<ChannelEvent>>,
    shutdown: CancellationToken,
    handle: Option<JoinHandle<Result<(), ChannelError>>>,
}

impl Session {
    fn start(channel: Arc<dyn Channel>) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let shutdown = CancellationToken::new();
        let token = shutdown.clone();
        let handle = tokio::spawn(async move { channel.run(tx, token).await });
        Self {
            events: Some(rx),
            shutdown,
            handle: Some(handle),
        }
    }

    /// The next event matching `pick`, skipping others, within the timeout.
    async fn next_event<T>(
        &mut self,
        what: &str,
        mut pick: impl FnMut(ChannelEvent) -> Option<T>,
    ) -> Result<T, String> {
        let Some(events) = self.events.as_mut() else {
            return Err("receiver already dropped".into());
        };
        let deadline = Instant::now() + EVENT_TIMEOUT;
        loop {
            match tokio::time::timeout_at(deadline, events.recv()).await {
                Err(_) => return Err(format!("no {what} arrived within {EVENT_TIMEOUT:?}")),
                Ok(None) => {
                    return Err(format!(
                        "run ended before {what} arrived (the event sender was dropped)"
                    ))
                }
                Ok(Some(event)) => {
                    if let Some(found) = pick(event) {
                        return Ok(found);
                    }
                }
            }
        }
    }

    async fn next_message(&mut self, marker: &str) -> Result<InboundMessage, String> {
        self.next_event(&format!("message containing {marker:?}"), |e| match e {
            ChannelEvent::Message(m) if m.text.contains(marker) => Some(m),
            _ => None,
        })
        .await
    }

    async fn next_approval(&mut self, prompt_id: &str) -> Result<ApprovalAnswer, String> {
        self.next_event(&format!("approval for prompt {prompt_id:?}"), |e| match e {
            ChannelEvent::Approval(a) if a.prompt_id == prompt_id => Some(a),
            _ => None,
        })
        .await
    }

    /// Wait for `run` to return; `Ok(())` only for a timely `Ok(())`.
    async fn join(&mut self) -> Result<(), String> {
        let Some(mut handle) = self.handle.take() else {
            return Err("run already joined".into());
        };
        match tokio::time::timeout(EVENT_TIMEOUT, &mut handle).await {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(e))) => Err(format!("run returned an error: {e}")),
            Ok(Err(e)) => Err(format!("run panicked: {e}")),
            Err(_) => {
                handle.abort();
                Err(format!("run did not return within {EVENT_TIMEOUT:?}"))
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

fn capability_result(
    what: &str,
    supported: bool,
    result: Result<(), ChannelError>,
) -> Result<(), String> {
    match (supported, result) {
        (true, Ok(())) => Ok(()),
        (true, Err(e)) => Err(format!("{what} is advertised but failed: {e}")),
        (false, Err(ChannelError::Unsupported(_))) => Ok(()),
        (false, Ok(())) => Err(format!(
            "{what} succeeded although the capability is false; return Unsupported or \
             advertise it"
        )),
        (false, Err(e)) => Err(format!(
            "{what} is not advertised, so it must return Unsupported, got: {e}"
        )),
    }
}

fn joined(problems: Vec<String>) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("; "))
    }
}

/// Plain words (no Markdown) in lines of twelve, at least `chars` long.
fn long_text(chars: usize) -> String {
    let mut out = String::with_capacity(chars + 16);
    let mut n = 0usize;
    // ASCII only, so bytes are chars; the trailing separator is trimmed.
    while out.len() <= chars {
        n += 1;
        out.push_str(&format!("w{n:05}"));
        out.push(if n.is_multiple_of(12) { '\n' } else { ' ' });
    }
    out.trim_end().to_string()
}

fn first_difference(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// [`ChannelHarness`] for [`TestChannel`]: the reference implementation, and
/// a check that the in-memory test double honours the channel contract.
pub struct TestChannelHarness {
    channel: Arc<TestChannel>,
    sender_id: String,
}

impl TestChannelHarness {
    pub fn new(channel: Arc<TestChannel>) -> Self {
        Self {
            channel,
            sender_id: "test:synthetic-user-1".to_string(),
        }
    }
}

#[async_trait]
impl ChannelHarness for TestChannelHarness {
    fn direct_conversation(&self) -> Conversation {
        crate::testing::direct("dm:synthetic-user-1")
    }

    async fn inject_direct_message(&self, text: &str) {
        self.channel.inject_message(crate::testing::message(
            self.direct_conversation(),
            &self.sender_id,
            text,
        ));
    }

    async fn inject_group_message(&self, text: &str, addressed: bool) {
        let mut message =
            crate::testing::message(crate::testing::group("group:acme"), &self.sender_id, text);
        message.addressed = addressed;
        self.channel.inject_message(message);
    }

    async fn press_button(&self, prompt: &ApprovalPrompt, choice_id: &str) {
        self.channel.inject(ChannelEvent::Approval(ApprovalAnswer {
            prompt_id: prompt.id.clone(),
            choice_id: choice_id.to_string(),
            sender: sender(&self.sender_id),
            conversation: prompt.conversation.clone(),
        }));
    }

    async fn sent_messages(&self) -> Vec<String> {
        self.channel
            .sent()
            .into_iter()
            .map(|m| m.message.text)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_text_is_long_enough_and_plain() {
        let text = long_text(300);
        assert!(text.chars().count() >= 300);
        assert!(text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c.is_whitespace()));
    }
}
