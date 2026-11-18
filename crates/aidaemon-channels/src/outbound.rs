//! The one redaction boundary for what the owner is shown.
//!
//! The [`ChannelRunner`](crate::ChannelRunner) wraps every channel it is
//! given in a [`RedactingChannel`], so everything it hands an adapter (a
//! message, an edit, a status, an approval prompt, a file's caption, a
//! command's answer) is already redacted, with the agent's
//! [`RedactionScope`](aidaemon_core::RedactionScope): a custom channel is
//! covered without doing anything. The built-in adapters also pass each
//! request they send through [`redact_body`] or [`redact_text`] at their
//! wire, the one place all of their traffic leaves through, so a path that
//! does not go through the runner (the daemon's own cards and statuses)
//! cannot reach a chat unredacted either; the conformance kit holds every
//! adapter to that (`channel.send_hides_credentials`). The redaction hides every
//! registered secret ([`aidaemon_core::redaction::register_secret`]: each
//! value a secret store handed out and each credential a tool printed) and
//! the credential shapes the kernel's `Redactor` knows, and registers a
//! labelled credential it sees for the first time.

use crate::channel::{
    ApprovalPrompt, Attachment, Channel, ChannelCapabilities, ChannelError, ChannelEvent,
    CommandInvocation, CommandSpec, Conversation, MessageRef, OutboundFile, OutboundMessage,
    RunProgress,
};

/// `text` as the owner may see it, with the process-wide secrets
/// ([`aidaemon_core::RedactionScope::process`]).
pub fn redact_text(text: &str) -> String {
    aidaemon_tools::sanitize::redact_for_owner(text)
}

/// [`redact_text`] with the secrets registered in `scope`.
pub fn redact_text_in(scope: &aidaemon_core::RedactionScope, text: &str) -> String {
    aidaemon_tools::sanitize::redact_for_owner_in(scope, text)
}

/// A platform request's body with every string in it redacted, whatever
/// its fields are called.
pub fn redact_body(body: &serde_json::Value) -> serde_json::Value {
    redact_body_in(&aidaemon_core::RedactionScope::process(), body)
}

/// [`redact_body`] with the secrets registered in `scope`.
pub fn redact_body_in(
    scope: &aidaemon_core::RedactionScope,
    body: &serde_json::Value,
) -> serde_json::Value {
    let mut body = body.clone();
    aidaemon_tools::sanitize::redact_json_for_owner_in(scope, &mut body);
    body
}

/// A channel whose every outbound text passes [`redact_text_in`] before
/// the adapter sees it: the message and its edits, the status's streamed
/// text, the approval prompt (body, button labels and card facts), a file's
/// name and caption, and a command's answer. The runner wraps each of its
/// channels in one; wrap an adapter yourself to drive it outside the
/// runner. What comes in (events, downloads) passes through untouched.
pub struct RedactingChannel {
    inner: std::sync::Arc<dyn Channel>,
    scope: aidaemon_core::RedactionScope,
}

impl RedactingChannel {
    /// `channel` behind the redaction of `scope`'s secrets (the agent's,
    /// [`aidaemon_core::Agent::redaction_scope`]) and the credential shapes.
    pub fn new(channel: std::sync::Arc<dyn Channel>, scope: aidaemon_core::RedactionScope) -> Self {
        Self {
            inner: channel,
            scope,
        }
    }

    /// The adapter behind the redaction.
    pub fn inner(&self) -> &std::sync::Arc<dyn Channel> {
        &self.inner
    }

    fn text(&self, text: &str) -> String {
        redact_text_in(&self.scope, text)
    }

    fn progress(&self, progress: &RunProgress) -> RunProgress {
        let mut progress = progress.clone();
        progress.streamed = self.text(&progress.streamed);
        progress
    }
}

#[async_trait::async_trait]
impl Channel for RedactingChannel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        self.inner.capabilities()
    }

    async fn run(
        &self,
        events: tokio::sync::mpsc::Sender<ChannelEvent>,
        shutdown: tokio_util::sync::CancellationToken,
    ) -> Result<(), ChannelError> {
        self.inner.run(events, shutdown).await
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        let mut message = message.clone();
        message.text = self.text(&message.text);
        self.inner.send(conversation, &message).await
    }

    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        self.inner.edit(message, &self.text(text)).await
    }

    async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        let mut prompt = prompt.clone();
        prompt.text = self.text(&prompt.text);
        for choice in &mut prompt.choices {
            choice.label = self.text(&choice.label);
        }
        // A card that cannot round-trip is dropped, never sent
        // unredacted: the prompt's text still says it all.
        prompt.details = prompt.details.take().and_then(|details| {
            let value = serde_json::to_value(&details).ok()?;
            serde_json::from_value(redact_body_in(&self.scope, &value)).ok()
        });
        self.inner.ask(&prompt).await
    }

    async fn remove_buttons(&self, message: &MessageRef) -> Result<(), ChannelError> {
        self.inner.remove_buttons(message).await
    }

    async fn send_status(
        &self,
        conversation: &Conversation,
        progress: &RunProgress,
    ) -> Result<MessageRef, ChannelError> {
        self.inner
            .send_status(conversation, &self.progress(progress))
            .await
    }

    async fn update_status(
        &self,
        message: &MessageRef,
        progress: &RunProgress,
    ) -> Result<(), ChannelError> {
        self.inner
            .update_status(message, &self.progress(progress))
            .await
    }

    fn status_text(&self, progress: &RunProgress) -> String {
        self.text(&self.inner.status_text(progress))
    }

    async fn typing(&self, conversation: &Conversation) -> Result<(), ChannelError> {
        self.inner.typing(conversation).await
    }

    async fn delete(&self, message: &MessageRef) -> Result<(), ChannelError> {
        self.inner.delete(message).await
    }

    async fn send_file(
        &self,
        conversation: &Conversation,
        file: &OutboundFile,
    ) -> Result<MessageRef, ChannelError> {
        let mut file = file.clone();
        file.name = self.text(&file.name);
        file.caption = file.caption.as_deref().map(|caption| self.text(caption));
        self.inner.send_file(conversation, &file).await
    }

    async fn download(&self, attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
        self.inner.download(attachment).await
    }

    async fn download_limited(
        &self,
        attachment: &Attachment,
        max_bytes: u64,
    ) -> Result<Vec<u8>, ChannelError> {
        self.inner.download_limited(attachment, max_bytes).await
    }

    async fn register_commands(&self, commands: &[CommandSpec]) -> Result<(), ChannelError> {
        self.inner.register_commands(commands).await
    }

    fn strips_latex(&self) -> bool {
        self.inner.strips_latex()
    }

    async fn display_name(&self, user_id: &str) -> Result<Option<String>, ChannelError> {
        self.inner.display_name(user_id).await
    }

    async fn answer_command(
        &self,
        command: &CommandInvocation,
        text: &str,
    ) -> Result<(), ChannelError> {
        self.inner.answer_command(command, &self.text(text)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live 2026-11-13: a helper's card showed `drush site:install`'s
    /// "User password: …" line. Whatever path builds the text, the wire
    /// hides the value, and once seen it stays hidden unlabelled too.
    #[test]
    fn a_password_line_never_reaches_a_chat() {
        let body = serde_json::json!({
            "chat_id": 42,
            "text": "Running a command · [success] User name: admin  User password: Wire-Synthetic-91",
            "reply_markup": {"inline_keyboard": [[{"text": "OK", "callback_data": "a:1"}]]},
        });
        let sent = redact_body(&body);
        let text = sent["text"].as_str().unwrap();
        assert!(!text.contains("Wire-Synthetic-91"), "{text}");
        assert!(text.contains("User password: [REDACTED:"), "{text}");
        assert_eq!(sent["chat_id"], 42);
        assert_eq!(sent["reply_markup"], body["reply_markup"]);
        let later = redact_text("Your admin login is admin / Wire-Synthetic-91.");
        assert!(!later.contains("Wire-Synthetic-91"), "{later}");
    }

    /// Records what it is handed, redacting nothing, as a hand-written
    /// adapter might.
    #[derive(Default)]
    struct Raw {
        shown: std::sync::Mutex<Vec<String>>,
    }

    impl Raw {
        fn note(&self, text: impl Into<String>) -> MessageRef {
            self.shown.lock().unwrap().push(text.into());
            MessageRef {
                conversation_id: "c".into(),
                message_id: "m".into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Channel for Raw {
        fn name(&self) -> &str {
            "raw"
        }
        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities::default()
        }
        async fn run(
            &self,
            _events: tokio::sync::mpsc::Sender<ChannelEvent>,
            _shutdown: tokio_util::sync::CancellationToken,
        ) -> Result<(), ChannelError> {
            Ok(())
        }
        async fn send(
            &self,
            _conversation: &Conversation,
            message: &OutboundMessage,
        ) -> Result<MessageRef, ChannelError> {
            Ok(self.note(&message.text))
        }
        async fn edit(&self, _message: &MessageRef, text: &str) -> Result<(), ChannelError> {
            self.note(text);
            Ok(())
        }
        async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
            self.note(serde_json::to_string(prompt).unwrap());
            Ok(self.note(&prompt.text))
        }
        async fn send_file(
            &self,
            _conversation: &Conversation,
            file: &OutboundFile,
        ) -> Result<MessageRef, ChannelError> {
            self.note(&file.name);
            Ok(self.note(file.caption.clone().unwrap_or_default()))
        }
    }

    /// Every outbound path through the wrapper reaches an adapter that
    /// redacts nothing without the secret: message, edit, status, prompt
    /// (text, buttons, card), file name and caption, command answer.
    #[tokio::test]
    async fn a_wrapped_adapter_is_handed_nothing_unredacted() {
        const SECRET: &str = "tok_wrapped_9a8b7c6d";
        let scope = aidaemon_core::RedactionScope::isolated();
        scope.register(SECRET).unwrap();
        let raw = std::sync::Arc::new(Raw::default());
        let channel = RedactingChannel::new(raw.clone(), scope);
        let conversation = Conversation {
            id: "c".into(),
            kind: crate::ConversationKind::Direct,
        };
        let line = format!("token {SECRET}; User password: Wrapped-Pass-77");
        let shown = channel
            .send(&conversation, &OutboundMessage::text(&line))
            .await
            .unwrap();
        channel.edit(&shown, &line).await.unwrap();
        let mut progress = RunProgress::working();
        progress.streamed = line.clone();
        channel.send_status(&conversation, &progress).await.unwrap();
        channel.update_status(&shown, &progress).await.unwrap();
        let prompt = ApprovalPrompt::new(
            "p1",
            conversation.clone(),
            &line,
            vec![crate::ApprovalChoice {
                id: "approve".into(),
                label: line.clone(),
            }],
        );
        channel.ask(&prompt).await.unwrap();
        channel
            .send_file(
                &conversation,
                &OutboundFile {
                    name: format!("{SECRET}.txt"),
                    bytes: Vec::new(),
                    mime_type: None,
                    caption: Some(line.clone()),
                },
            )
            .await
            .unwrap();
        let command = CommandInvocation {
            conversation: conversation.clone(),
            sender: crate::Sender {
                id: "u".into(),
                display_name: None,
                is_bot: false,
            },
            name: "token".into(),
            args: String::new(),
            handle: "h".into(),
        };
        channel.answer_command(&command, &line).await.unwrap();
        let shown = raw.shown.lock().unwrap().clone();
        assert!(shown.len() >= 9, "{shown:?}");
        for text in &shown {
            assert!(!text.contains(SECRET), "{text}");
            assert!(!text.contains("Wrapped-Pass-77"), "{text}");
        }
        assert_eq!(channel.name(), "raw");
    }
}
