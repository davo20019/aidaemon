//! An in-memory [`Channel`] for tests (feature `testing`).
//!
//! Inject [`ChannelEvent`]s with [`TestChannel::inject`] and inspect what the
//! runner sent, edited and asked. Capabilities are configurable so tests can
//! cover channels with and without edits, buttons and typing indicators.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

mod cards;
pub use cards::{CardChannel, CardChat};

use crate::channel::{
    ApprovalAnswer, ApprovalPrompt, Attachment, Channel, ChannelCapabilities, ChannelError,
    ChannelEvent, CommandInvocation, CommandSpec, Conversation, ConversationKind, InboundMessage,
    MessageRef, OutboundMessage, Sender,
};

/// A message the channel was asked to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentMessage {
    pub conversation: Conversation,
    pub message: OutboundMessage,
    pub message_ref: MessageRef,
}

/// An edit the channel was asked to make.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditedMessage {
    pub message_ref: MessageRef,
    pub text: String,
    pub at: Instant,
}

pub struct TestChannel {
    name: String,
    caps: ChannelCapabilities,
    inbound_tx: mpsc::UnboundedSender<ChannelEvent>,
    inbound_rx: tokio::sync::Mutex<mpsc::UnboundedReceiver<ChannelEvent>>,
    sent: Mutex<Vec<SentMessage>>,
    edits: Mutex<Vec<EditedMessage>>,
    prompts: Mutex<Vec<ApprovalPrompt>>,
    typing: AtomicUsize,
    runs: AtomicUsize,
    next_id: AtomicU64,
    run_failures: Mutex<VecDeque<ChannelError>>,
    files: Mutex<HashMap<String, Vec<u8>>>,
    downloads: Mutex<Vec<String>>,
    prompt_refs: Mutex<HashMap<String, MessageRef>>,
    removed_buttons: Mutex<Vec<MessageRef>>,
    strips_latex: bool,
    registered_commands: Mutex<Option<Vec<CommandSpec>>>,
    command_answers: Mutex<Vec<(String, String)>>,
}

impl TestChannel {
    /// A channel with no edits, buttons or typing, and a 4000-char limit.
    pub fn new(name: impl Into<String>) -> Self {
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        Self {
            name: name.into(),
            caps: ChannelCapabilities::default(),
            inbound_tx,
            inbound_rx: tokio::sync::Mutex::new(inbound_rx),
            sent: Mutex::new(Vec::new()),
            edits: Mutex::new(Vec::new()),
            prompts: Mutex::new(Vec::new()),
            typing: AtomicUsize::new(0),
            runs: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
            run_failures: Mutex::new(VecDeque::new()),
            files: Mutex::new(HashMap::new()),
            downloads: Mutex::new(Vec::new()),
            prompt_refs: Mutex::new(HashMap::new()),
            removed_buttons: Mutex::new(Vec::new()),
            strips_latex: true,
            registered_commands: Mutex::new(None),
            command_answers: Mutex::new(Vec::new()),
        }
    }

    pub fn with_capabilities(mut self, caps: ChannelCapabilities) -> Self {
        self.caps = caps;
        self
    }

    pub fn with_edits(mut self, on: bool) -> Self {
        self.caps.edit_messages = on;
        self
    }

    pub fn with_buttons(mut self, on: bool) -> Self {
        self.caps.buttons = on;
        self
    }

    pub fn with_typing(mut self, on: bool) -> Self {
        self.caps.typing_indicator = on;
        self
    }

    pub fn with_files(mut self, on: bool) -> Self {
        self.caps.files = on;
        self
    }

    /// Advertise a command menu.
    pub fn with_commands(mut self, on: bool) -> Self {
        self.caps.commands = on;
        self
    }

    /// Whether the runner should rewrite LaTeX for this channel.
    pub fn with_latex_stripping(mut self, on: bool) -> Self {
        self.strips_latex = on;
        self
    }

    /// The last command menu registered, if any.
    pub fn registered_commands(&self) -> Option<Vec<CommandSpec>> {
        self.registered_commands.lock().expect("poisoned").clone()
    }

    /// `(handle, text)` of every answered command invocation, in order.
    pub fn command_answers(&self) -> Vec<(String, String)> {
        self.command_answers.lock().expect("poisoned").clone()
    }

    /// Make `source` downloadable as `bytes`.
    pub fn add_file(&self, source: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.files
            .lock()
            .expect("poisoned")
            .insert(source.into(), bytes.into());
    }

    /// The sources downloaded so far, in order.
    pub fn downloads(&self) -> Vec<String> {
        self.downloads.lock().expect("poisoned").clone()
    }

    /// The ref [`Channel::ask`] returned for this prompt.
    pub fn prompt_ref(&self, prompt_id: &str) -> Option<MessageRef> {
        self.prompt_refs
            .lock()
            .expect("poisoned")
            .get(prompt_id)
            .cloned()
    }

    /// Messages whose buttons were removed, in order.
    pub fn removed_buttons(&self) -> Vec<MessageRef> {
        self.removed_buttons.lock().expect("poisoned").clone()
    }

    pub fn with_max_len(mut self, max: usize) -> Self {
        self.caps.max_message_len = max;
        self
    }

    /// Make the next calls to [`Channel::run`] fail with these errors, in
    /// order, before running normally.
    pub fn fail_runs(&self, errors: impl IntoIterator<Item = ChannelError>) {
        self.run_failures.lock().expect("poisoned").extend(errors);
    }

    /// Deliver an event as if the platform sent it.
    pub fn inject(&self, event: ChannelEvent) {
        let _ = self.inbound_tx.send(event);
    }

    pub fn inject_message(&self, message: InboundMessage) {
        self.inject(ChannelEvent::Message(message));
    }

    /// Press a button on (or answer) an approval prompt.
    pub fn press(
        &self,
        prompt_id: &str,
        choice_id: &str,
        sender_id: &str,
        conversation: Conversation,
    ) {
        self.inject(ChannelEvent::Approval(ApprovalAnswer::new(
            prompt_id,
            choice_id,
            sender(sender_id),
            conversation,
        )));
    }

    /// Everything sent, in order.
    pub fn sent(&self) -> Vec<SentMessage> {
        self.sent.lock().expect("poisoned").clone()
    }

    /// Texts sent to one conversation, in order.
    pub fn sent_texts(&self, conversation_id: &str) -> Vec<String> {
        self.sent()
            .into_iter()
            .filter(|m| m.conversation.id == conversation_id)
            .map(|m| m.message.text)
            .collect()
    }

    pub fn edits(&self) -> Vec<EditedMessage> {
        self.edits.lock().expect("poisoned").clone()
    }

    pub fn prompts(&self) -> Vec<ApprovalPrompt> {
        self.prompts.lock().expect("poisoned").clone()
    }

    pub fn typing_count(&self) -> usize {
        self.typing.load(Ordering::SeqCst)
    }

    /// How many times [`Channel::run`] was entered.
    pub fn run_count(&self) -> usize {
        self.runs.load(Ordering::SeqCst)
    }

    /// Poll `pred` until it holds or `timeout` passes. Returns whether it
    /// held.
    pub async fn wait_for(&self, timeout: Duration, pred: impl Fn(&Self) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if pred(self) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn next_ref(&self, conversation_id: &str) -> MessageRef {
        MessageRef {
            conversation_id: conversation_id.to_string(),
            message_id: format!("m{}", self.next_id.fetch_add(1, Ordering::SeqCst)),
        }
    }
}

/// A human sender with a synthetic display name.
pub fn sender(id: &str) -> Sender {
    Sender {
        id: id.to_string(),
        display_name: Some(format!("Synthetic {id}")),
        is_bot: false,
    }
}

pub fn direct(id: &str) -> Conversation {
    Conversation {
        id: id.to_string(),
        kind: ConversationKind::Direct,
    }
}

pub fn group(id: &str) -> Conversation {
    Conversation {
        id: id.to_string(),
        kind: ConversationKind::Group,
    }
}

/// An inbound message. `addressed` is true for direct conversations.
pub fn message(conversation: Conversation, sender_id: &str, text: &str) -> InboundMessage {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let addressed = conversation.kind == ConversationKind::Direct;
    InboundMessage {
        conversation,
        sender: sender(sender_id),
        message_id: format!("in{}", NEXT.fetch_add(1, Ordering::SeqCst)),
        text: text.to_string(),
        attachments: Vec::new(),
        addressed,
        parent_message_id: None,
        thread_id: None,
    }
}

#[async_trait]
impl Channel for TestChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ChannelCapabilities {
        self.caps
    }

    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        let failure = self.run_failures.lock().expect("poisoned").pop_front();
        if let Some(err) = failure {
            return Err(err);
        }
        let mut rx = self.inbound_rx.lock().await;
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                event = rx.recv() => match event {
                    Some(event) => {
                        if events.send(event).await.is_err() {
                            return Ok(());
                        }
                    }
                    None => return Ok(()),
                },
            }
        }
    }

    async fn send(
        &self,
        conversation: &Conversation,
        message: &OutboundMessage,
    ) -> Result<MessageRef, ChannelError> {
        // Like a real adapter, split text over the advertised limit; each
        // part is recorded as its own sent message. Text within the limit
        // (everything the runner sends, since it splits first) is recorded
        // unchanged.
        let parts = if message.text.chars().count() > self.caps.max_message_len {
            crate::split::split_message(&message.text, self.caps.max_message_len)
        } else {
            vec![message.text.clone()]
        };
        let mut sent = self.sent.lock().expect("poisoned");
        let mut last = None;
        for (i, text) in parts.into_iter().enumerate() {
            let message_ref = self.next_ref(&conversation.id);
            sent.push(SentMessage {
                conversation: conversation.clone(),
                message: OutboundMessage {
                    text,
                    reply_to: if i == 0 {
                        message.reply_to.clone()
                    } else {
                        None
                    },
                },
                message_ref: message_ref.clone(),
            });
            last = Some(message_ref);
        }
        Ok(last.unwrap_or_else(|| self.next_ref(&conversation.id)))
    }

    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        if !self.caps.edit_messages {
            return Err(ChannelError::Unsupported("edit"));
        }
        self.edits.lock().expect("poisoned").push(EditedMessage {
            message_ref: message.clone(),
            text: crate::split::truncate_for_edit(text, self.caps.max_message_len),
            at: Instant::now(),
        });
        Ok(())
    }

    async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        if !self.caps.buttons {
            return Err(ChannelError::Unsupported("buttons"));
        }
        self.prompts.lock().expect("poisoned").push(prompt.clone());
        let message_ref = self.next_ref(&prompt.conversation.id);
        self.prompt_refs
            .lock()
            .expect("poisoned")
            .insert(prompt.id.clone(), message_ref.clone());
        Ok(message_ref)
    }

    async fn remove_buttons(&self, message: &MessageRef) -> Result<(), ChannelError> {
        self.removed_buttons
            .lock()
            .expect("poisoned")
            .push(message.clone());
        Ok(())
    }

    async fn download(&self, attachment: &Attachment) -> Result<Vec<u8>, ChannelError> {
        if !self.caps.files {
            return Err(ChannelError::Unsupported("files"));
        }
        self.downloads
            .lock()
            .expect("poisoned")
            .push(attachment.source.clone());
        self.files
            .lock()
            .expect("poisoned")
            .get(&attachment.source)
            .cloned()
            .ok_or_else(|| ChannelError::Other("no such test file".into()))
    }

    fn strips_latex(&self) -> bool {
        self.strips_latex
    }

    async fn register_commands(&self, commands: &[CommandSpec]) -> Result<(), ChannelError> {
        *self.registered_commands.lock().expect("poisoned") = Some(commands.to_vec());
        Ok(())
    }

    /// Recorded, and also sent like the default answer.
    async fn answer_command(
        &self,
        command: &CommandInvocation,
        text: &str,
    ) -> Result<(), ChannelError> {
        self.command_answers
            .lock()
            .expect("poisoned")
            .push((command.handle.clone(), text.to_string()));
        self.send(&command.conversation, &OutboundMessage::text(text))
            .await
            .map(|_| ())
    }

    async fn typing(&self, _conversation: &Conversation) -> Result<(), ChannelError> {
        if self.caps.typing_indicator {
            self.typing.fetch_add(1, Ordering::SeqCst);
            Ok(())
        } else {
            Err(ChannelError::Unsupported("typing"))
        }
    }
}
