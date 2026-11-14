//! Your own channel: a transport that renders the runner's typed status and
//! approval requests itself instead of the default Markdown, the way a web
//! UI or a webhook consumer would. Here the "platform" is an in-process
//! outbox that prints what it would draw.
//!
//! - `send_status` / `update_status` receive a `RunProgress` (phase, tool
//!   calls with their receipt status, streamed text); the defaults send
//!   `RunProgress::to_markdown`.
//! - `ask` receives an `ApprovalPrompt` whose `details` hold the call as
//!   data (tool, arguments, reason, the owner request and its card). The
//!   runner calls `ask` only when `capabilities().buttons` is true.
//! - `run` must be re-runnable: it forwards inbound updates until shutdown,
//!   and may be called again later with a new sender.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example own_channel --features "testing channels"
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_sdk::CancellationToken;
use aidaemon_sdk::channels::{
    APPROVE_CHOICE, AccessPolicy, ApprovalAnswer, ApprovalPrompt, Channel, ChannelCapabilities,
    ChannelError, ChannelEvent, ChannelRunner, Conversation, ConversationKind, EventRouter,
    InboundMessage, MessageRef, OutboundMessage, ProgressPhase, RunProgress, Sender, ToolState,
};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

const OWNER: &str = "synthetic-user-1";

/// Restart a service on the synthetic staging host.
#[tool(destructive)]
async fn restart_service(service: String) -> String {
    format!("{service} restarted on staging (simulated)")
}

/// The "platform": inbound updates in, drawn widgets out.
struct Outbox {
    inbound: tokio::sync::Mutex<mpsc::UnboundedReceiver<ChannelEvent>>,
    drawn: Mutex<Vec<String>>,
    prompts: Mutex<Vec<ApprovalPrompt>>,
    next_id: Mutex<u64>,
}

impl Outbox {
    fn draw(&self, conversation: &str, widget: String) -> MessageRef {
        println!("[{conversation}] {widget}");
        self.drawn.lock().unwrap().push(widget);
        let mut next = self.next_id.lock().unwrap();
        *next += 1;
        MessageRef {
            conversation_id: conversation.to_string(),
            message_id: next.to_string(),
        }
    }
}

/// A status as the platform would draw it: a phase badge and a chip per
/// tool call, from the typed fields.
fn status_widget(progress: &RunProgress) -> String {
    let phase = match progress.phase {
        ProgressPhase::Working => "working".to_string(),
        ProgressPhase::Ended(ending) => format!("ended: {}", ending.word()),
        _ => "unknown".to_string(),
    };
    let chips: Vec<String> = progress
        .tools
        .iter()
        .map(|tool| match tool.state {
            ToolState::Running => format!("{} (running)", tool.tool),
            ToolState::Ended(status) => format!("{} ({status:?})", tool.tool),
            _ => tool.tool.clone(),
        })
        .collect();
    format!("<status {phase}> [{}]", chips.join(", "))
}

#[async_trait]
impl Channel for Outbox {
    fn name(&self) -> &str {
        "outbox"
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            edit_messages: true,
            buttons: true,
            max_message_len: 2000,
            ..Default::default()
        }
    }

    async fn run(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        let mut inbound = self.inbound.lock().await;
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                event = inbound.recv() => match event {
                    Some(event) => {
                        if events.send(event).await.is_err() {
                            return Ok(()); // the runner stopped listening
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
        Ok(self.draw(&conversation.id, format!("<message> {}", message.text)))
    }

    async fn edit(&self, message: &MessageRef, text: &str) -> Result<(), ChannelError> {
        self.draw(&message.conversation_id, format!("<edit> {text}"));
        Ok(())
    }

    async fn send_status(
        &self,
        conversation: &Conversation,
        progress: &RunProgress,
    ) -> Result<MessageRef, ChannelError> {
        Ok(self.draw(&conversation.id, status_widget(progress)))
    }

    async fn update_status(
        &self,
        message: &MessageRef,
        progress: &RunProgress,
    ) -> Result<(), ChannelError> {
        self.draw(&message.conversation_id, status_widget(progress));
        Ok(())
    }

    async fn ask(&self, prompt: &ApprovalPrompt) -> Result<MessageRef, ChannelError> {
        let widget = match &prompt.details {
            Some(details) => format!(
                "<approve? tool={} arguments={} risk={:?}>",
                details.tool,
                details.arguments,
                details.request.as_ref().map(|request| request.risk),
            ),
            None => format!("<approve? {}>", prompt.text),
        };
        self.prompts.lock().unwrap().push(prompt.clone());
        Ok(self.draw(&prompt.conversation.id, widget))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let router = EventRouter::new();
    let sink = router.clone();
    let agent = aidaemon_sdk::preset("ops-bot")
        .provider_from_env_or(ScriptedProvider::new([
            tool_calls([call("c1", "restart_service", json!({"service": "billing"}))]),
            text("billing is restarted on staging."),
        ]))
        .instructions("You run operations on staging. Keep replies to one sentence.")
        .tool(RestartService)
        .configure(move |builder| builder.event_sink(sink))
        .build()
        .await?;

    let (inbound, receiver) = mpsc::unbounded_channel();
    let outbox = Arc::new(Outbox {
        inbound: tokio::sync::Mutex::new(receiver),
        drawn: Mutex::new(Vec::new()),
        prompts: Mutex::new(Vec::new()),
        next_id: Mutex::new(0),
    });
    let runner = ChannelRunner::builder(agent)
        .channel(outbox.clone())
        .access(AccessPolicy::new().owner("outbox", OWNER))
        .event_router(router)
        .build()?;
    let shutdown = CancellationToken::new();
    let task = tokio::spawn({
        let shutdown = shutdown.clone();
        async move { runner.run(shutdown).await }
    });

    let dm = Conversation {
        id: "dm-1".to_string(),
        kind: ConversationKind::Direct,
    };
    let me = Sender {
        id: OWNER.to_string(),
        display_name: None,
        is_bot: false,
    };
    inbound.send(ChannelEvent::Message(InboundMessage::new(
        dm.clone(),
        me.clone(),
        "m1",
        "Restart billing on staging",
    )))?;

    // The destructive call waits for the owner: press Approve.
    let prompt = loop {
        if let Some(prompt) = outbox.prompts.lock().unwrap().first().cloned() {
            break prompt;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    inbound.send(ChannelEvent::Approval(ApprovalAnswer::new(
        prompt.id.clone(),
        APPROVE_CHOICE,
        me,
        dm,
    )))?;
    while !outbox
        .drawn
        .lock()
        .unwrap()
        .iter()
        .any(|widget| widget.contains("billing is restarted"))
    {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    shutdown.cancel();
    task.await?;
    Ok(())
}
