//! A chat bot on a channel. `ChannelRunner` handles access, one session per
//! conversation, queueing and approvals as buttons; the channel only moves
//! messages. Here the channel is an in-process one (`TestChannel`) so the
//! example runs anywhere; swap in `channels::telegram`, `slack` or
//! `discord` for a real platform.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example channel_bot --features "testing channels"
use std::sync::Arc;
use std::time::Duration;

use aidaemon_sdk::CancellationToken;
use aidaemon_sdk::channels::testing::{TestChannel, direct, message};
use aidaemon_sdk::channels::{APPROVE_CHOICE, AccessPolicy, ChannelRunner, EventRouter};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use serde_json::json;

const OWNER: &str = "synthetic-user-1";
const WAIT: Duration = Duration::from_secs(60);

/// Restart a service on the synthetic staging host.
#[tool(destructive)]
async fn restart_service(service: String) -> String {
    format!("{service} restarted on staging (simulated)")
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

    let channel = Arc::new(TestChannel::new("chat").with_buttons(true));
    let runner = ChannelRunner::builder(agent)
        .channel(channel.clone())
        .access(AccessPolicy::new().owner("chat", OWNER))
        .event_router(router)
        .build()?;
    let shutdown = CancellationToken::new();
    let task = tokio::spawn({
        let shutdown = shutdown.clone();
        async move { runner.run(shutdown).await }
    });

    channel.inject_message(message(direct("dm-1"), OWNER, "Restart billing on staging"));
    // A destructive call: the owner gets an approval prompt with buttons.
    let asked = |c: &TestChannel| !c.prompts().is_empty() || !c.sent_texts("dm-1").is_empty();
    channel.wait_for(WAIT, asked).await;
    if let Some(prompt) = channel.prompts().first() {
        println!("prompt: {}", prompt.text.lines().next().unwrap_or_default());
        channel.press(&prompt.id, APPROVE_CHOICE, OWNER, direct("dm-1"));
    }
    channel
        .wait_for(WAIT, |c| !c.sent_texts("dm-1").is_empty())
        .await;
    for sent in channel.sent_texts("dm-1") {
        println!("bot: {sent}");
    }
    shutdown.cancel();
    task.await?;
    Ok(())
}
