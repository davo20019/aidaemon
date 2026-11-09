//! Slash commands a host adds to a [`crate::ChannelRunner`].
//!
//! Register one with [`crate::ChannelRunnerBuilder::command`]. The runner
//! offers it in the platform's command menu next to the built-ins and calls
//! the handler after the same access check as a message, so a sender who may
//! not talk to the agent cannot run a command either. A `/word` that is
//! neither a built-in nor registered goes to the agent as text.
//!
//! ```ignore
//! let runner = ChannelRunner::builder(agent)
//!     .channel(channel)
//!     .access(access)
//!     .command("uptime", "Show how long the host has been up", |_ctx: CommandContext| async {
//!         CommandReply::text("Up for 3 days.")
//!     })
//!     .build()?;
//! ```

use std::future::Future;

use async_trait::async_trait;

use crate::channel::{CommandSpec, Conversation, Sender};
use crate::policy::SenderRole;

/// Longest command name every platform accepts.
pub const MAX_COMMAND_NAME_CHARS: usize = 32;
/// Longest description every platform accepts.
pub const MAX_COMMAND_DESCRIPTION_CHARS: usize = 100;

/// One invocation of a command, after the access check. Decide anything
/// about the sender from `sender.id` and `role`, never from the display name.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CommandContext {
    /// The channel's name.
    pub channel: String,
    pub conversation: Conversation,
    pub sender: Sender,
    pub role: SenderRole,
    /// The session this conversation belongs to.
    pub session_id: String,
    /// Lowercase, without the leading `/`.
    pub name: String,
    /// Everything after the command name, trimmed.
    pub args: String,
}

/// What a command answers.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandReply {
    /// Markdown shown to the sender's conversation.
    Text(String),
    /// Nothing to show.
    Silent,
}

impl CommandReply {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
}

/// Runs a registered command. Implemented for any
/// `async fn(CommandContext) -> CommandReply` closure. Each call has its own
/// task, so a slow handler holds up nothing else on the channel.
#[async_trait]
pub trait CommandHandler: Send + Sync {
    async fn handle(&self, context: CommandContext) -> CommandReply;
}

#[async_trait]
impl<F, Fut> CommandHandler for F
where
    F: Fn(CommandContext) -> Fut + Send + Sync,
    Fut: Future<Output = CommandReply> + Send,
{
    async fn handle(&self, context: CommandContext) -> CommandReply {
        self(context).await
    }
}

/// The commands the runner itself answers. `/approve` and `/deny` are only
/// on the menu of a channel without buttons, where they are how a prompt is
/// answered.
pub fn builtin_commands(buttons: bool) -> Vec<CommandSpec> {
    let mut commands = vec![
        CommandSpec::new("stop", "Stop the current run"),
        CommandSpec::new("new", "Start a new conversation"),
        CommandSpec::new("status", "Show what this conversation is doing"),
    ];
    if !buttons {
        commands.push(CommandSpec::new("approve", "Approve a pending action"));
        commands.push(CommandSpec::new("deny", "Deny a pending action"));
    }
    commands
}

/// The built-ins, including those a channel with buttons does not list.
pub(crate) fn is_builtin(name: &str) -> bool {
    builtin_commands(false).iter().any(|c| c.name == name)
}

/// Why a name or description would be refused by some platform.
pub(crate) fn invalid_spec(spec: &CommandSpec) -> Option<&'static str> {
    let name = &spec.name;
    let starts_with_letter = name.starts_with(|c: char| c.is_ascii_lowercase());
    let allowed = name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if name.len() > MAX_COMMAND_NAME_CHARS || !starts_with_letter || !allowed {
        return Some(
            "a command name is 1 to 32 lowercase letters, digits or underscores, starting with a letter",
        );
    }
    let description = spec.description.trim();
    if description.is_empty() || description.chars().count() > MAX_COMMAND_DESCRIPTION_CHARS {
        return Some("a command description is 1 to 100 characters");
    }
    None
}
