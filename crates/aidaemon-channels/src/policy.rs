//! Who may talk to the agent through which channel.

use std::collections::{HashMap, HashSet};

use crate::channel::{ConversationKind, InboundMessage};

/// The default reply to senders who are not on the allowlist.
pub const DEFAULT_REFUSAL: &str = "Sorry, I can only respond to people I have been set up for.";

/// How the runner treats a sender, decided from their platform user id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenderRole {
    Owner,
    Allowed,
}

/// What the runner should do with an inbound message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Hand it to the agent (or command handler).
    Accept(SenderRole),
    /// Drop it silently: a bot, or an unaddressed group message.
    Ignore,
    /// Not allowed: reply with the refusal message (if any) and drop it.
    Refuse,
}

/// Owners and allowlisted users per channel name. Ids are platform user ids
/// (never display names). Everyone else is refused and never reaches the
/// agent; bot senders are always ignored.
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    owners: HashMap<String, HashSet<String>>,
    allowed: HashMap<String, HashSet<String>>,
    refusal: Option<String>,
    require_addressed_in_groups: bool,
    approvals_require_owner: bool,
}

impl Default for AccessPolicy {
    fn default() -> Self {
        Self {
            owners: HashMap::new(),
            allowed: HashMap::new(),
            refusal: Some(DEFAULT_REFUSAL.to_string()),
            require_addressed_in_groups: true,
            approvals_require_owner: false,
        }
    }
}

impl AccessPolicy {
    /// An empty policy: nobody is allowed until added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an owner of `channel` (the channel's [`crate::Channel::name`]).
    pub fn owner(mut self, channel: impl Into<String>, user_id: impl Into<String>) -> Self {
        self.owners
            .entry(channel.into())
            .or_default()
            .insert(user_id.into());
        self
    }

    /// Allow a non-owner user on `channel`.
    pub fn allow(mut self, channel: impl Into<String>, user_id: impl Into<String>) -> Self {
        self.allowed
            .entry(channel.into())
            .or_default()
            .insert(user_id.into());
        self
    }

    /// The reply sent to unknown senders. `None` refuses silently.
    pub fn refusal_message(mut self, message: Option<String>) -> Self {
        self.refusal = message;
        self
    }

    /// In group and thread conversations, handle only messages that address
    /// the bot (`InboundMessage::addressed`). Default: true.
    pub fn require_addressed_in_groups(mut self, required: bool) -> Self {
        self.require_addressed_in_groups = required;
        self
    }

    /// Only owners may answer approval prompts. Default: false (owners and
    /// allowlisted users may).
    pub fn approvals_require_owner(mut self, required: bool) -> Self {
        self.approvals_require_owner = required;
        self
    }

    pub fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }

    pub fn role(&self, channel: &str, user_id: &str) -> Option<SenderRole> {
        let has = |map: &HashMap<String, HashSet<String>>| {
            map.get(channel).is_some_and(|ids| ids.contains(user_id))
        };
        if has(&self.owners) {
            Some(SenderRole::Owner)
        } else if has(&self.allowed) {
            Some(SenderRole::Allowed)
        } else {
            None
        }
    }

    /// May this user settle approval prompts on `channel`?
    pub fn may_approve(&self, channel: &str, user_id: &str) -> bool {
        match self.role(channel, user_id) {
            Some(SenderRole::Owner) => true,
            Some(SenderRole::Allowed) => !self.approvals_require_owner,
            None => false,
        }
    }

    /// Decide what to do with an inbound message. Bots and unaddressed group
    /// messages are ignored before authorization, so the bot never answers
    /// group chatter with refusals.
    pub fn admit(&self, channel: &str, message: &InboundMessage) -> Admission {
        if message.sender.is_bot {
            return Admission::Ignore;
        }
        let grouped = message.conversation.kind != ConversationKind::Direct;
        if grouped && self.require_addressed_in_groups && !message.addressed {
            return Admission::Ignore;
        }
        match self.role(channel, &message.sender.id) {
            Some(role) => Admission::Accept(role),
            None => Admission::Refuse,
        }
    }
}
