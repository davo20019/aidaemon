//! Who a conversation reaches and who is speaking: the shared vocabulary
//! hosts, tools, channels and memory use to scope what an agent may see and
//! remember.

/// Visibility level of the channel the message originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelVisibility {
    /// Direct message (1-on-1). Full memory, no restrictions.
    Private,
    /// Small private group (e.g., Telegram group, Slack MPIM). Cautious with sensitive info.
    PrivateGroup,
    /// Public channel visible to many users. No personal memory injected.
    Public,
    /// Untrusted public platform (Twitter, public APIs). Hardened security, minimal memory.
    PublicExternal,
    /// Internal/system-initiated (scheduler, triggers, sub-agents default). Full memory.
    Internal,
}

impl std::fmt::Display for ChannelVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelVisibility::Private => write!(f, "private"),
            ChannelVisibility::PrivateGroup => write!(f, "private_group"),
            ChannelVisibility::Public => write!(f, "public"),
            ChannelVisibility::PublicExternal => write!(f, "public_external"),
            ChannelVisibility::Internal => write!(f, "internal"),
        }
    }
}

impl ChannelVisibility {
    /// Parse from a string (e.g., from tool args). Falls back to `Internal`.
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "private" => Self::Private,
            "private_group" => Self::PrivateGroup,
            "public" => Self::Public,
            "public_external" => Self::PublicExternal,
            "internal" => Self::Internal,
            _ => Self::Internal,
        }
    }
}

/// Role assigned to a user based on config (owner_ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserRole {
    /// Full access — listed in `users.owner_ids`.
    Owner,
    /// Allowed by channel allowlist but not an owner.
    Guest,
    /// Non-whitelisted user — conversational access only, no tools.
    Public,
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserRole::Owner => write!(f, "owner"),
            UserRole::Guest => write!(f, "guest"),
            UserRole::Public => write!(f, "public"),
        }
    }
}

impl UserRole {
    /// Only explicit owner traffic is allowed to update durable owner memory.
    pub fn can_persist_owner_memory(self) -> bool {
        matches!(self, Self::Owner)
    }
}
