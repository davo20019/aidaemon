use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who authored a message in the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Who put a user-role message into the transcript. Policies use this
/// instead of matching text to tell real user input from harness feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOrigin {
    /// Typed by the user (or the calling application).
    User,
    /// Injected by the kernel, such as a stuck-detection nudge.
    Harness,
    /// Injected because an `on_stop` hook rejected a final answer.
    StopHook,
}

/// One tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id. The loop fills one in if the provider sent none.
    pub id: String,
    pub name: String,
    /// Arguments as JSON. Providers that send a string should parse it first.
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// A provider-neutral transcript entry.
///
/// The transcript stored in [`crate::RunState`] never contains the agent's
/// instructions; they are prepended on every request so they survive any
/// future compaction and keep the cacheable prefix stable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Set on `Role::Tool` messages: the call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Set on `Role::Tool` messages whose receipt is not a success, so
    /// providers can flag the result (for example Anthropic's `is_error`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Who injected a `Role::User` message. `None` on other roles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<MessageOrigin>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::from_origin(MessageOrigin::User, content)
    }

    /// A user-role message injected by the harness or a hook.
    pub fn from_origin(origin: MessageOrigin, content: impl Into<String>) -> Self {
        let mut message = Self::plain(Role::User, content);
        message.origin = Some(origin);
        message
    }

    /// True for user-role messages the harness or a hook injected.
    pub fn is_injected(&self) -> bool {
        matches!(
            self.origin,
            Some(MessageOrigin::Harness | MessageOrigin::StopHook)
        )
    }

    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
            is_error: false,
            origin: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
            is_error: false,
            origin: None,
        }
    }

    /// Mark a tool result as an error.
    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }

    fn plain(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            is_error: false,
            origin: None,
        }
    }
}
