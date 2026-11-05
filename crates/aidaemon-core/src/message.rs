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

/// Who put a message into the transcript. Policies use this instead of
/// matching text to tell real user input from harness feedback, and
/// providers use it to tell a hook's per-request note from the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOrigin {
    /// Typed by the user (or the calling application).
    User,
    /// Injected by the kernel, such as a stuck-detection nudge.
    Harness,
    /// Injected because an `on_stop` hook rejected a final answer.
    StopHook,
    /// A system note a `before_model` hook added to one model request (for
    /// example recalled memory); it is not part of the run's transcript.
    HookNote,
}

/// One tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id. The loop fills one in if the provider sent none.
    pub id: String,
    pub name: String,
    /// Arguments as JSON. Providers that send a string should parse it first.
    pub arguments: Value,
    /// Opaque data the same provider needs back when this call is replayed
    /// in a later request (a reasoning signature, thinking blocks). Other
    /// providers ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_data: Option<Value>,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            provider_data: None,
        }
    }

    pub fn with_provider_data(mut self, data: Value) -> Self {
        self.provider_data = Some(data);
        self
    }
}

/// An image the model should look at. Carried on user messages and on tool
/// results; providers map it to their own wire format and ignore it on other
/// roles. `Debug` omits the encoded bytes so transcripts stay loggable.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum Image {
    /// Inline bytes, base64-encoded (standard alphabet, with padding).
    Base64 { media_type: String, data: String },
    /// A URL the provider fetches itself.
    Url { url: String },
}

impl Image {
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Base64 {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    /// The image as a URL: the URL itself, or a `data:` URL for inline bytes.
    pub fn to_url(&self) -> String {
        match self {
            Self::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
            Self::Url { url } => url.clone(),
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base64 { media_type, data } => f
                .debug_struct("Base64")
                .field("media_type", media_type)
                .field("bytes_b64", &data.len())
                .finish(),
            Self::Url { url } => f.debug_struct("Url").field("url", url).finish(),
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
    /// Images on `Role::User` and `Role::Tool` messages, in order after the
    /// text. Providers ignore them on other roles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    /// Set on `Role::Tool` messages whose receipt is not a success, so
    /// providers can flag the result (for example Anthropic's `is_error`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Who injected a `Role::User` message, or [`MessageOrigin::HookNote`]
    /// on a hook's system note. `None` otherwise.
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

    /// A system note a `before_model` hook adds to one request. Providers
    /// that place notes apart from the transcript find them by this origin.
    pub fn hook_note(content: impl Into<String>) -> Self {
        let mut message = Self::plain(Role::System, content);
        message.origin = Some(MessageOrigin::HookNote);
        message
    }

    /// True for a [`Message::hook_note`].
    pub fn is_hook_note(&self) -> bool {
        self.origin == Some(MessageOrigin::HookNote)
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
            images: Vec::new(),
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
            images: Vec::new(),
            is_error: false,
            origin: None,
        }
    }

    /// Attach images (see [`Image`]).
    pub fn with_images(mut self, images: impl IntoIterator<Item = Image>) -> Self {
        self.images.extend(images);
        self
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
            images: Vec::new(),
            is_error: false,
            origin: None,
        }
    }
}

/// Plain text becomes a user message, so `Agent::run` and
/// `RunState::with_user_message` take either text or a built [`Message`]
/// (for example one with images).
impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Self::user(text)
    }
}

impl From<&String> for Message {
    fn from(text: &String) -> Self {
        Self::user(text.as_str())
    }
}

impl From<String> for Message {
    fn from(text: String) -> Self {
        Self::user(text)
    }
}
