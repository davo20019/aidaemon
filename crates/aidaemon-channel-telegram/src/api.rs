//! Minimal Telegram Bot API client: the types and methods this adapter uses.

use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use aidaemon_channels::ChannelError;

/// An error returned by a Bot API call.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ApiError {
    /// Telegram answered with `ok: false`.
    #[error("telegram API error {code}: {description}")]
    Api {
        code: u16,
        description: String,
        retry_after: Option<u64>,
    },
    /// The request did not complete (connect, timeout, bad body).
    #[error("telegram transport error: {0}")]
    Transport(String),
}

impl ApiError {
    pub(crate) fn code(&self) -> Option<u16> {
        match self {
            ApiError::Api { code, .. } => Some(*code),
            ApiError::Transport(_) => None,
        }
    }

    pub(crate) fn description_contains(&self, needle: &str) -> bool {
        match self {
            ApiError::Api { description, .. } => description.to_ascii_lowercase().contains(needle),
            ApiError::Transport(_) => false,
        }
    }

    pub(crate) fn is_parse_error(&self) -> bool {
        self.code() == Some(400)
            && (self.description_contains("can't parse entities")
                || self.description_contains("can't find end of the entity")
                || self.description_contains("unsupported start tag")
                || self.description_contains("unexpected end tag"))
    }

    pub(crate) fn is_not_modified(&self) -> bool {
        self.code() == Some(400) && self.description_contains("message is not modified")
    }

    /// True for errors that mean the token is not valid (401, or 404 which
    /// Telegram returns for a malformed token).
    pub(crate) fn is_auth(&self) -> bool {
        matches!(self.code(), Some(401) | Some(404))
    }
}

impl From<ApiError> for ChannelError {
    fn from(e: ApiError) -> Self {
        match e {
            ApiError::Api {
                code: 401 | 404,
                description,
                ..
            } => ChannelError::Auth(description),
            ApiError::Api {
                code: 429,
                retry_after,
                ..
            } => ChannelError::RateLimited {
                retry_after_secs: retry_after,
            },
            ApiError::Api {
                code, description, ..
            } => ChannelError::Other(format!("telegram API error {code}: {description}")),
            ApiError::Transport(msg) => ChannelError::Transport(msg),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Envelope {
    ok: bool,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error_code: Option<u16>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<ResponseParameters>,
}

#[derive(Debug, Deserialize)]
struct ResponseParameters {
    #[serde(default)]
    retry_after: Option<u64>,
}

#[derive(Clone)]
pub(crate) struct BotApi {
    client: reqwest::Client,
    base: String,
    /// Where `getFile` paths are downloaded from (holds the token).
    file_base: String,
}

impl BotApi {
    pub(crate) fn new(client: reqwest::Client, api_base: &str, token: &str) -> Self {
        let api_base = api_base.trim_end_matches('/');
        Self {
            client,
            base: format!("{api_base}/bot{token}"),
            file_base: format!("{api_base}/file/bot{token}"),
        }
    }

    pub(crate) async fn call<T: DeserializeOwned>(
        &self,
        method: &str,
        body: &Value,
        timeout: Duration,
    ) -> Result<T, ApiError> {
        let url = format!("{}/{}", self.base, method);
        let request = self.client.post(&url).json(body).timeout(timeout);
        self.finish(request).await
    }

    /// A call with a `multipart/form-data` body (file uploads).
    pub(crate) async fn call_multipart<T: DeserializeOwned>(
        &self,
        method: &str,
        form: reqwest::multipart::Form,
        timeout: Duration,
    ) -> Result<T, ApiError> {
        let url = format!("{}/{}", self.base, method);
        let request = self.client.post(&url).multipart(form).timeout(timeout);
        self.finish(request).await
    }

    /// Where a `file_path` from `getFile` is fetched. Holds the token.
    pub(crate) fn file_url(&self, file_path: &str) -> String {
        format!("{}/{}", self.file_base, file_path.trim_start_matches('/'))
    }

    async fn finish<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, ApiError> {
        let resp = request
            .send()
            .await
            .map_err(|e| ApiError::Transport(redact(&e.without_url().to_string())))?;
        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ApiError::Transport(redact(&e.without_url().to_string())))?;
        let envelope: Envelope = match serde_json::from_slice(&bytes) {
            Ok(env) => env,
            Err(_) if !status.is_success() => {
                return Err(ApiError::Api {
                    code: status.as_u16(),
                    description: status.canonical_reason().unwrap_or("error").to_string(),
                    retry_after: None,
                })
            }
            Err(e) => return Err(ApiError::Transport(format!("invalid response body: {e}"))),
        };
        if !envelope.ok {
            return Err(ApiError::Api {
                code: envelope.error_code.unwrap_or(status.as_u16()),
                description: envelope.description.unwrap_or_default(),
                retry_after: envelope.parameters.and_then(|p| p.retry_after),
            });
        }
        serde_json::from_value(envelope.result.unwrap_or(Value::Null))
            .map_err(|e| ApiError::Transport(format!("unexpected result shape: {e}")))
    }
}

/// Never echo anything that looks like the bot path (which holds the token).
fn redact(msg: &str) -> String {
    match msg.find("/bot") {
        Some(pos) => format!("{}/bot<redacted>", &msg[..pos]),
        None => msg.to_string(),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct User {
    pub id: i64,
    #[serde(default)]
    pub is_bot: bool,
    #[serde(default)]
    pub first_name: String,
    #[serde(default)]
    pub last_name: Option<String>,
    #[serde(default)]
    pub username: Option<String>,
}

impl User {
    pub(crate) fn display_name(&self) -> Option<String> {
        let name = match &self.last_name {
            Some(last) if !last.is_empty() => format!("{} {}", self.first_name, last),
            _ => self.first_name.clone(),
        };
        if name.trim().is_empty() {
            self.username.clone()
        } else {
            Some(name)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Chat {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessageEntity {
    #[serde(rename = "type")]
    pub kind: String,
    pub offset: usize,
    pub length: usize,
    #[serde(default)]
    pub user: Option<User>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PhotoSize {
    pub file_id: String,
    #[serde(default)]
    pub file_unique_id: String,
    #[serde(default)]
    pub file_size: Option<u64>,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FileLike {
    pub file_id: String,
    #[serde(default)]
    pub file_unique_id: String,
    #[serde(default)]
    pub file_name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub file_size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Message {
    pub message_id: i64,
    #[serde(default)]
    pub message_thread_id: Option<i64>,
    #[serde(default)]
    pub is_topic_message: bool,
    #[serde(default)]
    pub from: Option<User>,
    #[serde(default)]
    pub sender_chat: Option<Chat>,
    pub chat: Chat,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub entities: Vec<MessageEntity>,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default)]
    pub caption_entities: Vec<MessageEntity>,
    #[serde(default)]
    pub reply_to_message: Option<Box<Message>>,
    #[serde(default)]
    pub forum_topic_created: Option<Value>,
    #[serde(default)]
    pub photo: Vec<PhotoSize>,
    #[serde(default)]
    pub document: Option<FileLike>,
    #[serde(default)]
    pub audio: Option<FileLike>,
    #[serde(default)]
    pub voice: Option<FileLike>,
    #[serde(default)]
    pub video: Option<FileLike>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CallbackQuery {
    pub id: String,
    pub from: User,
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub data: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Update {
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub callback_query: Option<CallbackQuery>,
}

/// The subset of a sent message we read back.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SentMessage {
    pub message_id: i64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct InlineKeyboardButton {
    pub text: String,
    pub callback_data: String,
}
