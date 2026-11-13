//! The Bot API calls and types the Telegram bot uses beyond the channel
//! contract (reply keyboards, web-app buttons, callback answers, chat
//! actions), sent through the adapter's HTTP client.
//!
//! The shapes follow the Bot API's own JSON: requests are built as JSON
//! bodies and responses read with serde.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use crate::TelegramChannel as Adapter;
use aidaemon_channels::ChannelError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// A failed Bot API call.
#[derive(Debug)]
pub struct RequestError(pub ChannelError);

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for RequestError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChatId(pub i64);

impl std::fmt::Display for ChatId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageId(pub i32);

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UserId(pub u64);

#[derive(Debug, Clone, Deserialize)]
pub struct User {
    pub id: UserId,
    #[serde(default)]
    pub first_name: String,
    #[serde(default)]
    pub last_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Chat {
    pub id: ChatId,
    /// `private`, `group`, `supergroup` or `channel`.
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    title: Option<String>,
    /// Set for public supergroups and channels.
    #[serde(default)]
    pub username: Option<String>,
}

impl Chat {
    pub fn is_private(&self) -> bool {
        self.kind == "private"
    }

    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }
}

/// A document, audio, video or voice file.
#[derive(Debug, Clone, Deserialize)]
pub struct FileMeta {
    pub file_id: String,
    #[serde(default)]
    pub file_size: Option<u64>,
    #[serde(default)]
    pub file_name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WebAppData {
    pub data: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    #[serde(rename = "message_id")]
    pub id: MessageId,
    #[serde(with = "chrono::serde::ts_seconds", default = "epoch")]
    pub date: chrono::DateTime<chrono::Utc>,
    pub chat: Chat,
    #[serde(default)]
    pub from: Option<User>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    document: Option<FileMeta>,
    /// Every size of a picture, smallest first.
    #[serde(default)]
    photo: Option<Vec<FileMeta>>,
    #[serde(default)]
    audio: Option<FileMeta>,
    #[serde(default)]
    video: Option<FileMeta>,
    #[serde(default)]
    voice: Option<FileMeta>,
    #[serde(default)]
    web_app_data: Option<WebAppData>,
}

fn epoch() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::UNIX_EPOCH
}

impl Message {
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }
    pub fn caption(&self) -> Option<&str> {
        self.caption.as_deref()
    }
    pub fn document(&self) -> Option<&FileMeta> {
        self.document.as_ref()
    }
    pub fn photo(&self) -> Option<&[FileMeta]> {
        self.photo.as_deref()
    }
    pub fn audio(&self) -> Option<&FileMeta> {
        self.audio.as_ref()
    }
    pub fn video(&self) -> Option<&FileMeta> {
        self.video.as_ref()
    }
    pub fn voice(&self) -> Option<&FileMeta> {
        self.voice.as_ref()
    }
    pub fn web_app_data(&self) -> Option<&WebAppData> {
        self.web_app_data.as_ref()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CallbackQuery {
    pub id: String,
    pub from: User,
    /// The message the button was on (only its chat and id, if old).
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub data: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Update {
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub callback_query: Option<CallbackQuery>,
}

impl Update {
    /// The chat the update belongs to.
    pub fn chat_id(&self) -> Option<ChatId> {
        self.message.as_ref().map(|m| m.chat.id).or_else(|| {
            self.callback_query
                .as_ref()
                .and_then(|q| q.message.as_ref().map(|m| m.chat.id))
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Me {
    #[serde(default)]
    pub username: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum ParseMode {
    Html,
}

#[derive(Debug, Clone, Copy)]
pub enum ChatAction {
    Typing,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinkPreviewOptions {
    pub is_disabled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct WebAppInfo {
    #[serde(serialize_with = "url_as_str")]
    pub url: reqwest::Url,
}

fn url_as_str<S: serde::Serializer>(url: &reqwest::Url, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(url.as_str())
}

#[derive(Debug, Clone, Serialize)]
pub struct InlineKeyboardButton {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    callback_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_app: Option<WebAppInfo>,
}

impl InlineKeyboardButton {
    pub fn callback(text: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            callback_data: Some(data.into()),
            web_app: None,
        }
    }

    pub fn web_app(text: impl Into<String>, web_app: WebAppInfo) -> Self {
        Self {
            text: text.into(),
            callback_data: None,
            web_app: Some(web_app),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InlineKeyboardMarkup {
    inline_keyboard: Vec<Vec<InlineKeyboardButton>>,
}

impl InlineKeyboardMarkup {
    pub fn new(rows: Vec<Vec<InlineKeyboardButton>>) -> Self {
        Self {
            inline_keyboard: rows,
        }
    }
}

/// A reply-keyboard button that opens a web app.
#[derive(Debug, Clone, Serialize)]
pub struct KeyboardButton {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_app: Option<WebAppInfo>,
}

pub enum ButtonRequest {
    WebApp(WebAppInfo),
}

impl KeyboardButton {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            web_app: None,
        }
    }

    pub fn request(mut self, request: ButtonRequest) -> Self {
        let ButtonRequest::WebApp(info) = request;
        self.web_app = Some(info);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct KeyboardMarkup {
    keyboard: Vec<Vec<KeyboardButton>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    resize_keyboard: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    one_time_keyboard: bool,
}

impl KeyboardMarkup {
    pub fn new(rows: Vec<Vec<KeyboardButton>>) -> Self {
        Self {
            keyboard: rows,
            resize_keyboard: false,
            one_time_keyboard: false,
        }
    }

    pub fn resize_keyboard(mut self) -> Self {
        self.resize_keyboard = true;
        self
    }

    pub fn one_time_keyboard(mut self) -> Self {
        self.one_time_keyboard = true;
        self
    }
}

/// A command in the bot's `/` menu.
pub struct BotCommand {
    command: String,
    description: String,
}

impl BotCommand {
    pub fn new(command: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            description: description.into(),
        }
    }
}

/// The Bot API, through the adapter's client.
#[derive(Clone)]
pub struct Bot {
    api: Arc<Adapter>,
}

impl Bot {
    pub fn new(api: Arc<Adapter>) -> Self {
        Self { api }
    }

    fn request<T>(&self, method: &'static str, body: Value) -> Request<T> {
        Request {
            api: self.api.clone(),
            method,
            body,
            _result: PhantomData,
        }
    }

    pub fn send_message(&self, chat_id: ChatId, text: impl Into<String>) -> Request<Message> {
        self.request(
            "sendMessage",
            json!({ "chat_id": chat_id, "text": text.into() }),
        )
    }

    pub fn edit_message_text(
        &self,
        chat_id: ChatId,
        message_id: MessageId,
        text: impl Into<String>,
    ) -> Request<Value> {
        self.request(
            "editMessageText",
            json!({ "chat_id": chat_id, "message_id": message_id, "text": text.into() }),
        )
    }

    pub fn delete_message(&self, chat_id: ChatId, message_id: MessageId) -> Request<Value> {
        self.request(
            "deleteMessage",
            json!({ "chat_id": chat_id, "message_id": message_id }),
        )
    }

    pub fn send_chat_action(&self, chat_id: ChatId, action: ChatAction) -> Request<Value> {
        let ChatAction::Typing = action;
        self.request(
            "sendChatAction",
            json!({ "chat_id": chat_id, "action": "typing" }),
        )
    }

    pub fn answer_callback_query(&self, id: impl Into<String>) -> Request<Value> {
        self.request(
            "answerCallbackQuery",
            json!({ "callback_query_id": id.into() }),
        )
    }

    pub fn get_me(&self) -> Request<Me> {
        self.request("getMe", json!({}))
    }

    pub fn delete_webhook(&self) -> Request<Value> {
        self.request("deleteWebhook", json!({}))
    }

    pub fn set_my_commands(&self, commands: Vec<BotCommand>) -> Request<Value> {
        let commands: Vec<Value> = commands
            .into_iter()
            .map(|c| json!({ "command": c.command, "description": c.description }))
            .collect();
        self.request("setMyCommands", json!({ "commands": commands }))
    }
}

/// One Bot API call, with optional parameters set before it is awaited.
#[must_use = "a request does nothing until awaited"]
pub struct Request<T> {
    api: Arc<Adapter>,
    method: &'static str,
    body: Value,
    _result: PhantomData<fn() -> T>,
}

impl<T> Clone for Request<T> {
    fn clone(&self) -> Self {
        Self {
            api: self.api.clone(),
            method: self.method,
            body: self.body.clone(),
            _result: PhantomData,
        }
    }
}

impl<T> Request<T> {
    fn with(mut self, key: &str, value: Value) -> Self {
        self.body[key] = value;
        self
    }

    pub fn parse_mode(self, mode: ParseMode) -> Self {
        let ParseMode::Html = mode;
        self.with("parse_mode", json!("HTML"))
    }

    pub fn reply_markup(self, markup: impl Serialize) -> Self {
        let markup = serde_json::to_value(markup).unwrap_or(Value::Null);
        self.with("reply_markup", markup)
    }

    pub fn link_preview_options(self, options: LinkPreviewOptions) -> Self {
        let options = serde_json::to_value(options).unwrap_or(Value::Null);
        self.with("link_preview_options", options)
    }

    /// The text of a callback answer.
    pub fn text(self, text: impl Into<String>) -> Self {
        self.with("text", json!(text.into()))
    }

    /// Show a callback answer as an alert rather than a toast.
    pub fn show_alert(self, show: bool) -> Self {
        self.with("show_alert", json!(show))
    }
}

impl<T: DeserializeOwned + Send + 'static> IntoFuture for Request<T> {
    type Output = Result<T, RequestError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let result = self
                .api
                .call_bot_api(self.method, &self.body)
                .await
                .map_err(RequestError)?;
            serde_json::from_value(result).map_err(|e| {
                RequestError(ChannelError::Other(format!(
                    "{}: unexpected result: {e}",
                    self.method
                )))
            })
        })
    }
}

/// Check a bot token with `getMe`; returns the bot's username.
pub async fn validate_token(token: &str) -> Result<String, String> {
    let test_bot = Bot::new(Arc::new(Adapter::new(token.to_string())));
    let me = test_bot.get_me().await.map_err(|e| {
        format!(
            "Invalid token: {}\n\nMake sure you copied the full token from @BotFather.",
            e
        )
    })?;

    Ok(me.username.clone().unwrap_or_else(|| "unknown".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_parse_whatever_the_key_order() {
        let update: Update = serde_json::from_value(json!({
            "message": {
                "message_id": 10,
                "date": 1700000000,
                "chat": { "id": 7, "type": "private" },
                "from": { "id": 7, "is_bot": false, "first_name": "Alice" },
                "text": "hi"
            },
            "update_id": 1
        }))
        .unwrap();
        let message = update.message.unwrap();
        assert_eq!(message.text(), Some("hi"));
        assert!(message.chat.is_private());
        assert_eq!(message.date.timestamp(), 1700000000);
    }

    #[test]
    fn an_old_callback_message_carries_only_its_chat_and_id() {
        let query: CallbackQuery = serde_json::from_value(json!({
            "id": "cb",
            "from": { "id": 7, "is_bot": false, "first_name": "Alice" },
            "message": { "message_id": 3, "date": 0, "chat": { "id": 7, "type": "private" } },
            "data": "x"
        }))
        .unwrap();
        assert_eq!(query.message.unwrap().id, MessageId(3));
    }

    #[test]
    fn keyboards_serialize_as_the_bot_api_expects() {
        let url = reqwest::Url::parse("https://app.example.test/").unwrap();
        let inline = InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::web_app("Open", WebAppInfo { url: url.clone() }),
            InlineKeyboardButton::callback("Share", "agent:share"),
        ]]);
        assert_eq!(
            serde_json::to_value(inline).unwrap(),
            json!({ "inline_keyboard": [[
                { "text": "Open", "web_app": { "url": "https://app.example.test/" } },
                { "text": "Share", "callback_data": "agent:share" }
            ]] })
        );
        let reply = KeyboardMarkup::new(vec![vec![
            KeyboardButton::new("Open").request(ButtonRequest::WebApp(WebAppInfo { url })),
        ]])
        .resize_keyboard()
        .one_time_keyboard();
        assert_eq!(
            serde_json::to_value(reply).unwrap(),
            json!({
                "keyboard": [[{ "text": "Open", "web_app": { "url": "https://app.example.test/" } }]],
                "resize_keyboard": true,
                "one_time_keyboard": true
            })
        );
    }
}
