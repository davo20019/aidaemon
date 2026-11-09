//! Translation of Telegram updates into channel events.

use aidaemon_channels::{
    Attachment, AttachmentKind, Conversation, ConversationKind, InboundMessage, Sender,
};

use crate::api::{Chat, Message, MessageEntity, User};

/// Who the bot is, for mention and reply detection.
#[derive(Debug, Clone, Default)]
pub(crate) struct BotIdentity {
    pub id: Option<i64>,
    /// Without the leading `@`.
    pub username: String,
}

impl BotIdentity {
    fn is_me(&self, user: &User) -> bool {
        if let Some(id) = self.id {
            return user.id == id;
        }
        user.is_bot
            && user
                .username
                .as_deref()
                .is_some_and(|u| u.eq_ignore_ascii_case(&self.username))
    }

    fn is_my_username(&self, name: &str) -> bool {
        !self.username.is_empty() && name.eq_ignore_ascii_case(&self.username)
    }
}

/// Conversation for a message's chat (and forum topic, if any).
pub(crate) fn conversation_for(msg: &Message) -> Conversation {
    conversation_for_chat(&msg.chat, msg.is_topic_message, msg.message_thread_id)
}

pub(crate) fn conversation_for_chat(
    chat: &Chat,
    is_topic: bool,
    thread_id: Option<i64>,
) -> Conversation {
    match (chat.kind.as_str(), is_topic, thread_id) {
        ("private", _, _) => Conversation {
            id: chat.id.to_string(),
            kind: ConversationKind::Direct,
        },
        (_, true, Some(thread)) => Conversation {
            id: format!("{}:{}", chat.id, thread),
            kind: ConversationKind::Thread,
        },
        _ => Conversation {
            id: chat.id.to_string(),
            kind: ConversationKind::Group,
        },
    }
}

/// Parse a conversation id back into `(chat_id, message_thread_id)`.
pub(crate) fn parse_conversation_id(id: &str) -> Option<(i64, Option<i64>)> {
    match id.split_once(':') {
        Some((chat, thread)) => Some((chat.parse().ok()?, Some(thread.parse().ok()?))),
        None => Some((id.parse().ok()?, None)),
    }
}

pub(crate) fn sender_for(user: Option<&User>, sender_chat: Option<&Chat>) -> Option<Sender> {
    if let Some(chat) = sender_chat {
        // Anonymous admins and channel posts speak as a chat.
        if user.is_none_or(|u| u.is_bot) {
            return Some(Sender {
                id: chat.id.to_string(),
                display_name: chat.title.clone(),
                is_bot: false,
            });
        }
    }
    let user = user?;
    Some(Sender {
        id: user.id.to_string(),
        display_name: user.display_name(),
        is_bot: user.is_bot,
    })
}

/// Byte range of a UTF-16 `(offset, length)` span in `text`.
fn utf16_span(text: &str, offset: usize, length: usize) -> Option<(usize, usize)> {
    let end_units = offset.checked_add(length)?;
    let mut units = 0;
    let mut start = None;
    for (byte, c) in text.char_indices() {
        if units == offset && start.is_none() {
            start = Some(byte);
        }
        if units == end_units {
            return Some((start?, byte));
        }
        units += c.len_utf16();
    }
    if units == offset && start.is_none() {
        start = Some(text.len());
    }
    if units == end_units {
        return Some((start?, text.len()));
    }
    None
}

/// Map a Telegram message to an inbound message. Returns `None` for
/// messages with neither text nor attachments (service messages, stickers).
pub(crate) fn map_message(msg: &Message, bot: &BotIdentity) -> Option<InboundMessage> {
    let (raw_text, entities): (&str, &[MessageEntity]) = match (&msg.text, &msg.caption) {
        (Some(text), _) => (text, &msg.entities),
        (None, Some(caption)) => (caption, &msg.caption_entities),
        (None, None) => ("", &[]),
    };
    let attachments = attachments_for(msg);
    if raw_text.trim().is_empty() && attachments.is_empty() {
        return None;
    }
    let sender = sender_for(msg.from.as_ref(), msg.sender_chat.as_ref())?;
    let conversation = conversation_for(msg);

    let mut addressed = conversation.kind == ConversationKind::Direct;
    // Byte ranges to cut out of the text (the bot's own mention).
    let mut cuts: Vec<(usize, usize)> = Vec::new();

    for entity in entities {
        let Some((start, end)) = utf16_span(raw_text, entity.offset, entity.length) else {
            continue;
        };
        let span = &raw_text[start..end];
        match entity.kind.as_str() {
            "mention" if bot.is_my_username(span.trim_start_matches('@')) => {
                addressed = true;
                cuts.push((start, end));
            }
            "text_mention" if entity.user.as_ref().is_some_and(|u| bot.is_me(u)) => {
                addressed = true;
            }
            "bot_command" => {
                // `/cmd` or `/cmd@botname`.
                match span.split_once('@') {
                    Some((_, target)) if bot.is_my_username(target) => {
                        addressed = true;
                        let at = start + span.find('@').unwrap_or(span.len());
                        cuts.push((at, end));
                    }
                    Some(_) => {} // aimed at another bot
                    None => {
                        if entity.offset == 0 {
                            addressed = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(reply) = &msg.reply_to_message {
        // In forum topics every message "replies" to the topic-creation
        // service message; that is not a reply to the bot.
        if reply.forum_topic_created.is_none() && reply.from.as_ref().is_some_and(|u| bot.is_me(u))
        {
            addressed = true;
        }
    }

    let text = cut_spans(raw_text, &mut cuts);

    // A forum topic's service message is not something this one answers.
    let parent = msg
        .reply_to_message
        .as_ref()
        .filter(|reply| reply.forum_topic_created.is_none())
        .map(|reply| reply.message_id.to_string());
    let thread = msg
        .is_topic_message
        .then_some(msg.message_thread_id)
        .flatten()
        .map(|thread| thread.to_string());

    Some(
        InboundMessage::new(conversation, sender, msg.message_id.to_string(), text)
            .with_attachments(attachments)
            .with_addressed(addressed)
            .with_parent_message_id(parent)
            .with_thread_id(thread),
    )
}

fn cut_spans(text: &str, cuts: &mut [(usize, usize)]) -> String {
    if cuts.is_empty() {
        return text.to_string();
    }
    cuts.sort_unstable();
    let mut out = String::with_capacity(text.len());
    let mut pos = 0;
    for &(start, end) in cuts.iter() {
        if start < pos {
            continue;
        }
        out.push_str(&text[pos..start]);
        pos = end;
        // Swallow one space after a removed mention so "@bot hi" -> "hi".
        if text[pos..].starts_with(' ') && (out.is_empty() || out.ends_with(' ')) {
            pos += 1;
        }
    }
    out.push_str(&text[pos..]);
    out.trim().to_string()
}

fn attachments_for(msg: &Message) -> Vec<Attachment> {
    let mut out = Vec::new();
    // Telegram sends several sizes; the last is the largest.
    if let Some(photo) = msg
        .photo
        .iter()
        .max_by_key(|p| (u64::from(p.width) * u64::from(p.height), p.file_size))
    {
        let id = if photo.file_unique_id.is_empty() {
            &photo.file_id
        } else {
            &photo.file_unique_id
        };
        out.push(
            Attachment::new(format!("photo_{id}.jpg"), photo.file_id.clone())
                .with_mime_type(Some("image/jpeg".to_string()))
                .with_size_bytes(photo.file_size)
                .with_kind(AttachmentKind::Image),
        );
    }
    // A document says what it is by its MIME type; the others by their field.
    let files = [
        (&msg.document, "document", None, None),
        (
            &msg.audio,
            "audio",
            Some("audio/mpeg"),
            Some(AttachmentKind::Audio),
        ),
        (
            &msg.voice,
            "voice",
            Some("audio/ogg"),
            Some(AttachmentKind::Audio),
        ),
        (
            &msg.video,
            "video",
            Some("video/mp4"),
            Some(AttachmentKind::Video),
        ),
    ];
    for (file, kind, default_mime, typed) in files {
        let Some(file) = file else { continue };
        let name = file.file_name.clone().unwrap_or_else(|| {
            let id = if file.file_unique_id.is_empty() {
                &file.file_id
            } else {
                &file.file_unique_id
            };
            format!("{kind}_{id}")
        });
        let mime_type = file
            .mime_type
            .clone()
            .or_else(|| default_mime.map(str::to_string));
        let mut attachment = Attachment::new(name, file.file_id.clone())
            .with_mime_type(mime_type)
            .with_size_bytes(file.file_size);
        if let Some(typed) = typed {
            attachment = attachment.with_kind(typed);
        }
        out.push(attachment);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_spans() {
        let text = "😀 @bot hi";
        // 😀 is 2 UTF-16 units; "@bot" starts at unit 3.
        assert_eq!(utf16_span(text, 3, 4), Some((5, 9)));
        assert_eq!(&text[5..9], "@bot");
        assert_eq!(utf16_span(text, 0, 2), Some((0, 4)));
        assert_eq!(utf16_span(text, 1, 1), None);
        assert_eq!(utf16_span(text, 8, 2), Some((10, 12)));
        assert_eq!(utf16_span(text, 8, 50), None);
    }

    #[test]
    fn conversation_ids_roundtrip() {
        assert_eq!(parse_conversation_id("100000001"), Some((100000001, None)));
        assert_eq!(
            parse_conversation_id("-1000000000001:42"),
            Some((-1000000000001, Some(42)))
        );
        assert_eq!(parse_conversation_id("nope"), None);
    }

    #[test]
    fn cut_mention() {
        let mut cuts = vec![(0, 4)];
        assert_eq!(cut_spans("@bot hello", &mut cuts), "hello");
        let mut cuts = vec![(6, 10)];
        assert_eq!(cut_spans("hello @bot", &mut cuts), "hello");
        let mut cuts = vec![(4, 8)];
        assert_eq!(cut_spans("hey @bot there", &mut cuts), "hey there");
    }
}
