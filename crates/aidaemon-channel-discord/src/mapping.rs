//! Gateway payloads to [`ChannelEvent`]s, and the approval `custom_id` codec.

use std::collections::HashSet;

use aidaemon_channels::{
    ApprovalAnswer, Attachment, CommandInvocation, Conversation, ConversationKind, InboundMessage,
    Sender,
};
use serde_json::Value;

/// Discord limit for a component `custom_id`.
pub const CUSTOM_ID_MAX: usize = 100;
const CUSTOM_ID_PREFIX: &str = "aid1:";

/// Message types that carry user text: DEFAULT (0) and REPLY (19).
const USER_MESSAGE_TYPES: [u64; 2] = [0, 19];

/// Channel types that are threads (announcement, public, private).
pub const THREAD_CHANNEL_TYPES: [u64; 3] = [10, 11, 12];

fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// Encode a prompt id and choice id into a button `custom_id`:
/// `aid1:<prompt id length>:<prompt id><choice id>`. The length prefix keeps
/// decoding unambiguous whatever characters the ids contain. Returns `None`
/// when the result would exceed Discord's 100-character limit.
pub fn encode_custom_id(prompt_id: &str, choice_id: &str) -> Option<String> {
    let id = format!(
        "{CUSTOM_ID_PREFIX}{}:{prompt_id}{choice_id}",
        prompt_id.len()
    );
    (id.chars().count() <= CUSTOM_ID_MAX).then_some(id)
}

/// Inverse of [`encode_custom_id`]. `None` for ids this adapter did not make.
pub fn decode_custom_id(custom_id: &str) -> Option<(String, String)> {
    let rest = custom_id.strip_prefix(CUSTOM_ID_PREFIX)?;
    let (len, ids) = rest.split_once(':')?;
    let len: usize = len.parse().ok()?;
    if len > ids.len() || !ids.is_char_boundary(len) {
        return None;
    }
    let (prompt, choice) = ids.split_at(len);
    Some((prompt.to_string(), choice.to_string()))
}

/// Remove `<@id>` and `<@!id>` mentions of the bot and tidy whitespace.
pub fn strip_bot_mentions(text: &str, bot_id: Option<&str>) -> String {
    let Some(bot_id) = bot_id else {
        return text.trim().to_string();
    };
    let stripped = text
        .replace(&format!("<@{bot_id}>"), "")
        .replace(&format!("<@!{bot_id}>"), "");
    // Collapse the gaps a removed mention leaves, line by line.
    stripped
        .lines()
        .map(|line| {
            line.split(' ')
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn sender_from(user: &Value, member: Option<&Value>) -> Option<Sender> {
    let id = str_field(user, "id")?.to_string();
    let display_name = member
        .and_then(|m| str_field(m, "nick"))
        .or_else(|| str_field(user, "global_name"))
        .or_else(|| str_field(user, "username"))
        .map(str::to_string);
    Some(Sender {
        id,
        display_name,
        is_bot: user.get("bot").and_then(Value::as_bool).unwrap_or(false),
    })
}

fn conversation_kind(d: &Value, channel_id: &str, threads: &HashSet<String>) -> ConversationKind {
    if d.get("guild_id").is_none_or(Value::is_null) {
        return ConversationKind::Direct;
    }
    let typed_thread = d
        .get("channel_type")
        .and_then(Value::as_u64)
        .is_some_and(|t| THREAD_CHANNEL_TYPES.contains(&t));
    if typed_thread || threads.contains(channel_id) {
        ConversationKind::Thread
    } else {
        ConversationKind::Group
    }
}

/// Map a MESSAGE_CREATE payload. Returns `None` for bot authors (including
/// this bot), system messages and payloads missing required fields.
pub fn map_message(
    d: &Value,
    bot_id: Option<&str>,
    threads: &HashSet<String>,
) -> Option<InboundMessage> {
    let author = d.get("author")?;
    let sender = sender_from(author, d.get("member"))?;
    if sender.is_bot || Some(sender.id.as_str()) == bot_id {
        return None;
    }
    if d.get("webhook_id").is_some_and(|w| !w.is_null()) {
        return None;
    }
    let msg_type = d.get("type").and_then(Value::as_u64).unwrap_or(0);
    if !USER_MESSAGE_TYPES.contains(&msg_type) {
        return None;
    }
    let channel_id = str_field(d, "channel_id")?;
    let message_id = str_field(d, "id")?.to_string();
    let kind = conversation_kind(d, channel_id, threads);

    let mentioned = bot_id.is_some_and(|bot| {
        d.get("mentions")
            .and_then(Value::as_array)
            .is_some_and(|ms| ms.iter().any(|m| str_field(m, "id") == Some(bot)))
    });
    let reply_to_bot = bot_id.is_some_and(|bot| {
        d.get("referenced_message")
            .and_then(|r| r.get("author"))
            .and_then(|a| str_field(a, "id"))
            == Some(bot)
    });
    let addressed = kind == ConversationKind::Direct || mentioned || reply_to_bot;

    let attachments = d
        .get("attachments")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|a| {
                    Some(Attachment {
                        name: str_field(a, "filename")?.to_string(),
                        mime_type: str_field(a, "content_type").map(str::to_string),
                        size_bytes: a.get("size").and_then(Value::as_u64),
                        source: str_field(a, "url")?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Some(InboundMessage {
        conversation: Conversation {
            id: channel_id.to_string(),
            kind,
        },
        sender,
        message_id,
        text: strip_bot_mentions(str_field(d, "content").unwrap_or(""), bot_id),
        attachments,
        addressed,
    })
}

/// The conversation an interaction happened in.
fn interaction_conversation(d: &Value, threads: &HashSet<String>) -> Option<Conversation> {
    let channel_id =
        str_field(d, "channel_id").or_else(|| d.get("channel").and_then(|c| str_field(c, "id")))?;
    let mut kind = conversation_kind(d, channel_id, threads);
    if kind == ConversationKind::Group
        && d.get("channel")
            .and_then(|c| c.get("type"))
            .and_then(Value::as_u64)
            .is_some_and(|t| THREAD_CHANNEL_TYPES.contains(&t))
    {
        kind = ConversationKind::Thread;
    }
    Some(Conversation {
        id: channel_id.to_string(),
        kind,
    })
}

/// Map an INTERACTION_CREATE payload of type 2 (application command). The
/// handle is `<application id>/<interaction token>`, which answers it.
pub fn map_command(d: &Value, threads: &HashSet<String>) -> Option<CommandInvocation> {
    if d.get("type").and_then(Value::as_u64) != Some(2) {
        return None;
    }
    let data = d.get("data")?;
    let name = str_field(data, "name")?.to_string();
    // Every string option, in order, as the command's arguments.
    let args = data
        .get("options")
        .and_then(Value::as_array)
        .map(|options| {
            options
                .iter()
                .filter_map(|option| option.get("value").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let member = d.get("member");
    let user = member
        .and_then(|m| m.get("user"))
        .or_else(|| d.get("user"))?;
    let sender = sender_from(user, member)?;
    let handle = format!(
        "{}/{}",
        str_field(d, "application_id")?,
        str_field(d, "token")?
    );
    Some(CommandInvocation {
        conversation: interaction_conversation(d, threads)?,
        sender,
        name,
        args: args.trim().to_string(),
        handle,
    })
}

/// Map an INTERACTION_CREATE payload of type 3 (message component) whose
/// `custom_id` this adapter produced.
pub fn map_interaction(d: &Value, threads: &HashSet<String>) -> Option<ApprovalAnswer> {
    if d.get("type").and_then(Value::as_u64) != Some(3) {
        return None;
    }
    let custom_id = d
        .get("data")
        .and_then(|data| str_field(data, "custom_id"))?;
    let (prompt_id, choice_id) = decode_custom_id(custom_id)?;
    let member = d.get("member");
    let user = member
        .and_then(|m| m.get("user"))
        .or_else(|| d.get("user"))?;
    let sender = sender_from(user, member)?;
    Some(ApprovalAnswer {
        prompt_id,
        choice_id,
        sender,
        conversation: interaction_conversation(d, threads)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BOT: &str = "100000000000000009";

    fn alice() -> Value {
        json!({"id": "100000000000000001", "username": "alice", "global_name": "Alice Rivera"})
    }

    fn guild_msg(content: &str) -> Value {
        json!({
            "id": "100000000000000100",
            "type": 0,
            "channel_id": "100000000000000200",
            "guild_id": "100000000000000300",
            "author": alice(),
            "content": content,
            "mentions": [],
        })
    }

    #[test]
    fn dm_is_direct_and_addressed() {
        let mut d = guild_msg("hi");
        d.as_object_mut().unwrap().remove("guild_id");
        let m = map_message(&d, Some(BOT), &HashSet::new()).unwrap();
        assert_eq!(m.conversation.kind, ConversationKind::Direct);
        assert!(m.addressed);
        assert_eq!(m.sender.display_name.as_deref(), Some("Alice Rivera"));
        assert_eq!(m.sender.id, "100000000000000001");
    }

    #[test]
    fn guild_message_addressed_only_by_structural_mention() {
        let plain = map_message(&guild_msg("hello there"), Some(BOT), &HashSet::new()).unwrap();
        assert_eq!(plain.conversation.kind, ConversationKind::Group);
        assert!(!plain.addressed);

        // A mention token in the text alone is not enough; `mentions` decides.
        let mut d = guild_msg(&format!("<@{BOT}> deploy please"));
        let unlisted = map_message(&d, Some(BOT), &HashSet::new()).unwrap();
        assert!(!unlisted.addressed);
        d["mentions"] = json!([{"id": BOT, "username": "aidaemon", "bot": true}]);
        let m = map_message(&d, Some(BOT), &HashSet::new()).unwrap();
        assert!(m.addressed);
        assert_eq!(m.text, "deploy please");
    }

    #[test]
    fn strips_nickname_mentions_and_keeps_other_mentions() {
        let text = strip_bot_mentions(
            &format!("hey <@!{BOT}>  ask <@100000000000000001>\nnext <@{BOT}>"),
            Some(BOT),
        );
        assert_eq!(text, "hey ask <@100000000000000001>\nnext");
    }

    #[test]
    fn reply_to_bot_is_addressed() {
        let mut d = guild_msg("and then?");
        d["type"] = json!(19);
        d["referenced_message"] =
            json!({"id": "100000000000000099", "author": {"id": BOT, "bot": true}});
        let m = map_message(&d, Some(BOT), &HashSet::new()).unwrap();
        assert!(m.addressed);

        d["referenced_message"]["author"] = alice();
        assert!(
            !map_message(&d, Some(BOT), &HashSet::new())
                .unwrap()
                .addressed
        );
    }

    #[test]
    fn bot_authors_and_self_are_skipped() {
        let mut d = guild_msg("beep");
        d["author"] = json!({"id": "100000000000000005", "bot": true});
        assert!(map_message(&d, Some(BOT), &HashSet::new()).is_none());
        d["author"] = json!({"id": BOT});
        assert!(map_message(&d, Some(BOT), &HashSet::new()).is_none());
    }

    #[test]
    fn system_messages_are_skipped() {
        let mut d = guild_msg("");
        d["type"] = json!(7); // member join
        assert!(map_message(&d, Some(BOT), &HashSet::new()).is_none());
    }

    #[test]
    fn threads_detected_from_known_set_or_channel_type() {
        let d = guild_msg("in a thread");
        let mut threads = HashSet::new();
        threads.insert("100000000000000200".to_string());
        assert_eq!(
            map_message(&d, Some(BOT), &threads)
                .unwrap()
                .conversation
                .kind,
            ConversationKind::Thread
        );
        let mut typed = guild_msg("typed");
        typed["channel_type"] = json!(11);
        assert_eq!(
            map_message(&typed, Some(BOT), &HashSet::new())
                .unwrap()
                .conversation
                .kind,
            ConversationKind::Thread
        );
    }

    #[test]
    fn attachments_are_mapped() {
        let mut d = guild_msg("see file");
        d["attachments"] = json!([{
            "id": "100000000000000400",
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "size": 2048,
            "url": "https://cdn.example.invalid/report.pdf"
        }]);
        let m = map_message(&d, Some(BOT), &HashSet::new()).unwrap();
        assert_eq!(
            m.attachments,
            vec![Attachment {
                name: "report.pdf".into(),
                mime_type: Some("application/pdf".into()),
                size_bytes: Some(2048),
                source: "https://cdn.example.invalid/report.pdf".into(),
            }]
        );
    }

    #[test]
    fn custom_id_round_trips_with_separators() {
        let id = encode_custom_id("run:42:tool", "allow:once").unwrap();
        assert!(id.len() <= CUSTOM_ID_MAX);
        assert_eq!(
            decode_custom_id(&id),
            Some(("run:42:tool".to_string(), "allow:once".to_string()))
        );
        assert!(decode_custom_id("someone_else:button").is_none());
        assert!(encode_custom_id(&"x".repeat(95), "allow").is_none());
    }

    #[test]
    fn interaction_maps_guild_member_user() {
        let d = json!({
            "id": "100000000000000500",
            "type": 3,
            "token": "synthetic-interaction-token",
            "guild_id": "100000000000000300",
            "channel_id": "100000000000000200",
            "member": {"nick": "Ali", "user": alice()},
            "data": {"custom_id": encode_custom_id("p1", "deny").unwrap(), "component_type": 2}
        });
        let a = map_interaction(&d, &HashSet::new()).unwrap();
        assert_eq!(a.prompt_id, "p1");
        assert_eq!(a.choice_id, "deny");
        assert_eq!(a.sender.display_name.as_deref(), Some("Ali"));
        assert_eq!(a.conversation.kind, ConversationKind::Group);
    }
}
