//! Translation of Socket Mode payloads into [`ChannelEvent`]s.
//!
//! Pure functions over JSON so they can be tested without a socket.

use std::collections::{HashSet, VecDeque};

use aidaemon_channels::{
    ApprovalAnswer, Attachment, ChannelEvent, Conversation, ConversationKind, InboundMessage,
    Sender,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Prefix of the `action_id` on approval buttons rendered by `ask`.
pub(crate) const APPROVAL_ACTION_PREFIX: &str = "aidaemon_approval";

/// What a button's `value` carries: the prompt id and the choice id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ApprovalValue {
    #[serde(rename = "p")]
    pub prompt_id: String,
    #[serde(rename = "c")]
    pub choice_id: String,
}

/// Remembers recently seen `channel:ts` pairs. Slack delivers a message that
/// mentions the bot twice (`message` and `app_mention`); only the first one
/// is forwarded.
#[derive(Debug, Default)]
pub(crate) struct SeenMessages {
    order: VecDeque<String>,
    set: HashSet<String>,
}

impl SeenMessages {
    const CAPACITY: usize = 1024;

    /// Returns true the first time a key is seen.
    pub(crate) fn insert(&mut self, key: String) -> bool {
        if self.set.contains(&key) {
            return false;
        }
        if self.order.len() >= Self::CAPACITY {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        self.set.insert(key.clone());
        self.order.push_back(key);
        true
    }
}

pub(crate) struct MapContext<'a> {
    pub bot_user_id: Option<&'a str>,
    pub reply_in_thread: bool,
}

fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Parse a conversation id (`channel` or `channel:thread_ts`).
pub(crate) fn split_conversation_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once(':') {
        Some((channel, ts)) if !ts.is_empty() => (channel, Some(ts)),
        Some((channel, _)) => (channel, None),
        None => (id, None),
    }
}

fn conversation_for(
    channel: &str,
    is_dm: bool,
    thread_ts: Option<&str>,
    new_thread_ts: Option<&str>,
) -> Conversation {
    if is_dm {
        return Conversation {
            id: match thread_ts {
                Some(ts) => format!("{channel}:{ts}"),
                None => channel.to_string(),
            },
            kind: ConversationKind::Direct,
        };
    }
    match thread_ts.or(new_thread_ts) {
        Some(ts) => Conversation {
            id: format!("{channel}:{ts}"),
            kind: ConversationKind::Thread,
        },
        None => Conversation {
            id: channel.to_string(),
            kind: ConversationKind::Group,
        },
    }
}

/// True if `text` contains the structural mention token for `bot_id`
/// (`<@BOTID>` or `<@BOTID|label>`).
pub(crate) fn mentions(text: &str, bot_id: &str) -> bool {
    text.contains(&format!("<@{bot_id}>")) || text.contains(&format!("<@{bot_id}|"))
}

/// Remove every `<@BOTID>` / `<@BOTID|label>` token and tidy whitespace.
pub(crate) fn strip_mention(text: &str, bot_id: &str) -> String {
    let open = format!("<@{bot_id}");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(&open) {
        let after = &rest[pos + open.len()..];
        let token_end = match after.chars().next() {
            Some('>') => Some(1),
            Some('|') => after.find('>').map(|e| e + 1),
            _ => None,
        };
        match token_end {
            Some(end) => {
                out.push_str(&rest[..pos]);
                rest = &after[end..];
            }
            None => {
                out.push_str(&rest[..pos + open.len()]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    // Collapse the double space a mid-sentence mention leaves behind.
    let mut tidy = String::with_capacity(out.len());
    let mut prev_space = false;
    for c in out.trim().chars() {
        if c == ' ' {
            if prev_space {
                continue;
            }
            prev_space = true;
        } else {
            prev_space = false;
        }
        tidy.push(c);
    }
    tidy
}

fn attachments(event: &Value) -> Vec<Attachment> {
    event
        .get("files")
        .and_then(Value::as_array)
        .map(|files| {
            files
                .iter()
                .filter_map(|file| {
                    let source = str_field(file, "url_private")?;
                    Some(Attachment {
                        name: str_field(file, "name")
                            .or_else(|| str_field(file, "title"))
                            .unwrap_or("file")
                            .to_string(),
                        mime_type: str_field(file, "mimetype").map(str::to_string),
                        size_bytes: file.get("size").and_then(Value::as_u64),
                        source: source.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Map an `events_api` payload. Returns `None` for events the runner should
/// never see (bot messages, edits, other subtypes, duplicates).
pub(crate) fn map_events_api(
    payload: &Value,
    ctx: &MapContext<'_>,
    seen: &mut SeenMessages,
) -> Option<ChannelEvent> {
    let event = payload.get("event")?;
    let event_type = str_field(event, "type")?;
    if event_type != "message" && event_type != "app_mention" {
        return None;
    }
    if event.get("bot_id").is_some_and(|v| !v.is_null()) {
        return None;
    }
    match str_field(event, "subtype") {
        None | Some("file_share") => {}
        Some(_) => return None,
    }
    if event.get("edited").is_some_and(|v| !v.is_null()) {
        return None;
    }

    let user = str_field(event, "user")?;
    if ctx.bot_user_id == Some(user) {
        return None;
    }
    let channel = str_field(event, "channel")?;
    let ts = str_field(event, "ts")?;

    let raw_text = event.get("text").and_then(Value::as_str).unwrap_or("");
    let files = attachments(event);
    let mentioned = ctx.bot_user_id.is_some_and(|bot| mentions(raw_text, bot));
    let text = match ctx.bot_user_id {
        Some(bot) => strip_mention(raw_text, bot),
        None => raw_text.trim().to_string(),
    };
    if text.is_empty() && files.is_empty() {
        return None;
    }

    if !seen.insert(format!("{channel}:{ts}")) {
        return None;
    }

    let is_dm = match str_field(event, "channel_type") {
        Some(kind) => kind == "im",
        // `app_mention` carries no channel_type; DM ids start with `D`.
        None => channel.starts_with('D'),
    };
    let addressed = is_dm || event_type == "app_mention" || mentioned;
    let thread_ts = str_field(event, "thread_ts");
    // The bot answers an addressed top-level channel message in a new
    // thread rooted at it, so that thread is the conversation.
    let new_thread = (ctx.reply_in_thread && addressed).then_some(ts);
    let conversation = conversation_for(channel, is_dm, thread_ts, new_thread);

    let display_name = event.get("user_profile").and_then(|profile| {
        str_field(profile, "display_name")
            .or_else(|| str_field(profile, "real_name"))
            .map(str::to_string)
    });

    Some(ChannelEvent::Message(InboundMessage {
        conversation,
        sender: Sender {
            id: user.to_string(),
            display_name,
            is_bot: false,
        },
        message_id: ts.to_string(),
        text,
        attachments: files,
        addressed,
    }))
}

/// Map an `interactive` payload. Only `block_actions` on approval buttons
/// produced by `ask` become events.
pub(crate) fn map_interactive(payload: &Value) -> Option<ChannelEvent> {
    if str_field(payload, "type") != Some("block_actions") {
        return None;
    }
    let user = payload.get("user")?;
    let user_id = str_field(user, "id")?;
    let container = payload.get("container");
    let channel = payload
        .get("channel")
        .and_then(|c| str_field(c, "id"))
        .or_else(|| container.and_then(|c| str_field(c, "channel_id")))?;
    let thread_ts = payload
        .get("message")
        .and_then(|m| str_field(m, "thread_ts"))
        .or_else(|| container.and_then(|c| str_field(c, "thread_ts")));

    let value = payload
        .get("actions")?
        .as_array()?
        .iter()
        .filter(|action| {
            str_field(action, "action_id").is_some_and(|id| id.starts_with(APPROVAL_ACTION_PREFIX))
        })
        .find_map(|action| {
            serde_json::from_str::<ApprovalValue>(str_field(action, "value")?).ok()
        })?;

    let is_dm = channel.starts_with('D');
    Some(ChannelEvent::Approval(ApprovalAnswer {
        prompt_id: value.prompt_id,
        choice_id: value.choice_id,
        sender: Sender {
            id: user_id.to_string(),
            display_name: str_field(user, "name")
                .or_else(|| str_field(user, "username"))
                .map(str::to_string),
            is_bot: false,
        },
        conversation: conversation_for(channel, is_dm, thread_ts, None),
    }))
}

/// Map a `slash_commands` envelope payload. The handle is the command's
/// `response_url`, which answers it.
pub(crate) fn map_slash_command(payload: &Value) -> Option<ChannelEvent> {
    let command = str_field(payload, "command")?;
    let name = command.trim_start_matches('/').trim();
    if name.is_empty() {
        return None;
    }
    let channel = str_field(payload, "channel_id")?;
    let user_id = str_field(payload, "user_id")?;
    let handle = str_field(payload, "response_url")?;
    let kind = if channel.starts_with('D') {
        ConversationKind::Direct
    } else {
        ConversationKind::Group
    };
    Some(ChannelEvent::Command(
        aidaemon_channels::CommandInvocation {
            conversation: Conversation {
                id: channel.to_string(),
                kind,
            },
            sender: Sender {
                id: user_id.to_string(),
                display_name: str_field(payload, "user_name").map(str::to_string),
                is_bot: false,
            },
            name: name.to_string(),
            args: payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string(),
            handle: handle.to_string(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BOT: &str = "U000BOT";

    fn ctx(reply_in_thread: bool) -> MapContext<'static> {
        MapContext {
            bot_user_id: Some(BOT),
            reply_in_thread,
        }
    }

    fn message(extra: Value) -> Value {
        let mut event = json!({
            "type": "message",
            "user": "U000ALICE",
            "channel": "C000SYNTH",
            "channel_type": "channel",
            "ts": "1700000000.000100",
            "text": "hello there",
        });
        for (k, v) in extra.as_object().unwrap() {
            event[k] = v.clone();
        }
        json!({ "team_id": "T000SYNTH", "event": event })
    }

    fn msg(event: Option<ChannelEvent>) -> InboundMessage {
        match event {
            Some(ChannelEvent::Message(m)) => m,
            other => panic!("expected message, got {other:?}"),
        }
    }

    #[test]
    fn unaddressed_channel_message_is_group() {
        let m = msg(map_events_api(
            &message(json!({})),
            &ctx(true),
            &mut SeenMessages::default(),
        ));
        assert_eq!(m.conversation.kind, ConversationKind::Group);
        assert_eq!(m.conversation.id, "C000SYNTH");
        assert!(!m.addressed);
        assert_eq!(m.sender.id, "U000ALICE");
        assert_eq!(m.message_id, "1700000000.000100");
    }

    #[test]
    fn mention_addresses_strips_and_opens_thread() {
        let payload = message(json!({ "text": "<@U000BOT> deploy  please" }));
        let m = msg(map_events_api(
            &payload,
            &ctx(true),
            &mut SeenMessages::default(),
        ));
        assert!(m.addressed);
        assert_eq!(m.text, "deploy please");
        assert_eq!(m.conversation.kind, ConversationKind::Thread);
        assert_eq!(m.conversation.id, "C000SYNTH:1700000000.000100");

        let m = msg(map_events_api(
            &payload,
            &ctx(false),
            &mut SeenMessages::default(),
        ));
        assert_eq!(m.conversation.kind, ConversationKind::Group);
        assert_eq!(m.conversation.id, "C000SYNTH");
    }

    #[test]
    fn labelled_mention_token_counts_but_prose_does_not() {
        let m = msg(map_events_api(
            &message(json!({ "text": "hey <@U000BOT|aidaemon>, status?" })),
            &ctx(false),
            &mut SeenMessages::default(),
        ));
        assert!(m.addressed);
        assert_eq!(m.text, "hey , status?");

        let m = msg(map_events_api(
            &message(json!({ "text": "ask U000BOT or @aidaemon" })),
            &ctx(false),
            &mut SeenMessages::default(),
        ));
        assert!(!m.addressed);
    }

    #[test]
    fn thread_reply_uses_thread_ts() {
        let m = msg(map_events_api(
            &message(json!({ "thread_ts": "1699999999.000001" })),
            &ctx(true),
            &mut SeenMessages::default(),
        ));
        assert_eq!(m.conversation.kind, ConversationKind::Thread);
        assert_eq!(m.conversation.id, "C000SYNTH:1699999999.000001");
    }

    #[test]
    fn dm_is_direct_and_addressed() {
        let m = msg(map_events_api(
            &message(json!({ "channel": "D000SYNTH", "channel_type": "im" })),
            &ctx(true),
            &mut SeenMessages::default(),
        ));
        assert_eq!(m.conversation.kind, ConversationKind::Direct);
        assert_eq!(m.conversation.id, "D000SYNTH");
        assert!(m.addressed);
    }

    #[test]
    fn app_mention_and_message_are_deduplicated() {
        let mut seen = SeenMessages::default();
        let text = json!({ "text": "<@U000BOT> hi" });
        let mention = {
            let mut p = message(text.clone());
            p["event"]["type"] = json!("app_mention");
            p["event"].as_object_mut().unwrap().remove("channel_type");
            p
        };
        let m = msg(map_events_api(&mention, &ctx(true), &mut seen));
        assert!(m.addressed);
        assert_eq!(m.conversation.kind, ConversationKind::Thread);
        assert!(map_events_api(&message(text), &ctx(true), &mut seen).is_none());
    }

    #[test]
    fn skips_bot_edited_subtype_and_self() {
        let mut seen = SeenMessages::default();
        for extra in [
            json!({ "bot_id": "B000SYNTH" }),
            json!({ "subtype": "message_changed" }),
            json!({ "subtype": "channel_join" }),
            json!({ "edited": { "user": "U000ALICE", "ts": "1" } }),
            json!({ "user": BOT }),
        ] {
            assert!(
                map_events_api(&message(extra.clone()), &ctx(true), &mut seen).is_none(),
                "{extra}"
            );
        }
    }

    #[test]
    fn file_share_becomes_attachment() {
        let m = msg(map_events_api(
            &message(json!({
                "subtype": "file_share",
                "text": "",
                "files": [{
                    "name": "report.pdf",
                    "mimetype": "application/pdf",
                    "size": 1234,
                    "url_private": "https://files.slack.com/files-pri/T000SYNTH-F000SYNTH/report.pdf"
                }]
            })),
            &ctx(true),
            &mut SeenMessages::default(),
        ));
        assert_eq!(m.attachments.len(), 1);
        assert_eq!(m.attachments[0].name, "report.pdf");
        assert_eq!(m.attachments[0].size_bytes, Some(1234));
        assert!(m.attachments[0]
            .source
            .starts_with("https://files.slack.com/"));
    }

    #[test]
    fn block_actions_map_to_approval() {
        let value = serde_json::to_string(&ApprovalValue {
            prompt_id: "prompt-1".into(),
            choice_id: "approve".into(),
        })
        .unwrap();
        let payload = json!({
            "type": "block_actions",
            "user": { "id": "U000ALICE", "name": "alice" },
            "channel": { "id": "C000SYNTH" },
            "message": { "ts": "1700000000.000200", "thread_ts": "1700000000.000100" },
            "actions": [{ "action_id": "aidaemon_approval_0", "value": value }]
        });
        match map_interactive(&payload) {
            Some(ChannelEvent::Approval(a)) => {
                assert_eq!(a.prompt_id, "prompt-1");
                assert_eq!(a.choice_id, "approve");
                assert_eq!(a.sender.id, "U000ALICE");
                assert_eq!(a.conversation.kind, ConversationKind::Thread);
                assert_eq!(a.conversation.id, "C000SYNTH:1700000000.000100");
            }
            other => panic!("expected approval, got {other:?}"),
        }

        let mut foreign = payload.clone();
        foreign["actions"][0]["action_id"] = json!("someone_else");
        assert!(map_interactive(&foreign).is_none());
    }

    #[test]
    fn conversation_id_round_trip() {
        assert_eq!(split_conversation_id("C1"), ("C1", None));
        assert_eq!(split_conversation_id("C1:17.5"), ("C1", Some("17.5")));
    }
}
