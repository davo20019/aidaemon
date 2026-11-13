//! Conversation helpers kept at their old path: event -> turn projection is
//! `aidaemon_events::turn_from_event`, truncation `truncate_with_anchor`.

pub use aidaemon_events::conversation::truncate_with_anchor;

#[cfg(test)]
mod tests {
    use super::truncate_with_anchor;
    use crate::traits::Message;

    fn message(index: usize, role: &str) -> Message {
        let mut message = Message::new_runtime(format!("m{index}"), "s", role);
        message.content = Some(format!("{role} {index}"));
        message
    }

    #[test]
    fn pinned_anchor_keeps_the_newest_messages() {
        // user, then alternating assistant/user; the tail window would start
        // on an assistant message, so the first user message is pinned.
        let messages: Vec<Message> = (0..9)
            .map(|index| message(index, if index % 2 == 0 { "user" } else { "assistant" }))
            .collect();
        let truncated = truncate_with_anchor(messages, 4);
        let ids: Vec<&str> = truncated.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["m0", "m6", "m7", "m8"]);
    }
}
