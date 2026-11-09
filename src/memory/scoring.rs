use crate::traits::Message;

/// Importance (0.1-1.0) from the message's role and length.
pub fn score_message(msg: &Message) -> f32 {
    score_role_and_content(&msg.role, msg.content.as_deref())
}

/// Calculate importance score for event-native conversation turns.
pub fn score_turn(turn: &crate::events::ConversationTurn) -> f32 {
    score_role_and_content(turn.role.as_str(), turn.content.as_deref())
}

fn score_role_and_content(role: &str, content: Option<&str>) -> f32 {
    aidaemon_memory::turn_importance(role, content)
}
