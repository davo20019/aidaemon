use crate::traits::Message;

/// Episodic search ranking: semantic relevance with a small bounded
/// reinforcement for recalled episodes; age is intentionally absent, so an
/// old conversation stays discoverable.
pub fn episode_search_score(similarity: f32, recall_count: i32) -> f32 {
    similarity * aidaemon_memory::episode_recall_boost(recall_count.max(0) as u32)
}

/// Degraded-mode relevance when the embedding service is unavailable:
/// exact word overlap, every word counting.
pub use aidaemon_memory::word_overlap as lexical_relevance;

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
