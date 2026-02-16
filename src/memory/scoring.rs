use crate::traits::Message;
use chrono::{DateTime, Utc};

/// Calculate a memory score that combines similarity, recency, and recall patterns.
///
/// This function implements a decay-and-reinforcement model where:
/// - Recent memories are more accessible
/// - Frequently recalled memories are stronger
/// - Recently recalled memories get a boost
///
/// Returns a score between 0.0 and 1.0+ (can exceed 1.0 with high recall counts).
pub fn memory_score(
    similarity: f32,
    created_at: DateTime<Utc>,
    recall_count: i32,
    last_recalled: Option<DateTime<Utc>>,
) -> f32 {
    let now = Utc::now();

    // Recency decay: memories lose relevance over time
    // Half-life of roughly 10 days
    let days_since_created = (now - created_at).num_hours() as f32 / 24.0;
    let recency_decay = 1.0 / (1.0 + days_since_created * 0.1);

    // Recall boost: frequently recalled memories are stronger
    // Capped at 50% boost (after 5 recalls)
    let recall_boost = 1.0 + (recall_count as f32 * 0.1).min(0.5);

    // Recall recency: recently recalled memories get an additional boost
    // Decays over time since last recall
    let recall_recency = last_recalled
        .map(|t| {
            let days_since_recall = (now - t).num_hours() as f32 / 24.0;
            1.0 / (1.0 + days_since_recall * 0.05)
        })
        .unwrap_or(1.0);

    similarity * recency_decay * recall_boost * recall_recency
}

/// Calculate episode importance based on various factors.
pub fn calculate_episode_importance(
    message_count: i32,
    has_errors: bool,
    has_decisions: bool,
    has_goals: bool,
    emotional_intensity: f32, // 0.0-1.0 scale
) -> f32 {
    let mut importance = 0.5; // Base importance

    // More messages = more substantial session
    if message_count > 20 {
        importance += 0.2;
    } else if message_count > 10 {
        importance += 0.1;
    }

    // Error resolution is important to remember
    if has_errors {
        importance += 0.15;
    }

    // Decisions made during session
    if has_decisions {
        importance += 0.1;
    }

    // Goals mentioned or worked on
    if has_goals {
        importance += 0.1;
    }

    // Emotional sessions (frustration, excitement) are memorable
    importance += emotional_intensity * 0.2;

    importance.clamp(0.1, 1.0)
}

/// Calculate importance score (0.0 - 1.0) based on heuristics.
pub fn score_message(msg: &Message) -> f32 {
    score_role_and_content(&msg.role, msg.content.as_deref())
}

/// Calculate importance score for event-native conversation turns.
pub fn score_turn(turn: &crate::events::ConversationTurn) -> f32 {
    score_role_and_content(turn.role.as_str(), turn.content.as_deref())
}

fn score_role_and_content(role: &str, content: Option<&str>) -> f32 {
    let mut score: f32 = 0.5;

    // 1. Role-based baseline
    match role {
        "system" => return 0.0, // System prompts are not "memories"
        "tool" => score = 0.3,  // Tool outputs are less important unless they are results
        "assistant" => score = 0.5,
        "user" => score = 0.6, // User messages slightly more important
        _ => {}
    }

    if let Some(content) = content {
        let text = content.to_lowercase();
        let len = content.len();

        // 2. Length heuristic (Longer = usually more info)
        if len > 200 {
            score += 0.2;
        } else if len < 20 {
            score -= 0.2;
        }

        // 3. Explicit memory keywords
        if text.contains("important")
            || text.contains("remember")
            || text.contains("do not forget")
            || text.contains("don't forget")
            || text.contains("keep in mind")
            || text.contains("note that")
        {
            score += 0.3;
        }

        // 4. Questions (often set intent/context)
        if text.contains('?')
            || text.starts_with("how ")
            || text.starts_with("what ")
            || text.starts_with("why ")
            || text.starts_with("where ")
            || text.starts_with("when ")
            || text.starts_with("can you ")
            || text.starts_with("could you ")
        {
            score += 0.1;
        }

        // 5. Decisions and commitments
        if text.contains("let's go with")
            || text.contains("i chose")
            || text.contains("i decided")
            || text.contains("we'll use")
            || text.contains("let's use")
            || text.contains("switch to")
            || text.contains("going with")
            || text.contains("the plan is")
        {
            score += 0.25;
        }

        // 6. Corrections and preferences
        if text.contains("actually")
            || text.contains("no, ")
            || text.contains("instead")
            || text.contains("i prefer")
            || text.contains("don't use")
            || text.contains("always use")
            || text.contains("never use")
        {
            score += 0.2;
        }

        // 7. URLs and file paths (references worth recalling)
        if text.contains("http://")
            || text.contains("https://")
            || text.contains("github.com")
            || text.contains(".rs:")
            || text.contains(".ts:")
            || text.contains(".py:")
            || text.contains("~/")
            || text.contains("/src/")
            || text.contains("/home/")
        {
            score += 0.1;
        }

        // 8. Code blocks
        if text.contains("```") {
            score += 0.2;
        }

        // 9. Structured data (JSON objects/arrays)
        if ((text.contains('{') && text.contains('}'))
            || (text.contains('[') && text.contains(']')))
            && len > 50
        {
            score += 0.1;
        }

        // 10. Negative signals (Chit-chat / acknowledgements)
        if len < 15
            && (text == "ok"
                || text == "okay"
                || text == "thanks"
                || text == "cool"
                || text == "got it"
                || text == "sure"
                || text == "yes"
                || text == "no"
                || text == "k"
                || text == "ty"
                || text == "thx"
                || text == "yep")
        {
            score -= 0.3;
        }
    }

    // Clamp between 0.1 and 1.0
    score.clamp(0.1, 1.0)
}
