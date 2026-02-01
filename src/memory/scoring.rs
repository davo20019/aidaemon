use crate::traits::Message;

/// Calculate importance score (0.0 - 1.0) based on heuristics.
pub fn score_message(msg: &Message) -> f32 {
    let mut score: f32 = 0.5;

    // 1. Role-based baseline
    match msg.role.as_str() {
        "system" => return 0.0, // System prompts are not "memories"
        "tool" => score = 0.3,  // Tool outputs are less important unless they are results
        "assistant" => score = 0.5,
        "user" => score = 0.6, // User messages slightly more important
        _ => {}
    }

    if let Some(content) = &msg.content {
        let text = content.to_lowercase();
        
        // 2. Length heuristic (Longer = usually more info)
        if content.len() > 200 {
            score += 0.2;
        } else if content.len() < 20 {
            score -= 0.2;
        }

        // 3. Keyword boosting (Refined)
        if text.contains("important") 
            || text.contains("remember") 
            || text.contains("do not forget")
        {
            score += 0.3;
        }

        // 4. Negative signals (Chit-chat)
        if text.len() < 10 && (
            text.contains("ok") || text.contains("thanks") || text.contains("cool")
        ) {
            score -= 0.3;
        }

        // 4. Code detection
        if text.contains("```") {
            score += 0.2;
        }
    }

    // Clamp between 0.0 and 1.0
    score.clamp(0.1, 1.0)
}
