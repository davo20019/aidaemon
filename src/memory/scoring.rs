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
        if (text.contains('{') && text.contains('}')) || (text.contains('[') && text.contains(']')) {
            if len > 50 {
                score += 0.1;
            }
        }

        // 10. Negative signals (Chit-chat / acknowledgements)
        if len < 15 && (
            text == "ok" || text == "okay" || text == "thanks"
            || text == "cool" || text == "got it" || text == "sure"
            || text == "yes" || text == "no" || text == "k"
            || text == "ty" || text == "thx" || text == "yep"
        ) {
            score -= 0.3;
        }
    }

    // Clamp between 0.1 and 1.0
    score.clamp(0.1, 1.0)
}
