//! LLM-based plan step generation.

use serde_json::json;

use crate::traits::ModelProvider;

const PLAN_GENERATION_PROMPT: &str = r#"You are a task planner. Given a user request, break it down into discrete, actionable steps.

Each step should be:
- Atomic (can succeed or fail independently)
- Verifiable (clear what "done" means)
- Ordered (dependencies respected)
- Actionable (starts with a verb)

Guidelines:
- Keep it concise: 3-8 steps typical, max 10
- Do NOT include meta-steps like "understand the request" or "report results"
- Do NOT include "verify" steps after every action - only include critical verification points
- Be specific enough to be useful, but not so detailed that it's brittle

Return ONLY a JSON array of step description strings. No other text.

Example 1:
User: "Deploy the app to production"
["Run the test suite", "Build the production container", "Push to container registry", "Deploy to staging and verify", "Deploy to production", "Verify production health"]

Example 2:
User: "Refactor the auth module to use JWT"
["Add JWT library dependency", "Create JWT token generation utility", "Create JWT verification middleware", "Update login endpoint to return JWT", "Update protected routes to use JWT middleware", "Remove old session-based auth code", "Update tests"]

Example 3:
User: "Set up CI/CD pipeline"
["Create GitHub Actions workflow file", "Configure build step", "Configure test step", "Configure deployment step", "Add required secrets to repository", "Test pipeline with a sample commit"]

Now break down this request:
"#;

/// Generate plan steps from a user message using an LLM.
pub async fn generate_plan_steps(
    provider: &dyn ModelProvider,
    model: &str,
    user_message: &str,
) -> anyhow::Result<Vec<String>> {
    let prompt = format!("{}{}", PLAN_GENERATION_PROMPT, user_message);

    let messages = vec![json!({"role": "user", "content": prompt})];

    let response = provider.chat(model, &messages, &[]).await?;
    let text = response
        .content
        .ok_or_else(|| anyhow::anyhow!("Empty response from plan generation"))?;

    parse_json_array(&text)
}

/// Parse a JSON array from LLM response text.
/// Handles markdown code blocks and extracts the array.
fn parse_json_array(text: &str) -> anyhow::Result<Vec<String>> {
    let trimmed = text.trim();

    // Try to find JSON array in the response
    let json_str = extract_json_array(trimmed)?;

    // Parse the array
    let steps: Vec<String> = serde_json::from_str(&json_str).map_err(|e| {
        anyhow::anyhow!(
            "Failed to parse plan steps as JSON array: {}. Text was: {}",
            e,
            json_str
        )
    })?;

    // Validate we got reasonable steps
    if steps.is_empty() {
        anyhow::bail!("Plan generation returned empty step list");
    }

    if steps.len() > 15 {
        anyhow::bail!(
            "Plan generation returned too many steps ({}). Max is 15.",
            steps.len()
        );
    }

    // Validate each step is non-empty and reasonable length
    for (i, step) in steps.iter().enumerate() {
        if step.trim().is_empty() {
            anyhow::bail!("Step {} is empty", i + 1);
        }
        if step.len() > 500 {
            anyhow::bail!("Step {} is too long ({} chars). Keep steps concise.", i + 1, step.len());
        }
    }

    Ok(steps)
}

/// Extract JSON array from text, handling markdown code blocks.
fn extract_json_array(text: &str) -> anyhow::Result<String> {
    // First, try to extract from markdown code block
    if let Some(start) = text.find("```json") {
        let after_marker = &text[start + 7..];
        if let Some(end) = after_marker.find("```") {
            let content = after_marker[..end].trim();
            if content.starts_with('[') && content.ends_with(']') {
                return Ok(content.to_string());
            }
        }
    }

    // Try generic code block
    if let Some(start) = text.find("```") {
        let after_marker = &text[start + 3..];
        // Skip language identifier if present
        let content_start = after_marker.find('\n').map(|i| i + 1).unwrap_or(0);
        let after_lang = &after_marker[content_start..];
        if let Some(end) = after_lang.find("```") {
            let content = after_lang[..end].trim();
            if content.starts_with('[') && content.ends_with(']') {
                return Ok(content.to_string());
            }
        }
    }

    // Try to find raw JSON array
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return Ok(text[start..=end].to_string());
            }
        }
    }

    anyhow::bail!(
        "Could not find JSON array in response. Expected format: [\"step1\", \"step2\", ...]. Got: {}",
        if text.len() > 200 { &text[..200] } else { text }
    )
}

/// Generate a simple description from the user message for the plan.
pub fn extract_plan_description(user_message: &str) -> String {
    // Take first sentence or first 100 chars, whichever is shorter
    let first_sentence_end = user_message
        .find(['.', '!', '?'])
        .map(|i| i + 1)
        .unwrap_or(user_message.len());

    let description = &user_message[..first_sentence_end.min(100)];
    let mut result = description.trim().to_string();

    // Ensure it doesn't end awkwardly if truncated
    if result.len() == 100 && !result.ends_with('.') {
        if let Some(last_space) = result.rfind(' ') {
            result.truncate(last_space);
            result.push_str("...");
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_array_simple() {
        let text = r#"["Step 1", "Step 2", "Step 3"]"#;
        let result = parse_json_array(text).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "Step 1");
    }

    #[test]
    fn test_parse_json_array_with_code_block() {
        let text = r#"Here's the plan:

```json
["Step 1", "Step 2", "Step 3"]
```

Let me know if you want changes."#;

        let result = parse_json_array(text).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_parse_json_array_with_surrounding_text() {
        let text = r#"I've broken down your request into these steps:
["Run tests", "Build container", "Deploy"]
These steps should cover everything."#;

        let result = parse_json_array(text).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "Run tests");
    }

    #[test]
    fn test_parse_json_array_empty_fails() {
        let text = "[]";
        let result = parse_json_array(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_no_array_fails() {
        let text = "Here are some steps: first do this, then do that";
        let result = parse_json_array(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_plan_description() {
        assert_eq!(
            extract_plan_description("Deploy the app to production."),
            "Deploy the app to production."
        );

        assert_eq!(
            extract_plan_description("Deploy to prod"),
            "Deploy to prod"
        );

        // Long message truncation
        let long = "a".repeat(150);
        let desc = extract_plan_description(&long);
        assert!(desc.len() <= 103); // 100 + "..."
    }
}
