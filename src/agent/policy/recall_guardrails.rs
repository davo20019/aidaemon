#[cfg(test)]
use serde_json::Value;

use super::intent_routing::contains_keyword_as_words;
use crate::traits::Fact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CriticalFactQuery {
    OwnerName,
    AssistantName,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CriticalFactSummary {
    pub owner_name: Option<String>,
    pub assistant_name: Option<String>,
}

pub(super) fn is_personal_memory_tool(name: &str) -> bool {
    matches!(
        name,
        "manage_people" | "manage_memories" | "remember_fact" | "search_history"
    )
}

fn normalize_name_candidate(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'));
    if trimmed.is_empty() || trimmed.len() > 80 {
        return None;
    }
    if trimmed
        .chars()
        .any(|c| matches!(c, '\n' | '\r' | '[' | ']' | '{' | '}'))
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn extract_name_from_phrase(value: &str) -> Option<String> {
    let lower = value.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    for prefix in ["my name is ", "i am ", "i'm ", "call me "] {
        if lower.starts_with(prefix) {
            let name = value.trim()[prefix.len()..].trim();
            return normalize_name_candidate(name);
        }
    }

    if let Some(idx) = lower.find(" is myself") {
        return normalize_name_candidate(value[..idx].trim());
    }

    None
}

pub(super) fn detect_critical_fact_query(user_text: &str) -> Option<CriticalFactQuery> {
    let lower = user_text
        .trim()
        .trim_end_matches(['?', '!', '.'])
        .trim()
        .to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    // Multi-part questions should NOT be handled deterministically.
    // Questions like "What's my name, what do I like, and what's my dog's name?"
    // need the full LLM with context to answer comprehensively.
    let comma_count = lower.matches(',').count();
    let question_mark_count = lower.matches('?').count();
    let has_conjunction_joining = lower.contains(" and what")
        || lower.contains(" and who")
        || lower.contains(" and tell")
        || lower.contains(" and my");
    if comma_count >= 2 || question_mark_count >= 2 || has_conjunction_joining {
        return None;
    }

    let asks_owner_name = matches!(
        lower.as_str(),
        "what is my name" | "what's my name" | "who am i" | "tell me my name" | "my full name"
    );
    if asks_owner_name {
        return Some(CriticalFactQuery::OwnerName);
    }

    let asks_assistant_name = matches!(
        lower.as_str(),
        "what is your name"
            | "what's your name"
            | "who are you"
            | "what should i call you"
            | "what is your bot name"
            | "what's your bot name"
    );
    if asks_assistant_name {
        return Some(CriticalFactQuery::AssistantName);
    }

    None
}

pub(super) fn extract_critical_fact_summary(facts: &[Fact]) -> CriticalFactSummary {
    let mut summary = CriticalFactSummary::default();

    for fact in facts {
        let key = fact.key.trim();
        let value = fact.value.trim();
        if key.is_empty() || value.is_empty() {
            continue;
        }
        let lower_key = key.to_ascii_lowercase();
        let lower_cat = fact.category.trim().to_ascii_lowercase();

        if summary.owner_name.is_none() {
            let owner_name_key = matches!(
                lower_key.as_str(),
                "name" | "owner_name" | "user_name" | "full_name" | "my_name" | "owner"
            );
            let owner_name_category = matches!(
                lower_cat.as_str(),
                "user" | "personal" | "profile" | "identity"
            );
            if (owner_name_key && owner_name_category) || lower_key == "owner_name" {
                summary.owner_name = normalize_name_candidate(value);
            } else if let Some(name) = extract_name_from_phrase(value) {
                if lower_key.contains("name") || lower_key.contains("owner") {
                    summary.owner_name = Some(name);
                }
            }
        }

        if summary.assistant_name.is_none() {
            let assistant_key = matches!(
                lower_key.as_str(),
                "assistant_name" | "bot_name" | "ai_name" | "daemon_name"
            ) || (lower_key == "name"
                && matches!(lower_cat.as_str(), "assistant" | "bot"));
            if assistant_key {
                summary.assistant_name = normalize_name_candidate(value);
            }
        }
    }

    summary
}

#[cfg(test)]
pub(super) fn deterministic_reply_for_critical_query(
    query: CriticalFactQuery,
    summary: &CriticalFactSummary,
) -> String {
    match query {
        CriticalFactQuery::OwnerName => summary.owner_name.as_ref().map_or_else(
            || {
                "I don't have your name saved in critical memory yet. Tell me \"my name is ...\" and I'll pin it.".to_string()
            },
            |name| format!("Your name is {}.", name),
        ),
        CriticalFactQuery::AssistantName => summary.assistant_name.as_ref().map_or_else(
            || "I don't have a pinned assistant name in critical memory right now.".to_string(),
            |name| format!("My name is {}.", name),
        ),
    }
}

pub(super) fn build_critical_facts_prompt_block(summary: &CriticalFactSummary) -> Option<String> {
    let mut lines = vec![
        "═══ CRITICAL FACTS — USE THESE EXACT VALUES ═══".to_string(),
        "These pinned values cover ONLY your user's identity and your own — no other entity."
            .to_string(),
        "When asked about a fact below, reply with the EXACT value shown here.".to_string(),
        "Do NOT substitute, paraphrase, or infer different values from training data.".to_string(),
        "They do NOT apply to subjects from the current conversation: if the question refers to \
         an entity just discussed (e.g. \"the owner\"/\"the founder\" of a company), answer from \
         that conversation context, not from this block."
            .to_string(),
    ];

    let mut fact_count = 0;
    if let Some(owner_name) = summary.owner_name.as_ref() {
        lines.push(format!("• Owner name → {}", owner_name));
        fact_count += 1;
    }
    if let Some(assistant_name) = summary.assistant_name.as_ref() {
        lines.push(format!("• Assistant name → {}", assistant_name));
        fact_count += 1;
    }
    lines.push("═══════════════════════════════════════════════".to_string());

    if fact_count == 0 {
        None
    } else {
        Some(lines.join("\n"))
    }
}

pub(super) fn text_relates_to_critical_identity(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if detect_critical_fact_query(&lower).is_some() {
        return true;
    }

    contains_keyword_as_words(&lower, "my name is")
        || contains_keyword_as_words(&lower, "call me")
        || contains_keyword_as_words(&lower, "i am")
        || contains_keyword_as_words(&lower, "owner name")
        || contains_keyword_as_words(&lower, "bot name")
        || contains_keyword_as_words(&lower, "assistant name")
        || contains_keyword_as_words(&lower, "wife")
        || contains_keyword_as_words(&lower, "husband")
        || contains_keyword_as_words(&lower, "spouse")
        || contains_keyword_as_words(&lower, "daughter")
        || contains_keyword_as_words(&lower, "son")
        || contains_keyword_as_words(&lower, "children")
        || lower.contains(" is myself")
        || (lower.contains("[user]") && lower.contains("name:"))
        || (lower.contains("[user]") && lower.contains("name ="))
        || (lower.contains("saved fact") && lower.contains("name"))
        || (lower.contains("remembered:") && lower.contains("name"))
}

#[cfg(test)]
pub(super) fn filter_tool_defs_for_personal_memory(defs: &[Value]) -> Vec<Value> {
    defs.iter()
        .filter_map(|def| {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())?;
            if is_personal_memory_tool(name) {
                Some(def.clone())
            } else {
                None
            }
        })
        .collect()
}

/// Execution tools blocked when delegation mode is active.
/// Keep spawn_agent available for task-lead orchestration.
pub(super) fn is_delegation_blocked_tool(name: &str) -> bool {
    matches!(name, "terminal" | "browser" | "run_command")
}

#[cfg(test)]
pub(super) fn filter_tool_defs_for_delegation(defs: &[Value]) -> Vec<Value> {
    defs.iter()
        .filter_map(|def| {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())?;
            if is_delegation_blocked_tool(name) {
                None
            } else {
                Some(def.clone())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FactPrivacy;
    use chrono::Utc;
    use serde_json::json;

    fn make_fact(category: &str, key: &str, value: &str) -> Fact {
        Fact {
            id: 1,
            category: category.to_string(),
            key: key.to_string(),
            value: value.to_string(),
            source: "test".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            superseded_at: None,
            recall_count: 0,
            last_recalled_at: None,
            channel_id: None,
            privacy: FactPrivacy::Global,
            first_seen_at: None,
            source_excerpt: None,
        }
    }

    #[test]
    fn filters_tool_defs_to_personal_memory_scope() {
        let defs = vec![
            json!({"type":"function","function":{"name":"manage_people"}}),
            json!({"type":"function","function":{"name":"manage_memories"}}),
            json!({"type":"function","function":{"name":"browser"}}),
            json!({"type":"function","function":{"name":"terminal"}}),
        ];
        let filtered = filter_tool_defs_for_personal_memory(&defs);
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|d| d.get("function"))
            .filter_map(|f| f.get("name"))
            .filter_map(|n| n.as_str())
            .collect();
        assert_eq!(names, vec!["manage_people", "manage_memories"]);
    }

    #[test]
    fn identifies_delegation_blocked_tools() {
        assert!(is_delegation_blocked_tool("terminal"));
        assert!(is_delegation_blocked_tool("browser"));
        assert!(is_delegation_blocked_tool("run_command"));
        assert!(!is_delegation_blocked_tool("spawn_agent"));
        assert!(!is_delegation_blocked_tool("cli_agent"));
        assert!(!is_delegation_blocked_tool("web_search"));
    }

    #[test]
    fn filters_tool_defs_for_delegation_mode() {
        let defs = vec![
            json!({"type":"function","function":{"name":"terminal"}}),
            json!({"type":"function","function":{"name":"cli_agent"}}),
            json!({"type":"function","function":{"name":"web_search"}}),
            json!({"type":"function","function":{"name":"browser"}}),
            json!({"type":"function","function":{"name":"run_command"}}),
            json!({"type":"function","function":{"name":"spawn_agent"}}),
            json!({"type":"function","function":{"name":"remember_fact"}}),
        ];
        let filtered = filter_tool_defs_for_delegation(&defs);
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|d| d.get("function"))
            .filter_map(|f| f.get("name"))
            .filter_map(|n| n.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["cli_agent", "web_search", "spawn_agent", "remember_fact"]
        );
    }

    #[test]
    fn detects_critical_fact_queries() {
        assert_eq!(
            detect_critical_fact_query("What's my name?"),
            Some(CriticalFactQuery::OwnerName)
        );
        assert_eq!(
            detect_critical_fact_query("What is your bot name?"),
            Some(CriticalFactQuery::AssistantName)
        );
    }

    #[test]
    fn multi_part_questions_bypass_deterministic_resolver() {
        // Multi-part questions should go to the LLM for comprehensive answers
        assert_eq!(
            detect_critical_fact_query(
                "What's my name, what programming languages do I love, and what's my dog's name?"
            ),
            None
        );
        assert_eq!(
            detect_critical_fact_query("What's my name and what do I do for work?"),
            None
        );
        assert_eq!(
            detect_critical_fact_query("Who am I? What do I like? Where do I live?"),
            None
        );
        // Single-part questions still work
        assert_eq!(
            detect_critical_fact_query("What's my name?"),
            Some(CriticalFactQuery::OwnerName)
        );
    }

    #[test]
    fn extracts_critical_fact_summary() {
        let facts = vec![
            make_fact("user", "name", "Test Owner"),
            make_fact("assistant", "bot_name", "TestBot"),
            make_fact("user", "daughter_name", "Sofia"),
        ];
        let summary = extract_critical_fact_summary(&facts);
        assert_eq!(summary.owner_name.as_deref(), Some("Test Owner"));
        assert_eq!(summary.assistant_name.as_deref(), Some("TestBot"));
    }

    #[test]
    fn deterministic_reply_uses_critical_facts() {
        let summary = CriticalFactSummary {
            owner_name: Some("Test Owner".to_string()),
            assistant_name: Some("TestBot".to_string()),
        };
        assert_eq!(
            deterministic_reply_for_critical_query(CriticalFactQuery::OwnerName, &summary),
            "Your name is Test Owner."
        );
        assert_eq!(
            deterministic_reply_for_critical_query(CriticalFactQuery::AssistantName, &summary),
            "My name is TestBot."
        );
    }

    #[test]
    fn critical_facts_block_scopes_to_user_and_assistant_identity() {
        // Telemetry case: after a SpaceX conversation, "Who's the owner?" was
        // answered with the pinned owner name instead of the company's owner.
        // The block must say its values cover ONLY the user's/assistant's own
        // identity and that conversational antecedents win.
        let summary = CriticalFactSummary {
            owner_name: Some("Test Owner".to_string()),
            assistant_name: None,
        };
        let block = build_critical_facts_prompt_block(&summary).expect("block should render");
        assert!(
            block.contains("ONLY your user's identity and your own"),
            "block must scope pinned values to user/assistant identity: {}",
            block
        );
        assert!(
            block.contains("current conversation"),
            "block must defer to conversational antecedents: {}",
            block
        );
        // Existing consumers depend on these markers.
        assert!(block.contains("CRITICAL FACTS"));
        assert!(block.contains("• Owner name → Test Owner"));
    }

    #[test]
    fn critical_facts_block_empty_summary_renders_nothing() {
        let block = build_critical_facts_prompt_block(&CriticalFactSummary::default());
        assert!(block.is_none(), "no pinned facts → no block: {:?}", block);
    }

    #[test]
    fn detects_identity_related_text_snippets() {
        assert!(text_relates_to_critical_identity("my name is David"));
        assert!(text_relates_to_critical_identity(
            "Saved fact [user] name: David"
        ));
        assert!(!text_relates_to_critical_identity("run the tests"));
    }
}
