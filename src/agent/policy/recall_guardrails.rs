#[cfg(test)]
use serde_json::Value;

use super::intent_routing::contains_keyword_as_words;
#[cfg(test)]
use aidaemon_memory::render::{
    build_critical_facts_prompt_block, extract_critical_fact_summary, CriticalFactSummary,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CriticalFactQuery {
    OwnerName,
    AssistantName,
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
    use crate::traits::Fact;
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
