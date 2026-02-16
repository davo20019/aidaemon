use serde_json::Value;

use super::intent_routing::contains_keyword_as_words;
use crate::traits::Fact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CriticalFactQuery {
    OwnerName,
    AssistantName,
    CoreRelationships,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CriticalFactSummary {
    pub owner_name: Option<String>,
    pub assistant_name: Option<String>,
    pub relationships: Vec<String>,
}

pub(super) fn is_personal_memory_tool(name: &str) -> bool {
    matches!(name, "manage_people" | "manage_memories")
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

fn relationship_label_for_key(lower_key: &str) -> Option<&'static str> {
    if lower_key.contains("wife")
        || lower_key.contains("husband")
        || lower_key.contains("spouse")
        || lower_key.contains("partner")
    {
        return Some("partner");
    }
    if lower_key.contains("daughter")
        || lower_key.contains("son")
        || lower_key.contains("children")
        || lower_key.contains("child")
        || lower_key.contains("kids")
    {
        return Some("children");
    }
    None
}

pub(super) fn detect_critical_fact_query(user_text: &str) -> Option<CriticalFactQuery> {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    let asks_owner_name = contains_keyword_as_words(&lower, "what is my name")
        || contains_keyword_as_words(&lower, "what's my name")
        || contains_keyword_as_words(&lower, "who am i")
        || contains_keyword_as_words(&lower, "tell me my name")
        || contains_keyword_as_words(&lower, "my full name");
    if asks_owner_name {
        return Some(CriticalFactQuery::OwnerName);
    }

    let asks_assistant_name = contains_keyword_as_words(&lower, "what is your name")
        || contains_keyword_as_words(&lower, "what's your name")
        || contains_keyword_as_words(&lower, "who are you")
        || contains_keyword_as_words(&lower, "what should i call you")
        || contains_keyword_as_words(&lower, "what is your bot name")
        || contains_keyword_as_words(&lower, "what's your bot name");
    if asks_assistant_name {
        return Some(CriticalFactQuery::AssistantName);
    }

    let asks_relationships = contains_keyword_as_words(&lower, "who is my wife")
        || contains_keyword_as_words(&lower, "who is my husband")
        || contains_keyword_as_words(&lower, "who is my spouse")
        || contains_keyword_as_words(&lower, "who is my partner")
        || contains_keyword_as_words(&lower, "do i have daughters")
        || contains_keyword_as_words(&lower, "do i have daughter")
        || contains_keyword_as_words(&lower, "do i have sons")
        || contains_keyword_as_words(&lower, "do i have kids")
        || contains_keyword_as_words(&lower, "who are my children");
    if asks_relationships {
        return Some(CriticalFactQuery::CoreRelationships);
    }

    None
}

pub(super) fn extract_critical_fact_summary(facts: &[Fact]) -> CriticalFactSummary {
    let mut summary = CriticalFactSummary::default();
    let mut seen_relationships: std::collections::HashSet<String> =
        std::collections::HashSet::new();

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

        if summary.relationships.len() < 4 {
            if let Some(label) = relationship_label_for_key(&lower_key) {
                let clean_value = value
                    .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
                    .trim();
                if !clean_value.is_empty() && clean_value.len() <= 160 {
                    let line = format!("{}: {}", label, clean_value);
                    let dedupe = line.to_ascii_lowercase();
                    if seen_relationships.insert(dedupe) {
                        summary.relationships.push(line);
                    }
                }
            }
        }
    }

    summary
}

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
        CriticalFactQuery::CoreRelationships => {
            if summary.relationships.is_empty() {
                "I don't have core relationship details pinned yet.".to_string()
            } else {
                format!(
                    "Here are the core relationship details I have pinned:\n- {}",
                    summary.relationships.join("\n- ")
                )
            }
        }
    }
}

pub(super) fn build_critical_facts_prompt_block(summary: &CriticalFactSummary) -> Option<String> {
    let mut lines = vec![
        "[Critical Facts — Highest Priority For Recall]".to_string(),
        "When asked about identity/profile basics, answer from this block first.".to_string(),
    ];

    if let Some(owner_name) = summary.owner_name.as_ref() {
        lines.push(format!("- Owner name: {}", owner_name));
    }
    if let Some(assistant_name) = summary.assistant_name.as_ref() {
        lines.push(format!("- Assistant name: {}", assistant_name));
    }
    if !summary.relationships.is_empty() {
        lines.push("- Key relationships:".to_string());
        for rel in summary.relationships.iter().take(4) {
            lines.push(format!("  - {}", rel));
        }
    }

    if lines.len() <= 2 {
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

pub(super) fn looks_like_personal_memory_recall_question(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let mentions_personal_entities = contains_keyword_as_words(&lower, "daughter")
        || contains_keyword_as_words(&lower, "daughters")
        || contains_keyword_as_words(&lower, "son")
        || contains_keyword_as_words(&lower, "sons")
        || contains_keyword_as_words(&lower, "kid")
        || contains_keyword_as_words(&lower, "kids")
        || contains_keyword_as_words(&lower, "child")
        || contains_keyword_as_words(&lower, "children")
        || contains_keyword_as_words(&lower, "pet")
        || contains_keyword_as_words(&lower, "pets")
        || contains_keyword_as_words(&lower, "dog")
        || contains_keyword_as_words(&lower, "cat")
        || contains_keyword_as_words(&lower, "family")
        || contains_keyword_as_words(&lower, "wife")
        || contains_keyword_as_words(&lower, "husband")
        || contains_keyword_as_words(&lower, "mom")
        || contains_keyword_as_words(&lower, "dad")
        || contains_keyword_as_words(&lower, "mother")
        || contains_keyword_as_words(&lower, "father");

    contains_keyword_as_words(&lower, "what do you know about me")
        || contains_keyword_as_words(&lower, "about me")
        || (contains_keyword_as_words(&lower, "do i have") && mentions_personal_entities)
        || (contains_keyword_as_words(&lower, "what about") && mentions_personal_entities)
        || (contains_keyword_as_words(&lower, "do i") && mentions_personal_entities)
}

pub(super) fn user_is_reaffirmation_challenge(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    contains_keyword_as_words(&lower, "are you sure")
        || contains_keyword_as_words(&lower, "really")
        || contains_keyword_as_words(&lower, "you sure")
        || contains_keyword_as_words(&lower, "certain")
}

pub(super) fn user_requests_external_verification(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    contains_keyword_as_words(&lower, "actually check")
        || contains_keyword_as_words(&lower, "double check")
        || contains_keyword_as_words(&lower, "double-check")
        || contains_keyword_as_words(&lower, "fact check")
        || contains_keyword_as_words(&lower, "fact-check")
        || contains_keyword_as_words(&lower, "verify")
        || contains_keyword_as_words(&lower, "verify this")
        || contains_keyword_as_words(&lower, "look it up")
        || contains_keyword_as_words(&lower, "look this up")
        || contains_keyword_as_words(&lower, "check online")
        || contains_keyword_as_words(&lower, "search the web")
        || contains_keyword_as_words(&lower, "use tools")
}

pub(super) fn tool_result_indicates_no_evidence(result_text: &str) -> bool {
    let lower = result_text.to_ascii_lowercase();
    lower.contains("no matches found")
        || lower.contains("person not found")
        || lower.contains("no active fact found")
        || lower.contains("none recorded")
        || lower.contains("no results")
        || lower.contains("not found")
        || lower.contains("couldn't find")
        || lower.contains("could not find")
        || lower.contains("i don't have any information")
        || lower.contains("i don't have information")
        || lower.contains("no information")
        || lower.contains("no relevant")
        || lower.contains("no evidence")
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
        }
    }

    #[test]
    fn detects_personal_memory_recall_questions() {
        assert!(looks_like_personal_memory_recall_question(
            "Do I have daughters?"
        ));
        assert!(looks_like_personal_memory_recall_question(
            "What about pets?"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Do I have node installed?"
        ));
    }

    #[test]
    fn distinguishes_challenge_vs_external_verification() {
        assert!(user_is_reaffirmation_challenge("Are you sure?"));
        assert!(!user_requests_external_verification("Are you sure?"));
        assert!(user_requests_external_verification(
            "Please check online and verify this."
        ));
    }

    #[test]
    fn detects_no_evidence_tool_results() {
        assert!(tool_result_indicates_no_evidence(
            "No matches found (40 files scanned)"
        ));
        assert!(tool_result_indicates_no_evidence(
            "Person 'Alice' not found."
        ));
        assert!(!tool_result_indicates_no_evidence(
            "Found 2 matches in profile data."
        ));
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
    fn detects_critical_fact_queries() {
        assert_eq!(
            detect_critical_fact_query("What's my name?"),
            Some(CriticalFactQuery::OwnerName)
        );
        assert_eq!(
            detect_critical_fact_query("What is your bot name?"),
            Some(CriticalFactQuery::AssistantName)
        );
        assert_eq!(
            detect_critical_fact_query("Do I have daughters?"),
            Some(CriticalFactQuery::CoreRelationships)
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
        assert_eq!(summary.relationships.len(), 1);
        assert!(summary.relationships[0].contains("children"));
    }

    #[test]
    fn deterministic_reply_uses_critical_facts() {
        let summary = CriticalFactSummary {
            owner_name: Some("Test Owner".to_string()),
            assistant_name: Some("TestBot".to_string()),
            relationships: vec!["children: Sofia".to_string()],
        };
        assert_eq!(
            deterministic_reply_for_critical_query(CriticalFactQuery::OwnerName, &summary),
            "Your name is Test Owner."
        );
        assert_eq!(
            deterministic_reply_for_critical_query(CriticalFactQuery::AssistantName, &summary),
            "My name is TestBot."
        );
        assert!(deterministic_reply_for_critical_query(
            CriticalFactQuery::CoreRelationships,
            &summary
        )
        .contains("children: Sofia"));
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
