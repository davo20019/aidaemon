#[cfg(test)]
use serde_json::Value;

#[cfg(test)]
use aidaemon_memory::render::{
    CriticalFactSummary, build_critical_facts_prompt_block, extract_critical_fact_summary,
};

/// Execution tools blocked when delegation mode is active.
/// Keep spawn_agent available for task-lead orchestration.
pub fn is_delegation_blocked_tool(name: &str) -> bool {
    matches!(name, "terminal" | "browser")
}

#[cfg(test)]
pub fn filter_tool_defs_for_delegation(defs: &[Value]) -> Vec<Value> {
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
    use aidaemon_memory::records::{Fact, FactPrivacy};
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
}
