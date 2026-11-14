//! The tool roster (`aidaemon_runtime::engine::tool_roster`) on the daemon's
//! agent and test harness.

use super::*;
use crate::testing::{MockProvider, MockTool, setup_full_stack_test_agent_with_extra_tools};
use crate::traits::Tool;
use crate::types::{WorkspaceAccessLevel, WorkspaceGrant};
use aidaemon_runtime::engine::tool_roster::{ToolRoster as _, validate_tool_definition_contract};
use aidaemon_runtime::turn::helpers::{
    restrict_to_channel_reach, sort_tool_definitions_by_name, tool_name_from_definition,
};
use aidaemon_tools::semantics::ChannelReach;
use proptest::prelude::*;
use std::sync::Arc;

struct UnavailableMockTool;

#[async_trait::async_trait]
impl Tool for UnavailableMockTool {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "unavailable cli_agent for tests"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "cli_agent",
            "description": "unavailable cli_agent for tests",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> = async move { Ok("ok".to_string()) }.await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for UnavailableMockTool {
    fn is_available(&self) -> bool {
        false
    }
}

fn valid_tool_def() -> Value {
    named_tool_def("demo_tool")
}

fn named_tool_def(name: &str) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "demo",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"],
                "additionalProperties": false
            }
        }
    })
}

#[test]
fn tool_definition_contract_accepts_valid_definition() {
    let def = valid_tool_def();
    assert!(validate_tool_definition_contract(&def).is_ok());
}

#[test]
fn channel_reach_says_where_a_tool_may_be_offered() {
    use ChannelVisibility::*;
    // Desktop control: only direct messages and internal/system sessions.
    // A group or public conversation must not reach it: a participant's
    // message (or the bot's own echoed content) could otherwise launch it.
    for visibility in [Private, Internal] {
        assert!(ChannelReach::DirectOnly.offered_in(visibility));
    }
    for visibility in [PrivateGroup, Public, PublicExternal] {
        assert!(!ChannelReach::DirectOnly.offered_in(visibility));
    }
    // A public external channel sees only the public tools.
    assert!(ChannelReach::Public.offered_in(PublicExternal));
    assert!(!ChannelReach::Shared.offered_in(PublicExternal));
    assert!(ChannelReach::Shared.offered_in(Public));
}

/// The reach the tools in these tests declare.
fn reach(name: &str) -> ChannelReach {
    match name {
        "computer_use" => ChannelReach::DirectOnly,
        "web_search" => ChannelReach::Public,
        _ => ChannelReach::Shared,
    }
}

#[test]
fn channel_reach_strips_desktop_control_in_channels() {
    for visibility in [
        ChannelVisibility::PrivateGroup,
        ChannelVisibility::Public,
        ChannelVisibility::PublicExternal,
    ] {
        let mut defs = vec![named_tool_def("computer_use"), named_tool_def("web_search")];
        let mut caps = HashMap::from([
            ("computer_use".to_string(), ToolCapabilities::default()),
            ("web_search".to_string(), ToolCapabilities::default()),
        ]);
        restrict_to_channel_reach(&mut defs, &mut caps, visibility, reach);
        let names: Vec<&str> = defs.iter().filter_map(tool_name_from_definition).collect();
        assert!(
            !names.contains(&"computer_use"),
            "computer_use should be stripped in {visibility:?}"
        );
        assert!(
            names.contains(&"web_search"),
            "unrelated tools must survive in {visibility:?}"
        );
        assert!(!caps.contains_key("computer_use"));
        assert!(caps.contains_key("web_search"));
    }
}

#[test]
fn channel_reach_keeps_computer_use_in_dm_and_internal() {
    for visibility in [ChannelVisibility::Private, ChannelVisibility::Internal] {
        let mut defs = vec![named_tool_def("computer_use"), named_tool_def("web_search")];
        let mut caps = HashMap::from([("computer_use".to_string(), ToolCapabilities::default())]);
        restrict_to_channel_reach(&mut defs, &mut caps, visibility, reach);
        let names: Vec<&str> = defs.iter().filter_map(tool_name_from_definition).collect();
        assert!(
            names.contains(&"computer_use"),
            "computer_use must remain available in {visibility:?}"
        );
        assert!(caps.contains_key("computer_use"));
    }
}

proptest! {
    #[test]
    fn tool_definition_contract_rejects_invalid_required_keys(required_key in "[a-z]{1,12}") {
        let mut def = valid_tool_def();
        def["function"]["parameters"]["required"] = json!([required_key, "missing_key"]);
        let result = validate_tool_definition_contract(&def);
        prop_assert!(result.is_err());
    }

    #[test]
    fn tool_definition_contract_rejects_non_boolean_additional_properties(flag in ".*") {
        let mut def = valid_tool_def();
        def["function"]["parameters"]["additionalProperties"] = json!(flag);
        let result = validate_tool_definition_contract(&def);
        prop_assert!(result.is_err());
    }
}

#[tokio::test]
async fn tool_definitions_skip_unavailable_tools() {
    let available = Arc::new(MockTool::new("web_search", "search", "ok")) as Arc<dyn Tool>;
    let unavailable = Arc::new(UnavailableMockTool) as Arc<dyn Tool>;
    let harness = setup_full_stack_test_agent_with_extra_tools(
        MockProvider::new(),
        vec![available, unavailable],
    )
    .await
    .unwrap();

    let (defs, caps) = harness
        .agent
        .tool_definitions_with_capabilities("test query")
        .await;
    let names: Vec<String> = defs
        .iter()
        .filter_map(tool_name_from_definition)
        .map(ToString::to_string)
        .collect();

    assert!(names.contains(&"web_search".to_string()));
    assert!(!names.contains(&"cli_agent".to_string()));
    assert!(caps.contains_key("web_search"));
    assert!(!caps.contains_key("cli_agent"));
    assert!(
        !harness
            .agent
            .tools()
            .iter()
            .any(|tool| tool.name() == "cli_agent" && tool.is_available())
    );
}

#[tokio::test]
async fn session_static_roster_is_deterministic_and_sorted() {
    let t1 = Arc::new(MockTool::new("zebra_tool", "z", "ok")) as Arc<dyn Tool>;
    let t2 = Arc::new(MockTool::new("alpha_tool", "a", "ok")) as Arc<dyn Tool>;
    let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![t1, t2])
        .await
        .unwrap();
    let channel_ctx = ChannelContext::private("test");

    let roster_a = harness
        .agent
        .session_static_tool_roster(UserRole::Owner, &channel_ctx);
    let roster_b = harness
        .agent
        .session_static_tool_roster(UserRole::Owner, &channel_ctx);
    // Deterministic: identical output on repeated calls (function takes no query).
    assert_eq!(roster_a, roster_b);

    let names: Vec<&str> = roster_a.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"alpha_tool"));
    assert!(names.contains(&"zebra_tool"));
    // Sorted by name.
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
}

#[tokio::test]
async fn session_static_roster_empty_for_non_owner() {
    let t1 = Arc::new(MockTool::new("alpha_tool", "a", "ok")) as Arc<dyn Tool>;
    let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![t1])
        .await
        .unwrap();
    let channel_ctx = ChannelContext::private("test");

    assert!(
        harness
            .agent
            .session_static_tool_roster(UserRole::Guest, &channel_ctx)
            .is_empty()
    );
    assert!(
        harness
            .agent
            .session_static_tool_roster(UserRole::Public, &channel_ctx)
            .is_empty()
    );
}

#[tokio::test]
async fn session_static_roster_limits_scoped_guest_to_granted_file_tools() {
    let read = Arc::new(MockTool::new("read_file", "read", "ok")) as Arc<dyn Tool>;
    let write = Arc::new(MockTool::new("write_file", "write", "ok")) as Arc<dyn Tool>;
    let terminal = Arc::new(MockTool::new("terminal", "shell", "ok")) as Arc<dyn Tool>;
    let harness = setup_full_stack_test_agent_with_extra_tools(
        MockProvider::new(),
        vec![read, write, terminal],
    )
    .await
    .unwrap();
    let grant = WorkspaceGrant {
        platform: "slack".to_string(),
        workspace_id: "T_TEST".to_string(),
        channel_id: "C_PRIVATE".to_string(),
        user_id: "U_GUEST".to_string(),
        project_root: "/project".to_string(),
        access: WorkspaceAccessLevel::Read,
        expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
    };
    let channel_ctx = ChannelContext {
        visibility: ChannelVisibility::PrivateGroup,
        platform: "slack".to_string(),
        channel_name: Some("project-chat".to_string()),
        channel_id: Some("slack:C_PRIVATE".to_string()),
        workspace_id: Some("T_TEST".to_string()),
        sender_name: Some("Guest".to_string()),
        sender_id: Some("slack:U_GUEST".to_string()),
        channel_member_names: vec![],
        user_id_map: HashMap::new(),
        workspace_grant: Some(grant),
        trusted: false,
    };

    let roster = harness
        .agent
        .session_static_tool_roster(UserRole::Guest, &channel_ctx);
    let names = roster
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"read_file"));
    assert!(!names.contains(&"write_file"));
    assert!(!names.contains(&"terminal"));
}

#[tokio::test]
async fn session_static_roster_applies_public_external_allowlist() {
    // A public tool (the real web_search declares itself so) and one that
    // is not.
    let web = Arc::new(
        MockTool::new("web_search", "search", "ok").with_dispatch_facts(
            crate::traits::ToolDispatchFacts {
                channel_reach: ChannelReach::Public,
                ..Default::default()
            },
        ),
    ) as Arc<dyn Tool>;
    let other = Arc::new(MockTool::new("alpha_tool", "a", "ok")) as Arc<dyn Tool>;
    let harness =
        setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![web, other])
            .await
            .unwrap();
    let mut channel_ctx = ChannelContext::private("test");
    channel_ctx.visibility = ChannelVisibility::PublicExternal;

    let roster = harness
        .agent
        .session_static_tool_roster(UserRole::Owner, &channel_ctx);
    let names: Vec<&str> = roster.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"web_search"));
    assert!(!names.contains(&"alpha_tool"));
}

#[test]
fn sort_tool_definitions_by_name_orders_by_name() {
    let mut defs = vec![
        named_tool_def("zebra"),
        named_tool_def("alpha"),
        named_tool_def("mango"),
    ];
    sort_tool_definitions_by_name(&mut defs);
    let names: Vec<&str> = defs.iter().filter_map(tool_name_from_definition).collect();
    assert_eq!(names, vec!["alpha", "mango", "zebra"]);
}

#[test]
fn sort_tool_definitions_by_name_tiebreaks_on_serialized_bytes() {
    // Same name, different bodies — deterministic order by serialized bytes.
    let mut a = named_tool_def("dup");
    a["function"]["description"] = json!("aaa");
    let mut b = named_tool_def("dup");
    b["function"]["description"] = json!("bbb");

    let mut defs1 = vec![b.clone(), a.clone()];
    let mut defs2 = vec![a.clone(), b.clone()];
    sort_tool_definitions_by_name(&mut defs1);
    sort_tool_definitions_by_name(&mut defs2);
    assert_eq!(defs1, defs2);
}

#[tokio::test]
async fn private_owner_roster_keeps_safe_capabilities_without_keyword_routing() {
    let extra_tools = [
        "manage_mandates",
        "manage_memories",
        "http_request",
        "manage_api",
    ]
    .into_iter()
    .map(|name| Arc::new(MockTool::new(name, "test capability", "ok")) as Arc<dyn Tool>)
    .collect();
    let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), extra_tools)
        .await
        .unwrap();

    for request in [
        "Prepare a mandate draft. Do not schedule or activate it.",
        "Review every 15 minutes and schedule the work.",
        "Help me decide what to do next.",
    ] {
        let (defs, _) = harness
            .agent
            .tool_definitions_with_capabilities(request)
            .await;
        let names = defs
            .iter()
            .filter_map(tool_name_from_definition)
            .collect::<Vec<_>>();
        assert!(names.contains(&"manage_mandates"));
        assert!(names.contains(&"manage_memories"));
        assert!(names.contains(&"http_request"));
        assert!(names.contains(&"manage_api"));
    }
}

#[tokio::test]
async fn task_lead_policy_filter_keeps_delegation_tools_exposed() {
    // The delegation surface, stating the fact the real tools state.
    let delegation = |name: &str| -> Arc<dyn crate::traits::Tool> {
        Arc::new(
            crate::testing::MockTool::new(name, "delegation", "ok").with_dispatch_facts(
                crate::traits::ToolDispatchFacts {
                    task_lead_delegation: true,
                    ..Default::default()
                },
            ),
        )
    };
    let mut harness = setup_full_stack_test_agent_with_extra_tools(
        MockProvider::new(),
        vec![
            delegation("manage_goal_tasks"),
            delegation("cli_agent"),
            delegation("spawn_agent"),
        ],
    )
    .await
    .unwrap();
    harness.agent.set_test_task_lead_mode();

    let defs = vec![
        named_tool_def("system_info"),
        named_tool_def("remember_fact"),
        named_tool_def("policy_metrics"),
        named_tool_def("search_files"),
        named_tool_def("read_file"),
        named_tool_def("check_environment"),
        named_tool_def("goal_trace"),
        named_tool_def("manage_goal_tasks"),
        named_tool_def("cli_agent"),
        named_tool_def("spawn_agent"),
    ];
    let capabilities: HashMap<String, ToolCapabilities> = HashMap::from([
        (
            "system_info".to_string(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
        ),
        (
            "remember_fact".to_string(),
            ToolCapabilities {
                read_only: false,
                external_side_effect: false,
                needs_approval: false,
                idempotent: false,
                high_impact_write: false,
            },
        ),
        (
            "policy_metrics".to_string(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
        ),
        (
            "search_files".to_string(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
        ),
        (
            "read_file".to_string(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
        ),
        (
            "check_environment".to_string(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
        ),
        (
            "goal_trace".to_string(),
            ToolCapabilities {
                read_only: true,
                external_side_effect: false,
                needs_approval: false,
                idempotent: true,
                high_impact_write: false,
            },
        ),
        (
            "manage_goal_tasks".to_string(),
            ToolCapabilities {
                read_only: false,
                external_side_effect: false,
                needs_approval: false,
                idempotent: false,
                high_impact_write: false,
            },
        ),
        (
            "cli_agent".to_string(),
            ToolCapabilities {
                read_only: false,
                external_side_effect: true,
                needs_approval: true,
                idempotent: false,
                high_impact_write: true,
            },
        ),
        (
            "spawn_agent".to_string(),
            ToolCapabilities {
                read_only: false,
                external_side_effect: false,
                needs_approval: false,
                idempotent: false,
                high_impact_write: true,
            },
        ),
    ]);

    let filtered = harness.agent.filter_tool_definitions_for_policy(
        &defs,
        &capabilities,
        &ExecutionPolicy::for_profile(ModelProfile::Balanced),
    );
    let names: Vec<String> = filtered
        .iter()
        .filter_map(tool_name_from_definition)
        .map(ToString::to_string)
        .collect();

    assert!(names.contains(&"manage_goal_tasks".to_string()));
    assert!(names.contains(&"cli_agent".to_string()));
    assert!(names.contains(&"spawn_agent".to_string()));
}
