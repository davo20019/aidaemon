use super::*;
use crate::testing::MockTool;
use crate::traits::ToolRole;

/// Mock tool that returns a specific ToolRole.
struct MockRoleTool {
    tool_name: String,
    role: ToolRole,
}

impl MockRoleTool {
    fn new(name: &str, role: ToolRole) -> Self {
        Self {
            tool_name: name.to_string(),
            role,
        }
    }
}

#[async_trait::async_trait]
impl Tool for MockRoleTool {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> &str {
        "mock"
    }
    fn schema(&self) -> Value {
        json!({
            "name": self.tool_name,
            "description": "mock",
            "parameters": { "type": "object", "properties": {} }
        })
    }
    fn tool_role(&self) -> ToolRole {
        self.role
    }
    async fn call(&self, _args: &str) -> anyhow::Result<String> {
        Ok("ok".to_string())
    }
}

#[test]
fn test_tool_scoping_task_lead() {
    // Simulate tool filtering for task lead role
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(MockRoleTool::new("terminal", ToolRole::Action)),
        Arc::new(MockRoleTool::new("web_search", ToolRole::Action)),
        Arc::new(MockRoleTool::new("system_info", ToolRole::Universal)),
        Arc::new(MockRoleTool::new("remember_fact", ToolRole::Universal)),
        Arc::new(MockRoleTool::new("plan_manager", ToolRole::Management)),
    ];

    // Task lead filter: Management + Universal only
    let tl_tools: Vec<String> = tools
        .iter()
        .filter(|t| t.name() != "spawn_agent")
        .filter(|t| matches!(t.tool_role(), ToolRole::Management | ToolRole::Universal))
        .map(|t| t.name().to_string())
        .collect();

    assert!(tl_tools.contains(&"system_info".to_string()));
    assert!(tl_tools.contains(&"remember_fact".to_string()));
    assert!(tl_tools.contains(&"plan_manager".to_string()));
    assert!(!tl_tools.contains(&"terminal".to_string()));
    assert!(!tl_tools.contains(&"web_search".to_string()));
    assert_eq!(tl_tools.len(), 3);
}

#[test]
fn test_tool_scoping_executor() {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(MockRoleTool::new("terminal", ToolRole::Action)),
        Arc::new(MockRoleTool::new("web_search", ToolRole::Action)),
        Arc::new(MockRoleTool::new("system_info", ToolRole::Universal)),
        Arc::new(MockRoleTool::new("remember_fact", ToolRole::Universal)),
        Arc::new(MockRoleTool::new("plan_manager", ToolRole::Management)),
    ];

    // Executor filter: Action + Universal only
    let exec_tools: Vec<String> = tools
        .iter()
        .filter(|t| t.name() != "spawn_agent")
        .filter(|t| matches!(t.tool_role(), ToolRole::Action | ToolRole::Universal))
        .map(|t| t.name().to_string())
        .collect();

    assert!(exec_tools.contains(&"terminal".to_string()));
    assert!(exec_tools.contains(&"web_search".to_string()));
    assert!(exec_tools.contains(&"system_info".to_string()));
    assert!(exec_tools.contains(&"remember_fact".to_string()));
    assert!(!exec_tools.contains(&"plan_manager".to_string()));
    assert_eq!(exec_tools.len(), 4);
}

#[test]
fn test_tool_scoping_legacy_no_filter() {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(MockRoleTool::new("terminal", ToolRole::Action)),
        Arc::new(MockRoleTool::new("system_info", ToolRole::Universal)),
        Arc::new(MockRoleTool::new("plan_manager", ToolRole::Management)),
        Arc::new(MockRoleTool::new("spawn_agent", ToolRole::Action)),
    ];

    // Legacy: filter out spawn_agent only, keep everything else
    let legacy_tools: Vec<String> = tools
        .iter()
        .filter(|t| t.name() != "spawn_agent")
        .map(|t| t.name().to_string())
        .collect();

    assert_eq!(legacy_tools.len(), 3);
    assert!(legacy_tools.contains(&"terminal".to_string()));
    assert!(legacy_tools.contains(&"system_info".to_string()));
    assert!(legacy_tools.contains(&"plan_manager".to_string()));
}

#[test]
fn test_agent_role_default() {
    assert_eq!(AgentRole::Orchestrator, AgentRole::Orchestrator);
    assert_ne!(AgentRole::TaskLead, AgentRole::Executor);
}

#[test]
fn test_tool_role_default() {
    // Verify that MockTool (from testing.rs) defaults to Action
    let mock = MockTool::new("test", "desc", "result");
    assert_eq!(mock.tool_role(), ToolRole::Action);
}

#[test]
fn test_system_info_tool_is_universal() {
    let tool = crate::tools::SystemInfoTool;
    assert_eq!(tool.tool_role(), ToolRole::Universal);
}
