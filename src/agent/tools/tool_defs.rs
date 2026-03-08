use super::*;
use std::collections::{HashMap, HashSet};

fn validate_required_fields_contract(parameters: &Value) -> Result<(), String> {
    let properties = parameters
        .get("properties")
        .ok_or_else(|| "missing parameters.properties".to_string())?
        .as_object()
        .ok_or_else(|| "parameters.properties must be an object".to_string())?;

    if let Some(required) = parameters.get("required") {
        let required_items = required
            .as_array()
            .ok_or_else(|| "parameters.required must be an array".to_string())?;
        for item in required_items {
            let key = item
                .as_str()
                .ok_or_else(|| "parameters.required entries must be strings".to_string())?;
            if !properties.contains_key(key) {
                return Err(format!(
                    "parameters.required references unknown property '{}'",
                    key
                ));
            }
        }
    }

    if let Some(additional) = parameters.get("additionalProperties") {
        if !additional.is_boolean() {
            return Err("parameters.additionalProperties must be a boolean".to_string());
        }
    }

    Ok(())
}

impl Agent {
    pub(crate) fn validate_tool_definition_contract(def: &Value) -> Result<(), String> {
        let func = def
            .get("function")
            .ok_or_else(|| "missing function object".to_string())?;
        let name = func
            .get("name")
            .and_then(|n| n.as_str())
            .map(str::trim)
            .ok_or_else(|| "missing function.name".to_string())?;
        if name.is_empty() {
            return Err("function.name must be non-empty".to_string());
        }
        if func
            .get("description")
            .and_then(|d| d.as_str())
            .is_none_or(|d| d.trim().is_empty())
        {
            return Err(format!("tool '{}' is missing function.description", name));
        }
        let parameters = func
            .get("parameters")
            .ok_or_else(|| format!("tool '{}' is missing function.parameters", name))?;
        if parameters.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Err(format!("tool '{}' must use object parameters schema", name));
        }
        validate_required_fields_contract(parameters)?;
        Ok(())
    }

    /// Build OpenAI-format tool definitions plus capability metadata map.
    pub(super) async fn tool_definitions_with_capabilities(
        &self,
        user_message: &str,
    ) -> (Vec<Value>, HashMap<String, ToolCapabilities>) {
        let mut defs: Vec<Value> = Vec::new();
        let mut capabilities: HashMap<String, ToolCapabilities> = HashMap::new();

        for tool in &self.tools {
            if !tool.is_available() {
                continue;
            }
            let name = tool.name().to_string();
            capabilities.insert(name.clone(), tool.capabilities());
            let candidate = json!({
                "type": "function",
                "function": tool.schema()
            });
            match Self::validate_tool_definition_contract(&candidate) {
                Ok(()) => defs.push(candidate),
                Err(reason) => {
                    POLICY_METRICS
                        .tool_schema_contract_rejections_total
                        .fetch_add(1, Ordering::Relaxed);
                    warn!(
                        tool = %name,
                        error = %reason,
                        "Dropping tool definition that violates schema contract"
                    );
                }
            }
        }

        // MCP composition stage 1: explicit trigger matching
        if let Some(ref registry) = self.mcp_registry {
            let mcp_tools = registry.match_tools(user_message).await;
            for tool in mcp_tools {
                let name = tool.name().to_string();
                capabilities.entry(name.clone()).or_default();
                let candidate = json!({
                    "type": "function",
                    "function": tool.schema()
                });
                match Self::validate_tool_definition_contract(&candidate) {
                    Ok(()) => defs.push(candidate),
                    Err(reason) => {
                        POLICY_METRICS
                            .tool_schema_contract_rejections_total
                            .fetch_add(1, Ordering::Relaxed);
                        warn!(
                            tool = %name,
                            error = %reason,
                            "Dropping MCP tool definition that violates schema contract"
                        );
                    }
                }
            }
        }

        (defs, capabilities)
    }

    /// Build the OpenAI-format tool definitions.
    #[allow(dead_code)]
    async fn tool_definitions(&self, user_message: &str) -> Vec<Value> {
        self.tool_definitions_with_capabilities(user_message)
            .await
            .0
    }

    pub(super) fn has_available_tool(&self, tool_name: &str) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.name() == tool_name && tool.is_available())
    }

    pub(super) fn has_cli_agents_available(&self) -> bool {
        self.has_available_tool("cli_agent")
    }

    pub(super) fn tool_name_from_definition(def: &Value) -> Option<&str> {
        def.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
    }

    fn connected_api_tools_to_pin(user_message: &str) -> Option<&'static [&'static str]> {
        match crate::agent::intent_routing::classify_connected_api_intent(user_message) {
            Some(crate::agent::intent_routing::ConnectedApiIntent::RuntimeCapabilityValidation)
            | Some(crate::agent::intent_routing::ConnectedApiIntent::ReadAction)
            | Some(crate::agent::intent_routing::ConnectedApiIntent::WriteAction) => Some(&[
                "manage_api",
                "manage_oauth",
                "manage_http_auth",
                "manage_skills",
                "http_request",
            ]),
            None => None,
        }
    }

    pub(super) fn ensure_connected_api_tools_exposed(
        &self,
        user_message: &str,
        filtered_defs: &[Value],
        base_defs: &[Value],
    ) -> Vec<Value> {
        let Some(pinned_names) = Self::connected_api_tools_to_pin(user_message) else {
            return filtered_defs.to_vec();
        };
        let base_by_name: HashMap<String, Value> = base_defs
            .iter()
            .filter_map(|def| {
                let name = Self::tool_name_from_definition(def)?.to_string();
                Some((name, def.clone()))
            })
            .collect();

        let mut exposed: Vec<Value> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for name in pinned_names {
            if let Some(def) = base_by_name.get(*name) {
                seen.insert((*name).to_string());
                exposed.push(def.clone());
            }
        }

        for def in filtered_defs {
            let Some(name) = Self::tool_name_from_definition(def) else {
                continue;
            };
            if seen.insert(name.to_string()) {
                exposed.push(def.clone());
            }
        }

        exposed
    }

    pub(super) fn filter_tool_definitions_for_policy(
        &self,
        defs: &[Value],
        capabilities: &HashMap<String, ToolCapabilities>,
        policy: &ExecutionPolicy,
        risk_score: f32,
        widen: bool,
    ) -> Vec<Value> {
        let mut ordered: Vec<(Value, String, ToolCapabilities)> = defs
            .iter()
            .filter_map(|def| {
                let name = Self::tool_name_from_definition(def)?.to_string();
                let caps = capabilities.get(&name).copied().unwrap_or_default();
                Some((def.clone(), name, caps))
            })
            .collect();

        // Essential tools that must always be available regardless of profile/approval filters.
        // Without these, the agent can read files but never write them — rendering coding useless.
        // Memory tools are included because the agent's core personal-assistant function
        // depends on being able to store and manage facts/people at any risk level.
        // Web tools are essential because a personal assistant must be able to search the web
        // and fetch URLs — without these, the model resorts to terminal curl/grep workarounds.
        const ESSENTIAL_TOOLS: &[&str] = &[
            "write_file",
            "edit_file",
            "terminal",
            "remember_fact",
            "manage_memories",
            "manage_people",
            "web_search",
            "web_fetch",
            "manage_api",
            "http_request",
            "manage_http_auth",
            "manage_oauth",
            "send_file",
        ];

        // Stable prioritization: essential tools first, then read-only + idempotent.
        // Essential tools must sort before truncation cuts them off.
        ordered.sort_by_key(|(_, name, caps)| {
            let is_essential = ESSENTIAL_TOOLS.contains(&name.as_str());
            (
                !is_essential, // essential tools first
                !caps.read_only,
                caps.needs_approval,
                !caps.idempotent,
                caps.high_impact_write,
                caps.external_side_effect,
            )
        });

        if widen {
            return ordered.into_iter().map(|(d, _, _)| d).collect();
        }

        let mut filtered: Vec<(Value, String, ToolCapabilities)> = ordered;
        let low_risk = risk_score < 0.34 && matches!(policy.model_profile, ModelProfile::Cheap);

        if low_risk {
            // Start with essential tools (always available) + read-only tools.
            let mut keep: Vec<_> = filtered
                .iter()
                .filter(|(_, name, c)| c.read_only || ESSENTIAL_TOOLS.contains(&name.as_str()))
                .cloned()
                .collect();
            // Fill up to a minimum of 5 with remaining tools from the sorted list.
            if keep.len() < 5 {
                for candidate in filtered.iter().cloned() {
                    if keep.iter().any(|(_, n, _)| n == &candidate.1) {
                        continue;
                    }
                    keep.push(candidate);
                    if keep.len() >= 5 {
                        break;
                    }
                }
            }
            if keep.len() > 16 {
                keep.truncate(16);
            }
            return keep.into_iter().map(|(d, _, _)| d).collect();
        }

        match policy.model_profile {
            ModelProfile::Cheap => {
                filtered.retain(|(_, name, caps)| {
                    ESSENTIAL_TOOLS.contains(&name.as_str())
                        || caps.read_only
                        || !caps.high_impact_write
                });
                filtered.truncate(16);
            }
            ModelProfile::Balanced => {
                if risk_score < 0.55 {
                    filtered.retain(|(_, name, caps)| {
                        ESSENTIAL_TOOLS.contains(&name.as_str())
                            || caps.read_only
                            || !caps.high_impact_write
                    });
                }
                filtered.truncate(20);
            }
            ModelProfile::Strong => {
                // Keep strong turns capable, but avoid exposing an unbounded tool surface.
                filtered.truncate(28);
            }
        }

        if matches!(policy.approval_mode, ApprovalMode::Auto) {
            filtered.retain(|(_, name, caps)| {
                ESSENTIAL_TOOLS.contains(&name.as_str()) || caps.read_only || !caps.needs_approval
            });
        }

        filtered.into_iter().map(|(d, _, _)| d).collect()
    }

    pub(super) async fn load_policy_tool_set(
        &self,
        user_message: &str,
        channel_visibility: ChannelVisibility,
        policy: &ExecutionPolicy,
        risk_score: f32,
        enforce_filter: bool,
    ) -> (Vec<Value>, Vec<Value>, HashMap<String, ToolCapabilities>) {
        let (mut defs, mut caps) = self.tool_definitions_with_capabilities(user_message).await;

        if channel_visibility == ChannelVisibility::PublicExternal {
            let allowed = ["web_search", "remember_fact", "system_info"];
            defs.retain(|d| {
                Self::tool_name_from_definition(d).is_some_and(|name| allowed.contains(&name))
            });
            caps.retain(|name, _| allowed.contains(&name.as_str()));
        }

        let base_defs = defs.clone();
        if enforce_filter {
            defs = self.filter_tool_definitions_for_policy(&defs, &caps, policy, risk_score, false);
            defs = self.ensure_connected_api_tools_exposed(user_message, &defs, &base_defs);
        }

        (defs, base_defs, caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{setup_full_stack_test_agent_with_extra_tools, MockProvider, MockTool};
    use crate::traits::Tool;
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

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }

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
        assert!(Agent::validate_tool_definition_contract(&def).is_ok());
    }

    proptest! {
        #[test]
        fn tool_definition_contract_rejects_invalid_required_keys(required_key in "[a-z]{1,12}") {
            let mut def = valid_tool_def();
            def["function"]["parameters"]["required"] = json!([required_key, "missing_key"]);
            let result = Agent::validate_tool_definition_contract(&def);
            prop_assert!(result.is_err());
        }

        #[test]
        fn tool_definition_contract_rejects_non_boolean_additional_properties(flag in ".*") {
            let mut def = valid_tool_def();
            def["function"]["parameters"]["additionalProperties"] = json!(flag);
            let result = Agent::validate_tool_definition_contract(&def);
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
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"web_search".to_string()));
        assert!(!names.contains(&"cli_agent".to_string()));
        assert!(caps.contains_key("web_search"));
        assert!(!caps.contains_key("cli_agent"));
        assert!(!harness.agent.has_cli_agents_available());
    }

    #[tokio::test]
    async fn runtime_validation_queries_pin_connected_api_tools() {
        let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();

        let filtered = vec![named_tool_def("search_files"), named_tool_def("terminal")];
        let base = vec![
            named_tool_def("search_files"),
            named_tool_def("manage_api"),
            named_tool_def("http_request"),
            named_tool_def("manage_http_auth"),
            named_tool_def("manage_skills"),
            named_tool_def("manage_oauth"),
            named_tool_def("terminal"),
        ];

        let exposed = harness.agent.ensure_connected_api_tools_exposed(
            "Can you verify whether you can post to Twitter/X right now before answering?",
            &filtered,
            &base,
        );
        let names: Vec<String> = exposed
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"manage_api".to_string()));
        assert!(names.contains(&"http_request".to_string()));
        assert!(names.contains(&"manage_http_auth".to_string()));
        assert!(names.contains(&"manage_skills".to_string()));
        assert!(names.contains(&"manage_oauth".to_string()));
        assert_eq!(names.first().map(String::as_str), Some("manage_api"));
    }

    #[tokio::test]
    async fn connected_api_write_queries_pin_connected_api_tools() {
        let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();

        let filtered = vec![named_tool_def("search_files"), named_tool_def("terminal")];
        let base = vec![
            named_tool_def("search_files"),
            named_tool_def("manage_api"),
            named_tool_def("http_request"),
            named_tool_def("manage_http_auth"),
            named_tool_def("manage_skills"),
            named_tool_def("manage_oauth"),
            named_tool_def("terminal"),
        ];

        let exposed = harness.agent.ensure_connected_api_tools_exposed(
            "Create a GitHub issue for this regression.",
            &filtered,
            &base,
        );
        let names: Vec<String> = exposed
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"manage_api".to_string()));
        assert!(names.contains(&"http_request".to_string()));
        assert!(names.contains(&"manage_http_auth".to_string()));
        assert!(names.contains(&"manage_skills".to_string()));
        assert!(names.contains(&"manage_oauth".to_string()));
    }

    #[tokio::test]
    async fn connected_api_read_queries_pin_connected_api_tools() {
        let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();

        let filtered = vec![named_tool_def("search_files"), named_tool_def("terminal")];
        let base = vec![
            named_tool_def("search_files"),
            named_tool_def("manage_api"),
            named_tool_def("http_request"),
            named_tool_def("manage_http_auth"),
            named_tool_def("manage_skills"),
            named_tool_def("manage_oauth"),
            named_tool_def("terminal"),
        ];

        let exposed = harness.agent.ensure_connected_api_tools_exposed(
            "List my open GitHub issues.",
            &filtered,
            &base,
        );

        let names: Vec<String> = exposed
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"manage_api".to_string()));
        assert!(names.contains(&"http_request".to_string()));
        assert!(names.contains(&"manage_http_auth".to_string()));
        assert!(names.contains(&"manage_skills".to_string()));
        assert!(names.contains(&"manage_oauth".to_string()));
    }

    #[tokio::test]
    async fn non_connected_api_queries_do_not_pin_connected_api_tools() {
        let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();

        let filtered = vec![named_tool_def("search_files"), named_tool_def("terminal")];
        let base = vec![
            named_tool_def("search_files"),
            named_tool_def("manage_api"),
            named_tool_def("http_request"),
            named_tool_def("manage_http_auth"),
            named_tool_def("manage_skills"),
            named_tool_def("manage_oauth"),
            named_tool_def("terminal"),
        ];

        let exposed = harness.agent.ensure_connected_api_tools_exposed(
            "What's your twitter account?",
            &filtered,
            &base,
        );
        let names: Vec<String> = exposed
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert_eq!(
            names,
            vec!["search_files".to_string(), "terminal".to_string()]
        );
    }

    #[tokio::test]
    async fn policy_filter_keeps_connected_api_tools_exposed() {
        let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();

        let defs = vec![
            named_tool_def("search_files"),
            named_tool_def("read_file"),
            named_tool_def("manage_api"),
            named_tool_def("http_request"),
            named_tool_def("manage_http_auth"),
            named_tool_def("manage_oauth"),
        ];
        let capabilities: HashMap<String, ToolCapabilities> = HashMap::from([
            ("search_files".to_string(), ToolCapabilities::default()),
            ("read_file".to_string(), ToolCapabilities::default()),
            (
                "manage_api".to_string(),
                ToolCapabilities {
                    read_only: false,
                    external_side_effect: true,
                    needs_approval: true,
                    idempotent: false,
                    high_impact_write: true,
                },
            ),
            (
                "http_request".to_string(),
                ToolCapabilities {
                    read_only: false,
                    external_side_effect: true,
                    needs_approval: true,
                    idempotent: false,
                    high_impact_write: false,
                },
            ),
            (
                "manage_http_auth".to_string(),
                ToolCapabilities {
                    read_only: false,
                    external_side_effect: true,
                    needs_approval: true,
                    idempotent: false,
                    high_impact_write: true,
                },
            ),
            (
                "manage_oauth".to_string(),
                ToolCapabilities {
                    read_only: false,
                    external_side_effect: true,
                    needs_approval: true,
                    idempotent: false,
                    high_impact_write: true,
                },
            ),
        ]);

        let filtered = harness.agent.filter_tool_definitions_for_policy(
            &defs,
            &capabilities,
            &ExecutionPolicy::for_profile(ModelProfile::Cheap),
            0.2,
            false,
        );
        let names: Vec<String> = filtered
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"manage_api".to_string()));
        assert!(names.contains(&"http_request".to_string()));
        assert!(names.contains(&"manage_http_auth".to_string()));
        assert!(names.contains(&"manage_oauth".to_string()));
    }
}
