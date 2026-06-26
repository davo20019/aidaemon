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

// impl-Agent justification: tool definition assembly and lookup over tools/mcp_registry/skill_cache/role.
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

    /// Build OpenAI-format tool definitions plus capability metadata for the
    /// statically registered tools only. This collection is query-independent:
    /// it iterates `self.tools` (the fixed registered set), validates each
    /// schema, and performs NO user-message gating, NO MCP-trigger matching, and
    /// NO per-turn policy/personal-memory restrictions. It is the shared source
    /// for both `tool_definitions_with_capabilities` (which appends MCP-triggered
    /// tools) and `session_static_tool_roster` (which hashes into the Pillar A
    /// core). Keep these two callers using this helper so the static collection
    /// is never duplicated.
    pub(super) fn registered_tool_definitions_with_capabilities(
        &self,
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

        (defs, capabilities)
    }

    /// Build OpenAI-format tool definitions plus capability metadata map.
    pub(super) async fn tool_definitions_with_capabilities(
        &self,
        user_message: &str,
    ) -> (Vec<Value>, HashMap<String, ToolCapabilities>) {
        let (mut defs, mut capabilities) = self.registered_tool_definitions_with_capabilities();

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

    pub(super) fn has_registered_tool(&self, tool_name: &str) -> bool {
        self.tools.iter().any(|tool| tool.name() == tool_name)
    }

    pub(super) fn has_cli_agents_available(&self) -> bool {
        self.has_available_tool("cli_agent")
    }

    pub(super) fn tool_name_from_definition(def: &Value) -> Option<&str> {
        def.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
    }

    /// Session-static tool roster for the Pillar A core prompt: `(tool name,
    /// serialized schema)` pairs, sorted by tool name.
    ///
    /// This is intentionally query-independent. It is built from the statically
    /// registered, currently-available tools (`registered_tool_definitions_with_capabilities`),
    /// applies ONLY the session-static channel-visibility class (the
    /// `PublicExternal` allowlist), and deliberately EXCLUDES:
    /// - MCP-trigger matching (per user message),
    /// - per-turn policy filtering, personal-memory restriction, and
    ///   untrusted-external-reference restriction.
    ///
    /// Non-owners receive tools based on role (no tool access), so the roster is
    /// empty for any non-`Owner` role — matching the `tools_allowed_for_user`
    /// gate in bootstrap.
    pub(super) fn session_static_tool_roster(
        &self,
        user_role: UserRole,
        visibility: ChannelVisibility,
    ) -> Vec<(String, String)> {
        if user_role != UserRole::Owner {
            return Vec::new();
        }

        let (mut defs, _caps) = self.registered_tool_definitions_with_capabilities();

        // Channel-visibility class is session-static and so stays in the core.
        if visibility == ChannelVisibility::PublicExternal {
            let allowed = ["web_search", "remember_fact", "system_info"];
            defs.retain(|d| {
                Self::tool_name_from_definition(d).is_some_and(|name| allowed.contains(&name))
            });
        }
        // Keep the core-prompt roster consistent with the per-turn tool set:
        // desktop control is never advertised outside DMs/internal sessions.
        if !Self::visibility_allows_desktop_control(visibility) {
            defs.retain(|d| {
                Self::tool_name_from_definition(d)
                    .map(|name| !Self::DESKTOP_CONTROL_TOOLS.contains(&name))
                    .unwrap_or(true)
            });
        }

        let mut roster: Vec<(String, String)> = defs
            .iter()
            .filter_map(|d| {
                let name = Self::tool_name_from_definition(d)?.to_string();
                // Deterministic serialization: serde_json preserves the schema's
                // per-construction-site key order, so this is stable for a given
                // tool version.
                Some((name, d.to_string()))
            })
            .collect();
        roster.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        roster
    }

    /// Sort tool definitions in place by tool name, using serialized definition
    /// bytes as a deterministic tie-breaker. Consumed by Task 6 payload assembly
    /// (both after bootstrap assembly and on the final effective provider
    /// subset) to enforce a stable provider tool-array ordering.
    ///
    /// Sort keys (name, serialized bytes) are computed ONCE per element so
    /// serialization cost is O(n), not O(n log n).
    pub(super) fn sort_tool_definitions_by_name(defs: &mut Vec<Value>) {
        // Compute each element's sort key exactly once, then sort by key.
        let mut keyed: Vec<(String, String, Value)> = defs
            .drain(..)
            .map(|d| {
                let name = Self::tool_name_from_definition(&d)
                    .unwrap_or("")
                    .to_string();
                let serialized = d.to_string();
                (name, serialized, d)
            })
            .collect();
        keyed.sort_by(|(an, as_, _), (bn, bs, _)| an.cmp(bn).then_with(|| as_.cmp(bs)));
        defs.extend(keyed.into_iter().map(|(_, _, d)| d));
    }

    fn request_requires_connected_api_setup_tools(user_message: &str) -> bool {
        crate::agent::intent_routing::user_text_requests_auth_or_integration_management(
            user_message,
        ) || crate::agent::intent_routing::classify_connected_api_intent(user_message).is_some()
    }

    pub(super) fn restrict_connected_api_setup_tools_for_request(
        &self,
        user_message: &str,
        defs: &[Value],
    ) -> Vec<Value> {
        if Self::request_requires_connected_api_setup_tools(user_message) {
            return defs.to_vec();
        }

        defs.iter()
            .filter(|def| {
                !matches!(
                    Self::tool_name_from_definition(def),
                    Some("manage_api" | "manage_http_auth" | "manage_oauth")
                )
            })
            .cloned()
            .collect()
    }

    fn connected_api_tools_to_pin(user_message: &str) -> Option<&'static [&'static str]> {
        if crate::agent::intent_routing::user_text_requests_auth_or_integration_management(
            user_message,
        ) {
            return Some(&[
                "manage_api",
                "manage_oauth",
                "manage_http_auth",
                "http_request",
            ]);
        }

        match crate::agent::intent_routing::classify_connected_api_intent(user_message) {
            Some(crate::agent::intent_routing::ConnectedApiIntent::RuntimeCapabilityValidation)
            | Some(crate::agent::intent_routing::ConnectedApiIntent::ReadAction)
            | Some(crate::agent::intent_routing::ConnectedApiIntent::WriteAction) => Some(&[
                "manage_api",
                "manage_oauth",
                "manage_http_auth",
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
            "http_request",
            "send_file",
        ];
        let role_required_tools: &[&str] = match self.role() {
            // Task leads must retain their delegation surface after policy pruning.
            AgentRole::TaskLead => &["spawn_agent", "cli_agent", "manage_goal_tasks"],
            _ => &[],
        };
        let is_policy_essential =
            |name: &str| ESSENTIAL_TOOLS.contains(&name) || role_required_tools.contains(&name);

        // Stable prioritization: essential tools first, then read-only + idempotent.
        // Essential tools must sort before truncation cuts them off.
        ordered.sort_by_key(|(_, name, caps)| {
            let is_essential = is_policy_essential(name);
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
                .filter(|(_, name, c)| c.read_only || is_policy_essential(name))
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
                    is_policy_essential(name) || caps.read_only || !caps.high_impact_write
                });
                filtered.truncate(16);
            }
            ModelProfile::Balanced => {
                if risk_score < 0.55 {
                    filtered.retain(|(_, name, caps)| {
                        is_policy_essential(name) || caps.read_only || !caps.high_impact_write
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
                is_policy_essential(name) || caps.read_only || !caps.needs_approval
            });
        }

        filtered.into_iter().map(|(d, _, _)| d).collect()
    }

    /// Tools that drive the owner's physical machine (desktop GUI automation).
    /// These are powerful, owner-machine actions: an inbound chat message — or
    /// the bot's own posted content echoed back in a shared channel — must never
    /// be able to launch them. They are therefore offered only in 1-on-1 DMs and
    /// internal/system sessions, never in group or public conversations.
    const DESKTOP_CONTROL_TOOLS: &'static [&'static str] = &["computer_use"];

    /// Whether desktop-control tools may be exposed in a conversation of this
    /// visibility. Only direct messages (`Private`) and internal/system sessions
    /// (`Internal`, e.g. the scheduler or spawned sub-agents) qualify. Group and
    /// public channels never do, so another participant's message cannot reach
    /// the desktop.
    pub(crate) fn visibility_allows_desktop_control(visibility: ChannelVisibility) -> bool {
        matches!(
            visibility,
            ChannelVisibility::Private | ChannelVisibility::Internal
        )
    }

    /// Strip desktop-control tools from `defs`/`caps` when the channel visibility
    /// does not permit them. No-op for `Private`/`Internal`.
    pub(crate) fn restrict_desktop_control_for_visibility(
        defs: &mut Vec<Value>,
        caps: &mut HashMap<String, ToolCapabilities>,
        visibility: ChannelVisibility,
    ) {
        if Self::visibility_allows_desktop_control(visibility) {
            return;
        }
        defs.retain(|d| {
            Self::tool_name_from_definition(d)
                .map(|name| !Self::DESKTOP_CONTROL_TOOLS.contains(&name))
                .unwrap_or(true)
        });
        caps.retain(|name, _| !Self::DESKTOP_CONTROL_TOOLS.contains(&name.as_str()));
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

        // Desktop control is owner-machine-only: never expose it outside DMs and
        // internal sessions, so a shared-channel message can't trigger it.
        Self::restrict_desktop_control_for_visibility(&mut defs, &mut caps, channel_visibility);

        let base_defs = defs.clone();
        defs = self.restrict_connected_api_setup_tools_for_request(user_message, &defs);
        if enforce_filter {
            defs = self.filter_tool_definitions_for_policy(&defs, &caps, policy, risk_score, false);
            defs = self.restrict_connected_api_setup_tools_for_request(user_message, &defs);
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

    #[test]
    fn desktop_control_allowed_only_in_dms_and_internal() {
        // Direct messages and internal/system sessions may drive the desktop.
        assert!(Agent::visibility_allows_desktop_control(
            ChannelVisibility::Private
        ));
        assert!(Agent::visibility_allows_desktop_control(
            ChannelVisibility::Internal
        ));
        // Group and public conversations must not — a participant's message
        // (or the bot's own echoed content) could otherwise launch it.
        assert!(!Agent::visibility_allows_desktop_control(
            ChannelVisibility::PrivateGroup
        ));
        assert!(!Agent::visibility_allows_desktop_control(
            ChannelVisibility::Public
        ));
        assert!(!Agent::visibility_allows_desktop_control(
            ChannelVisibility::PublicExternal
        ));
    }

    #[test]
    fn restrict_desktop_control_strips_computer_use_in_channels() {
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
            Agent::restrict_desktop_control_for_visibility(&mut defs, &mut caps, visibility);
            let names: Vec<&str> = defs
                .iter()
                .filter_map(Agent::tool_name_from_definition)
                .collect();
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
    fn restrict_desktop_control_keeps_computer_use_in_dm_and_internal() {
        for visibility in [ChannelVisibility::Private, ChannelVisibility::Internal] {
            let mut defs = vec![named_tool_def("computer_use"), named_tool_def("web_search")];
            let mut caps =
                HashMap::from([("computer_use".to_string(), ToolCapabilities::default())]);
            Agent::restrict_desktop_control_for_visibility(&mut defs, &mut caps, visibility);
            let names: Vec<&str> = defs
                .iter()
                .filter_map(Agent::tool_name_from_definition)
                .collect();
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
    async fn session_static_roster_is_deterministic_and_sorted() {
        let t1 = Arc::new(MockTool::new("zebra_tool", "z", "ok")) as Arc<dyn Tool>;
        let t2 = Arc::new(MockTool::new("alpha_tool", "a", "ok")) as Arc<dyn Tool>;
        let harness =
            setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![t1, t2])
                .await
                .unwrap();

        let roster_a = harness
            .agent
            .session_static_tool_roster(UserRole::Owner, ChannelVisibility::Private);
        let roster_b = harness
            .agent
            .session_static_tool_roster(UserRole::Owner, ChannelVisibility::Private);
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

        assert!(harness
            .agent
            .session_static_tool_roster(UserRole::Guest, ChannelVisibility::Private)
            .is_empty());
        assert!(harness
            .agent
            .session_static_tool_roster(UserRole::Public, ChannelVisibility::Private)
            .is_empty());
    }

    #[tokio::test]
    async fn session_static_roster_applies_public_external_allowlist() {
        let web = Arc::new(MockTool::new("web_search", "search", "ok")) as Arc<dyn Tool>;
        let other = Arc::new(MockTool::new("alpha_tool", "a", "ok")) as Arc<dyn Tool>;
        let harness =
            setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![web, other])
                .await
                .unwrap();

        let roster = harness
            .agent
            .session_static_tool_roster(UserRole::Owner, ChannelVisibility::PublicExternal);
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
        Agent::sort_tool_definitions_by_name(&mut defs);
        let names: Vec<&str> = defs
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .collect();
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
        Agent::sort_tool_definitions_by_name(&mut defs1);
        Agent::sort_tool_definitions_by_name(&mut defs2);
        assert_eq!(defs1, defs2);
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
    async fn auth_management_queries_pin_connected_api_tools() {
        let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();

        let filtered = vec![named_tool_def("search_files"), named_tool_def("terminal")];
        let base = vec![
            named_tool_def("search_files"),
            named_tool_def("manage_api"),
            named_tool_def("http_request"),
            named_tool_def("manage_http_auth"),
            named_tool_def("manage_oauth"),
            named_tool_def("terminal"),
        ];

        let exposed = harness.agent.ensure_connected_api_tools_exposed(
            "Reconnect my GitHub OAuth integration.",
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
        assert!(names.contains(&"manage_oauth".to_string()));
    }

    #[tokio::test]
    async fn drafting_queries_strip_connected_api_setup_tools() {
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
        let filtered = harness
            .agent
            .restrict_connected_api_setup_tools_for_request(
                "Can you post a tweet about your new stuff and make it engaging?",
                &filtered,
            );
        let names: Vec<String> = filtered
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        // "post a tweet ... make it engaging" now classifies as WriteAction
        // (DraftThenDeliver), so restrict_connected_api_setup_tools_for_request
        // keeps all defs. However, the upstream policy filter (Cheap, risk=0.2)
        // already removed manage_oauth via the low-risk truncation. The
        // restrict step can only preserve what survived the policy filter.
        assert!(names.contains(&"manage_api".to_string()));
        assert!(names.contains(&"http_request".to_string()));
        assert!(names.contains(&"manage_http_auth".to_string()));
        // manage_oauth was removed by Cheap low-risk policy truncation
        assert!(!names.contains(&"manage_oauth".to_string()));
    }

    #[tokio::test]
    async fn task_lead_policy_filter_keeps_delegation_tools_exposed() {
        let mut harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), vec![])
            .await
            .unwrap();
        harness.agent.set_test_task_lead_mode();

        let defs = vec![
            named_tool_def("system_info"),
            named_tool_def("remember_fact"),
            named_tool_def("policy_metrics"),
            named_tool_def("project_inspect"),
            named_tool_def("git_info"),
            named_tool_def("check_environment"),
            named_tool_def("service_status"),
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
                "project_inspect".to_string(),
                ToolCapabilities {
                    read_only: true,
                    external_side_effect: false,
                    needs_approval: false,
                    idempotent: true,
                    high_impact_write: false,
                },
            ),
            (
                "git_info".to_string(),
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
                "service_status".to_string(),
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
            0.3419,
            false,
        );
        let names: Vec<String> = filtered
            .iter()
            .filter_map(Agent::tool_name_from_definition)
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"manage_goal_tasks".to_string()));
        assert!(names.contains(&"cli_agent".to_string()));
        assert!(names.contains(&"spawn_agent".to_string()));
    }
}
