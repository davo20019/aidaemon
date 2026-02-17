use super::*;

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

    pub(super) fn tool_name_from_definition(def: &Value) -> Option<&str> {
        def.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
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

        // Stable prioritization: read-only + idempotent first for low-risk turns.
        ordered.sort_by_key(|(_, _, caps)| {
            (
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
            let readonly: Vec<_> = filtered
                .iter()
                .filter(|(_, _, c)| c.read_only)
                .cloned()
                .collect();
            let mut keep = readonly;
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
            if keep.len() > 10 {
                keep.truncate(10);
            }
            return keep.into_iter().map(|(d, _, _)| d).collect();
        }

        match policy.model_profile {
            ModelProfile::Cheap => {
                filtered.retain(|(_, _, caps)| caps.read_only || !caps.high_impact_write);
                filtered.truncate(12);
            }
            ModelProfile::Balanced => {
                if risk_score < 0.55 {
                    filtered.retain(|(_, _, caps)| caps.read_only || !caps.high_impact_write);
                }
                filtered.truncate(20);
            }
            ModelProfile::Strong => {}
        }

        if matches!(policy.approval_mode, ApprovalMode::Auto) {
            filtered.retain(|(_, _, caps)| caps.read_only || !caps.needs_approval);
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
        }

        (defs, base_defs, caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn valid_tool_def() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "demo_tool",
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
}
