//! `manage_auth`: one tool for an API's credentials, split by `kind`.
//! `kind: "profile"` runs the manual auth profile actions
//! ([`ManageHttpAuthTool`]: bearer, header, basic and OAuth 1.0a tokens in
//! the keychain); `kind: "oauth"` runs the OAuth connection and provider
//! actions ([`ManageOAuthTool`]). Every action, its owner card and its
//! declared effects are the inner tool's: this tool only routes the call
//! (without `kind`) to the one its `kind` names.

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::OutcomeTool as Tool;
use crate::semantics::{
    ApprovalCall, OwnerApprovalNeed, StableObservationSubject, ToolCallContext, ToolCallOutcome,
    ToolCallSemantics, ToolCapabilities,
};
use crate::services::manage_http_auth::ManageHttpAuthTool;
use crate::services::manage_oauth::ManageOAuthTool;

pub struct ManageAuthTool {
    profile: ManageHttpAuthTool,
    oauth: ManageOAuthTool,
}

impl ManageAuthTool {
    pub fn new(profile: ManageHttpAuthTool, oauth: ManageOAuthTool) -> Self {
        Self { profile, oauth }
    }

    /// The inner tool a call's `kind` names, and the call without `kind`.
    fn route(&self, arguments: &Value) -> Option<(&dyn Tool, Value)> {
        let mut forwarded = arguments.as_object()?.clone();
        let tool: &dyn Tool = match forwarded.remove("kind")?.as_str()? {
            "profile" => &self.profile,
            "oauth" => &self.oauth,
            _ => return None,
        };
        Some((tool, Value::Object(forwarded)))
    }

    /// [`Self::route`] over arguments as the model sent them.
    fn route_str(&self, arguments: &str) -> Option<(&dyn Tool, String)> {
        let parsed = serde_json::from_str::<Value>(arguments).ok()?;
        let (tool, forwarded) = self.route(&parsed)?;
        Some((tool, forwarded.to_string()))
    }
}

const TOOL_DESCRIPTION: &str = "API credentials. kind=profile: manual auth profiles (bearer, header, basic, oauth1a), verify. kind=oauth: OAuth connections and providers; inspect first, connect reauthorizes, bind_account proves identity. Secrets never in args: owner runs `aidaemon keychain set <key>`. New API end to end: manage_api.";

/// `kind` routes the call; every other property is the inner tools'
/// union (enums merged, the first description kept).
fn merged_schema(profile: &Value, oauth: &Value) -> Value {
    let mut properties = Map::new();
    properties.insert(
        "kind".to_string(),
        json!({
            "type": "string",
            "enum": ["profile", "oauth"],
            "description": "profile: actions list, describe, upsert, remove, verify. oauth: actions connect, list, bind_account, remove, set_credentials, refresh, providers, describe_provider, register_provider, remove_provider."
        }),
    );
    for schema in [profile, oauth] {
        let Some(inner) = schema["parameters"]["properties"].as_object() else {
            continue;
        };
        for (name, property) in inner {
            match properties.get_mut(name) {
                None => {
                    properties.insert(name.clone(), property.clone());
                }
                Some(existing) => {
                    if let (Some(have), Some(more)) = (
                        existing.get("enum").and_then(Value::as_array).cloned(),
                        property.get("enum").and_then(Value::as_array),
                    ) {
                        let mut merged = have;
                        for value in more {
                            if !merged.contains(value) {
                                merged.push(value.clone());
                            }
                        }
                        existing["enum"] = Value::Array(merged);
                    }
                    if existing.get("description").is_none()
                        && let Some(description) = property.get("description")
                    {
                        existing["description"] = description.clone();
                    }
                }
            }
        }
    }
    json!({
        "name": "manage_auth",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["kind", "action"],
            "additionalProperties": false
        }
    })
}

/// The answer to a call whose `kind` is missing or unknown; nothing ran.
const KIND_REQUIRED: &str = "manage_auth needs kind: \"profile\" (manual auth profiles) or \"oauth\" (OAuth connections and providers). Nothing ran.";

#[async_trait]
impl Tool for ManageAuthTool {
    fn name(&self) -> &str {
        "manage_auth"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        merged_schema(&self.profile.schema(), &self.oauth.schema())
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        match self.route_str(arguments) {
            Some((tool, forwarded)) => tool.execute(&forwarded, facts).await,
            None => Ok(ToolCallOutcome::from_output(KIND_REQUIRED.to_string())),
        }
    }
}

#[async_trait]
impl crate::semantics::ToolSemantics for ManageAuthTool {
    /// The inner tool's card: changing a profile, binding an account,
    /// disconnecting and registering or removing a provider ask the owner.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        let Some((tool, arguments)) = self.route(call.arguments) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let as_sent = self
            .route(call.as_sent)
            .map(|(_, as_sent)| as_sent)
            .unwrap_or_else(|| arguments.clone());
        tool.owner_approval_request(
            &ApprovalCall {
                arguments: &arguments,
                as_sent: &as_sent,
                refused: call.refused,
            },
            facts,
        )
        .await
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        match self.route_str(arguments) {
            Some((tool, forwarded)) => tool.call_effects(&forwarded),
            // A call without a kind runs nothing.
            None => aidaemon_core::CallEffects::none(),
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        match self.route_str(arguments) {
            Some((tool, forwarded)) => tool.call_semantics(&forwarded),
            None => ToolCallSemantics::observation(),
        }
    }

    fn stable_observation_subjects(&self) -> Vec<StableObservationSubject> {
        let mut subjects = self.profile.stable_observation_subjects();
        subjects.extend(self.oauth.stable_observation_subjects());
        subjects
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    fn output_trust(
        &self,
        arguments: &str,
        roots: &crate::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        match self.route_str(arguments) {
            Some((tool, forwarded)) => tool.output_trust(&forwarded, roots),
            None => aidaemon_core::SourceTrust::Trusted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schema_merges_both_tools_actions_and_auth_types() {
        let profile = json!({"parameters": {"properties": {
            "action": {"type": "string", "enum": ["list", "upsert", "remove"]},
            "auth_type": {"type": "string", "enum": ["bearer"]},
            "profile": {"type": "string"}
        }}});
        let oauth = json!({"parameters": {"properties": {
            "action": {"type": "string", "enum": ["connect", "list", "remove"]},
            "auth_type": {"type": "string", "enum": ["oauth2_pkce"]},
            "service": {"type": "string"}
        }}});
        let schema = merged_schema(&profile, &oauth);
        let properties = &schema["parameters"]["properties"];
        assert_eq!(
            properties["action"]["enum"],
            json!(["list", "upsert", "remove", "connect"])
        );
        assert_eq!(
            properties["auth_type"]["enum"],
            json!(["bearer", "oauth2_pkce"])
        );
        assert!(properties["profile"].is_object() && properties["service"].is_object());
        assert_eq!(schema["parameters"]["required"], json!(["kind", "action"]));
    }
}
