//! Which tools a turn is offered: the registered roster (schema-checked),
//! the MCP tools a message calls for, the session-static roster the core
//! prompt names, and the per-turn narrowing by execution policy
//! ([`ToolRoster`], on any [`TurnHost`]).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use aidaemon_channels::context::ChannelContext;
use aidaemon_core::ChannelVisibility;
use aidaemon_events::traits::AgentRole;
use aidaemon_tools::semantics::{ToolCapabilities, UserRole};
use serde_json::{json, Value};

use crate::host::TurnHost;
use crate::policy::execution::{ApprovalMode, ExecutionPolicy, ModelProfile};
use crate::policy::metrics::POLICY_METRICS;
use crate::turn::helpers::{tool_name_from_definition, visibility_allows_desktop_control};

/// The tools a public external channel may ever be offered.
pub const PUBLIC_EXTERNAL_TOOLS: &[&str] = &["web_search", "remember_fact", "system_info"];

/// Tools that drive the owner's physical machine (desktop GUI automation).
/// An inbound chat message (or the bot's own content echoed back in a shared
/// channel) must never launch them, so they are offered only in 1-on-1 DMs and
/// internal sessions.
pub const DESKTOP_CONTROL_TOOLS: &[&str] = &["computer_use"];

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

    if let Some(additional) = parameters.get("additionalProperties")
        && !additional.is_boolean() {
            return Err("parameters.additionalProperties must be a boolean".to_string());
        }

    Ok(())
}

/// Whether an OpenAI-format tool definition meets the schema contract every
/// offered tool must: a named, described function with an object parameters
/// schema whose `required` keys exist.
pub fn validate_tool_definition_contract(def: &Value) -> Result<(), String> {
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

/// Push `tool`'s definition onto `defs` when it meets the schema contract;
/// count and log it otherwise.
fn push_checked_definition(defs: &mut Vec<Value>, name: &str, schema: Value, source: &str) {
    let candidate = json!({
        "type": "function",
        "function": schema
    });
    match validate_tool_definition_contract(&candidate) {
        Ok(()) => defs.push(candidate),
        Err(reason) => {
            POLICY_METRICS
                .tool_schema_contract_rejections_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                tool = %name,
                error = %reason,
                "Dropping {source} definition that violates schema contract"
            );
        }
    }
}

/// The tools a turn of the host is offered.
#[async_trait::async_trait]
pub trait ToolRoster: TurnHost {
    /// OpenAI-format definitions plus capabilities of the registered,
    /// available tools only. Query-independent: no MCP matching, no per-turn
    /// policy. The shared source of [`Self::tool_definitions_with_capabilities`]
    /// and [`Self::session_static_tool_roster`].
    fn registered_tool_definitions_with_capabilities(
        &self,
    ) -> (Vec<Value>, HashMap<String, ToolCapabilities>) {
        let mut defs: Vec<Value> = Vec::new();
        let mut capabilities: HashMap<String, ToolCapabilities> = HashMap::new();
        for tool in self.tools() {
            if !tool.is_available() {
                continue;
            }
            let name = tool.name().to_string();
            capabilities.insert(name.clone(), tool.capabilities());
            push_checked_definition(&mut defs, &name, tool.schema(), "tool");
        }
        (defs, capabilities)
    }

    /// The registered tools plus the MCP tools `user_message` calls for.
    async fn tool_definitions_with_capabilities(
        &self,
        user_message: &str,
    ) -> (Vec<Value>, HashMap<String, ToolCapabilities>) {
        let (mut defs, mut capabilities) = self.registered_tool_definitions_with_capabilities();
        for tool in self.mcp_tools_matching(user_message).await {
            let name = tool.name().to_string();
            // The server's own hints, as for a daemon tool (a daemon tool of
            // the same name keeps its capabilities).
            capabilities
                .entry(name.clone())
                .or_insert_with(|| tool.capabilities());
            push_checked_definition(&mut defs, &name, tool.schema(), "MCP tool");
        }
        (defs, capabilities)
    }

    /// Drop from `defs` and `capabilities` the tools not offered on a turn of
    /// `session_id` (each tool's `offered_in_session`).
    fn retain_offered_in_session(
        &self,
        defs: &mut Vec<Value>,
        capabilities: &mut HashMap<String, ToolCapabilities>,
        session_id: &str,
    ) {
        let withheld: HashSet<&str> = self
            .tools()
            .iter()
            .filter(|tool| !tool.offered_in_session(session_id))
            .map(|tool| tool.name())
            .collect();
        if withheld.is_empty() {
            return;
        }
        defs.retain(|def| {
            tool_name_from_definition(def).is_none_or(|name| !withheld.contains(name))
        });
        capabilities.retain(|name, _| !withheld.contains(name.as_str()));
    }

    /// Whether a tool named `name` is registered.
    fn has_registered_tool(&self, name: &str) -> bool {
        self.tools().iter().any(|tool| tool.name() == name)
    }

    /// The session-static roster the core prompt names: `(name, serialized
    /// definition)`, sorted by name. Built from the registered tools with
    /// only the session-static narrowing (a workspace grant's allowlist,
    /// channel visibility, the memory plugin's own tools, desktop control);
    /// no MCP matching and no per-turn policy. Non-owners get no tools unless
    /// the context carries an active workspace grant.
    fn session_static_tool_roster(
        &self,
        user_role: UserRole,
        channel_ctx: &ChannelContext,
    ) -> Vec<(String, String)> {
        let workspace_grant = channel_ctx.active_workspace_grant(user_role);
        if user_role != UserRole::Owner && workspace_grant.is_none() {
            return Vec::new();
        }

        let (mut defs, _caps) = self.registered_tool_definitions_with_capabilities();

        if let Some(grant) = workspace_grant {
            defs.retain(|definition| {
                tool_name_from_definition(definition).is_some_and(|name| grant.allows_tool(name))
            });
        }

        // Channel-visibility class is session-static and so stays in the core.
        if channel_ctx.visibility == ChannelVisibility::PublicExternal {
            defs.retain(|d| {
                tool_name_from_definition(d)
                    .is_some_and(|name| PUBLIC_EXTERNAL_TOOLS.contains(&name))
            });
        }
        // Daemon memory tools follow the active memory plugin, as on turns.
        let owned_memory_tools = self.memory_plugin().owned_daemon_tools();
        defs = defs
            .into_iter()
            .filter_map(|d| {
                let offer = tool_name_from_definition(&d)
                    .map(|name| crate::turn::memory::daemon_tool_offer(name, owned_memory_tools));
                match offer {
                    Some(crate::turn::memory::DaemonToolOffer::Hidden) => None,
                    Some(crate::turn::memory::DaemonToolOffer::GoalActions) => {
                        Some(crate::turn::memory::goal_actions_view(&d))
                    }
                    _ => Some(d),
                }
            })
            .collect();
        // Desktop control is never advertised outside DMs/internal sessions.
        if !visibility_allows_desktop_control(channel_ctx.visibility) {
            defs.retain(|d| {
                tool_name_from_definition(d)
                    .map(|name| !DESKTOP_CONTROL_TOOLS.contains(&name))
                    .unwrap_or(true)
            });
        }

        let mut roster: Vec<(String, String)> = defs
            .iter()
            .filter_map(|d| {
                let name = tool_name_from_definition(d)?.to_string();
                // serde_json keeps each schema's construction-site key order,
                // so this is stable for a given tool version.
                Some((name, d.to_string()))
            })
            .collect();
        roster.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        roster
    }

    /// `defs` narrowed to what `policy` allows at `risk_score` (`widen`:
    /// only reorder). Essential tools (`always_offered`, and a task lead's
    /// `task_lead_delegation` tools) survive every filter and sort first.
    fn filter_tool_definitions_for_policy(
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
                let name = tool_name_from_definition(def)?.to_string();
                let caps = capabilities.get(&name).copied().unwrap_or_default();
                Some((def.clone(), name, caps))
            })
            .collect();

        // Without the essential tools the agent can read files but never
        // write them, cannot keep memory or reach the web (the model then
        // resorts to terminal curl/grep workarounds).
        let task_lead = matches!(self.role(), AgentRole::TaskLead);
        let essential: HashSet<&str> = self
            .tools()
            .iter()
            .filter(|tool| {
                let facts = tool.dispatch_facts();
                facts.always_offered || (task_lead && facts.task_lead_delegation)
            })
            .map(|tool| tool.name())
            .collect();
        let is_policy_essential = |name: &str| essential.contains(name);

        // Stable prioritization: essential tools first (before truncation can
        // cut them off), then read-only + idempotent.
        ordered.sort_by_key(|(_, name, caps)| {
            (
                !is_policy_essential(name),
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

        let mut filtered = ordered;
        let low_risk = risk_score < 0.34 && matches!(policy.model_profile, ModelProfile::Cheap);

        if low_risk {
            // Essential + read-only tools, filled up to 5 from the sorted list.
            let mut keep: Vec<_> = filtered
                .iter()
                .filter(|(_, name, c)| c.read_only || is_policy_essential(name))
                .cloned()
                .collect();
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
            keep.truncate(16);
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
                // Capable, but not an unbounded tool surface.
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
}

impl<H: TurnHost + ?Sized> ToolRoster for H {}
