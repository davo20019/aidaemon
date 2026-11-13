//! Agent turns on the aidaemon SDK kernel (`crates/aidaemon-core`).
//!
//! The turn engine lives in `aidaemon-runtime`
//! ([`aidaemon_runtime::engine::turn_engine::TurnEngine`]) and runs on the
//! agent as its `TurnHost`. This module reads the engine settings from the
//! daemon's config.

use tracing::{info, warn};

use super::Agent;
use crate::config::{AppConfig, ProviderKind};

pub(crate) use aidaemon_runtime::engine::settings::{
    log_configured_trust_roots, SdkEngineSettings, DEFAULT_MAX_TURNS,
};

/// What the engine tests reach through `super::*`.
#[cfg(test)]
pub(in crate::agent) use test_support::*;
#[cfg(test)]
mod test_support {

    pub(in crate::agent) use crate::config::VisionConfig;
    pub(in crate::agent) use crate::events::{EventType, TaskOutcome, TaskStatus};
    pub(in crate::agent) use crate::traits::MessageAttachment;
    pub(in crate::agent) use crate::types::{ChannelContext, UserRole};
    pub(in crate::agent) use aidaemon_core as sdk;
    pub(in crate::agent) use aidaemon_runtime::engine::budgets;
    pub(in crate::agent) use aidaemon_runtime::engine::budgets::*;
    pub(in crate::agent) use aidaemon_runtime::engine::calls::{
        HostCall as DaemonCall, KernelNotice,
    };
    pub(in crate::agent) use aidaemon_runtime::engine::outcome::TurnVerdict;
    pub(in crate::agent) use async_trait::async_trait;

    pub(in crate::agent) use aidaemon_runtime::engine::settings::{
        turn_trust_roots, TrustRootSources,
    };
    pub(in crate::agent) use aidaemon_runtime::engine::turn_engine::stop::*;
    pub(in crate::agent) use aidaemon_runtime::engine::turn_engine::turn_hooks as hooks;
    pub(in crate::agent) use aidaemon_runtime::engine::turn_engine::turn_hooks::*;
    pub(in crate::agent) use aidaemon_runtime::engine::turn_engine::{
        compaction, model_call, outcome, tool_calls,
    };
    pub(in crate::agent) use aidaemon_runtime::engine::turn_engine::{
        sdk_images, terminal_canonicalizer, TerminalCanonTurn,
    };

    pub(in crate::agent) use serde_json::Value;
    pub(in crate::agent) use std::collections::HashSet;
    pub(in crate::agent) use std::sync::atomic::{AtomicBool, Ordering};
    pub(in crate::agent) use std::sync::Arc;
    pub(in crate::agent) use std::time::Duration;
    pub(in crate::agent) use tokio::sync::mpsc;
}

/// The engine settings from `[agent.engine]`.
pub(crate) fn settings_from_config(
    config: &AppConfig,
    backend: &dyn crate::execution::ExecutionBackend,
) -> SdkEngineSettings {
    let engine = &config.agent.engine;
    if engine.kind.is_some() {
        warn!("[agent.engine] kind is ignored: the SDK kernel is the only agent loop");
    }
    if config.state.daily_token_budget.is_some() {
        warn!(
            "[state] daily_token_budget is ignored: no global daily token cap is enforced; \
             use per-goal daily budgets, subagents.task_token_budget or a provider spend limit"
        );
    }
    if config.subagents.task_timeout_secs.is_some() {
        warn!(
            "[subagents] task_timeout_secs is ignored: runs are bounded by progress and cost \
             (inactivity watchdog, per-call timeouts, limits, token budgets), not elapsed time"
        );
    }
    let mut model_override = engine
        .model
        .clone()
        .filter(|model| !model.trim().is_empty());
    if !engine.api_key.trim().is_empty() {
        warn!(
            "[agent.engine] api_key is ignored: the SDK engine uses the daemon's configured \
             providers"
        );
        // An engine api_key marks a config written for the engine's former
        // dedicated Anthropic client; its model pin named a model for that
        // client, which the daemon's own provider may not serve.
        if config.provider.kind != ProviderKind::Anthropic {
            if let Some(model) = model_override.take() {
                warn!(
                    model = %model,
                    "[agent.engine] model is ignored: it was configured for the removed \
                     Anthropic engine client; set it without api_key to pin a model the \
                     daemon's provider serves"
                );
            }
        }
    }
    let settings = SdkEngineSettings {
        model_override,
        max_turns: engine.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
        owner_default: config.authority.owner_default,
        trusted_roots: config.authority.trusted_roots.clone(),
        outbound_secrets: super::approvals::ConfiguredSecrets::new(config.outbound_secret_values()),
    };
    info!(model_override = ?settings.model_override, "Agent engine: SDK kernel");
    log_configured_trust_roots(
        &settings.trusted_roots,
        backend.workspace_root().as_str().to_string(),
    );
    settings
}

impl Agent {
    /// The trusted roots the owner configured: `[authority] trusted_roots`.
    /// `[path_aliases] projects` is a naming convenience, not a grant.
    pub(in crate::agent) fn configured_trust_roots(&self) -> Vec<String> {
        self.sdk_engine.trusted_roots.clone()
    }
}

#[cfg(test)]
#[path = "sdk_engine/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "sdk_engine/finish_tests.rs"]
mod finish_tests;
#[cfg(test)]
#[path = "sdk_engine/hook_port_tests.rs"]
mod hook_port_tests;
#[cfg(test)]
#[path = "sdk_engine/status_tests.rs"]
mod status_tests;
