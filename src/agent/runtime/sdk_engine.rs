//! Agent turns on the aidaemon SDK kernel (`crates/aidaemon-core`).
//!
//! The turn engine lives in `aidaemon-runtime`
//! ([`aidaemon_runtime::engine::turn_engine::TurnEngine`]) and runs on the
//! agent as its `TurnHost`; `crate::startup::engine` reads its settings from
//! the daemon's config. What stays here is the engine tests' support.

#[cfg(test)]
pub(crate) use aidaemon_runtime::engine::settings::SdkEngineSettings;

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
