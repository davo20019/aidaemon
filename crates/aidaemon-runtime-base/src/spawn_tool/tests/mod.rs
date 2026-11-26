//! Tests of `spawn_agent`. The names they share are imported here; the
//! tests are grouped by concern in the child modules.

use super::*;

use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::helper_run::card::*;
use crate::helper_run::delivery::*;
use crate::run_state::RunPhase;
use crate::run_state::timing::*;
use crate::spawn::{ChildAgentRequest, ChildAgentRun, ChildAgentRuntime, ChildTimedOut};
use crate::stores::StateStore;
use aidaemon_channels::ports::OutboundRouter;
use aidaemon_core::ChannelVisibility;
use aidaemon_core::SourceTrust;
use aidaemon_events::TaskOutcome;
use aidaemon_events::traits::AgentRole;
use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::internal_markers::MEASURED_TIMING_MARK;
use aidaemon_tools::semantics::{ToolCallContext, UserRole};
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

mod basics;
mod child_runtime;
mod specialist_args;
