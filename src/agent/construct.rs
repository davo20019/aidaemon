//! Builds the daemon's [`Agent`] from its parts and the daemon's host ports:
//! its memory over the daemon stores, its file policy, OTel export and
//! computer-use pins.

use std::sync::Arc;

use super::{Agent, AgentParts};
use aidaemon_runtime::agent::{HostPorts, TraceExport};

/// The daemon's root agent.
pub(crate) fn new_agent(parts: AgentParts) -> Agent {
    let ports = host_ports(&parts);
    Agent::new(parts, ports)
}

/// The daemon's ports for an agent built from `parts`.
fn host_ports(parts: &AgentParts) -> HostPorts {
    let deps = &parts.dependencies;
    let daemon_memory = super::memory_plugin::DaemonMemory::new(
        deps.state.clone(),
        deps.memory_store.clone(),
        deps.event_store.clone(),
        deps.llm_runtime.clone(),
        deps.side_calls.clone(),
        parts.context_window_config.clone(),
        parts.max_facts,
        parts.record_decision_points,
        &parts.system_prompt,
    );
    let transcript_writes = daemon_memory.writes();
    HostPorts {
        memory: Arc::new(daemon_memory),
        transcript_writes,
        file_policy: crate::tools::daemon_file_policy(parts.daemon_paths.clone()),
        trace: Some(Arc::new(DaemonTrace)),
        computer_use: computer_use(),
    }
}

/// The daemon's OTel export, read when a run starts.
struct DaemonTrace;

impl TraceExport for DaemonTrace {
    fn sink(&self) -> Option<Arc<dyn aidaemon_core::EventSink>> {
        crate::otel_export::sink()
    }

    fn observe_sent(&self, run: aidaemon_core::RunId, messages: &[serde_json::Value]) {
        crate::otel_export::observe_sent(run, messages);
    }
}

fn computer_use() -> Option<Arc<dyn aidaemon_runtime::host::ComputerUseState>> {
    #[cfg(feature = "computer_use")]
    {
        Some(Arc::new(super::computer_use::ComputerUsePins))
    }
    #[cfg(not(feature = "computer_use"))]
    None
}

/// Lower the token-pressure watermark in focused tests without weakening
/// production defaults; the turn's post-reply compaction reads the
/// transcript writer's own copy, rebuilt here with the lower one.
#[cfg(test)]
pub(crate) fn set_context_compaction_tokens_for_test(
    agent: &mut Agent,
    token_threshold: usize,
    recent_tokens: usize,
) {
    use aidaemon_runtime::host::TurnHost as _;
    agent.set_test_context_compaction_tokens(token_threshold, recent_tokens);
    let writes = super::memory_plugin::DaemonMemory::new(
        agent.state_store().clone(),
        agent.memory_stores().clone(),
        agent.event_store().clone(),
        agent.llm_runtime().clone(),
        agent.side_calls().clone(),
        agent.turn_runtime().context_window_config.clone(),
        0,
        false,
        "",
    )
    .writes();
    agent.set_test_transcript_writes(writes);
}
