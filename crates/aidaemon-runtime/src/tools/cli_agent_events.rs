//! A launched coding agent's typed events, kept as it runs.
//!
//! Each output line goes through the agent's own reader
//! ([`aidaemon_coding_agents::CodingAgent::read_events`]); each event is kept
//! on the run's invocation in order, its session id on the invocation
//! itself, and its latest step becomes the run record's activity, which the
//! owner's app shows while it runs.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use aidaemon_coding_agents::{ActionJudge, AgentAction, AgentEvent, CodingAgent, Verdict};
use async_trait::async_trait;
use serde_json::json;

use super::cli_agent::CliAgentStore;
use crate::run_state::RunStates;

pub(super) struct AgentEventRecorder {
    agent: Arc<dyn CodingAgent>,
    store: Arc<dyn CliAgentStore>,
    /// The invocation row the events are kept on (0: none was logged).
    invocation: i64,
    runs: Arc<RunStates>,
    run_id: String,
    /// The events' order, shared with the run's [`RecordingJudge`].
    seq: Arc<AtomicI64>,
    /// Steps already recorded, by the agent's own id: an item reported
    /// started and again completed is one step.
    steps: HashSet<String>,
    /// Its output read as typed events at least once: its steps come from
    /// them, not from how much it printed.
    structured: bool,
}

impl AgentEventRecorder {
    pub(super) fn new(
        agent: Arc<dyn CodingAgent>,
        store: Arc<dyn CliAgentStore>,
        invocation: i64,
        runs: Arc<RunStates>,
        run_id: String,
    ) -> Self {
        Self {
            agent,
            store,
            invocation,
            runs,
            run_id,
            seq: Arc::default(),
            steps: HashSet::new(),
            structured: false,
        }
    }

    /// `judge`, keeping each decision it makes as one of this run's events.
    pub(super) fn judging(&self, judge: Arc<dyn ActionJudge>) -> Arc<dyn ActionJudge> {
        Arc::new(RecordingJudge {
            judge,
            store: self.store.clone(),
            invocation: self.invocation,
            seq: self.seq.clone(),
        })
    }

    /// Read one output line (as the run reads it, after its host).
    pub(super) async fn read(&mut self, line: &str) {
        for event in self.agent.read_events(line) {
            self.structured = true;
            self.record(event).await;
        }
    }

    /// Whether its output reads as typed events ([`Self::read`] found one).
    pub(super) fn structured(&self) -> bool {
        self.structured
    }

    async fn record(&mut self, event: AgentEvent) {
        match &event {
            AgentEvent::Step(step) => {
                if let Some(id) = step.id.as_deref()
                    && !self.steps.insert(id.to_string())
                {
                    return;
                }
                self.runs.activity(&self.run_id, step.tool.words());
                if let aidaemon_coding_agents::StepTool::Edit { path: Some(path) } = &step.tool {
                    self.runs.edited(&self.run_id, path);
                }
            }
            AgentEvent::Session { id, .. } if self.invocation != 0 => {
                if let Err(error) = self.store.set_cli_agent_session(self.invocation, id).await {
                    tracing::warn!(%error, "Could not keep a coding agent's session id");
                }
            }
            _ => {}
        }
        if self.invocation == 0 {
            return;
        }
        let (kind, data) = event.to_record();
        keep(&*self.store, self.invocation, &self.seq, kind, data).await;
    }
}

async fn keep(
    store: &dyn CliAgentStore,
    invocation: i64,
    seq: &AtomicI64,
    kind: &str,
    data: serde_json::Value,
) {
    let seq = seq.fetch_add(1, Ordering::Relaxed) + 1;
    if let Err(error) = store
        .record_cli_agent_event(invocation, seq, kind, &data.to_string())
        .await
    {
        tracing::warn!(%error, "Could not keep a coding agent's event");
    }
}

/// The runtime's judge for a hosted agent's requests, keeping each
/// decision as a `permission` event: what was asked (its kind and the
/// owner's label for it), whether it was allowed, and why not.
struct RecordingJudge {
    judge: Arc<dyn ActionJudge>,
    store: Arc<dyn CliAgentStore>,
    invocation: i64,
    seq: Arc<AtomicI64>,
}

#[async_trait]
impl ActionJudge for RecordingJudge {
    async fn judge(&self, action: AgentAction) -> Verdict {
        let kind = match &action {
            AgentAction::RunCommand { .. } => "run",
            AgentAction::WriteFile { .. } => "write",
            AgentAction::EditFile { .. } => "edit",
            AgentAction::ReadFile { .. } => "read",
            AgentAction::Search { .. } => "search",
            AgentAction::Fetch { .. } => "fetch",
            AgentAction::ReachHost { .. } => "network",
        };
        let label = action.label();
        let verdict = self.judge.judge(action).await;
        if self.invocation != 0 {
            let (outcome, reason) = match &verdict {
                Verdict::Allow => ("allowed", None),
                Verdict::Deny { reason } => ("refused", Some(reason.clone())),
                Verdict::Declined { reason } => ("declined", Some(reason.clone())),
                Verdict::Unanswered { reason } => ("unanswered", Some(reason.clone())),
                // A verdict added later: not allowed, with its own reason.
                other => ("refused", other.reason().map(str::to_string)),
            };
            keep(
                &*self.store,
                self.invocation,
                &self.seq,
                "permission",
                json!({"action": kind, "label": label, "outcome": outcome, "reason": reason}),
            )
            .await;
        }
        verdict
    }
}
