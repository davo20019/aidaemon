//! The daemon's background work: its periodic jobs and services, and the
//! goal cycle from `aidaemon_work::scheduler`, run on one [`ServiceHost`].
//! The daemon gives the cycle its agent, its channels and its presentation.
//!
//! Wiring only: the job telemetry, its observer and the database health gate
//! are `aidaemon_work::heartbeat`'s; what stays here registers the daemon's
//! jobs and adapts its agent, hub and wording to the cycle's host traits.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::time::Duration;

use aidaemon_core::{CancellationToken, FnService, Schedule, Service, ServiceError, ServiceHost};
use async_trait::async_trait;

use tokio::sync::mpsc;

#[allow(unused_imports)]
pub(crate) use aidaemon_work::scheduler::*;

pub use aidaemon_work::heartbeat::{DbHealthGate, HeartbeatTelemetry, JobObserver};

use crate::agent::Agent;
use crate::agent::TurnNotes;
use crate::goal_tokens::GoalTokenRegistry;
use crate::runtime_ports::OutboundRouter;
use crate::traits::{Goal, StateStore, Task};
use crate::types::{ChannelContext, UserRole};

/// Type alias for the async closure that heartbeat jobs execute.
type HeartbeatRunFn =
    Box<dyn Fn() -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>> + Send + Sync>;

/// A registered periodic job.
pub struct HeartbeatJob {
    pub name: String,
    pub interval: Duration,
    /// The async function to call.
    pub run: HeartbeatRunFn,
}

/// The daemon's background work: its periodic jobs, and the goal cycle
/// (schedules, mandates, stuck tasks, notifications), run on a
/// [`ServiceHost`].
pub struct HeartbeatCoordinator {
    jobs: Vec<HeartbeatJob>,
    services: Vec<Arc<dyn Service>>,
    cycle: GoalCycle,
    state: Arc<dyn StateStore>,
    max_concurrent: usize,
    tick_interval: Duration,
    wake_rx: Option<mpsc::Receiver<()>>,
    hub: Option<Weak<dyn OutboundRouter>>,
    agent: Option<Weak<Agent>>,
    goal_token_registry: Option<GoalTokenRegistry>,
    telemetry: Option<Arc<HeartbeatTelemetry>>,
}

impl std::ops::Deref for HeartbeatCoordinator {
    type Target = GoalCycle;

    fn deref(&self) -> &GoalCycle {
        &self.cycle
    }
}

impl std::ops::DerefMut for HeartbeatCoordinator {
    fn deref_mut(&mut self) -> &mut GoalCycle {
        &mut self.cycle
    }
}

/// The daemon's agent, as the goal cycle hands it task leads.
struct DaemonTaskLeads {
    agent: Weak<Agent>,
    state: Arc<dyn StateStore>,
    hub: Option<Weak<dyn OutboundRouter>>,
    goal_token_registry: Option<GoalTokenRegistry>,
}

#[async_trait]
impl TaskLeadHost for DaemonTaskLeads {
    fn available(&self) -> bool {
        self.agent.strong_count() > 0
    }

    fn spawn_task_lead(
        &self,
        goal: Goal,
        task_text: String,
        task_id: String,
        running_surface_id: Option<String>,
    ) {
        let Some(agent) = self.agent.upgrade() else {
            return;
        };
        let session_id = goal.session_id.clone();
        crate::agent::spawn_background_task_lead(
            agent,
            goal,
            task_text,
            session_id,
            ChannelContext::internal(),
            UserRole::Owner,
            self.state.clone(),
            self.hub.clone(),
            self.goal_token_registry.clone(),
            Some(task_id),
            running_surface_id,
        );
    }

    fn next_model_answer_at(&self) -> Option<std::time::SystemTime> {
        self.agent.upgrade()?.next_model_answer_at()
    }

    async fn retry_deferred_provider_requests(&self) -> usize {
        match self.agent.upgrade() {
            Some(agent) => agent.engine().retry_deferred_provider_requests().await,
            None => 0,
        }
    }

    async fn tainted_note_turn(&self, session_id: &str) -> anyhow::Result<String> {
        self.live()?.tainted_note_turn(session_id).await
    }

    async fn record_parent_visible_result_note_in_turn(
        &self,
        session_id: &str,
        prefix: &str,
        text: &str,
        turn_id: Option<&str>,
    ) -> anyhow::Result<()> {
        self.live()?
            .record_parent_visible_result_note_in_turn(session_id, prefix, text, turn_id)
            .await
    }

    async fn record_mandate_owner_input_context(
        &self,
        session_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        action_token: &str,
    ) -> anyhow::Result<()> {
        self.live()?
            .record_mandate_owner_input_context(
                session_id,
                mandate_id,
                mandate_version,
                action_token,
            )
            .await
    }
}

impl DaemonTaskLeads {
    fn live(&self) -> anyhow::Result<Arc<Agent>> {
        self.agent
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("the agent runtime has stopped"))
    }
}

/// The daemon's channel hub, as the goal cycle sends notices.
struct HubDelivery(Weak<dyn OutboundRouter>);

impl HubDelivery {
    fn hub(&self) -> anyhow::Result<Arc<dyn OutboundRouter>> {
        self.0
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("the channel hub has stopped"))
    }
}

#[async_trait]
impl NoticeDelivery for HubDelivery {
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        self.hub()?.send_text(session_id, text).await
    }

    async fn send_text_tracked(
        &self,
        session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>> {
        self.hub()?.send_text_tracked(session_id, text).await
    }
}

/// How the daemon shows work to the owner.
struct DaemonPresenter;

impl WorkPresenter for DaemonPresenter {
    fn goal_label(&self, description: &str) -> String {
        crate::tools::sanitize::short_goal_label(description)
    }

    fn user_facing(&self, text: &str) -> String {
        crate::tools::sanitize::sanitize_user_facing_reply(text)
    }

    fn notification(&self, notification_type: &str, text: &str) -> String {
        crate::channels::present_notification(notification_type, text)
    }

    fn goal_task_results(
        &self,
        tasks: &[Task],
        root_task_id: Option<&str>,
        fallback: &str,
    ) -> String {
        crate::agent::build_goal_task_results_summary(tasks, root_task_id, fallback)
    }
}

impl HeartbeatCoordinator {
    pub fn new(
        state: Arc<dyn StateStore>,
        tick_interval_secs: u64,
        max_concurrent: usize,
        wake_rx: mpsc::Receiver<()>,
        hub: Option<Weak<dyn OutboundRouter>>,
        goal_token_registry: Option<GoalTokenRegistry>,
        telemetry: Option<Arc<HeartbeatTelemetry>>,
    ) -> Self {
        let mut coordinator = Self {
            jobs: Vec::new(),
            services: Vec::new(),
            cycle: GoalCycle::new(
                state.clone(),
                Arc::new(DaemonPresenter),
                goal_token_registry.clone(),
            ),
            state,
            max_concurrent,
            tick_interval: Duration::from_secs(tick_interval_secs),
            wake_rx: Some(wake_rx),
            hub: None,
            agent: None,
            goal_token_registry,
            telemetry,
        };
        if let Some(hub) = hub {
            coordinator.set_hub(hub);
        }
        coordinator
    }

    /// Set the hub reference (deferred, since hub is created after heartbeat).
    pub fn set_hub(&mut self, hub: Weak<dyn OutboundRouter>) {
        self.cycle
            .set_delivery(Some(Arc::new(HubDelivery(hub.clone()))));
        self.hub = Some(hub);
        self.wire_agent();
    }

    /// Set the agent reference (deferred, since agent is created before heartbeat starts).
    /// Used for dispatching orphaned pending tasks.
    pub fn set_agent(&mut self, agent: Weak<Agent>) {
        self.agent = Some(agent);
        self.wire_agent();
    }

    fn wire_agent(&mut self) {
        let host = self.agent.clone().map(|agent| {
            Arc::new(DaemonTaskLeads {
                agent,
                state: self.state.clone(),
                hub: self.hub.clone(),
                goal_token_registry: self.goal_token_registry.clone(),
            }) as Arc<dyn TaskLeadHost>
        });
        self.cycle.set_agent(host);
    }

    /// Register a periodic job with the heartbeat coordinator.
    pub fn register_job<F, Fut>(&mut self, name: &str, interval: Duration, f: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        self.jobs.push(HeartbeatJob {
            name: name.to_string(),
            interval,
            run: Box::new(move || Box::pin(f())),
        });
        if let Some(telemetry) = &self.telemetry {
            telemetry.register_job(name, interval);
        }
    }

    #[cfg(test)]
    pub(crate) fn scheduled_names(&self) -> Vec<String> {
        self.jobs
            .iter()
            .map(|job| job.name.clone())
            .chain(self.services.iter().map(|s| s.name().to_string()))
            .collect()
    }

    /// Run `service` alongside the jobs, under the same gate and telemetry.
    pub fn add_service(&mut self, service: Arc<dyn Service>) {
        if let (Some(telemetry), Schedule::Every(interval)) = (&self.telemetry, service.schedule())
        {
            telemetry.register_job(service.name(), interval);
        }
        self.services.push(service);
    }

    /// The host that runs this coordinator's jobs and goal cycle.
    pub(crate) fn into_host(mut self) -> ServiceHost {
        let mut host = ServiceHost::new(self.tick_interval)
            .max_concurrent(self.max_concurrent)
            .gate(Arc::new(DbHealthGate::new(self.state.clone())))
            .observer(Arc::new(JobObserver::new(self.telemetry.clone())));
        for job in std::mem::take(&mut self.jobs) {
            let HeartbeatJob {
                name,
                interval,
                run,
            } = job;
            let service = FnService::new(name, Schedule::Every(interval), move || {
                let run = run();
                async move { run.await.map_err(|e| ServiceError::new(e.to_string())) }
            });
            host = host.service(Arc::new(service));
        }
        for service in std::mem::take(&mut self.services) {
            host = host.service(service);
        }
        // A user message wakes the cycle at once.
        if let Some(mut wake_rx) = self.wake_rx.take() {
            let waker = host.waker();
            tokio::spawn(async move {
                while wake_rx.recv().await.is_some() {
                    waker.wake();
                }
            });
        }
        host.service(Arc::new(GoalCycleService::new(self.cycle)))
    }

    /// Run in the background until `shutdown` is cancelled.
    pub fn start(self, shutdown: CancellationToken) -> tokio::task::JoinHandle<()> {
        tokio::spawn(self.into_host().run(shutdown))
    }
}

#[cfg(test)]
mod tests;
