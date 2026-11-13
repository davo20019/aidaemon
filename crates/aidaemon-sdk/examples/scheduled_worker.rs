//! Background work: a service the host runs on a schedule, each run driving
//! an agent. The host gives every service the same treatment (one run at a
//! time, backoff after failures, a bounded drain on shutdown) and reports
//! each run once it ends: its number, how it ended and how long it took.
//!
//! `Schedule::Every` here; `Schedule::DailyUtc { hour: 2, minute: 0 }`
//! runs a nightly job instead. The run's number ties the agent run to it.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example scheduled_worker --features testing
use std::sync::Arc;
use std::time::Duration;

use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::ScriptedProvider;
use aidaemon_sdk::{
    CancellationToken, Schedule, Service, ServiceContext, ServiceError, ServiceEvent, ServiceHost,
    ServiceObserver,
};

/// Writes a status digest with an agent, every few seconds.
struct Digest {
    agent: Agent,
}

#[async_trait::async_trait]
impl Service for Digest {
    fn name(&self) -> &str {
        "digest"
    }

    fn schedule(&self) -> Schedule {
        Schedule::Every(Duration::from_secs(2))
    }

    async fn run(&self, ctx: &ServiceContext) -> Result<(), ServiceError> {
        // A state made here has a run id known before the run starts: every
        // event of the agent run carries it.
        let state = RunState::new(
            self.agent.id(),
            &format!("service:digest:{}", ctx.run()),
            "Write today's one-line status digest.",
        );
        let run_id = state.run_id;
        let outcome = self
            .agent
            .run_from(state, ctx.shutdown().child_token())
            .await
            .map_err(|err| ServiceError::new(err.to_string()))?;
        if !outcome.is_succeeded() {
            return Err(ServiceError::new(format!("{:?}", outcome.status)));
        }
        println!(
            "run {} (agent run {run_id}): {}",
            ctx.run(),
            outcome.final_text.unwrap_or_default()
        );
        Ok(())
    }
}

/// Prints how each run ended.
struct Log;

impl ServiceObserver for Log {
    fn event(&self, event: ServiceEvent) {
        if let ServiceEvent::Ended {
            service,
            run,
            outcome,
            elapsed,
            ..
        } = event
        {
            println!("{service} run {run} ended after {elapsed:?}: {outcome:?}");
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent =
        aidaemon_sdk::preset("digest")
            .provider_from_env_or(ScriptedProvider::new([]).then_repeat(
                aidaemon_sdk::testing::text("All systems normal (synthetic)."),
            ))
            .instructions("You write one-line status digests.")
            .build()
            .await?;

    let host = ServiceHost::new(Duration::from_millis(500))
        .observer(Arc::new(Log))
        .service(Arc::new(Digest { agent }));
    let shutdown = CancellationToken::new();
    let task = tokio::spawn(host.run(shutdown.clone()));
    tokio::time::sleep(Duration::from_secs(5)).await;
    shutdown.cancel();
    task.await?;
    Ok(())
}
