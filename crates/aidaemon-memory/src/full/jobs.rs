//! The full memory system's background jobs: embedding backfill, fact
//! consolidation, decay, episodes for idle and long sessions, and pattern
//! detection, with the schedule each runs on ([`MemoryJobs::services`]).
//!
//! The jobs read the host's conversation log through [`ActivityLog`] and
//! leave what is the host's to [`JobsHost`]: which model runs each call,
//! learning from a session's work trace, and filing the goals an episode
//! mentions.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::{FnService, Provider, Schedule, Service, ServiceError};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tracing::{info, warn};

use crate::Embedder;
use crate::episodes::TranscriptEntry;
use crate::history::{ActivityLog, LoggedTurn, TurnRole, UncapturedSessions};
use crate::stores::MemoryStores;

/// A background model call the jobs make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobCall {
    /// Durable facts from owner history.
    FactConsolidation,
    /// An episode's summary of a session.
    SessionAnalysis,
}

/// What the jobs leave to the host.
#[async_trait]
pub trait JobsHost: Send + Sync {
    /// The provider and model for `call`, asked on every pass.
    fn model(&self, call: JobCall) -> (Arc<dyn Provider>, String);

    /// Learn from the work recorded in `session_id` (procedures, error
    /// solutions, expertise) after its facts were consolidated. The host
    /// owns its work trace. Default: nothing.
    async fn learn_session(&self, _session_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// File `goal`, which episode `episode_id` of `session_id` mentioned.
    /// Goals are the host's. Default: nothing.
    async fn file_goal(
        &self,
        _goal: &str,
        _episode_id: i64,
        _session_id: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// How often each job runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobSchedule {
    pub embeddings: Duration,
    pub consolidation: Duration,
    pub decay: Duration,
    pub episodes: Duration,
    pub pattern_detection: Duration,
}

impl Default for JobSchedule {
    /// Embeddings every 5 s, consolidation every 6 h, decay daily, episodes
    /// every 30 min, pattern detection every 6 h.
    fn default() -> Self {
        Self {
            embeddings: Duration::from_secs(5),
            consolidation: Duration::from_secs(6 * 3600),
            decay: Duration::from_secs(24 * 3600),
            episodes: Duration::from_secs(30 * 60),
            pattern_detection: Duration::from_secs(6 * 3600),
        }
    }
}

/// A session goes idle this long after its last turn.
const IDLE_AFTER: chrono::Duration = chrono::Duration::minutes(30);
/// Uncaptured turns an idle session needs before it gets an episode.
const IDLE_MIN_TURNS: usize = 5;
/// Uncaptured turns that give an active session a mid-session episode.
const LONG_MIN_TURNS: usize = 20;
/// Turns an episode covers at most.
const EPISODE_TURNS: usize = 100;
/// Turns a mid-session episode needs once read.
const LONG_MIN_MESSAGES: usize = 10;
/// How far back pattern detection looks, and how much it reads.
const PATTERN_WINDOW: chrono::Duration = chrono::Duration::hours(24);
const PATTERN_TURNS: usize = 500;

/// The full memory system's background jobs over `store`.
pub struct MemoryJobs {
    store: Arc<dyn MemoryStores>,
    log: Arc<dyn ActivityLog>,
    host: Arc<dyn JobsHost>,
    embedder: Arc<dyn Embedder>,
    schedule: JobSchedule,
}

impl MemoryJobs {
    pub fn new(
        store: Arc<dyn MemoryStores>,
        log: Arc<dyn ActivityLog>,
        host: Arc<dyn JobsHost>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            store,
            log,
            host,
            embedder,
            schedule: JobSchedule::default(),
        }
    }

    pub fn with_schedule(mut self, schedule: JobSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// The jobs as services, named `embeddings`, `consolidation`,
    /// `memory_decay`, `episodes` and `pattern_detection`.
    pub fn services(self: &Arc<Self>) -> Vec<Arc<dyn Service>> {
        fn every<F, Fut>(name: &str, interval: Duration, run: F) -> Arc<dyn Service>
        where
            F: Fn() -> Fut + Send + Sync + 'static,
            Fut: std::future::Future<Output = anyhow::Result<()>> + Send + 'static,
        {
            Arc::new(FnService::new(name, Schedule::Every(interval), move || {
                let run = run();
                async move { run.await.map_err(|e| ServiceError::new(e.to_string())) }
            }))
        }
        let schedule = self.schedule;
        let jobs = self.clone();
        let embeddings = every("embeddings", schedule.embeddings, move || {
            let jobs = jobs.clone();
            async move { jobs.embeddings().await.map(|_| ()) }
        });
        let jobs = self.clone();
        let consolidation = every("consolidation", schedule.consolidation, move || {
            let jobs = jobs.clone();
            async move { jobs.consolidate(Utc::now()).await }
        });
        let jobs = self.clone();
        let decay = every("memory_decay", schedule.decay, move || {
            let jobs = jobs.clone();
            async move { jobs.decay().await }
        });
        let jobs = self.clone();
        let episodes = every("episodes", schedule.episodes, move || {
            let jobs = jobs.clone();
            async move { jobs.episodes(Utc::now()).await }
        });
        let jobs = self.clone();
        let patterns = every("pattern_detection", schedule.pattern_detection, move || {
            let jobs = jobs.clone();
            async move { jobs.detect_patterns(Utc::now()).await }
        });
        vec![embeddings, consolidation, decay, episodes, patterns]
    }

    /// Embed one batch of learned records still missing an embedding.
    /// True when there was something to embed.
    pub async fn embeddings(&self) -> anyhow::Result<bool> {
        Ok(
            crate::backfill_learning_embeddings(self.store.as_ref(), self.embedder.as_ref())
                .await?,
        )
    }

    /// Consolidate durable facts from owner history up to an hour before
    /// `now` ([`FactConsolidator`](crate::consolidation::FactConsolidator)),
    /// then let the host learn from the same sessions' work.
    pub async fn consolidate(&self, now: DateTime<Utc>) -> anyhow::Result<()> {
        let (provider, model) = self.host.model(JobCall::FactConsolidation);
        let report = crate::consolidation::FactConsolidator::new(provider, model)
            .consolidate(
                self.log.as_ref(),
                self.log.as_ref(),
                self.store.as_ref(),
                now,
            )
            .await?;
        for session_id in &report.consolidated_sessions {
            // Recheck, so a session classified isolated since never enters
            // learning.
            if self.log.is_memory_isolated(session_id).await? {
                continue;
            }
            if let Err(e) = self.host.learn_session(session_id).await {
                warn!(
                    "Event consolidation failed for session {}: {}",
                    session_id, e
                );
            }
        }
        Ok(())
    }

    /// Merge restated facts, then let unrecalled memory fade.
    pub async fn decay(&self) -> anyhow::Result<()> {
        crate::stores::decay_memory(self.store.as_ref(), &crate::stores::DecayPolicy::default())
            .await?;
        info!("Applied memory decay");
        Ok(())
    }

    /// Episodes for sessions idle 30 minutes with 5+ uncaptured turns, then
    /// mid-session ones for sessions with 20+.
    pub async fn episodes(&self, now: DateTime<Utc>) -> anyhow::Result<()> {
        if let Err(e) = self.episodes_for_idle_sessions(now).await {
            warn!(error = %e, "Failed to create episodes for idle sessions");
        }
        self.episodes_for_long_sessions().await
    }

    /// Record an episode of `session_id` from `turns`
    /// ([`SessionAnalyst`](crate::episodes::SessionAnalyst)), then hand the
    /// goals it found to the host.
    pub async fn create_episode(
        &self,
        session_id: &str,
        turns: &[LoggedTurn],
    ) -> anyhow::Result<i64> {
        anyhow::ensure!(
            !self.log.is_memory_isolated(session_id).await?,
            "Memory-isolated worker sessions cannot create global episodes"
        );
        let entries = transcript_entries(turns);
        let start_time = turns.first().map(|t| t.created_at).unwrap_or_else(Utc::now);
        let end_time = turns.last().map(|t| t.created_at).unwrap_or_else(Utc::now);
        let (provider, model) = self.host.model(JobCall::SessionAnalysis);
        let (episode_id, analysis) = crate::episodes::SessionAnalyst::new(provider, model)
            .record_episode(
                self.store.as_ref(),
                session_id,
                &entries,
                turns.len(),
                start_time,
                end_time,
                self.log.channel_id(session_id),
            )
            .await?;
        for goal_text in &analysis.goals_mentioned {
            if let Err(e) = self.host.file_goal(goal_text, episode_id, session_id).await {
                warn!("Failed to extract goal '{}': {}", goal_text, e);
            }
        }
        Ok(episode_id)
    }

    /// Habitual tool sequences and the owner's verbosity, from the last
    /// day's activity.
    pub async fn detect_patterns(&self, now: DateTime<Utc>) -> anyhow::Result<()> {
        let since = now - PATTERN_WINDOW;
        let recent = self.log.recent_turns(since, PATTERN_TURNS).await?;
        let owner_messages = self.log.recent_owner_messages(since, PATTERN_TURNS).await?;
        if recent.is_empty() {
            return Ok(());
        }
        let tools: Vec<String> = recent
            .iter()
            .flat_map(|turn| turn.tool_calls.iter().cloned())
            .collect();
        let patterns = crate::record_tool_sequence_patterns(self.store.as_ref(), &tools).await?;
        if !patterns.is_empty() {
            info!(count = patterns.len(), "Detected behavior patterns");
        }
        let owner_texts: Vec<&str> = owner_messages
            .iter()
            .filter(|turn| turn.role == TurnRole::User)
            .filter_map(|turn| turn.content.as_deref())
            .collect();
        if crate::update_communication_profile(self.store.as_ref(), &owner_texts)
            .await?
            .is_some()
        {
            info!("Updated user communication profile");
        }
        Ok(())
    }

    async fn uncaptured(&self, which: UncapturedSessions) -> anyhow::Result<Vec<String>> {
        let ends = self.store.latest_episode_ends().await?;
        let mut eligible = Vec::new();
        for session_id in self.log.uncaptured_sessions(&ends, which).await? {
            if !self.log.is_memory_isolated(&session_id).await? {
                eligible.push(session_id);
            }
        }
        Ok(eligible)
    }

    /// The session's turns since its latest episode, if they all came from
    /// the owner; `None` otherwise.
    async fn owner_turns_since_last_episode(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<Vec<LoggedTurn>>> {
        let bound = self.store.last_episode_bound(session_id).await?;
        if !self
            .log
            .owner_only_after(session_id, bound.as_ref())
            .await?
        {
            return Ok(None);
        }
        Ok(Some(
            self.log
                .turns_after(session_id, bound.as_ref(), EPISODE_TURNS)
                .await?,
        ))
    }

    async fn episodes_for_idle_sessions(&self, now: DateTime<Utc>) -> anyhow::Result<()> {
        let idle = self
            .uncaptured(UncapturedSessions {
                min_turns: IDLE_MIN_TURNS,
                idle_before: Some(now - IDLE_AFTER),
            })
            .await?;
        if idle.is_empty() {
            return Ok(());
        }
        info!(count = idle.len(), "Creating episodes for idle sessions");
        for session_id in idle {
            let Some(turns) = self.owner_turns_since_last_episode(&session_id).await? else {
                continue;
            };
            if turns.len() >= IDLE_MIN_TURNS {
                match self.create_episode(&session_id, &turns).await {
                    Ok(episode_id) => {
                        info!(session_id, episode_id, "Created episode for idle session");
                    }
                    Err(e) => {
                        warn!(session_id, error = %e, "Failed to create episode for idle session");
                    }
                }
            }
        }
        Ok(())
    }

    /// Sessions with many turns since their last episode risk losing
    /// context as the message window rotates: capture it now.
    async fn episodes_for_long_sessions(&self) -> anyhow::Result<()> {
        let long = self
            .uncaptured(UncapturedSessions {
                min_turns: LONG_MIN_TURNS,
                idle_before: None,
            })
            .await?;
        for session_id in long {
            let Some(turns) = self.owner_turns_since_last_episode(&session_id).await? else {
                continue;
            };
            if turns.len() >= LONG_MIN_MESSAGES {
                match self.create_episode(&session_id, &turns).await {
                    Ok(episode_id) => {
                        info!(
                            session_id,
                            episode_id, "Created mid-session episode for long-running session"
                        );
                    }
                    Err(e) => {
                        warn!(
                            session_id,
                            error = %e,
                            "Failed to create mid-session episode"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// Logged turns as episode transcript entries.
fn transcript_entries(turns: &[LoggedTurn]) -> Vec<TranscriptEntry> {
    turns
        .iter()
        .map(|turn| TranscriptEntry {
            role: turn.role,
            content: turn.content.clone(),
            tool_calls: turn.tool_calls.clone(),
            tool_name: turn.tool_name.clone(),
        })
        .collect()
}
