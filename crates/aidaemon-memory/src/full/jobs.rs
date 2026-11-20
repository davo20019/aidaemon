//! The full memory system's background jobs: embedding backfill, fact
//! consolidation, decay, episodes for idle and long sessions, and pattern
//! detection, with the schedule each runs on ([`MemoryJobs::services`]).
//!
//! The jobs read the host's conversation log through [`ActivityLog`] and
//! leave what is the host's to [`JobsHost`]: which model runs each call,
//! learning from a session's work trace, and filing the goals an episode
//! mentions.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
/// Longest wait between embedding passes that keep finding nothing.
const EMBEDDINGS_IDLE_CAP: Duration = Duration::from_secs(5 * 60);

/// Spaces out embedding passes that find nothing to embed: each empty pass
/// doubles the wait before the next, up to [`EMBEDDINGS_IDLE_CAP`]; a pass
/// that embeds something, or fails, goes back to the schedule's pace.
/// The pass itself is a scan for rows without an embedding, so running it
/// every few seconds on a fully embedded store only costs database time.
#[derive(Debug)]
struct IdleBackoff {
    base: Duration,
    cap: Duration,
    wait: Duration,
    next_due: Option<Instant>,
}

impl IdleBackoff {
    fn new(base: Duration, cap: Duration) -> Self {
        Self {
            base,
            cap: cap.max(base),
            wait: Duration::ZERO,
            next_due: None,
        }
    }

    fn due(&self, now: Instant) -> bool {
        self.next_due.is_none_or(|due| now >= due)
    }

    /// Record a pass at `now` that did (`true`) or did not find work.
    fn record(&mut self, now: Instant, found_work: bool) {
        if found_work {
            self.wait = Duration::ZERO;
            self.next_due = None;
        } else {
            self.wait = if self.wait.is_zero() {
                self.base
            } else {
                (self.wait * 2).min(self.cap)
            };
            self.next_due = Some(now + self.wait);
        }
    }

    fn reset(&mut self) {
        self.wait = Duration::ZERO;
        self.next_due = None;
    }
}

/// The full memory system's background jobs over `store`.
pub struct MemoryJobs {
    store: Arc<dyn MemoryStores>,
    log: Arc<dyn ActivityLog>,
    host: Arc<dyn JobsHost>,
    embedder: Arc<dyn Embedder>,
    schedule: JobSchedule,
    /// Project memory, for consolidation of sessions that work in a project.
    project_notes: Option<Arc<crate::project::ProjectNotes>>,
    /// Spaces out [`Self::embeddings`] passes while they find nothing.
    embeddings_backoff: Mutex<IdleBackoff>,
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
            project_notes: None,
            embeddings_backoff: Mutex::new(IdleBackoff::new(
                JobSchedule::default().embeddings,
                EMBEDDINGS_IDLE_CAP,
            )),
        }
    }

    /// Consolidation keeps project facts of a session that works in a
    /// project ([`ConversationSource::project`]) in `notes`.
    ///
    /// [`ConversationSource::project`]: crate::history::ConversationSource::project
    pub fn with_project_notes(mut self, notes: Option<Arc<crate::project::ProjectNotes>>) -> Self {
        self.project_notes = notes;
        self
    }

    pub fn with_schedule(mut self, schedule: JobSchedule) -> Self {
        self.schedule = schedule;
        self.embeddings_backoff =
            Mutex::new(IdleBackoff::new(schedule.embeddings, EMBEDDINGS_IDLE_CAP));
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
    /// True when there was something to embed. While passes keep finding
    /// nothing, later calls skip the scan (returning false) until the backoff
    /// is due again (doubling from 5 s up to 5 min while passes stay empty).
    pub async fn embeddings(&self) -> anyhow::Result<bool> {
        if !self.backoff().due(Instant::now()) {
            return Ok(false);
        }
        let result =
            crate::backfill_learning_embeddings(self.store.as_ref(), self.embedder.as_ref()).await;
        let mut backoff = self.backoff();
        match &result {
            Ok(found_work) => backoff.record(Instant::now(), *found_work),
            Err(_) => backoff.reset(),
        }
        Ok(result?)
    }

    fn backoff(&self) -> std::sync::MutexGuard<'_, IdleBackoff> {
        self.embeddings_backoff
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Consolidate durable facts from owner history up to an hour before
    /// `now` ([`FactConsolidator`](crate::consolidation::FactConsolidator)),
    /// then let the host learn from the same sessions' work.
    pub async fn consolidate(&self, now: DateTime<Utc>) -> anyhow::Result<()> {
        let (provider, model) = self.host.model(JobCall::FactConsolidation);
        let report = crate::consolidation::FactConsolidator::new(provider, model)
            .with_project_notes(self.project_notes.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: Duration = Duration::from_secs(5);
    const CAP: Duration = Duration::from_secs(60);

    #[test]
    fn empty_passes_double_the_wait_up_to_the_cap() {
        let start = Instant::now();
        let mut backoff = IdleBackoff::new(BASE, CAP);
        assert!(backoff.due(start));

        let mut now = start;
        let mut waits = Vec::new();
        for _ in 0..6 {
            backoff.record(now, false);
            waits.push(backoff.wait);
            assert!(!backoff.due(now + backoff.wait - Duration::from_millis(1)));
            now += backoff.wait;
            assert!(backoff.due(now));
        }
        let secs: Vec<u64> = waits.iter().map(Duration::as_secs).collect();
        assert_eq!(secs, [5, 10, 20, 40, 60, 60]);
    }

    #[test]
    fn a_pass_with_work_returns_to_the_schedule_pace() {
        let now = Instant::now();
        let mut backoff = IdleBackoff::new(BASE, CAP);
        for _ in 0..4 {
            backoff.record(now, false);
        }
        backoff.record(now, true);
        assert!(backoff.due(now));
        backoff.record(now, false);
        assert_eq!(backoff.wait, BASE);
    }

    #[test]
    fn a_failed_pass_clears_the_backoff() {
        let now = Instant::now();
        let mut backoff = IdleBackoff::new(BASE, CAP);
        backoff.record(now, false);
        backoff.reset();
        assert!(backoff.due(now));
    }
}
