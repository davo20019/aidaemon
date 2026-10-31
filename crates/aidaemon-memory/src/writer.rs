//! [`MemoryWriter`]: background fact extraction as an `on_stop` hook.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use aidaemon_core::{Hook, HookContext, StopDecision};
use async_trait::async_trait;
use tokio::task::JoinHandle;

use crate::memory::{IngestReport, Memory};

/// Result of one background extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct WriterReport {
    pub run_id: String,
    pub session_id: String,
    /// The ingest result, or the error message. Errors never reach the run.
    pub result: Result<IngestReport, String>,
}

const MAX_KEPT_REPORTS: usize = 64;
const MAX_SEEN_TURNS: usize = 1_024;

#[derive(Default)]
struct Shared {
    reports: Mutex<VecDeque<WriterReport>>,
}

/// Extracts durable facts from the latest user turn when the model gives a
/// final answer. Always returns [`StopDecision::Accept`] immediately: the
/// extraction runs on a spawned task and can neither block nor fail the
/// run. Register it after any hook that may reject a final answer.
///
/// Outcomes (including model usage) are logged with `tracing` and kept in
/// a small buffer; [`MemoryWriter::drain`] waits for pending work and
/// returns them, which is useful at shutdown and in tests.
///
/// Requires a [`Memory`] built with a [`crate::FactExtractor`]; without one
/// the hook does nothing.
pub struct MemoryWriter {
    memory: Memory,
    shared: Arc<Shared>,
    pending: Mutex<Vec<JoinHandle<()>>>,
    seen: Mutex<VecDeque<(String, usize)>>,
}

impl MemoryWriter {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            shared: Arc::new(Shared::default()),
            pending: Mutex::new(Vec::new()),
            seen: Mutex::new(VecDeque::new()),
        }
    }

    /// Number of extractions still running.
    pub fn pending(&self) -> usize {
        self.pending
            .lock()
            .map(|p| p.iter().filter(|h| !h.is_finished()).count())
            .unwrap_or(0)
    }

    /// Wait for every spawned extraction, then return (and clear) the
    /// buffered reports.
    pub async fn drain(&self) -> Vec<WriterReport> {
        let handles: Vec<JoinHandle<()>> = self
            .pending
            .lock()
            .map(|mut p| std::mem::take(&mut *p))
            .unwrap_or_default();
        for handle in handles {
            let _ = handle.await;
        }
        self.shared
            .reports
            .lock()
            .map(|mut r| r.drain(..).collect())
            .unwrap_or_default()
    }

    /// True the first time a (run, user message) pair is seen, so a final
    /// answer rejected by a later hook does not trigger a second extraction.
    fn first_time(&self, run_id: &str, index: usize) -> bool {
        let Ok(mut seen) = self.seen.lock() else {
            return true;
        };
        let key = (run_id.to_string(), index);
        if seen.contains(&key) {
            return false;
        }
        if seen.len() >= MAX_SEEN_TURNS {
            seen.pop_front();
        }
        seen.push_back(key);
        true
    }
}

#[async_trait]
impl Hook for MemoryWriter {
    async fn on_stop(&self, ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        if self.memory.extractor().is_none() {
            return StopDecision::Accept;
        }
        let Some(index) = aidaemon_policies::last_user_message_index(ctx.messages) else {
            return StopDecision::Accept;
        };
        let run_id = ctx.run_id.to_string();
        if !self.first_time(&run_id, index) {
            return StopDecision::Accept;
        }
        let turn = ctx.messages[index..].to_vec();
        let session_id = ctx.session_id.to_string();
        let memory = self.memory.clone();
        let shared = self.shared.clone();
        let handle = tokio::spawn(async move {
            let result = memory
                .ingest_turn(&session_id, Some(&run_id), &turn)
                .await
                .map_err(|e| e.to_string());
            match &result {
                Ok(report) => tracing::info!(
                    session_id = %session_id,
                    run_id = %run_id,
                    stored = report.facts.len(),
                    rejected = report.rejected,
                    input_tokens = report.usage.input_tokens,
                    output_tokens = report.usage.output_tokens,
                    "memory extraction finished"
                ),
                Err(error) => tracing::warn!(
                    session_id = %session_id,
                    run_id = %run_id,
                    %error,
                    "memory extraction failed"
                ),
            }
            if let Ok(mut reports) = shared.reports.lock() {
                if reports.len() >= MAX_KEPT_REPORTS {
                    reports.pop_front();
                }
                reports.push_back(WriterReport {
                    run_id,
                    session_id,
                    result,
                });
            }
        });
        if let Ok(mut pending) = self.pending.lock() {
            pending.retain(|h| !h.is_finished());
            pending.push(handle);
        }
        StopDecision::Accept
    }
}
