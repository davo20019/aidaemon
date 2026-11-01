//! [`MemoryWriter`]: background fact extraction as an `on_run_end` hook.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use aidaemon_core::{Hook, HookContext, RunOutcome};
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

#[derive(Default)]
struct Shared {
    reports: Mutex<VecDeque<WriterReport>>,
}

/// Extracts durable facts from the latest user turn once a run completes.
/// Failed, cancelled and interrupted runs write nothing. The extraction runs
/// on a spawned task and can neither block nor change the run, and because
/// it fires at run end, hook order does not matter: an answer another hook
/// rejected is never extracted.
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
}

impl MemoryWriter {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            shared: Arc::new(Shared::default()),
            pending: Mutex::new(Vec::new()),
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
}

#[async_trait]
impl Hook for MemoryWriter {
    async fn on_run_end(&self, ctx: &HookContext<'_>, outcome: &RunOutcome) {
        if !outcome.is_completed() || self.memory.extractor().is_none() {
            return;
        }
        let Some(index) = aidaemon_policies::last_user_message_index(ctx.messages) else {
            return;
        };
        let run_id = ctx.run_id.to_string();
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
    }
}
