//! Stream progress for the model-call watchdog: a provider call reports each
//! stream event, so a slow but live stream is not mistaken for a hang.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Liveness of the streaming call in flight: milliseconds since `started`
/// at the latest received chunk. A caller that bounds a call by idleness
/// rather than wall-clock time scopes a call with [`with_stream_progress`];
/// streaming providers report each chunk with [`note_stream_progress`].
#[derive(Debug, Clone)]
pub struct StreamProgress {
    started: Instant,
    last_chunk_ms: Arc<AtomicU64>,
}

impl Default for StreamProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamProgress {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            last_chunk_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    /// When the latest chunk arrived; the call's start if none has.
    pub fn last_activity(&self) -> Instant {
        self.started + std::time::Duration::from_millis(self.last_chunk_ms.load(Ordering::Relaxed))
    }

    fn note(&self) {
        let elapsed = self.started.elapsed().as_millis() as u64;
        self.last_chunk_ms.fetch_max(elapsed, Ordering::Relaxed);
    }
}

tokio::task_local! {
    static STREAM_PROGRESS: StreamProgress;
}

/// Run `call` with `progress` receiving the chunk reports of any streaming
/// provider it reaches on this task (retries and fallbacks included).
pub async fn with_stream_progress<F: Future>(progress: StreamProgress, call: F) -> F::Output {
    STREAM_PROGRESS.scope(progress, call).await
}

/// A streaming provider received a chunk. No-op outside a progress scope.
pub fn note_stream_progress() {
    let _ = STREAM_PROGRESS.try_with(StreamProgress::note);
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[tokio::test]
    async fn chunk_reports_reach_the_scoped_progress_only() {
        // Outside a scope a report is a no-op, not a panic.
        note_stream_progress();
        let progress = StreamProgress::new();
        let before = progress.last_activity();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        with_stream_progress(progress.clone(), async { note_stream_progress() }).await;
        assert!(progress.last_activity() > before);
    }
}
