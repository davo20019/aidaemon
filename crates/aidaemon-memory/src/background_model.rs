//! A heavyweight model that loads once in the background and serves one call
//! at a time to callers that each wait only as long as they can afford.
//!
//! Callers (a memory search under its rerank deadline) stop waiting by
//! dropping the future. Two things must survive that:
//! - the load: it runs in its own task and fills the shared cell, so a
//!   caller that gives up never throws a half-done load away and every later
//!   caller reuses the one load in flight or its finished model;
//! - the queue: a call whose caller gave up still runs to the end on its
//!   blocking thread (inference can't be interrupted), but later callers
//!   don't line up behind it. They are refused at once while it runs, and
//!   otherwise wait for the model only inside their own deadline, never
//!   queueing work that outlives them.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{Mutex, OnceCell};
use tokio::task;

type Loader<M> = Box<dyn Fn() -> anyhow::Result<M> + Send + Sync>;

pub(crate) struct BackgroundModel<M> {
    inner: Arc<Inner<M>>,
}

impl<M> Clone for BackgroundModel<M> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

struct Inner<M> {
    name: &'static str,
    load: Loader<M>,
    model: OnceCell<Arc<Mutex<M>>>,
    /// Id of the call holding the model (0: none yet).
    running: AtomicU64,
    /// Id of the last call whose caller stopped waiting before it finished.
    abandoned: AtomicU64,
    next_call: AtomicU64,
}

impl<M: Send + 'static> BackgroundModel<M> {
    /// `load` runs on a blocking thread the first time a call needs the model.
    pub(crate) fn new(
        name: &'static str,
        load: impl Fn() -> anyhow::Result<M> + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                name,
                load: Box::new(load),
                model: OnceCell::new(),
                running: AtomicU64::new(0),
                abandoned: AtomicU64::new(0),
                next_call: AtomicU64::new(1),
            }),
        }
    }

    /// The loaded model, starting the load if none is in flight. The load
    /// runs in a spawned task: a caller that stops waiting leaves it running,
    /// and it fills the cell for everyone. A failed load leaves the cell
    /// empty so a later call tries again.
    async fn model(&self) -> anyhow::Result<Arc<Mutex<M>>> {
        if let Some(model) = self.inner.model.get() {
            return Ok(model.clone());
        }
        let inner = self.inner.clone();
        tokio::spawn(async move {
            inner
                .model
                .get_or_try_init(|| async {
                    let loader = inner.clone();
                    let model = task::spawn_blocking(move || (loader.load)()).await??;
                    tracing::info!(model = inner.name, "model loaded");
                    Ok::<_, anyhow::Error>(Arc::new(Mutex::new(model)))
                })
                .await
                .cloned()
        })
        .await?
    }

    /// Runs `work` on the model on a blocking thread. Refused at once while
    /// the model is busy with a call whose caller gave up; otherwise waits for
    /// the model as long as the caller keeps waiting.
    pub(crate) async fn run<R: Send + 'static>(
        &self,
        work: impl FnOnce(&mut M) -> anyhow::Result<R> + Send + 'static,
    ) -> anyhow::Result<R> {
        let model = self.model().await?;
        let guard = match model.clone().try_lock_owned() {
            Ok(guard) => guard,
            Err(_) => {
                let running = self.inner.running.load(Ordering::SeqCst);
                if running != 0 && self.inner.abandoned.load(Ordering::SeqCst) == running {
                    anyhow::bail!(
                        "{} is still busy with a call its caller gave up on",
                        self.inner.name
                    );
                }
                // Cancellable: a caller that gives up here leaves no work behind.
                model.lock_owned().await
            }
        };
        let call = self.inner.next_call.fetch_add(1, Ordering::SeqCst);
        self.inner.running.store(call, Ordering::SeqCst);
        let mut watch = AbandonWatch {
            inner: &self.inner,
            call,
            armed: true,
        };
        let result = task::spawn_blocking(move || {
            let mut guard = guard;
            work(&mut guard)
        })
        .await;
        watch.armed = false;
        result?
    }

    /// Whether the model has finished loading.
    #[cfg(test)]
    pub(crate) fn is_loaded(&self) -> bool {
        self.inner.model.initialized()
    }
}

/// Marks its call abandoned when the caller drops the future before the
/// blocking work has finished.
struct AbandonWatch<'a, M> {
    inner: &'a Inner<M>,
    call: u64,
    armed: bool,
}

impl<M> Drop for AbandonWatch<'_, M> {
    fn drop(&mut self) {
        if self.armed {
            self.inner.abandoned.store(self.call, Ordering::SeqCst);
            tracing::warn!(
                model = self.inner.name,
                "caller stopped waiting; the call runs on and later callers skip the model until it ends"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    struct Fake;

    fn slow_loader(loads: Arc<AtomicUsize>, delay: Duration) -> BackgroundModel<Fake> {
        BackgroundModel::new("fake", move || {
            loads.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(delay);
            Ok(Fake)
        })
    }

    async fn call_within(
        model: &BackgroundModel<Fake>,
        deadline: Duration,
        work: Duration,
    ) -> Option<anyhow::Result<()>> {
        tokio::time::timeout(
            deadline,
            model.run(move |_| {
                std::thread::sleep(work);
                Ok(())
            }),
        )
        .await
        .ok()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timed_out_first_load_finishes_and_is_reused() {
        let loads = Arc::new(AtomicUsize::new(0));
        let model = slow_loader(loads.clone(), Duration::from_millis(400));

        // The first caller gives up while the model loads.
        assert!(
            call_within(&model, Duration::from_millis(50), Duration::ZERO)
                .await
                .is_none()
        );
        // A second caller during the load joins it rather than starting one.
        assert!(
            call_within(&model, Duration::from_millis(50), Duration::ZERO)
                .await
                .is_none()
        );
        assert!(!model.is_loaded());

        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(model.is_loaded(), "the abandoned load still finished");
        call_within(&model, Duration::from_millis(500), Duration::ZERO)
            .await
            .expect("a loaded, idle model answers in time")
            .unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 1, "exactly one load");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timed_out_call_does_not_hold_up_the_next_one() {
        let loads = Arc::new(AtomicUsize::new(0));
        let model = slow_loader(loads, Duration::ZERO);
        call_within(&model, Duration::from_secs(5), Duration::ZERO)
            .await
            .unwrap()
            .unwrap();

        // A slow inference outlives its caller's deadline...
        assert!(call_within(
            &model,
            Duration::from_millis(50),
            Duration::from_millis(800)
        )
        .await
        .is_none());
        // ...and the next caller is refused at once instead of waiting
        // behind it for its whole deadline.
        let started = std::time::Instant::now();
        let next = call_within(&model, Duration::from_secs(5), Duration::ZERO)
            .await
            .expect("answered before its own deadline");
        assert!(next.is_err(), "skipped while the abandoned call runs");
        assert!(started.elapsed() < Duration::from_millis(200));

        // Once the abandoned call ends, the model serves again.
        tokio::time::sleep(Duration::from_millis(900)).await;
        call_within(&model, Duration::from_millis(500), Duration::ZERO)
            .await
            .expect("an idle model answers in time")
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn busy_model_waits_only_within_the_callers_deadline() {
        let loads = Arc::new(AtomicUsize::new(0));
        let model = slow_loader(loads, Duration::ZERO);
        call_within(&model, Duration::from_secs(5), Duration::ZERO)
            .await
            .unwrap()
            .unwrap();

        // A live (not abandoned) call holds the model for 300 ms.
        let holder = {
            let model = model.clone();
            tokio::spawn(async move {
                call_within(&model, Duration::from_secs(5), Duration::from_millis(300)).await
            })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;

        // A caller with a short deadline gives up without queueing work...
        let started = std::time::Instant::now();
        assert!(
            call_within(&model, Duration::from_millis(50), Duration::ZERO)
                .await
                .is_none()
        );
        assert!(started.elapsed() < Duration::from_millis(200));
        // ...and one with room waits its turn and is served.
        call_within(&model, Duration::from_secs(5), Duration::ZERO)
            .await
            .expect("served after the live call")
            .unwrap();
        holder.await.unwrap().unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_load_is_retried_by_the_next_call() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let model = BackgroundModel::new("flaky", move || {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("download failed");
            }
            Ok(Fake)
        });
        assert!(model.run(|_| Ok(())).await.is_err());
        model.run(|_| Ok(())).await.unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }
}
