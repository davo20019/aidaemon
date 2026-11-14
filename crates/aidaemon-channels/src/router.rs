//! Routing agent events to the conversation whose run produced them.
//!
//! An [`aidaemon_core::Agent`] takes its [`EventSink`] at build time, but the
//! runner starts runs for many conversations on the same agent. The
//! [`EventRouter`] is the bridge: register a clone of it on the agent, give
//! the same router to the runner, and each run's events reach the
//! conversation that started it.
//!
//! ```ignore
//! let router = EventRouter::new();
//! let agent = Agent::builder("assistant").allow_unguarded()
//!     .provider(provider)
//!     .model("model-name")
//!     .event_sink(router.clone())      // 1. the agent emits into the router
//!     .build()?;
//! let runner = ChannelRunner::builder(agent)
//!     .channel(channel)
//!     .event_router(router)            // 2. the runner reads from it
//!     .build()?;
//! ```
//!
//! Without step 1 runs still work, but no streamed progress is shown.
//! Use [`EventRouter::with_next`] to keep another sink (for example a
//! tracing sink) on the agent as well.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use aidaemon_core::{Event, EventSink, RunId};
use tokio::sync::mpsc;

#[derive(Default)]
struct Routes {
    by_session: HashMap<String, mpsc::UnboundedSender<Event>>,
    by_run: HashMap<RunId, mpsc::UnboundedSender<Event>>,
}

/// An [`EventSink`] that forwards each event to the conversation whose run
/// emitted it. Cheap to clone; clones share routes.
#[derive(Clone, Default)]
pub struct EventRouter {
    routes: Arc<Mutex<Routes>>,
    next: Option<Arc<dyn EventSink>>,
}

impl EventRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Also forward every event to `sink` (after routing).
    pub fn with_next(mut self, sink: impl EventSink + 'static) -> Self {
        self.next = Some(Arc::new(sink));
        self
    }

    /// Route runs of `session_id` (bound when the run emits `RunStarted`)
    /// and, if known, the run `run_id` (resumes keep their run id).
    pub(crate) fn attach(
        &self,
        session_id: &str,
        run_id: Option<RunId>,
    ) -> mpsc::UnboundedReceiver<Event> {
        let (tx, rx) = mpsc::unbounded_channel();
        if let Ok(mut routes) = self.routes.lock() {
            if let Some(run_id) = run_id {
                routes.by_run.insert(run_id, tx.clone());
            }
            routes.by_session.insert(session_id.to_string(), tx);
        }
        rx
    }

    pub(crate) fn detach(&self, session_id: &str) {
        if let Ok(mut routes) = self.routes.lock()
            && let Some(tx) = routes.by_session.remove(session_id)
        {
            routes.by_run.retain(|_, other| !other.same_channel(&tx));
        }
    }

    #[cfg(test)]
    fn route_count(&self) -> (usize, usize) {
        let routes = self.routes.lock().expect("router poisoned");
        (routes.by_session.len(), routes.by_run.len())
    }
}

impl EventSink for EventRouter {
    fn emit(&self, run_id: RunId, event: &Event) {
        if let Ok(mut routes) = self.routes.lock() {
            if let Event::RunStarted { session_id } = event
                && let Some(tx) = routes.by_session.get(session_id).cloned()
            {
                routes.by_run.insert(run_id, tx);
            }
            let finished = matches!(event, Event::Finished { .. });
            if let Some(tx) = routes.by_run.get(&run_id) {
                let _ = tx.send(event.clone());
            }
            if finished {
                routes.by_run.remove(&run_id);
            }
        }
        if let Some(next) = &self.next {
            next.emit(run_id, event);
        }
    }

    fn observe_request(&self, run_id: RunId, request: &aidaemon_core::ModelRequest) {
        if let Some(next) = &self.next {
            next.observe_request(run_id, request);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_by_session_then_by_run_id() {
        let router = EventRouter::new();
        let mut rx = router.attach("telegram:1", None);
        let run = RunId::new();
        let other = RunId::new();
        router.emit(
            run,
            &Event::RunStarted {
                session_id: "telegram:1".into(),
            },
        );
        router.emit(run, &Event::TurnStarted { turn: 1 });
        router.emit(other, &Event::TurnStarted { turn: 9 });
        assert!(matches!(rx.try_recv(), Ok(Event::RunStarted { .. })));
        assert!(matches!(rx.try_recv(), Ok(Event::TurnStarted { turn: 1 })));
        assert!(rx.try_recv().is_err());
        router.detach("telegram:1");
        assert_eq!(router.route_count(), (0, 0));
    }

    #[test]
    fn known_run_ids_route_without_run_started() {
        let router = EventRouter::new();
        let run = RunId::new();
        let mut rx = router.attach("slack:C1", Some(run));
        router.emit(run, &Event::RunResumed);
        assert!(matches!(rx.try_recv(), Ok(Event::RunResumed)));
    }
}
