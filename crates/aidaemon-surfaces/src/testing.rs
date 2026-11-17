//! A scripted backend for tests: surfaces made by hand, frames pushed by
//! hand, input recorded.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::types::{
    Codec, ExclusiveGuard, FrameSink, HandoffPrep, InputEvent, PointedAt, Size, StreamHandle,
    StreamParams, SurfaceCaps, SurfaceDescriptor, SurfaceError, SurfaceId, SurfaceKind,
    SurfaceObservation,
};
use crate::{SurfaceBackend, SurfaceNotifier};

#[derive(Default)]
struct MockState {
    surfaces: Vec<SurfaceDescriptor>,
    sinks: HashMap<SurfaceId, FrameSink>,
    streams_started: usize,
    streams_stopped: Arc<Mutex<usize>>,
    inputs: Vec<(SurfaceId, InputEvent)>,
    locks: HashMap<SurfaceId, Arc<tokio::sync::Mutex<()>>>,
    observation: SurfaceObservation,
    reopen_on_handoff: bool,
}

/// A backend whose surfaces and frames a test controls.
#[derive(Clone, Default)]
pub struct MockSurface {
    state: Arc<Mutex<MockState>>,
    notifier: Arc<Mutex<Option<SurfaceNotifier>>>,
}

impl MockSurface {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> std::sync::MutexGuard<'_, MockState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_notifier(&self, notifier: SurfaceNotifier) {
        *self.notifier.lock().unwrap() = Some(notifier);
    }

    fn notify(&self) {
        if let Some(n) = self.notifier.lock().unwrap().as_ref() {
            n.notify();
        }
    }

    /// Add a browser-tab-like surface in `session`, known to the agent as
    /// `agent_ref`.
    pub fn add(&self, session: &str, agent_ref: &str) -> SurfaceId {
        let id = SurfaceId::random();
        let mut state = self.state();
        for other in state
            .surfaces
            .iter_mut()
            .filter(|s| s.session_id.as_deref() == Some(session))
        {
            other.active = false;
        }
        state.surfaces.push(SurfaceDescriptor {
            id: id.clone(),
            kind: SurfaceKind::BrowserTab,
            session_id: Some(session.to_string()),
            agent_ref: Some(agent_ref.to_string()),
            active: true,
            title: "Example".into(),
            site: Some("example.test".into()),
            size: Size { w: 1280, h: 720 },
            caps: SurfaceCaps::interactive(),
            opener: None,
        });
        state.locks.insert(id.clone(), Arc::default());
        drop(state);
        self.notify();
        id
    }

    pub fn remove(&self, id: &SurfaceId) {
        self.state().surfaces.retain(|s| &s.id != id);
        self.notify();
    }

    pub fn set_title(&self, id: &SurfaceId, title: &str) {
        if let Some(s) = self.state().surfaces.iter_mut().find(|s| &s.id == id) {
            s.title = title.to_string();
        }
        self.notify();
    }

    /// Push a frame to a surface's stream, if one runs.
    pub fn push_frame(&self, id: &SurfaceId, bytes: &[u8]) -> bool {
        let sink = self.state().sinks.get(id).cloned();
        sink.is_some_and(|sink| sink.push(Codec::Jpeg, 2, 2, bytes.to_vec()))
    }

    pub fn streams_started(&self) -> usize {
        self.state().streams_started
    }

    pub fn streams_stopped(&self) -> usize {
        *self.state().streams_stopped.lock().unwrap()
    }

    pub fn inputs(&self) -> Vec<(SurfaceId, InputEvent)> {
        self.state().inputs.clone()
    }

    pub fn set_observation(&self, observation: SurfaceObservation) {
        self.state().observation = observation;
    }

    /// Make `prepare_handoff` reopen the surface under a new id.
    pub fn reopen_on_handoff(&self) {
        self.state().reopen_on_handoff = true;
    }

    /// The agent's action lock for a surface: hold it to simulate an agent
    /// action in progress.
    pub fn agent_lock(&self, id: &SurfaceId) -> Arc<tokio::sync::Mutex<()>> {
        self.state().locks.entry(id.clone()).or_default().clone()
    }
}

#[async_trait]
impl SurfaceBackend for MockSurface {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::BrowserTab
    }

    async fn list(&self) -> Vec<SurfaceDescriptor> {
        self.state().surfaces.clone()
    }

    async fn start_stream(
        &self,
        id: &SurfaceId,
        _params: StreamParams,
        sink: FrameSink,
    ) -> Result<StreamHandle, SurfaceError> {
        let mut state = self.state();
        if !state.surfaces.iter().any(|s| &s.id == id) {
            return Err(SurfaceError::NotFound);
        }
        state.streams_started += 1;
        state.sinks.insert(id.clone(), sink);
        let stopped = state.streams_stopped.clone();
        let this = self.clone();
        let id = id.clone();
        Ok(StreamHandle::new(move || {
            *stopped.lock().unwrap() += 1;
            this.state().sinks.remove(&id);
        }))
    }

    async fn input(&self, id: &SurfaceId, event: &InputEvent) -> Result<(), SurfaceError> {
        self.state().inputs.push((id.clone(), event.clone()));
        Ok(())
    }

    async fn observe(&self, id: &SurfaceId) -> Result<SurfaceObservation, SurfaceError> {
        let state = self.state();
        let surface = state
            .surfaces
            .iter()
            .find(|s| &s.id == id)
            .ok_or(SurfaceError::NotFound)?;
        let mut observation = state.observation.clone();
        if observation.title.is_empty() {
            observation.title = surface.title.clone();
        }
        observation.agent_ref = surface.agent_ref.clone();
        Ok(observation)
    }

    async fn describe_point(
        &self,
        id: &SurfaceId,
        _x: f64,
        _y: f64,
    ) -> Result<PointedAt, SurfaceError> {
        let state = self.state();
        let surface = state
            .surfaces
            .iter()
            .find(|s| &s.id == id)
            .ok_or(SurfaceError::NotFound)?;
        Ok(PointedAt {
            what: "a button".into(),
            text: Some("Sign in".into()),
            selector: Some("#signin".into()),
            agent_ref: surface.agent_ref.clone(),
            rect: None,
        })
    }

    async fn focused(&self, _id: &SurfaceId) -> Result<Option<PointedAt>, SurfaceError> {
        Ok(Some(PointedAt {
            what: "a text field".into(),
            text: Some("Username".into()),
            selector: Some("#username".into()),
            agent_ref: None,
            rect: Some([0.04, 0.4, 0.5, 0.05]),
        }))
    }

    async fn exclusive(&self, id: &SurfaceId) -> Result<ExclusiveGuard, SurfaceError> {
        let lock = self.agent_lock(id);
        Ok(ExclusiveGuard::new(lock.lock_owned().await))
    }

    async fn prepare_handoff(&self, id: &SurfaceId) -> Result<HandoffPrep, SurfaceError> {
        let mut state = self.state();
        if !state.reopen_on_handoff {
            return Ok(HandoffPrep {
                id: id.clone(),
                keeps_sign_in: Some(true),
            });
        }
        let surface = state
            .surfaces
            .iter_mut()
            .find(|s| &s.id == id)
            .ok_or(SurfaceError::NotFound)?;
        let new_id = SurfaceId::random();
        surface.id = new_id.clone();
        state.locks.insert(new_id.clone(), Arc::default());
        Ok(HandoffPrep {
            id: new_id,
            keeps_sign_in: Some(true),
        })
    }
}
