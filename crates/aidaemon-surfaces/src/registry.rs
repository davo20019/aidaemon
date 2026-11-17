//! The registry: every surface the backends report, one control lease per
//! surface, the live sessions watching them and the handoffs waiting.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Duration;

use tokio::sync::{Notify, broadcast, watch};
use tokio::time::Instant;
use tracing::{debug, info, warn};

use crate::SurfaceBackend;
use crate::types::{
    Controller, ExclusiveGuard, Frame, FrameSink, HandoffPrep, InputEvent, StreamHandle,
    StreamParams, SurfaceDescriptor, SurfaceError, SurfaceId, SurfaceInfo, SurfaceObservation,
};

/// How live surfaces behave. Every limit on a person is an inactivity
/// bound, never a wall-clock one.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfacesConfig {
    /// Surfaces can be watched at all.
    pub enabled: bool,
    /// The owner may control a surface (not only watch it).
    pub allow_control: bool,
    /// The owner may take control without the agent having asked. Off: only
    /// while a handoff for that surface waits.
    pub control_anytime: bool,
    /// Live sessions open at once, across all surfaces.
    pub max_live_sessions: usize,
    /// The most any viewer gets.
    pub stream_cap: StreamParams,
    /// The owner's device holds control with no viewer open for this long,
    /// then control goes back to the agent.
    pub disconnect_grace: Duration,
    /// No input from the controlling device for this long gives control
    /// back to the agent.
    pub idle_release: Duration,
}

impl Default for SurfacesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_control: true,
            control_anytime: true,
            max_live_sessions: 8,
            stream_cap: StreamParams {
                max_fps: 20,
                ..StreamParams::default()
            },
            disconnect_grace: Duration::from_secs(30),
            idle_release: Duration::from_secs(15 * 60),
        }
    }
}

/// A change to the set of surfaces, for clients and audit.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceChange {
    /// Created, renamed, navigated, or its controller or handoff changed.
    Updated(SurfaceInfo),
    Closed(SurfaceId),
}

/// What a live session may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveMode {
    /// Watch only.
    View,
    /// Watch and take control on open.
    Control,
}

/// One device watching one surface. Read the current frame with
/// `frames.borrow_and_update()` first: a frame already in the slot when the
/// session opened is not reported as changed.
pub struct LiveSession {
    pub live_id: String,
    pub surface: SurfaceInfo,
    pub frames: watch::Receiver<Option<Arc<Frame>>>,
    /// Turns true when the session ends (closed, surface gone, registry
    /// stopped).
    pub ended: watch::Receiver<bool>,
}

/// Receivers for an open live session (see
/// [`SurfaceRegistry::subscribe_live`]).
pub struct LiveReceivers {
    pub frames: watch::Receiver<Option<Arc<Frame>>>,
    pub ended: watch::Receiver<bool>,
}

/// Lets a backend tell the registry its surfaces changed.
#[derive(Clone)]
pub struct SurfaceNotifier(Weak<Shared>);

impl SurfaceNotifier {
    /// Ask for a refresh soon. Cheap; many calls collapse into one.
    pub fn notify(&self) {
        if let Some(shared) = self.0.upgrade() {
            shared.notify.notify_one();
        }
    }

    /// A notifier that does nothing (a backend with no registry).
    pub fn detached() -> Self {
        Self(Weak::new())
    }
}

/// Every surface, its lease, its live sessions and its handoff.
#[derive(Clone)]
pub struct SurfaceRegistry {
    shared: Arc<Shared>,
}

struct Shared {
    config: RwLock<SurfacesConfig>,
    backends: RwLock<Vec<Arc<dyn SurfaceBackend>>>,
    state: Mutex<State>,
    /// Serializes starting and stopping streams.
    streams: tokio::sync::Mutex<()>,
    changes: broadcast::Sender<SurfaceChange>,
    notify: Notify,
}

#[derive(Default)]
struct State {
    surfaces: BTreeMap<SurfaceId, Entry>,
    lives: HashMap<String, Live>,
}

struct Entry {
    descriptor: SurfaceDescriptor,
    backend: usize,
    controller: Controller,
    handoff: bool,
    stream: Option<Stream>,
    /// Held while the owner controls the surface.
    exclusive: Option<ExclusiveGuard>,
    last_input: Instant,
    holder_left_at: Option<Instant>,
    /// Laid out for the controlling viewer's room, not its own size.
    fitted: bool,
}

impl Entry {
    fn info(&self) -> SurfaceInfo {
        SurfaceInfo::from_descriptor(
            self.descriptor.clone(),
            self.controller.clone(),
            self.handoff,
        )
    }
}

struct Stream {
    sink: FrameSink,
    _handle: StreamHandle,
    viewers: usize,
    /// Held for the stream's life: between a view opening and its socket
    /// subscribing, the backend must still see the stream as watched (or it
    /// stops sending pictures for good).
    _watching: watch::Receiver<Option<Arc<Frame>>>,
}

struct Live {
    surface: SurfaceId,
    device_id: String,
    /// The viewer's room for the picture, for laying the surface out while
    /// its device holds control.
    fit: Option<crate::types::Size>,
    ended: watch::Sender<bool>,
}

/// The smallest and largest room a surface is laid out for, in points.
const MIN_FIT: u32 = 240;
const MAX_FIT: u32 = 2048;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl Default for SurfaceRegistry {
    fn default() -> Self {
        Self::new(SurfacesConfig::default())
    }
}

impl std::fmt::Debug for SurfaceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.shared.state);
        f.debug_struct("SurfaceRegistry")
            .field("surfaces", &state.surfaces.len())
            .field("lives", &state.lives.len())
            .finish()
    }
}

impl SurfaceRegistry {
    pub fn new(config: SurfacesConfig) -> Self {
        let (changes, _) = broadcast::channel(256);
        Self {
            shared: Arc::new(Shared {
                config: RwLock::new(config),
                backends: RwLock::new(Vec::new()),
                state: Mutex::new(State::default()),
                streams: tokio::sync::Mutex::new(()),
                changes,
                notify: Notify::new(),
            }),
        }
    }

    pub fn config(&self) -> SurfacesConfig {
        self.shared
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Replace the settings (from the owner's Settings page). Turning
    /// control off gives every surface back to the agent.
    pub fn set_config(&self, config: SurfacesConfig) {
        let control_off = !config.enabled || !config.allow_control;
        let viewing_off = !config.enabled;
        *self
            .shared
            .config
            .write()
            .unwrap_or_else(|e| e.into_inner()) = config;
        if control_off {
            let held: Vec<(SurfaceId, String)> = lock(&self.shared.state)
                .surfaces
                .iter()
                .filter_map(|(id, e)| e.controller.owner_device().map(|d| (id.clone(), d.into())))
                .collect();
            for (id, device) in held {
                let _ = self.release_control(&id, &device);
            }
        }
        if viewing_off {
            let lives: Vec<String> = lock(&self.shared.state).lives.keys().cloned().collect();
            for live in lives {
                self.close_live(&live);
            }
        }
    }

    pub fn add_backend(&self, backend: Arc<dyn SurfaceBackend>) {
        self.shared
            .backends
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(backend);
        self.shared.notify.notify_one();
    }

    /// The handle a backend uses to report changes.
    pub fn notifier(&self) -> SurfaceNotifier {
        SurfaceNotifier(Arc::downgrade(&self.shared))
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SurfaceChange> {
        self.shared.changes.subscribe()
    }

    /// Run the background work: refresh on a backend's notice, and give
    /// control back after the owner's device leaves or goes quiet. Ends when
    /// the registry is dropped. Needs a tokio runtime.
    pub fn start(&self) {
        let weak = Arc::downgrade(&self.shared);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(2));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                let Some(shared) = weak.upgrade() else { return };
                let registry = SurfaceRegistry { shared };
                let notified = registry.shared.notify.notified();
                tokio::select! {
                    _ = notified => registry.refresh().await,
                    _ = tick.tick() => registry.reap(Instant::now()),
                }
                drop(registry);
            }
        });
    }

    fn emit(&self, change: SurfaceChange) {
        let _ = self.shared.changes.send(change);
    }

    fn backends(&self) -> Vec<Arc<dyn SurfaceBackend>> {
        self.shared
            .backends
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Ask every backend what exists now and publish the differences.
    pub async fn refresh(&self) {
        let backends = self.backends();
        let mut reported = Vec::with_capacity(backends.len());
        for (index, backend) in backends.iter().enumerate() {
            reported.push((index, backend.list().await));
        }
        let mut changes = Vec::new();
        {
            let mut state = lock(&self.shared.state);
            for (index, descriptors) in reported {
                let mut seen = Vec::with_capacity(descriptors.len());
                for descriptor in descriptors {
                    seen.push(descriptor.id.clone());
                    match state.surfaces.get_mut(&descriptor.id) {
                        Some(entry) if entry.descriptor == descriptor => {}
                        Some(entry) => {
                            entry.descriptor = descriptor;
                            changes.push(SurfaceChange::Updated(entry.info()));
                        }
                        None => {
                            let entry = Entry {
                                descriptor,
                                backend: index,
                                controller: Controller::Agent,
                                handoff: false,
                                stream: None,
                                exclusive: None,
                                last_input: Instant::now(),
                                holder_left_at: None,
                                fitted: false,
                            };
                            changes.push(SurfaceChange::Updated(entry.info()));
                            state.surfaces.insert(entry.descriptor.id.clone(), entry);
                        }
                    }
                }
                let gone: Vec<SurfaceId> = state
                    .surfaces
                    .iter()
                    .filter(|(id, e)| e.backend == index && !seen.contains(id))
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in gone {
                    state.surfaces.remove(&id);
                    let ended: Vec<String> = state
                        .lives
                        .iter()
                        .filter(|(_, live)| live.surface == id)
                        .map(|(live_id, _)| live_id.clone())
                        .collect();
                    for live_id in ended {
                        if let Some(live) = state.lives.remove(&live_id) {
                            let _ = live.ended.send(true);
                        }
                    }
                    changes.push(SurfaceChange::Closed(id));
                }
            }
        }
        for change in changes {
            self.emit(change);
        }
    }

    /// Every surface known now, by id.
    pub fn list(&self) -> Vec<SurfaceInfo> {
        lock(&self.shared.state)
            .surfaces
            .values()
            .map(Entry::info)
            .collect()
    }

    pub fn get(&self, id: &SurfaceId) -> Option<SurfaceInfo> {
        lock(&self.shared.state).surfaces.get(id).map(Entry::info)
    }

    /// The surface a tool call in `session_id` names: the one whose agent
    /// reference is `agent_ref`, or with none the session's active one.
    /// Refreshes first, so a tab opened a moment ago is found.
    pub async fn find_for_agent(
        &self,
        session_id: &str,
        agent_ref: Option<&str>,
    ) -> Option<SurfaceInfo> {
        self.refresh().await;
        let state = lock(&self.shared.state);
        let in_session = state
            .surfaces
            .values()
            .filter(|e| e.descriptor.session_id.as_deref() == Some(session_id));
        let mut found = None;
        for entry in in_session {
            match agent_ref {
                Some(r) if entry.descriptor.agent_ref.as_deref() == Some(r) => {
                    return Some(entry.info());
                }
                Some(r) if entry.descriptor.id.as_str() == r => return Some(entry.info()),
                None if entry.descriptor.active => found = Some(entry.info()),
                None if found.is_none() => found = Some(entry.info()),
                _ => {}
            }
        }
        if agent_ref.is_some() { None } else { found }
    }

    fn backend_of(&self, id: &SurfaceId) -> Result<Arc<dyn SurfaceBackend>, SurfaceError> {
        let index = lock(&self.shared.state)
            .surfaces
            .get(id)
            .map(|e| e.backend)
            .ok_or(SurfaceError::NotFound)?;
        self.backends()
            .get(index)
            .cloned()
            .ok_or(SurfaceError::NotFound)
    }

    /// What is at `(x, y)` on `id`, for the owner to point the agent at.
    pub async fn describe_point(
        &self,
        id: &SurfaceId,
        x: f64,
        y: f64,
    ) -> Result<crate::PointedAt, SurfaceError> {
        let inside = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        if !(inside(x) && inside(y)) {
            return Err(SurfaceError::BadInput("point outside the frame"));
        }
        if !self.config().enabled {
            return Err(SurfaceError::Disabled);
        }
        self.backend_of(id)?.describe_point(id, x, y).await
    }

    /// What has the keyboard on `id` now.
    pub async fn focused(&self, id: &SurfaceId) -> Result<Option<crate::PointedAt>, SurfaceError> {
        self.backend_of(id)?.focused(id).await
    }

    pub async fn observe(&self, id: &SurfaceId) -> Result<SurfaceObservation, SurfaceError> {
        self.backend_of(id)?.observe(id).await
    }

    /// Get a surface ready for a handoff; returns the surface to hand over.
    pub async fn prepare_handoff(&self, id: &SurfaceId) -> Result<HandoffPrep, SurfaceError> {
        let prep = self.backend_of(id)?.prepare_handoff(id).await?;
        if prep.id != *id {
            self.refresh().await;
        }
        Ok(prep)
    }

    /// The agent asked the owner to act on `id` and waits.
    pub fn begin_handoff(&self, id: &SurfaceId) -> Result<(), SurfaceError> {
        let info = {
            let mut state = lock(&self.shared.state);
            let entry = state.surfaces.get_mut(id).ok_or(SurfaceError::NotFound)?;
            entry.handoff = true;
            entry.info()
        };
        info!(surface = %id, "handoff opened");
        self.emit(SurfaceChange::Updated(info));
        Ok(())
    }

    /// The handoff ended (answered or not): control goes back to the agent.
    pub fn end_handoff(&self, id: &SurfaceId) {
        let (info, guard) = {
            let mut state = lock(&self.shared.state);
            let Some(entry) = state.surfaces.get_mut(id) else {
                return;
            };
            if !entry.handoff && entry.controller == Controller::Agent {
                return;
            }
            entry.handoff = false;
            entry.controller = Controller::Agent;
            (entry.info(), entry.exclusive.take())
        };
        drop(guard);
        info!(surface = %id, "handoff ended");
        self.emit(SurfaceChange::Updated(info));
    }

    pub fn handoff_pending(&self, id: &SurfaceId) -> bool {
        lock(&self.shared.state)
            .surfaces
            .get(id)
            .is_some_and(|e| e.handoff)
    }

    fn may_control(&self, id: &SurfaceId) -> Result<(), SurfaceError> {
        let config = self.config();
        if !config.enabled {
            return Err(SurfaceError::Disabled);
        }
        if !config.allow_control {
            return Err(SurfaceError::Refused(
                "Controlling screens is turned off in Settings.".into(),
            ));
        }
        let state = lock(&self.shared.state);
        let entry = state.surfaces.get(id).ok_or(SurfaceError::NotFound)?;
        if !entry.handoff && !config.control_anytime {
            return Err(SurfaceError::Refused(
                "You can control this screen only while the agent is waiting for you there.".into(),
            ));
        }
        Ok(())
    }

    /// Open a live session for `device_id`. `Control` also takes control.
    pub async fn open_live(
        &self,
        id: &SurfaceId,
        device_id: &str,
        mode: LiveMode,
        want: StreamParams,
    ) -> Result<LiveSession, SurfaceError> {
        let config = self.config();
        if !config.enabled {
            return Err(SurfaceError::Disabled);
        }
        if mode == LiveMode::Control {
            self.may_control(id)?;
        }
        if lock(&self.shared.state).lives.len() >= config.max_live_sessions {
            return Err(SurfaceError::TooMany);
        }
        let params = StreamParams {
            max_w: want.max_w.clamp(160, config.stream_cap.max_w),
            max_h: want.max_h.clamp(160, config.stream_cap.max_h),
            max_fps: want.max_fps.clamp(1, config.stream_cap.max_fps),
            quality: want.quality.clamp(10, config.stream_cap.quality.max(10)),
            fit: want.fit.map(|room| crate::types::Size {
                w: room.w.clamp(MIN_FIT, MAX_FIT),
                h: room.h.clamp(MIN_FIT, MAX_FIT),
            }),
        };
        let backend = self.backend_of(id)?;
        let frames = {
            let _streams = self.shared.streams.lock().await;
            let existing = lock(&self.shared.state)
                .surfaces
                .get(id)
                .ok_or(SurfaceError::NotFound)?
                .stream
                .as_ref()
                .map(|s| s.sink.clone());
            let sink = match existing {
                Some(sink) => sink,
                None => {
                    let (sink, watching) = FrameSink::new();
                    let handle = backend.start_stream(id, params, sink.clone()).await?;
                    let mut state = lock(&self.shared.state);
                    let entry = state.surfaces.get_mut(id).ok_or(SurfaceError::NotFound)?;
                    entry.stream = Some(Stream {
                        sink: sink.clone(),
                        _handle: handle,
                        viewers: 0,
                        _watching: watching,
                    });
                    sink
                }
            };
            let mut state = lock(&self.shared.state);
            let entry = state.surfaces.get_mut(id).ok_or(SurfaceError::NotFound)?;
            if let Some(stream) = entry.stream.as_mut() {
                stream.viewers += 1;
            }
            sink.subscribe()
        };
        let live_id = format!("live_{}", uuid::Uuid::new_v4().simple());
        let (ended_tx, ended) = watch::channel(false);
        {
            let mut state = lock(&self.shared.state);
            state.lives.insert(
                live_id.clone(),
                Live {
                    surface: id.clone(),
                    device_id: device_id.to_string(),
                    fit: params.fit,
                    ended: ended_tx,
                },
            );
            if let Some(entry) = state.surfaces.get_mut(id)
                && entry.controller.owner_device() == Some(device_id)
            {
                entry.holder_left_at = None;
            }
        }
        debug!(surface = %id, live = %live_id, ?mode, "live session opened");
        if mode == LiveMode::Control
            && let Err(e) = self.take_control(id, device_id).await
        {
            self.close_live(&live_id);
            return Err(e);
        }
        let surface = self.get(id).ok_or(SurfaceError::NotFound)?;
        Ok(LiveSession {
            live_id,
            surface,
            frames,
            ended,
        })
    }

    /// End a live session. The stream stops with its last viewer.
    pub fn close_live(&self, live_id: &str) {
        let stopped = {
            let mut state = lock(&self.shared.state);
            let Some(live) = state.lives.remove(live_id) else {
                return;
            };
            let _ = live.ended.send(true);
            let holder_still_watching = state
                .lives
                .values()
                .any(|l| l.surface == live.surface && l.device_id == live.device_id);
            let Some(entry) = state.surfaces.get_mut(&live.surface) else {
                return;
            };
            if entry.controller.owner_device() == Some(live.device_id.as_str())
                && !holder_still_watching
            {
                entry.holder_left_at = Some(Instant::now());
            }
            match entry.stream.as_mut() {
                Some(stream) if stream.viewers > 1 => {
                    stream.viewers -= 1;
                    None
                }
                Some(_) => entry.stream.take(),
                None => None,
            }
        };
        drop(stopped);
        debug!(live = %live_id, "live session closed");
    }

    /// Fresh receivers for a live session that is open: its frames (the
    /// current one reads as unseen) and its end.
    pub fn subscribe_live(&self, live_id: &str) -> Option<LiveReceivers> {
        let state = lock(&self.shared.state);
        let live = state.lives.get(live_id)?;
        let stream = state.surfaces.get(&live.surface)?.stream.as_ref()?;
        let mut frames = stream.sink.subscribe();
        frames.mark_changed();
        Some(LiveReceivers {
            frames,
            ended: live.ended.subscribe(),
        })
    }

    /// The surface and device a live session belongs to.
    pub fn live_owner(&self, live_id: &str) -> Option<(SurfaceId, String)> {
        lock(&self.shared.state)
            .lives
            .get(live_id)
            .map(|l| (l.surface.clone(), l.device_id.clone()))
    }

    /// Give `device_id` control. Waits for an agent action already running
    /// there to finish; the agent's next action waits until control comes
    /// back. Owner beats agent; between devices the newest wins.
    pub async fn take_control(
        &self,
        id: &SurfaceId,
        device_id: &str,
    ) -> Result<Controller, SurfaceError> {
        self.may_control(id)?;
        let already = {
            let mut state = lock(&self.shared.state);
            let entry = state.surfaces.get_mut(id).ok_or(SurfaceError::NotFound)?;
            match &entry.controller {
                Controller::Owner { device_id: holder } if holder == device_id => {
                    Some(entry.info())
                }
                Controller::Owner { .. } => {
                    entry.controller = Controller::Owner {
                        device_id: device_id.to_string(),
                    };
                    entry.last_input = Instant::now();
                    Some(entry.info())
                }
                Controller::Agent => None,
            }
        };
        if let Some(info) = already {
            self.fit_for(id, device_id).await;
            let controller = info.controller.clone();
            self.emit(SurfaceChange::Updated(info));
            return Ok(controller);
        }
        let guard = self.backend_of(id)?.exclusive(id).await?;
        let info = {
            let mut state = lock(&self.shared.state);
            let watching = state
                .lives
                .values()
                .any(|l| &l.surface == id && l.device_id == device_id);
            let entry = state.surfaces.get_mut(id).ok_or(SurfaceError::NotFound)?;
            if entry.exclusive.is_none() {
                entry.exclusive = Some(guard);
            }
            entry.controller = Controller::Owner {
                device_id: device_id.to_string(),
            };
            entry.last_input = Instant::now();
            entry.holder_left_at = (!watching).then(Instant::now);
            entry.info()
        };
        info!(surface = %id, device = device_id, "owner took control");
        self.fit_for(id, device_id).await;
        let controller = info.controller.clone();
        self.emit(SurfaceChange::Updated(info));
        Ok(controller)
    }

    /// Lay `id` out for the room of `device_id`'s newest viewer (its own
    /// size when that viewer said none). Best effort: a surface that could
    /// not still works, its picture only has more margin.
    async fn fit_for(&self, id: &SurfaceId, device_id: &str) {
        let (room, fitted) = {
            let state = lock(&self.shared.state);
            let room = state
                .lives
                .values()
                .filter(|l| &l.surface == id && l.device_id == device_id)
                .find_map(|l| l.fit);
            let fitted = state.surfaces.get(id).is_some_and(|e| e.fitted);
            (room, fitted)
        };
        if room.is_none() && !fitted {
            return;
        }
        let Ok(backend) = self.backend_of(id) else {
            return;
        };
        match backend.fit(id, room).await {
            Ok(()) => {
                if let Some(entry) = lock(&self.shared.state).surfaces.get_mut(id) {
                    entry.fitted = room.is_some();
                }
            }
            Err(e) => {
                warn!(surface = %id, error = %e, "could not lay the surface out for the owner's screen")
            }
        }
    }

    /// Let the agent act again (drop its hold). A surface laid out for the
    /// owner's screen goes back to its own size first, so the agent never
    /// sees the owner's layout.
    fn unfit_then_release(&self, id: &SurfaceId, guard: Option<ExclusiveGuard>, fitted: bool) {
        let Some(guard) = guard else { return };
        if !fitted {
            drop(guard);
            return;
        }
        let (Ok(backend), Ok(runtime)) =
            (self.backend_of(id), tokio::runtime::Handle::try_current())
        else {
            drop(guard);
            return;
        };
        let id = id.clone();
        runtime.spawn(async move {
            if let Err(e) = backend.fit(&id, None).await {
                warn!(surface = %id, error = %e, "could not put the surface back at its own size");
            }
            drop(guard);
        });
    }

    /// Give control back to the agent. Only the holding device may.
    pub fn release_control(&self, id: &SurfaceId, device_id: &str) -> Result<(), SurfaceError> {
        let (info, guard, fitted) = {
            let mut state = lock(&self.shared.state);
            let entry = state.surfaces.get_mut(id).ok_or(SurfaceError::NotFound)?;
            if entry.controller.owner_device() != Some(device_id) {
                return Ok(());
            }
            entry.controller = Controller::Agent;
            entry.holder_left_at = None;
            let fitted = std::mem::take(&mut entry.fitted);
            (entry.info(), entry.exclusive.take(), fitted)
        };
        self.unfit_then_release(id, guard, fitted);
        info!(surface = %id, device = device_id, "owner gave control back");
        self.emit(SurfaceChange::Updated(info));
        Ok(())
    }

    /// Apply input from a live session. Only the controlling device's input
    /// reaches the surface.
    pub async fn input(&self, live_id: &str, event: &InputEvent) -> Result<(), SurfaceError> {
        event.validate()?;
        let id = {
            let mut state = lock(&self.shared.state);
            let live = state
                .lives
                .get(live_id)
                .ok_or(SurfaceError::NoLiveSession)?;
            let (surface, device) = (live.surface.clone(), live.device_id.clone());
            let entry = state
                .surfaces
                .get_mut(&surface)
                .ok_or(SurfaceError::NotFound)?;
            if entry.controller.owner_device() != Some(device.as_str()) {
                return Err(SurfaceError::Refused(
                    "Take control before typing or tapping.".into(),
                ));
            }
            if !entry.descriptor.caps.accepts(event) {
                return Err(SurfaceError::NotSupported(
                    "this screen does not take that input",
                ));
            }
            entry.last_input = Instant::now();
            surface
        };
        self.backend_of(&id)?.input(&id, event).await
    }

    /// Give control back where the owner's device left or went quiet.
    pub fn reap(&self, now: Instant) {
        let config = self.config();
        let mut released = Vec::new();
        {
            let mut state = lock(&self.shared.state);
            let State { surfaces, lives } = &mut *state;
            for (id, entry) in surfaces.iter_mut() {
                let Some(device) = entry.controller.owner_device().map(str::to_string) else {
                    continue;
                };
                let watching = lives
                    .values()
                    .any(|l| &l.surface == id && l.device_id == device);
                let left_too_long = !watching
                    && entry
                        .holder_left_at
                        .get_or_insert(now)
                        .checked_add(config.disconnect_grace)
                        .is_some_and(|deadline| now >= deadline);
                if watching {
                    entry.holder_left_at = None;
                }
                let quiet_too_long =
                    now.saturating_duration_since(entry.last_input) >= config.idle_release;
                if left_too_long || quiet_too_long {
                    entry.controller = Controller::Agent;
                    entry.holder_left_at = None;
                    let fitted = std::mem::take(&mut entry.fitted);
                    released.push((entry.info(), entry.exclusive.take(), device, fitted));
                }
            }
        }
        for (info, guard, device, fitted) in released {
            self.unfit_then_release(&info.id, guard, fitted);
            warn!(surface = %info.id, device, "control went back to the agent: the owner's device left or went quiet");
            self.emit(SurfaceChange::Updated(info));
        }
    }

    /// Open live sessions (for the gateway's status).
    pub fn live_count(&self) -> usize {
        lock(&self.shared.state).lives.len()
    }
}

#[cfg(test)]
mod tests;
