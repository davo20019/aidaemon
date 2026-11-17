//! Screens on other machines (protocol 1.31): a paired device shares its
//! own display (`aidaemon share-screen`), and the daemon lists it as a live
//! surface like its own desktop.
//!
//! The device connects to `/client/v1/surfaces/provide/{surface_id}/ws`
//! with the single-use ticket `share_screen` gave it, sends frames while
//! someone watches, and takes the owner's input while the owner controls it.
//! The device decides what it shares and shows its own notice when it is
//! driven; it disappears from the list the moment its socket closes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_client_protocol::media::{self, MediaHeader, ProviderCommand, SurfaceInput};
use aidaemon_surfaces::{
    Codec, ExclusiveGuard, FrameSink, InputEvent, KeyAction, Modifier, PointerAction,
    PointerButton, Size, StreamHandle, StreamParams, SurfaceBackend, SurfaceCaps,
    SurfaceDescriptor, SurfaceError, SurfaceId, SurfaceKind, SurfaceNotifier, SurfaceObservation,
    clean_title,
};
use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket};
use tokio::sync::mpsc;
use tracing::info;

/// A device's screen silent for this long (no frame, no pong) is dropped.
const QUIET_LIMIT: Duration = Duration::from_secs(90);
const PING_EVERY: Duration = Duration::from_secs(20);
/// The largest frame a device may send.
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

struct Remote {
    name: String,
    size: Size,
    device_id: String,
    /// Set while the device is connected.
    to_device: Option<mpsc::UnboundedSender<ProviderCommand>>,
    /// Set while someone watches.
    sink: Option<FrameSink>,
}

type Remotes = Arc<Mutex<HashMap<SurfaceId, Remote>>>;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The screens paired devices share.
#[derive(Default)]
pub struct RemoteScreens {
    remotes: Remotes,
    notifier: Mutex<Option<SurfaceNotifier>>,
}

/// The owner's input as the protocol carries it.
pub(crate) fn wire_input(event: &InputEvent) -> SurfaceInput {
    match event {
        InputEvent::Pointer {
            action,
            x,
            y,
            button,
        } => SurfaceInput::Pointer {
            action: match action {
                PointerAction::Down => "down",
                PointerAction::Move => "move",
                PointerAction::Up => "up",
                PointerAction::Cancel => "cancel",
            }
            .into(),
            x: *x,
            y: *y,
            button: Some(
                match button {
                    PointerButton::Left => "left",
                    PointerButton::Right => "right",
                    PointerButton::Middle => "middle",
                }
                .into(),
            ),
        },
        InputEvent::Wheel { x, y, dx, dy } => SurfaceInput::Wheel {
            x: *x,
            y: *y,
            dx: *dx,
            dy: *dy,
        },
        InputEvent::Key {
            action,
            code,
            key,
            mods,
        } => SurfaceInput::Key {
            action: if *action == KeyAction::Down {
                "down"
            } else {
                "up"
            }
            .into(),
            code: code.clone(),
            key: key.clone(),
            mods: mods
                .iter()
                .map(|m| {
                    match m {
                        Modifier::Shift => "shift",
                        Modifier::Ctrl => "ctrl",
                        Modifier::Alt => "alt",
                        Modifier::Meta => "meta",
                    }
                    .to_string()
                })
                .collect(),
        },
        InputEvent::Text { text } => SurfaceInput::Text { text: text.clone() },
    }
}

impl RemoteScreens {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_notifier(&self, notifier: SurfaceNotifier) {
        *lock(&self.notifier) = Some(notifier);
    }

    fn notify(&self) {
        if let Some(notifier) = lock(&self.notifier).as_ref() {
            notifier.notify();
        }
    }

    /// A device offers its screen: the surface it will be. A device that
    /// offered before replaces its earlier offer.
    pub fn offer(&self, device_id: &str, name: &str, width: u32, height: u32) -> SurfaceId {
        let id = SurfaceId::random();
        let mut remotes = lock(&self.remotes);
        remotes.retain(|_, r| r.device_id != device_id);
        remotes.insert(
            id.clone(),
            Remote {
                name: clean_title(if name.trim().is_empty() {
                    "Another computer"
                } else {
                    name
                }),
                size: Size {
                    w: width.clamp(1, 16_384),
                    h: height.clamp(1, 16_384),
                },
                device_id: device_id.to_string(),
                to_device: None,
                sink: None,
            },
        );
        id
    }

    /// The device an offered screen belongs to.
    pub fn device_of(&self, id: &SurfaceId) -> Option<String> {
        lock(&self.remotes).get(id).map(|r| r.device_id.clone())
    }

    fn send(&self, id: &SurfaceId, command: ProviderCommand) -> Result<(), SurfaceError> {
        let remotes = lock(&self.remotes);
        let tx = remotes
            .get(id)
            .and_then(|r| r.to_device.as_ref())
            .ok_or(SurfaceError::NotFound)?;
        tx.send(command)
            .map_err(|_| SurfaceError::Backend("that computer went away".into()))
    }

    /// Serve a device's authenticated provider socket until it closes.
    pub(crate) async fn serve(&self, id: SurfaceId, mut socket: WebSocket) {
        let (tx, mut rx) = mpsc::unbounded_channel::<ProviderCommand>();
        {
            let mut remotes = lock(&self.remotes);
            let Some(remote) = remotes.get_mut(&id) else {
                return;
            };
            remote.to_device = Some(tx);
        }
        info!(surface = %id, "A paired computer shares its screen");
        self.notify();
        let mut last_heard = tokio::time::Instant::now();
        let mut ping = tokio::time::interval(PING_EVERY);
        loop {
            tokio::select! {
                command = rx.recv() => {
                    let Some(command) = command else { break };
                    let Ok(text) = serde_json::to_string(&command) else { continue };
                    if socket.send(Message::Text(text.into())).await.is_err() {
                        break;
                    }
                }
                incoming = socket.recv() => {
                    let Some(Ok(message)) = incoming else { break };
                    last_heard = tokio::time::Instant::now();
                    let Message::Binary(bytes) = message else { continue };
                    if bytes.len() > MAX_FRAME_BYTES {
                        continue;
                    }
                    let Ok((MediaHeader::Frame { codec, w, h, .. }, payload)) = media::decode(&bytes) else {
                        continue;
                    };
                    let codec = match codec.as_str() {
                        "jpeg" => Codec::Jpeg,
                        "png" => Codec::Png,
                        _ => continue,
                    };
                    let sink = lock(&self.remotes).get(&id).and_then(|r| r.sink.clone());
                    if let Some(sink) = sink {
                        sink.push(codec, w, h, payload.to_vec());
                    }
                }
                _ = ping.tick() => {
                    if last_heard.elapsed() >= QUIET_LIMIT {
                        break;
                    }
                    if socket.send(Message::Ping(Vec::new().into())).await.is_err() {
                        break;
                    }
                }
            }
        }
        lock(&self.remotes).remove(&id);
        info!(surface = %id, "A paired computer stopped sharing its screen");
        self.notify();
    }
}

#[async_trait]
impl SurfaceBackend for RemoteScreens {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::Node
    }

    async fn list(&self) -> Vec<SurfaceDescriptor> {
        lock(&self.remotes)
            .iter()
            .filter(|(_, r)| r.to_device.is_some())
            .map(|(id, r)| SurfaceDescriptor {
                id: id.clone(),
                kind: SurfaceKind::Node,
                session_id: None,
                agent_ref: None,
                active: false,
                title: r.name.clone(),
                site: None,
                size: r.size,
                caps: SurfaceCaps::interactive(),
                opener: None,
            })
            .collect()
    }

    async fn start_stream(
        &self,
        id: &SurfaceId,
        params: StreamParams,
        sink: FrameSink,
    ) -> Result<StreamHandle, SurfaceError> {
        {
            let mut remotes = lock(&self.remotes);
            let remote = remotes.get_mut(id).ok_or(SurfaceError::NotFound)?;
            remote.sink = Some(sink);
        }
        self.send(
            id,
            ProviderCommand::Start {
                max_w: params.max_w,
                max_h: params.max_h,
                max_fps: params.max_fps,
                quality: params.quality,
            },
        )?;
        let remotes = self.remotes.clone();
        let id = id.clone();
        Ok(StreamHandle::new(move || {
            let mut remotes = lock(&remotes);
            if let Some(remote) = remotes.get_mut(&id) {
                remote.sink = None;
                if let Some(tx) = &remote.to_device {
                    let _ = tx.send(ProviderCommand::Stop);
                }
            }
        }))
    }

    async fn input(&self, id: &SurfaceId, event: &InputEvent) -> Result<(), SurfaceError> {
        self.send(
            id,
            ProviderCommand::Input {
                event: wire_input(event),
            },
        )
    }

    async fn observe(&self, id: &SurfaceId) -> Result<SurfaceObservation, SurfaceError> {
        let remotes = lock(&self.remotes);
        let remote = remotes.get(id).ok_or(SurfaceError::NotFound)?;
        Ok(SurfaceObservation {
            site: None,
            title: remote.name.clone(),
            password_field_present: false,
            agent_ref: None,
        })
    }

    async fn exclusive(&self, id: &SurfaceId) -> Result<ExclusiveGuard, SurfaceError> {
        self.send(id, ProviderCommand::Controlled { on: true })?;
        struct Released(Remotes, SurfaceId);
        impl Drop for Released {
            fn drop(&mut self) {
                if let Some(tx) = lock(&self.0).get(&self.1).and_then(|r| r.to_device.clone()) {
                    let _ = tx.send(ProviderCommand::Controlled { on: false });
                }
            }
        }
        Ok(ExclusiveGuard::new(Released(
            self.remotes.clone(),
            id.clone(),
        )))
    }
}
