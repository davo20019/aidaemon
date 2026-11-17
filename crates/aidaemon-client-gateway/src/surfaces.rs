//! Live surfaces through the gateway (protocol 1.30): the surface commands,
//! the per-session frame socket, and the surface events.
//!
//! The control plane (open, close, take and release control) runs on the
//! main socket as ordinary commands, acked and scope-checked. Frames run on
//! a socket of their own, `/client/v1/surfaces/{live_id}/ws`, so they never
//! wait behind chat events or enter the resume log. That socket opens with a
//! single-use ticket from `surface_opened`, bound to the device and the live
//! session, sent as its first message (never in the URL).
//!
//! Nothing here stores, logs or forwards a frame's bytes or the owner's
//! input beyond the surface itself.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_client_protocol::media::{self, MediaClientMessage, MediaHeader, SurfaceInput};
use aidaemon_client_protocol::{
    self as wire, CommandResult, ErrorCode, Event, ProtocolError, StreamAccept, SurfaceMode,
    SurfaceTransport, TransportKind, ViewAction,
};
use aidaemon_surfaces::{
    Controller, InputEvent, KeyAction, LiveMode, Modifier, PointerAction, PointerButton,
    StreamParams, SurfaceChange, SurfaceError, SurfaceId, SurfaceInfo, SurfaceKind,
    SurfaceRegistry,
};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::response::Response;
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::pairing::Principal;
use crate::views::ViewTargets;
use crate::{ClientGateway, Publisher};

/// How long a ticket from `surface_opened` waits to be used.
const TICKET_TTL: Duration = Duration::from_secs(60);
/// How long the frame socket waits for its ticket.
const AUTH_WITHIN: Duration = Duration::from_secs(10);
/// A frame socket that hears nothing (not even a pong) for this long closes.
const QUIET_LIMIT: Duration = Duration::from_secs(60);
/// How often the server pings the frame socket.
const PING_EVERY: Duration = Duration::from_secs(20);
/// The most input events one live session may send per second.
const INPUT_PER_SECOND: u32 = 120;
/// The largest message a client may send on the frame socket.
const MAX_CLIENT_MESSAGE: usize = 64 * 1024;
/// The device id the owner's local token acts as.
const LOCAL_DEVICE: &str = "local";

struct Ticket {
    /// The ticket's hash; the ticket itself is never kept.
    hash: String,
    live_id: String,
    device_id: String,
    expires: Instant,
    revoked: CancellationToken,
}

/// The gateway's side of live surfaces.
pub struct SurfaceDesk {
    registry: SurfaceRegistry,
    /// Screens paired devices share.
    remote: Arc<crate::remote_screens::RemoteScreens>,
    tickets: Mutex<Vec<Ticket>>,
    /// Device names, for "controlled from …".
    names: Mutex<HashMap<String, String>>,
}

fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The protocol error for a surface failure.
fn refusal(error: SurfaceError) -> ProtocolError {
    let code = match &error {
        SurfaceError::NotFound | SurfaceError::NoLiveSession => ErrorCode::NotFound,
        SurfaceError::Disabled | SurfaceError::Refused(_) => ErrorCode::Forbidden,
        SurfaceError::TooMany => ErrorCode::RateLimited,
        SurfaceError::BadInput(_) | SurfaceError::NotSupported(_) => ErrorCode::Invalid,
        SurfaceError::Backend(_) => ErrorCode::Internal,
    };
    ProtocolError::new(code, error.to_string())
}

impl SurfaceDesk {
    pub fn new(registry: SurfaceRegistry) -> Arc<Self> {
        let remote = crate::remote_screens::RemoteScreens::new();
        remote.set_notifier(registry.notifier());
        registry.add_backend(remote.clone());
        Arc::new(Self {
            registry,
            remote,
            tickets: Mutex::new(Vec::new()),
            names: Mutex::new(HashMap::new()),
        })
    }

    pub fn registry(&self) -> &SurfaceRegistry {
        &self.registry
    }

    /// Whether clients are offered surfaces at all.
    pub fn enabled(&self) -> bool {
        self.registry.config().enabled
    }

    fn device_of(&self, principal: &Principal) -> String {
        let id = principal
            .device_id
            .clone()
            .unwrap_or_else(|| LOCAL_DEVICE.to_string());
        let name = principal
            .device_name
            .clone()
            .unwrap_or_else(|| "this computer".to_string());
        lock(&self.names).insert(id.clone(), name);
        id
    }

    /// A surface as the protocol shows it.
    pub fn wire(&self, info: &SurfaceInfo) -> wire::SurfaceInfo {
        wire::SurfaceInfo {
            id: info.id.to_string(),
            kind: match info.kind {
                SurfaceKind::BrowserTab => wire::SurfaceKind::BrowserTab,
                SurfaceKind::Desktop => wire::SurfaceKind::Desktop,
                SurfaceKind::Window => wire::SurfaceKind::Window,
                SurfaceKind::Node => wire::SurfaceKind::Node,
                _ => wire::SurfaceKind::Unknown,
            },
            session_id: info.session_id.clone(),
            title: info.title.clone(),
            site: info.site.clone(),
            width: info.size.w,
            height: info.size.h,
            controller: self.wire_controller(&info.controller),
            handoff_pending: info.handoff_pending,
            interactive: info.caps.pointer || info.caps.keys || info.caps.text,
            opener: info.opener.as_ref().map(ToString::to_string),
        }
    }

    fn wire_controller(&self, controller: &Controller) -> wire::SurfaceController {
        match controller {
            Controller::Agent => wire::SurfaceController::Agent,
            Controller::Owner { device_id } => wire::SurfaceController::Owner {
                device_id: device_id.clone(),
                device: lock(&self.names).get(device_id).cloned(),
            },
        }
    }

    pub fn list(&self) -> Vec<wire::SurfaceInfo> {
        if !self.enabled() {
            return Vec::new();
        }
        self.registry.list().iter().map(|s| self.wire(s)).collect()
    }

    /// `open_surface`: start a live session and hand out its ticket.
    pub async fn open(
        &self,
        principal: &Principal,
        surface_id: String,
        mode: SurfaceMode,
        accept: StreamAccept,
    ) -> Result<CommandResult, ProtocolError> {
        let device = self.device_of(principal);
        let mode = match mode {
            SurfaceMode::View => LiveMode::View,
            SurfaceMode::Control => LiveMode::Control,
            SurfaceMode::Unknown => {
                return Err(ProtocolError::new(ErrorCode::Invalid, "Unknown mode"));
            }
        };
        let codecs: Vec<String> = if accept.codecs.is_empty() {
            vec!["jpeg".into()]
        } else {
            accept.codecs.clone()
        };
        if !codecs.iter().any(|c| c == "jpeg" || c == "png") {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                "This daemon sends jpeg or png frames",
            ));
        }
        let defaults = StreamParams::default();
        let want = StreamParams {
            max_w: accept.max_w.unwrap_or(defaults.max_w),
            max_h: accept.max_h.unwrap_or(defaults.max_h),
            max_fps: accept.max_fps.unwrap_or(defaults.max_fps),
            quality: accept.quality.unwrap_or(defaults.quality),
        };
        let id = SurfaceId::from_string(surface_id);
        let live = self
            .registry
            .open_live(&id, &device, mode, want)
            .await
            .map_err(refusal)?;
        let ticket = format!("tkt_{}", uuid::Uuid::new_v4().simple());
        lock(&self.tickets).push(Ticket {
            hash: sha(&ticket),
            live_id: live.live_id.clone(),
            device_id: device,
            expires: Instant::now() + TICKET_TTL,
            revoked: principal.revoked.clone(),
        });
        let surface = self.wire(&live.surface);
        let live_id = live.live_id.clone();
        // The socket subscribes again when it opens; this session's own
        // receivers are not needed until then.
        drop(live);
        Ok(CommandResult::SurfaceOpened {
            transports: vec![SurfaceTransport {
                kind: TransportKind::WsFrames,
                url: format!("/client/v1/surfaces/{live_id}/ws"),
                ticket,
                codecs: vec!["jpeg".into(), "png".into()],
            }],
            live_id,
            surface,
        })
    }

    /// `close_surface`: only the device that opened it may.
    pub fn close(
        &self,
        principal: &Principal,
        live_id: &str,
    ) -> Result<CommandResult, ProtocolError> {
        let device = self.device_of(principal);
        match self.registry.live_owner(live_id) {
            Some((_, owner)) if owner == device => {
                self.registry.close_live(live_id);
                lock(&self.tickets).retain(|t| t.live_id != live_id);
                Ok(CommandResult::Reply {
                    text: "Closed".into(),
                })
            }
            Some(_) => Err(ProtocolError::new(
                ErrorCode::Forbidden,
                "Another device opened that view",
            )),
            None => Ok(CommandResult::Reply {
                text: "Already closed".into(),
            }),
        }
    }

    pub async fn take(
        &self,
        principal: &Principal,
        surface_id: String,
    ) -> Result<CommandResult, ProtocolError> {
        let device = self.device_of(principal);
        let id = SurfaceId::from_string(surface_id);
        self.registry
            .take_control(&id, &device)
            .await
            .map_err(refusal)?;
        let info = self
            .registry
            .get(&id)
            .ok_or_else(|| refusal(SurfaceError::NotFound))?;
        Ok(CommandResult::SurfaceControl {
            surface: self.wire(&info),
        })
    }

    pub fn release(
        &self,
        principal: &Principal,
        surface_id: String,
    ) -> Result<CommandResult, ProtocolError> {
        let device = self.device_of(principal);
        let id = SurfaceId::from_string(surface_id);
        self.registry
            .release_control(&id, &device)
            .map_err(refusal)?;
        let info = self
            .registry
            .get(&id)
            .ok_or_else(|| refusal(SurfaceError::NotFound))?;
        Ok(CommandResult::SurfaceControl {
            surface: self.wire(&info),
        })
    }

    /// What the owner pointed at, as their message to the surface's
    /// conversation: the conversation and the words.
    pub async fn point_message(
        &self,
        surface_id: String,
        x: f64,
        y: f64,
        note: Option<String>,
    ) -> Result<(String, String), ProtocolError> {
        let id = SurfaceId::from_string(surface_id);
        let session = self
            .registry
            .get(&id)
            .and_then(|s| s.session_id)
            .ok_or_else(|| {
                ProtocolError::new(ErrorCode::Invalid, "That screen belongs to no conversation")
            })?;
        let at = self
            .registry
            .describe_point(&id, x, y)
            .await
            .map_err(refusal)?;
        let mut thing = at.what.clone();
        if let Some(words) = &at.text {
            thing.push_str(&format!(" “{words}”"));
        }
        let tab = at
            .agent_ref
            .as_deref()
            .map(|tab| format!(" on browser tab {tab}"))
            .unwrap_or_default();
        let selector = at
            .selector
            .as_deref()
            .map(|selector| format!(" (selector `{selector}`)"))
            .unwrap_or_default();
        let note: String = note
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(|n| n.chars().take(2000).collect())
            .unwrap_or_else(|| "Look here.".to_string());
        Ok((
            session,
            format!("{note}\n\n[I pointed at {thing}{tab}{selector}.]"),
        ))
    }

    /// `share_screen`: this device offers its own screen; where it sends
    /// frames, and the ticket for it.
    pub fn share(
        &self,
        principal: &Principal,
        name: String,
        width: u32,
        height: u32,
    ) -> Result<CommandResult, ProtocolError> {
        if !self.enabled() {
            return Err(refusal(SurfaceError::Disabled));
        }
        let device = self.device_of(principal);
        let id = self.remote.offer(&device, &name, width, height);
        let ticket = format!("tkt_{}", uuid::Uuid::new_v4().simple());
        lock(&self.tickets).push(Ticket {
            hash: sha(&ticket),
            live_id: provider_key(&id),
            device_id: device,
            expires: Instant::now() + TICKET_TTL,
            revoked: principal.revoked.clone(),
        });
        Ok(CommandResult::ScreenShared {
            url: format!("/client/v1/surfaces/provide/{id}/ws"),
            surface_id: id.to_string(),
            ticket,
        })
    }

    /// Use a ticket: the device it was given to and what revokes it.
    fn redeem(&self, live_id: &str, ticket: &str) -> Option<(String, CancellationToken)> {
        let hash = sha(ticket);
        let now = Instant::now();
        let mut tickets = lock(&self.tickets);
        let index = tickets
            .iter()
            .position(|t| t.hash == hash && t.live_id == live_id && t.expires > now)?;
        let ticket = tickets.swap_remove(index);
        (!ticket.revoked.is_cancelled()).then_some((ticket.device_id, ticket.revoked))
    }

    /// Close live sessions whose ticket was never used.
    fn expire_tickets(&self) {
        let now = Instant::now();
        let expired: Vec<String> = {
            let mut tickets = lock(&self.tickets);
            let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *tickets)
                .into_iter()
                .partition(|t| t.expires <= now || t.revoked.is_cancelled());
            *tickets = kept;
            gone.into_iter().map(|t| t.live_id).collect()
        };
        for live_id in expired {
            self.registry.close_live(&live_id);
        }
    }

    /// Publish the registry's changes as events, and expire unused tickets.
    /// A surface waiting for the owner is brought forward on the app that
    /// started the conversation (a view move only: no stream, no control).
    pub(crate) fn spawn_pump(self: &Arc<Self>, publisher: Publisher, views: Arc<ViewTargets>) {
        let desk = Arc::downgrade(self);
        let mut changes = self.registry.subscribe();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            let mut shown: HashSet<SurfaceId> = HashSet::new();
            loop {
                tokio::select! {
                    change = changes.recv() => {
                        let Some(desk) = desk.upgrade() else { return };
                        match change {
                            Ok(SurfaceChange::Updated(info)) => {
                                if !info.handoff_pending {
                                    shown.remove(&info.id);
                                } else if shown.insert(info.id.clone())
                                    && let Some(session) = info.session_id.as_deref()
                                {
                                    let _ = views.show(
                                        session,
                                        ViewAction::ShowSurface {
                                            surface_id: info.id.to_string(),
                                        },
                                        "the agent is waiting for you there",
                                    );
                                }
                                publisher.publish(
                                    info.session_id.clone(),
                                    Event::SurfaceUpdated { surface: desk.wire(&info) },
                                );
                            }
                            Ok(SurfaceChange::Closed(id)) => {
                                shown.remove(&id);
                                publisher.publish(
                                    None,
                                    Event::SurfaceClosed { surface_id: id.to_string() },
                                );
                            }
                            Err(broadcast::error::RecvError::Lagged(_)) => {
                                for info in desk.registry.list() {
                                    publisher.publish(
                                        info.session_id.clone(),
                                        Event::SurfaceUpdated { surface: desk.wire(&info) },
                                    );
                                }
                            }
                            Err(broadcast::error::RecvError::Closed) => return,
                        }
                    }
                    _ = tick.tick() => {
                        let Some(desk) = desk.upgrade() else { return };
                        desk.expire_tickets();
                    }
                }
            }
        });
    }
}

/// The ticket key of a screen a device provides (tickets otherwise name
/// live sessions).
fn provider_key(id: &SurfaceId) -> String {
    format!("provide:{id}")
}

/// `GET /client/v1/surfaces/provide/{surface_id}/ws`: a paired device's
/// own screen.
pub(crate) async fn provider_socket(
    State(gateway): State<Arc<ClientGateway>>,
    Path(surface_id): Path<String>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade
        .max_message_size(8 * 1024 * 1024 + 64 * 1024)
        .on_upgrade(move |mut socket| async move {
            let Some(desk) = gateway.surface_desk() else {
                return;
            };
            let id = SurfaceId::from_string(surface_id);
            let auth = match tokio::time::timeout(AUTH_WITHIN, socket.recv()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    serde_json::from_str::<MediaClientMessage>(&text).ok()
                }
                _ => None,
            };
            let Some(MediaClientMessage::Auth { ticket }) = auth else {
                let _ = close_with(&mut socket, "Send the ticket first").await;
                return;
            };
            let Some((device, revoked)) = desk.redeem(&provider_key(&id), &ticket) else {
                let _ = close_with(&mut socket, "That ticket is not valid").await;
                return;
            };
            if desk.remote.device_of(&id).as_deref() != Some(device.as_str()) {
                let _ = close_with(&mut socket, "That ticket is not valid").await;
                return;
            }
            tokio::select! {
                _ = desk.remote.serve(id, socket) => {}
                _ = revoked.cancelled() => {}
            }
        })
}

/// The owner's input as the surfaces crate takes it.
pub fn input_of(input: SurfaceInput) -> Result<InputEvent, &'static str> {
    Ok(match input {
        SurfaceInput::Pointer {
            action,
            x,
            y,
            button,
        } => InputEvent::Pointer {
            action: match action.as_str() {
                "down" => PointerAction::Down,
                "move" => PointerAction::Move,
                "up" => PointerAction::Up,
                "cancel" => PointerAction::Cancel,
                _ => return Err("unknown pointer action"),
            },
            x,
            y,
            button: match button.as_deref() {
                None | Some("left") => PointerButton::Left,
                Some("right") => PointerButton::Right,
                Some("middle") => PointerButton::Middle,
                Some(_) => return Err("unknown button"),
            },
        },
        SurfaceInput::Wheel { x, y, dx, dy } => InputEvent::Wheel { x, y, dx, dy },
        SurfaceInput::Key {
            action,
            code,
            key,
            mods,
        } => InputEvent::Key {
            action: match action.as_str() {
                "down" => KeyAction::Down,
                "up" => KeyAction::Up,
                _ => return Err("unknown key action"),
            },
            code,
            key,
            mods: mods
                .iter()
                .map(|m| match m.as_str() {
                    "shift" => Ok(Modifier::Shift),
                    "ctrl" => Ok(Modifier::Ctrl),
                    "alt" => Ok(Modifier::Alt),
                    "meta" => Ok(Modifier::Meta),
                    _ => Err("unknown modifier"),
                })
                .collect::<Result<_, _>>()?,
        },
        SurfaceInput::Text { text } => InputEvent::Text { text },
    })
}

/// `GET /client/v1/surfaces/{live_id}/ws`.
pub(crate) async fn media_socket(
    State(gateway): State<Arc<ClientGateway>>,
    Path(live_id): Path<String>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade
        .max_message_size(MAX_CLIENT_MESSAGE)
        .on_upgrade(move |socket| async move {
            let Some(desk) = gateway.surface_desk() else {
                return;
            };
            if let Err(error) = run_media(desk, socket, live_id).await {
                debug!(%error, "Surface socket closed");
            }
        })
}

async fn send_header(socket: &mut WebSocket, header: &MediaHeader) -> anyhow::Result<()> {
    socket
        .send(Message::Binary(media::encode(header, &[]).into()))
        .await?;
    Ok(())
}

async fn close_with(socket: &mut WebSocket, reason: &str) -> anyhow::Result<()> {
    let _ = send_header(
        socket,
        &MediaHeader::Closed {
            reason: reason.to_string(),
        },
    )
    .await;
    let _ = socket.send(Message::Close(None)).await;
    Ok(())
}

async fn run_media(
    desk: Arc<SurfaceDesk>,
    mut socket: WebSocket,
    live_id: String,
) -> anyhow::Result<()> {
    let auth = match tokio::time::timeout(AUTH_WITHIN, socket.recv()).await {
        Ok(Some(Ok(Message::Text(text)))) => serde_json::from_str::<MediaClientMessage>(&text).ok(),
        _ => None,
    };
    let Some(MediaClientMessage::Auth { ticket }) = auth else {
        return close_with(&mut socket, "Send the ticket first").await;
    };
    let Some((device_id, revoked)) = desk.redeem(&live_id, &ticket) else {
        return close_with(&mut socket, "That ticket is not valid").await;
    };
    let Some((surface_id, owner)) = desk.registry.live_owner(&live_id) else {
        return close_with(&mut socket, "That view has ended").await;
    };
    if owner != device_id {
        return close_with(&mut socket, "That ticket is not valid").await;
    }
    let result = pump_media(
        &desk,
        &mut socket,
        &live_id,
        &surface_id,
        &device_id,
        revoked,
    )
    .await;
    desk.registry.close_live(&live_id);
    info!(surface = %surface_id, device = %device_id, "Surface view closed");
    result
}

async fn pump_media(
    desk: &SurfaceDesk,
    socket: &mut WebSocket,
    live_id: &str,
    surface_id: &SurfaceId,
    device_id: &str,
    revoked: CancellationToken,
) -> anyhow::Result<()> {
    // The session's receivers: a fresh subscription to the same stream.
    let Some(mut live) = desk.registry.subscribe_live(live_id) else {
        return close_with(socket, "That view has ended").await;
    };
    let mut changes = desk.registry.subscribe();
    send_header(
        socket,
        &MediaHeader::Ready {
            surface_id: surface_id.to_string(),
        },
    )
    .await?;
    if let Some(info) = desk.registry.get(surface_id) {
        send_header(
            socket,
            &MediaHeader::Controller {
                controller: desk.wire_controller(&info.controller),
            },
        )
        .await?;
    }
    // One frame in flight: the next goes once the client acks this one.
    let mut in_flight: Option<u64> = None;
    let mut waiting = true;
    let mut last_heard = Instant::now();
    let mut ping = tokio::time::interval(PING_EVERY);
    let mut window_start = Instant::now();
    let mut inputs_in_window = 0u32;
    loop {
        if waiting && in_flight.is_none() {
            let frame = live.frames.borrow_and_update().clone();
            if let Some(frame) = frame {
                let header = MediaHeader::Frame {
                    v: media::MEDIA_VERSION,
                    seq: frame.seq,
                    at_ms: frame.at_ms,
                    codec: frame.codec.name().to_string(),
                    w: frame.w,
                    h: frame.h,
                    key: frame.key,
                };
                socket
                    .send(Message::Binary(media::encode(&header, &frame.data).into()))
                    .await?;
                in_flight = Some(frame.seq);
            }
            waiting = false;
        }
        tokio::select! {
            changed = live.frames.changed() => {
                if changed.is_err() {
                    return close_with(socket, "The screen closed").await;
                }
                waiting = true;
            }
            _ = live.ended.changed() => {
                return close_with(socket, "The view ended").await;
            }
            change = changes.recv() => match change {
                Ok(SurfaceChange::Updated(info)) if &info.id == surface_id => {
                    send_header(socket, &MediaHeader::Controller {
                        controller: desk.wire_controller(&info.controller),
                    }).await?;
                }
                Ok(SurfaceChange::Closed(id)) if &id == surface_id => {
                    return close_with(socket, "The screen closed").await;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return close_with(socket, "The daemon is stopping").await;
                }
                _ => {}
            },
            incoming = socket.recv() => {
                let Some(Ok(message)) = incoming else { return Ok(()) };
                last_heard = Instant::now();
                let text = match message {
                    Message::Text(text) => text,
                    Message::Close(_) => return Ok(()),
                    _ => continue,
                };
                match serde_json::from_str::<MediaClientMessage>(&text) {
                    Ok(MediaClientMessage::Ack { seq }) => {
                        if in_flight.is_some_and(|sent| seq >= sent) {
                            in_flight = None;
                        }
                    }
                    Ok(MediaClientMessage::Input { event }) => {
                        let now = Instant::now();
                        if now.duration_since(window_start) >= Duration::from_secs(1) {
                            window_start = now;
                            inputs_in_window = 0;
                        }
                        inputs_in_window += 1;
                        if inputs_in_window > INPUT_PER_SECOND {
                            continue;
                        }
                        let Ok(event) = input_of(event) else { continue };
                        let says_focus = matches!(
                            event,
                            InputEvent::Pointer { action: PointerAction::Up, .. }
                                | InputEvent::Key { .. }
                                | InputEvent::Text { .. }
                        );
                        if let Err(error) = desk.registry.input(live_id, &event).await {
                            debug!(device = device_id, %error, "Surface input refused");
                            continue;
                        }
                        // Say where typing goes now, so the owner knows a
                        // box has the keyboard before typing into it.
                        if says_focus
                            && let Ok(focused) = desk.registry.focused(surface_id).await
                        {
                            send_header(socket, &MediaHeader::Focus {
                                what: focused.as_ref().map(|f| f.what.clone()),
                                rect: focused.as_ref().and_then(|f| f.rect),
                                label: focused.and_then(|f| f.text),
                            }).await?;
                        }
                    }
                    _ => {}
                }
            }
            _ = ping.tick() => {
                if last_heard.elapsed() >= QUIET_LIMIT {
                    return close_with(socket, "Nothing heard for a minute").await;
                }
                socket.send(Message::Ping(Vec::new().into())).await?;
            }
            _ = revoked.cancelled() => {
                return close_with(socket, "This device was revoked").await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_input_maps_to_surface_input() {
        let event = input_of(SurfaceInput::Key {
            action: "down".into(),
            code: "KeyA".into(),
            key: "a".into(),
            mods: vec!["shift".into()],
        })
        .unwrap();
        assert!(matches!(event, InputEvent::Key { mods, .. } if mods == vec![Modifier::Shift]));
        assert!(
            input_of(SurfaceInput::Pointer {
                action: "hover".into(),
                x: 0.1,
                y: 0.1,
                button: None,
            })
            .is_err()
        );
    }
}
