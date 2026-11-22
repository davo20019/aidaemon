//! Share this computer's screen with a daemon (protocol 1.31): the device
//! side of "screens on other machines". The device offers its screen with
//! `share_screen`, connects to the provider socket it was given, sends
//! frames while someone watches and applies the owner's input. What it
//! captures and how it types are a [`ScreenSource`]'s, so any platform (or a
//! test) can share.
//!
//! The device must be paired with `control`. It shares only while this runs,
//! and its [`ScreenSource::controlled`] tells whoever sits at it.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_client_protocol::media::{
    self, MediaClientMessage, MediaHeader, ProviderCommand, SurfaceInput,
};
use aidaemon_client_protocol::{ClientFrame, Command, CommandResult, ServerFrame};
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::link::Auth;
use crate::signin::GrantCache;

/// A picture of the screen.
pub struct Captured {
    /// `jpeg` or `png`.
    pub codec: &'static str,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
}

/// What a device shares: its screen's pictures and its input.
pub trait ScreenSource: Send + 'static {
    /// The screen's size, in its own pixels or points.
    fn size(&self) -> (u32, u32);
    /// A picture, scaled to fit `max_w` by `max_h`.
    fn capture(&mut self, max_w: u32, max_h: u32, quality: u8) -> Result<Captured, String>;
    /// Apply one input event from the owner.
    fn input(&mut self, event: SurfaceInput) -> Result<(), String>;
    /// The owner controls this screen now, or gave it back.
    fn controlled(&mut self, _on: bool) {}
}

async fn token(base: &str, auth: &Auth) -> Result<String, String> {
    match auth {
        Auth::LocalToken(token) => Ok(token.clone()),
        Auth::Device(store) => {
            let device = store
                .load()
                .map_err(|e| e.to_string())?
                .ok_or("This computer is not paired: pair it first")?;
            let _ = base;
            #[cfg(feature = "remote")]
            if crate::remote::is_remote(&device.gateway) {
                let dialer =
                    crate::remote::Dialer::for_device(&device).map_err(|e| e.to_string())?;
                let http = crate::remote::RemoteHttp::new(&device.gateway, dialer)
                    .map_err(|e| e.to_string())?;
                return GrantCache::default()
                    .token_with(&http, &device, crate::signin::now_ms())
                    .await
                    .map_err(|e| e.to_string());
            }
            GrantCache::default()
                .token(&crate::tls::http_client(), &device, crate::signin::now_ms())
                .await
                .map_err(|e| e.to_string())
        }
    }
}

/// Offer the screen and serve it until the daemon or the connection ends
/// it. `gateway` is the gateway's HTTP base (`https://host`). `on_state`
/// hears what happens, in words for whoever runs it.
pub async fn share(
    gateway: &str,
    auth: &Auth,
    name: &str,
    source: Box<dyn ScreenSource>,
    on_state: impl Fn(&str),
) -> Result<(), String> {
    // A daemon reached over iroh is reached through the local bridge.
    let base = crate::local::reachable_base(gateway, auth)
        .await?
        .trim_end_matches('/')
        .to_string();
    let token = token(&base, auth).await?;
    let (width, height) = source.size();
    let reply: ServerFrame = crate::tls::http_client()
        .post(format!("{base}/client/v1/commands"))
        .bearer_auth(&token)
        .json(&ClientFrame::Command {
            request_id: crate::machine::request_id(),
            command: Command::ShareScreen {
                name: name.to_string(),
                width,
                height,
            },
        })
        .send()
        .await
        .map_err(|e| format!("could not reach the daemon: {e}"))?
        .json()
        .await
        .map_err(|e| format!("the daemon's answer was not readable: {e}"))?;
    let (url, ticket) = match reply {
        ServerFrame::Ack {
            result: CommandResult::ScreenShared { url, ticket, .. },
            ..
        } => (url, ticket),
        ServerFrame::Error { error, .. } => return Err(error.message),
        other => return Err(format!("unexpected answer: {other:?}")),
    };
    let ws = match base.split_once("://") {
        Some(("https", rest)) => format!("wss://{rest}{url}"),
        Some((_, rest)) => format!("ws://{rest}{url}"),
        None => format!("ws://{base}{url}"),
    };
    let tls = crate::tls::client_config().map_err(|e| e.to_string())?;
    let (mut socket, _) = tokio_tungstenite::connect_async_tls_with_config(
        ws.as_str(),
        None,
        false,
        Some(tokio_tungstenite::Connector::Rustls(tls)),
    )
    .await
    .map_err(|e| format!("could not open the screen link: {e}"))?;
    let auth_text = serde_json::to_string(&MediaClientMessage::Auth { ticket }).unwrap_or_default();
    socket
        .send(Message::text(auth_text))
        .await
        .map_err(|e| e.to_string())?;
    on_state("Sharing this screen. Nobody is watching yet.");

    let source = Arc::new(Mutex::new(source));
    let mut wanted: Option<(u32, u32, u32, u8)> = None;
    let mut last: Option<u64> = None;
    let mut seq = 0u64;
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    let mut next_frame = tokio::time::Instant::now();
    loop {
        tokio::select! {
            incoming = socket.next() => {
                let message = match incoming {
                    Some(Ok(message)) => message,
                    _ => return Ok(()),
                };
                let text = match message {
                    Message::Text(text) => text,
                    Message::Binary(bytes) => {
                        if let Ok((MediaHeader::Closed { reason }, _)) = media::decode(&bytes) {
                            return Err(reason);
                        }
                        continue;
                    }
                    Message::Close(_) => return Ok(()),
                    _ => continue,
                };
                match serde_json::from_str::<ProviderCommand>(&text) {
                    Ok(ProviderCommand::Start { max_w, max_h, max_fps, quality }) => {
                        wanted = Some((max_w, max_h, max_fps.max(1), quality));
                        last = None;
                        on_state("Someone is watching this screen.");
                    }
                    Ok(ProviderCommand::Stop) => {
                        wanted = None;
                        on_state("Nobody is watching this screen.");
                    }
                    Ok(ProviderCommand::Input { event }) => {
                        let source = source.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            source.lock().unwrap_or_else(|e| e.into_inner()).input(event)
                        })
                        .await;
                    }
                    Ok(ProviderCommand::Controlled { on }) => {
                        source.lock().unwrap_or_else(|e| e.into_inner()).controlled(on);
                        on_state(if on {
                            "The owner is controlling this screen from their device."
                        } else {
                            "The owner gave control back."
                        });
                    }
                    _ => {}
                }
            }
            _ = tick.tick() => {
                let Some((max_w, max_h, max_fps, quality)) = wanted else { continue };
                let now = tokio::time::Instant::now();
                if now < next_frame {
                    continue;
                }
                next_frame = now + Duration::from_millis(1000 / u64::from(max_fps));
                let source = source.clone();
                let captured = tokio::task::spawn_blocking(move || {
                    source
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .capture(max_w, max_h, quality)
                })
                .await
                .map_err(|e| e.to_string())?;
                let Ok(frame) = captured else { continue };
                // Only a changed picture is sent.
                let digest = {
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    frame.bytes.hash(&mut hasher);
                    hasher.finish()
                };
                if last == Some(digest) {
                    continue;
                }
                last = Some(digest);
                seq += 1;
                let header = MediaHeader::Frame {
                    v: media::MEDIA_VERSION,
                    seq,
                    at_ms: crate::signin::now_ms(),
                    codec: frame.codec.to_string(),
                    w: frame.w,
                    h: frame.h,
                    key: true,
                };
                if socket
                    .send(Message::Binary(media::encode(&header, &frame.bytes).into()))
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }
        }
    }
}
