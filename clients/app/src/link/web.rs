//! The browser: the shared [`LinkMachine`] over the page's WebSocket, signed
//! in with the device key the page keeps (WebCrypto, non-extractable). The
//! app is served by the gateway itself, so every request is same-origin.

use std::time::Duration;

use aidaemon_client_core::machine::{ClientIdentity, DropKind, LinkMachine, Tick, Update};
use aidaemon_client_core::signin::{
    self, GrantCache, Http, HttpReply, PairRequest, SignInError, Signer,
};
use aidaemon_client_protocol::{Capability, ClientKind, Command};
use futures::StreamExt;
use futures::channel::mpsc;
use nsh_core::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::store::Store;

#[wasm_bindgen(module = "/js/browser.js")]
extern "C" {
    fn origin() -> String;
    #[wasm_bindgen(js_name = deviceId)]
    fn device_id() -> js_sys::Promise;
    #[wasm_bindgen(js_name = newKey)]
    fn new_key() -> js_sys::Promise;
    #[wasm_bindgen(js_name = keepKey)]
    fn keep_key(id: &str) -> js_sys::Promise;
    fn sign(message: &str) -> js_sys::Promise;
    fn forget() -> js_sys::Promise;
    #[wasm_bindgen(js_name = pairingLink)]
    fn pairing_link() -> Vec<String>;
    #[wasm_bindgen(js_name = forgetPairingLink)]
    fn forget_pairing_link();
    #[wasm_bindgen(js_name = takePicture)]
    fn take_picture_js(max_dimension: u32) -> js_sys::Promise;
    #[wasm_bindgen(js_name = downloadUrl)]
    fn download_url_js(url: &str, name: &str);
    #[wasm_bindgen(js_name = canShare)]
    fn can_share_js() -> bool;
    #[wasm_bindgen(js_name = prefetchShare)]
    fn prefetch_share_js(url: &str);
    #[wasm_bindgen(js_name = shareFile)]
    fn share_file_js(url: &str, name: &str) -> js_sys::Promise;
    #[wasm_bindgen(js_name = markUp)]
    fn mark_up_js(
        url: &str,
        strokes: &str,
        width: f32,
        color: &str,
        max_dimension: u32,
    ) -> js_sys::Promise;

    #[wasm_bindgen(js_name = appleDevice)]
    fn apple_device_js() -> bool;
    #[wasm_bindgen(js_name = pageVisible)]
    fn page_visible() -> bool;
    #[wasm_bindgen(js_name = onWake)]
    fn on_wake(wake: &Closure<dyn FnMut()>);

    type MediaSocket;
    #[wasm_bindgen(constructor)]
    fn new(url: &str, ticket: &str, event: &Closure<dyn FnMut(String, String)>) -> MediaSocket;
    #[wasm_bindgen(method, js_name = send)]
    fn send_media(this: &MediaSocket, text: &str);
    #[wasm_bindgen(method, js_name = close)]
    fn close_media(this: &MediaSocket);

    type KeyCapture;
    #[wasm_bindgen(constructor)]
    fn new(event: &Closure<dyn FnMut(bool, String, String, String) -> bool>) -> KeyCapture;
    #[wasm_bindgen(method, js_name = stop)]
    fn stop_keys(this: &KeyCapture);

    type Socket;
    #[wasm_bindgen(constructor)]
    fn new(url: &str, event: &Closure<dyn FnMut(String, String)>) -> Socket;
    #[wasm_bindgen(method)]
    fn send(this: &Socket, text: &str);
    #[wasm_bindgen(method)]
    fn close(this: &Socket);
}

/// A picture taken with the device's camera: its JPEG bytes, scaled to fit
/// `max_dimension`; `None` when nothing was taken.
/// Save a same-origin file under `name` (the browser's download).
pub fn download(url: &str, name: &str) {
    download_url_js(url, name);
}

/// Whether the browser has a share sheet.
/// An iPhone, iPad or Mac: its system font is Apple's own.
pub fn apple_device() -> bool {
    apple_device_js()
}

pub fn can_share() -> bool {
    can_share_js()
}

/// Fetch a picture ahead, so a tap on Share can open the sheet at once.
pub fn prefetch_share(url: &str) {
    prefetch_share_js(url);
}

/// Share a picture through the system's share sheet (or save it, where
/// files cannot be shared). Called in the tap: the sheet opens only then.
pub fn share_file(url: &str, name: &str) {
    drop(share_file_js(url, name));
}

/// The picture with `strokes` (JSON) drawn over it, as JPEG bytes.
pub async fn mark_up(url: &str, strokes: &str, width: f32, color: &str) -> Option<Vec<u8>> {
    let made = JsFuture::from(mark_up_js(url, strokes, width, color, 2048))
        .await
        .ok()?;
    (!made.is_null() && !made.is_undefined()).then(|| js_sys::Uint8Array::new(&made).to_vec())
}

pub async fn take_picture(max_dimension: u32) -> Option<Vec<u8>> {
    let taken = JsFuture::from(take_picture_js(max_dimension)).await.ok()?;
    (!taken.is_null() && !taken.is_undefined()).then(|| js_sys::Uint8Array::new(&taken).to_vec())
}

async fn text_of(promise: js_sys::Promise) -> Result<String, String> {
    JsFuture::from(promise)
        .await
        .map(|v| v.as_string().unwrap_or_default())
        .map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}")))
}

/// Requests through NSH's `fetch`, the page's own.
struct PageHttp;

fn reply(result: Result<HttpResponse, String>) -> Result<HttpReply, SignInError> {
    match result {
        Ok(response) => Ok(HttpReply {
            status: response.status(),
            body: response.text(),
        }),
        Err(why) => Err(SignInError::Unreachable(why)),
    }
}

impl Http for PageHttp {
    async fn post_json(&self, url: &str, body: String) -> Result<HttpReply, SignInError> {
        reply(fetch(HttpRequest::post(url, body).header("content-type", "application/json")).await)
    }

    async fn get(&self, url: &str) -> Result<HttpReply, SignInError> {
        reply(fetch(HttpRequest::get(url)).await)
    }

    async fn sleep(&self, duration: Duration) {
        timer(duration).await;
    }
}

/// The page's key.
struct PageKey;

impl Signer for PageKey {
    async fn sign(&self, message: &str) -> Result<String, SignInError> {
        text_of(sign(message)).await.map_err(SignInError::Refused)
    }
}

fn now() -> u64 {
    crate::store::now_ms()
}

/// The pairing offer the opening link carried, if any.
pub fn offered_pairing() -> Option<(String, String, Option<String>)> {
    let parts = pairing_link();
    let (offer, code, fp) = (
        parts.first().cloned().unwrap_or_default(),
        parts.get(1).cloned().unwrap_or_default(),
        parts.get(2).cloned().unwrap_or_default(),
    );
    (!offer.is_empty()).then(|| (offer, code, (!fp.is_empty()).then_some(fp)))
}

/// Pair this browser, then connect.
pub async fn pair(
    store: Store,
    offer: String,
    code: String,
    name: String,
    fingerprint: Option<String>,
) -> Result<(), String> {
    let request = PairRequest {
        gateway: origin(),
        offer_id: offer,
        code,
        device_name: name,
        device_kind: ClientKind::Web,
        fingerprint,
    };
    let public = text_of(new_key()).await?;
    let id = signin::redeem_and_wait(&PageHttp, &request, public, Duration::from_millis(1500))
        .await
        .map_err(|e| e.to_string())?;
    text_of(keep_key(&id)).await?;
    forget_pairing_link();
    // The link (owned by the root, waiting for a key) picks it up.
    store.update(Update::Connecting);
    Ok(())
}

/// What a socket said.
enum Heard {
    Open,
    Message(String),
    Close(String),
}

/// The paired device's id, once there is one: the pairing screen saves it.
async fn paired_device(store: Store) -> String {
    loop {
        let device = text_of(device_id()).await.unwrap_or_default();
        if !device.is_empty() {
            return device;
        }
        if store.connection_untracked() != crate::store::Connection::NeedsPairing {
            store.needs_pairing();
        }
        timer(Duration::from_secs(1)).await;
    }
}

/// Forget this browser's key: the daemon refused it, and the owner pairs
/// again.
pub async fn unpair() {
    let _ = text_of(forget()).await;
}

pub fn start(store: Store) {
    // Files' links are on this page's own origin; opening one needs it
    // whole.
    store.set_gateway_base(origin());
    spawn(async move {
        let mut device = paired_device(store).await;
        let mut machine = LinkMachine::new(
            ClientIdentity {
                kind: ClientKind::Web,
                name: "aidaemon-app".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            },
            None,
            vec![Capability::ViewActions, Capability::Surfaces],
        );
        let (commands_tx, mut commands) = mpsc::unbounded::<(String, Command)>();
        store.attach(Box::new(move |id, command| {
            let _ = commands_tx.unbounded_send((id, command));
        }));
        // The page shown again or back online: check the socket now, or
        // stop waiting out the backoff.
        let (wake_tx, mut wakes) = mpsc::unbounded::<()>();
        let wake = Closure::<dyn FnMut()>::new(move || {
            let _ = wake_tx.unbounded_send(());
        });
        on_wake(&wake);
        wake.forget();
        let base = origin();
        let url = aidaemon_client_core::machine::socket_url(&base);
        let mut grant = GrantCache::default();
        loop {
            machine.set_seen(page_visible());
            store.update(Update::Connecting);
            let token = match grant_token(&mut grant, &base, &device).await {
                Ok(token) => token,
                Err((DropKind::Unauthorized, _)) if offered_pairing().is_some() => {
                    // Refused, with a new pairing offer in hand: pair again.
                    unpair().await;
                    grant.drop_grant();
                    device = paired_device(store).await;
                    continue;
                }
                Err((kind, why)) => {
                    let d = machine.failed(kind, why);
                    let wait = d.retry_in;
                    if kind == DropKind::Unauthorized {
                        grant.drop_grant();
                    }
                    store.update(Update::Disconnected(d));
                    wait_or_wake(&mut machine, wait, &mut wakes).await;
                    continue;
                }
            };
            let (events_tx, mut events) = mpsc::unbounded::<Heard>();
            let on_event =
                Closure::<dyn FnMut(String, String)>::new(move |kind: String, text: String| {
                    let heard = match kind.as_str() {
                        "open" => Heard::Open,
                        "message" => Heard::Message(text),
                        _ => Heard::Close(text),
                    };
                    let _ = events_tx.unbounded_send(heard);
                });
            let socket = Socket::new(&url, &on_event);
            let send = |frame: &aidaemon_client_protocol::ClientFrame| {
                if let Ok(text) = serde_json::to_string(frame) {
                    socket.send(&text);
                }
            };
            let disconnect = loop {
                let mut tick = Box::pin(timer(Duration::from_secs(1)));
                let event = futures::select! {
                    heard = events.next() => Some(heard),
                    _ = wakes.next() => {
                        if let Some(frame) = machine.woke(now()) {
                            send(&frame);
                        }
                        None
                    }
                    command = commands.next() => {
                        if let Some((id, command)) = command
                            && let Some(frame) = machine.command(id, command)
                        {
                            send(&frame);
                        }
                        None
                    }
                    _ = futures::FutureExt::fuse(&mut tick) => {
                        machine.set_seen(page_visible());
                        match machine.tick(now()) {
                            Tick::Ping(frame) => send(&frame),
                            Tick::Dead(d) => break d,
                            Tick::Wait => {}
                        }
                        None
                    }
                };
                match event {
                    Some(Some(Heard::Open)) => {
                        for frame in machine.opened_with_token(now(), Some(token.clone())) {
                            send(&frame);
                        }
                        store.update(Update::Connected);
                    }
                    Some(Some(Heard::Message(text))) => {
                        if let Some(frame) = machine.received(&text, now()) {
                            // Welcomed with the key it has: a pairing link
                            // still in the address has nothing left to do.
                            if matches!(
                                frame,
                                aidaemon_client_protocol::ServerFrame::Welcome { .. }
                            ) && offered_pairing().is_some()
                            {
                                forget_pairing_link();
                            }
                            store.update(Update::Frame(frame));
                        }
                    }
                    Some(heard @ (Some(Heard::Close(_)) | None)) => {
                        let why = match heard {
                            Some(Heard::Close(why)) => why,
                            _ => "the socket ended".to_owned(),
                        };
                        break if machine.is_open() {
                            machine.closed(why)
                        } else {
                            machine.failed(DropKind::DaemonNotRunning, why)
                        };
                    }
                    None => {}
                }
            };
            socket.close();
            drop(on_event);
            if disconnect.kind == DropKind::Unauthorized {
                grant.drop_grant();
                if offered_pairing().is_some() {
                    unpair().await;
                    device = paired_device(store).await;
                    continue;
                }
            }
            let wait = disconnect.retry_in;
            store.update(Update::Disconnected(disconnect));
            wait_or_wake(&mut machine, wait, &mut wakes).await;
        }
    });
}

/// The backoff's wait, cut short when the page wakes.
async fn wait_or_wake(
    machine: &mut LinkMachine,
    wait: Duration,
    wakes: &mut mpsc::UnboundedReceiver<()>,
) {
    let mut wait = Box::pin(futures::FutureExt::fuse(timer(wait)));
    futures::select! {
        _ = wait => {}
        _ = wakes.next() => {
            machine.woke(now());
        }
    }
}

async fn grant_token(
    grant: &mut GrantCache,
    base: &str,
    device: &str,
) -> Result<String, (DropKind, String)> {
    let stored = aidaemon_client_core::StoredDevice {
        gateway: base.to_owned(),
        device_id: device.to_owned(),
        secret_key: String::new(),
    };
    grant
        .token_signed(&PageHttp, &stored, &PageKey, now())
        .await
        .map_err(|error| match error {
            SignInError::Unreachable(why) => (DropKind::DaemonNotRunning, why),
            SignInError::Refused(why) => (DropKind::Unauthorized, why),
            SignInError::Failed(why) => (DropKind::Dropped, why),
        })
}

/// Open a live view's frame socket on the page's own WebSocket (protocol
/// 1.30). The page turns each frame into a `data:` URI and acks it.
pub fn open_live(store: Store, epoch: u64, url: String, ticket: String) {
    use aidaemon_client_core::surfaces::MediaEvent;
    use aidaemon_client_protocol::media::MediaHeader;
    let (events_tx, mut events) = mpsc::unbounded::<(String, String)>();
    let on_event = Closure::<dyn FnMut(String, String)>::new(move |kind: String, text: String| {
        let _ = events_tx.unbounded_send((kind, text));
    });
    let socket = std::rc::Rc::new(MediaSocket::new(&url, &ticket, &on_event));
    // With a mouse and keyboard, the page's keys go to the agent's browser
    // while the owner is in control.
    let keys = (!crate::ui::touch()).then(|| {
        let on_key = Closure::<dyn FnMut(bool, String, String, String) -> bool>::new(
            move |down: bool, code: String, key: String, mods: String| {
                if !store.live_controlling() || code.is_empty() {
                    return false;
                }
                let mods: Vec<&str> = mods
                    .split(',')
                    .filter_map(|m| match m {
                        "shift" => Some("shift"),
                        "ctrl" => Some("ctrl"),
                        "alt" => Some("alt"),
                        "meta" => Some("meta"),
                        _ => None,
                    })
                    .collect();
                store.live_key_event(down, &code, &key, &mods);
                true
            },
        );
        let capture = KeyCapture::new(&on_key);
        (capture, on_key)
    });
    let keep = std::rc::Rc::new(std::cell::RefCell::new(Some((on_event, keys))));
    store.media_attached(
        epoch,
        crate::store::MediaLink {
            send: Box::new({
                let socket = socket.clone();
                move |text: String| socket.send_media(&text)
            }),
            close: Box::new(move || {
                socket.close_media();
                if let Some((_, Some((capture, _)))) = keep.borrow_mut().take() {
                    capture.stop_keys();
                }
            }),
        },
    );
    spawn(async move {
        while let Some((kind, text)) = events.next().await {
            let event = match kind.as_str() {
                "frame" => {
                    let Ok(f) = serde_json::from_str::<serde_json::Value>(&text) else {
                        continue;
                    };
                    let number = |key: &str| f[key].as_u64().unwrap_or(0);
                    MediaEvent::Frame {
                        seq: number("seq"),
                        w: u32::try_from(number("w")).unwrap_or(0),
                        h: u32::try_from(number("h")).unwrap_or(0),
                        uri: f["uri"].as_str().unwrap_or_default().to_owned(),
                    }
                }
                "header" => match serde_json::from_str::<MediaHeader>(&text) {
                    Ok(MediaHeader::Ready { .. }) => MediaEvent::Ready,
                    Ok(MediaHeader::Controller { controller }) => {
                        MediaEvent::Controller(controller)
                    }
                    Ok(MediaHeader::Closed { reason }) => MediaEvent::Closed(reason),
                    Ok(MediaHeader::Focus { what, label, rect }) => MediaEvent::Focus(
                        aidaemon_client_core::surfaces::focus_words(what, label),
                        rect,
                    ),
                    _ => continue,
                },
                _ => MediaEvent::Closed(text),
            };
            let ended = matches!(event, MediaEvent::Closed(_));
            store.media_event(epoch, event);
            if ended {
                return;
            }
        }
    });
}
