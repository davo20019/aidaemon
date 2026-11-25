//! The browser: the shared [`LinkMachine`] over the page's WebSocket, signed
//! in with the device key the page keeps (WebCrypto, non-extractable).
//!
//! Served by the gateway itself, every request is same-origin. On a site of
//! its own (the project's web app), a pairing link names the daemon's
//! `iroh://` address: the page then addresses the daemon at
//! `<site>/_daemon` ([`base`]), and `js/browser.js` carries those requests
//! and sockets over iroh through a relay.

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
    #[wasm_bindgen(js_name = remoteAddress)]
    fn remote_address() -> String;
    #[wasm_bindgen(js_name = keepRemoteGateway)]
    fn keep_remote_gateway(url: &str);
    #[wasm_bindgen(js_name = unconfirmedRemote)]
    fn unconfirmed_remote() -> String;
    #[wasm_bindgen(js_name = irohKey)]
    fn iroh_key(url: &str) -> String;
    #[wasm_bindgen(js_name = switchRemote)]
    fn switch_remote(url: &str) -> js_sys::Promise;
    #[wasm_bindgen(js_name = daemonBase)]
    fn daemon_base() -> String;
    #[wasm_bindgen(js_name = remoteReady)]
    fn remote_ready() -> js_sys::Promise;
    #[wasm_bindgen(js_name = remoteEndpoint)]
    fn remote_endpoint() -> js_sys::Promise;
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
    #[wasm_bindgen(js_name = watchEndPull)]
    fn watch_end_pull_js(test_id: &str, on_pull: &Closure<dyn FnMut(f32, bool)>);

    type MediaSocket;
    #[wasm_bindgen(constructor)]
    fn new(url: &str, ticket: &str, event: &Closure<dyn FnMut(String, String)>) -> MediaSocket;
    #[wasm_bindgen(method, js_name = send)]
    fn send_media(this: &MediaSocket, text: &str);
    #[wasm_bindgen(method, js_name = close)]
    fn close_media(this: &MediaSocket);

    type ShellSocket;
    #[wasm_bindgen(constructor)]
    fn new(url: &str, ticket: &str, event: &Closure<dyn FnMut(String)>) -> ShellSocket;
    #[wasm_bindgen(method, js_name = send)]
    fn send_shell(this: &ShellSocket, text: &str);
    #[wasm_bindgen(method, js_name = close)]
    fn close_shell(this: &ShellSocket);

    type KeyCapture;
    #[wasm_bindgen(constructor)]
    fn new(event: &Closure<dyn FnMut(bool, String, String, String) -> bool>) -> KeyCapture;
    #[wasm_bindgen(method, js_name = stop)]
    fn stop_keys(this: &KeyCapture);

    #[wasm_bindgen(js_name = speechAvailable)]
    fn speech_available_js() -> bool;
    #[wasm_bindgen(js_name = Speaker)]
    type SpeakerJs;
    #[wasm_bindgen(constructor, js_class = "Speaker")]
    fn new(text: &str, lang: &str, ended: &Closure<dyn FnMut()>) -> SpeakerJs;
    #[wasm_bindgen(method, js_class = "Speaker", js_name = stop)]
    fn stop_speaker(this: &SpeakerJs);

    #[wasm_bindgen(js_name = dictationAvailable)]
    fn dictation_available_js() -> bool;
    #[wasm_bindgen(js_name = Dictation)]
    type DictationJs;
    #[wasm_bindgen(constructor, js_class = "Dictation")]
    fn new(lang: &str, interim: bool, event: &Closure<dyn FnMut(String, String)>) -> DictationJs;
    #[wasm_bindgen(method, js_class = "Dictation", js_name = stop)]
    fn stop_dictation(this: &DictationJs);
    #[wasm_bindgen(method, js_class = "Dictation", js_name = abort)]
    fn abort_dictation(this: &DictationJs);

    type Socket;
    #[wasm_bindgen(constructor)]
    fn new(
        url: &str,
        token: &str,
        sign_in: &str,
        event: &Closure<dyn FnMut(String, String)>,
    ) -> Socket;
    #[wasm_bindgen(method)]
    fn send(this: &Socket, text: &str);
    #[wasm_bindgen(method)]
    fn close(this: &Socket);
}

/// A picture taken with the device's camera: its JPEG bytes, scaled to fit
/// `max_dimension`; `None` when nothing was taken.
/// Save a same-origin file under `name` (the browser's download).
/// Whether this page reaches its daemon over iroh (it is on a site of its
/// own and a pairing link, or a past pairing, named the daemon's address).
fn over_iroh() -> bool {
    aidaemon_client_core::machine::transport_of(&remote_address())
        == aidaemon_client_core::machine::Transport::PeerToPeer
}

/// Where the app addresses its daemon: the page's own origin, or over iroh
/// the base the page carries there.
fn base() -> String {
    if over_iroh() { daemon_base() } else { origin() }
}

/// Over iroh, wait for the page's service worker, which carries what the
/// app addresses by URL; `Err` when this browser cannot.
async fn ready() -> Result<(), String> {
    if !over_iroh() {
        return Ok(());
    }
    JsFuture::from(remote_ready())
        .await
        .map(|_| ())
        .map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}")))
}

/// How this browser reaches the daemon: its page's own origin, or a relay
/// (a browser cannot take iroh's direct path).
pub fn transport() -> aidaemon_client_core::machine::Transport {
    if over_iroh() {
        aidaemon_client_core::machine::Transport::Relay
    } else {
        aidaemon_client_core::machine::transport_of(&origin())
    }
}

pub fn download(url: &str, name: &str) {
    download_url_js(url, name);
}

/// A finger pulling the scroll view marked `test_id` past its bottom:
/// how far, and whether it let go (the last report of a pull). The page
/// watches once per mark; the latest receiver hears it, earlier ones end.
pub fn end_pulls(test_id: &'static str) -> mpsc::UnboundedReceiver<(f32, bool)> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static PULLS: RefCell<HashMap<&'static str, mpsc::UnboundedSender<(f32, bool)>>> =
            RefCell::new(HashMap::new());
    }
    let (tx, rx) = mpsc::unbounded();
    let first = PULLS.with(|p| p.borrow_mut().insert(test_id, tx).is_none());
    if first {
        let on_pull = Closure::<dyn FnMut(f32, bool)>::new(move |pull: f32, released: bool| {
            PULLS.with(|p| {
                if let Some(tx) = p.borrow().get(test_id) {
                    let _ = tx.unbounded_send((pull, released));
                }
            });
        });
        watch_end_pull_js(test_id, &on_pull);
        // The page keeps watching for its whole life.
        on_pull.forget();
    }
    rx
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

/// Whether the page can speak text (the Web Speech API's synthesis).
pub fn speech_available() -> bool {
    speech_available_js()
}

/// Text read aloud with the page's voice; `ended` runs once when it is
/// done or stopped. Dropping it stops it.
pub struct Speaker {
    js: SpeakerJs,
    // Called by the page until the reading ends.
    _ended: Closure<dyn FnMut()>,
}

impl Speaker {
    pub fn start(text: &str, ended: impl FnOnce() + 'static) -> Self {
        let mut ended = Some(ended);
        let ended = Closure::<dyn FnMut()>::new(move || {
            if let Some(ended) = ended.take() {
                ended();
            }
        });
        let js = SpeakerJs::new(text, "", &ended);
        Self { js, _ended: ended }
    }

    pub fn stop(&self) {
        self.js.stop_speaker();
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.js.stop_speaker();
    }
}

/// Whether the page can recognize speech (the Web Speech API: Chrome,
/// Edge and Safari have it; Firefox does not).
pub fn dictation_available() -> bool {
    dictation_available_js()
}

/// A dictation session on the page's Web Speech API, reporting as NSH's
/// recognizer does. Dropping it aborts it.
pub struct Dictation {
    js: DictationJs,
    events: mpsc::UnboundedReceiver<nsh_extra::speech::SpeechEvent>,
    // Called by the page until the session ends; kept as long as it can be.
    _event: Closure<dyn FnMut(String, String)>,
}

impl Dictation {
    pub fn start(request: nsh_extra::speech::SpeechRequest) -> Self {
        use nsh_extra::speech::SpeechEvent;
        let (tx, events) = mpsc::unbounded();
        let event = Closure::<dyn FnMut(String, String)>::new(move |kind: String, text: String| {
            // `kind` is one of the names `Dictation` in js/browser.js sends
            // (a machine format of our own, not prose).
            let event = match kind.as_str() {
                "start" => SpeechEvent::Start,
                "result" => SpeechEvent::Result {
                    text,
                    is_final: false,
                },
                "final" => SpeechEvent::Result {
                    text,
                    is_final: true,
                },
                "end" => SpeechEvent::End {
                    error: speech_error(&text),
                },
                _ => return,
            };
            let _ = tx.unbounded_send(event);
        });
        let js = DictationJs::new(
            request.lang.as_deref().unwrap_or(""),
            request.interim,
            &event,
        );
        Self {
            js,
            events,
            _event: event,
        }
    }

    pub fn poll_next(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<nsh_extra::speech::SpeechEvent>> {
        let polled = self.events.poll_next_unpin(cx);
        // Nothing comes after the end.
        if let std::task::Poll::Ready(Some(nsh_extra::speech::SpeechEvent::End { .. })) = &polled {
            self.events.close();
        }
        polled
    }

    pub fn stop(&self) {
        self.js.stop_dictation();
    }
}

impl Drop for Dictation {
    fn drop(&mut self) {
        self.js.abort_dictation();
    }
}

/// A Web Speech API error name as NSH names it; `None` when the session
/// only ended or was aborted. Names are the API's own (a machine format).
fn speech_error(name: &str) -> Option<nsh_extra::speech::SpeechError> {
    use nsh_extra::speech::SpeechError;
    match name {
        "" | "aborted" => None,
        "service-not-allowed" => Some(SpeechError::NotAllowed),
        other => Some(SpeechError::from_name(other).unwrap_or(SpeechError::Failed)),
    }
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

    /// Over iroh the page's requests go to its service-worker base, but
    /// they reach the daemon at `remote_address()`: name that one in a
    /// challenge, or the daemon refuses it as meant for another.
    fn audience(&self, gateway: &str) -> Option<String> {
        if over_iroh() {
            aidaemon_client_protocol::sign_in_audience(&remote_address())
        } else {
            aidaemon_client_protocol::sign_in_audience(gateway)
        }
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

/// A link that names a daemon this browser is not paired with moves it only
/// on the owner's word, shown that daemon's key: a link alone (from a page,
/// a message, a model's reply) must not hand this browser, its messages and
/// the keys typed into it, to another daemon. Until then the page keeps its
/// own daemon and the link offers nothing (`js/browser.js`).
fn confirm_link_daemon() {
    let address = unconfirmed_remote();
    if address.is_empty() {
        return;
    }
    let key = iroh_key(&address);
    crate::ui::confirm(
        "Switch to another AIDaemon?",
        format!(
            "This link points to a different AIDaemon than the one this browser is paired with. \
             Its key is {key}. Switch only if you started this pairing yourself and that key is \
             your AIDaemon's: switching forgets this browser's pairing."
        ),
        "Switch",
        move || {
            spawn(async move {
                let _ = JsFuture::from(switch_remote(&address)).await;
            });
        },
    );
}

/// Pair this browser, then connect.
pub async fn pair(
    store: Store,
    offer: String,
    code: String,
    name: String,
    fingerprint: Option<String>,
) -> Result<(), String> {
    ready().await?;
    // The link's daemon address counts only while this browser has no key
    // (`remoteAddress` in js/browser.js), so it is read before the key is
    // kept: read after, it was lost and every later request went to the
    // site instead of the daemon.
    let remote = over_iroh().then(remote_address);
    let request = PairRequest {
        gateway: base(),
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
    if let Some(remote) = &remote {
        keep_remote_gateway(remote);
    }
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
    /// Over iroh: the daemon refused the token (in the upgrade).
    Refused(String),
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
    confirm_link_daemon();
    // Files' links are on this page's own origin; opening one needs it
    // whole.
    store.set_gateway_base(base());
    spawn(async move {
        let mut device = paired_device(store).await;
        let mut machine = LinkMachine::new(
            ClientIdentity {
                kind: ClientKind::Web,
                name: "aidaemon-app".into(),
                version: Some(crate::link::app_version()),
            },
            None,
            vec![
                Capability::ViewActions,
                Capability::Surfaces,
                Capability::Shells,
            ],
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
        let mut grant = GrantCache::default();
        // Over iroh the socket signs in by itself (one round trip) until
        // the daemon refuses that once; then this page signs in with a
        // challenge, as before.
        let mut by_challenge = false;
        loop {
            // Paired over iroh now, if a pairing link said so.
            let base = base();
            store.set_gateway_base(base.clone());
            let iroh = over_iroh();
            // Over iroh the base has a path, which `socket_url` drops.
            let url = if iroh {
                format!("{}/client/v1/ws", base.replacen("http", "ws", 1))
            } else {
                aidaemon_client_core::machine::socket_url(&base)
            };
            machine.set_seen(page_visible());
            store.update(Update::Connecting);
            if let Err(why) = ready().await {
                let d = machine.failed(DropKind::DaemonNotRunning, why);
                let wait = d.retry_in;
                store.update(Update::Disconnected(d));
                wait_or_wake(&mut machine, wait, &mut wakes).await;
                continue;
            }
            let sign_in = if iroh && !by_challenge {
                socket_sign_in(&device).await
            } else {
                None
            };
            let token = if sign_in.is_some() {
                Ok(String::new())
            } else {
                grant_token(&mut grant, &base, &device).await
            };
            let token = match token {
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
                    // The kinds `Socket` in js/browser.js sends (a machine
                    // format of our own).
                    let heard = match kind.as_str() {
                        "open" => Heard::Open,
                        "message" => Heard::Message(text),
                        "refused" => Heard::Refused(text),
                        _ => Heard::Close(text),
                    };
                    let _ = events_tx.unbounded_send(heard);
                });
            // Over iroh the token (or the sign-in) goes in the upgrade (the
            // daemon takes it only there); the page's own socket sends it
            // in its hello.
            let socket = Socket::new(
                &url,
                if iroh { &token } else { "" },
                sign_in.as_deref().unwrap_or_default(),
                &on_event,
            );
            let mut opened = false;
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
                        opened = true;
                        let opening = if iroh {
                            machine.opened(now())
                        } else {
                            machine.opened_with_token(now(), Some(token.clone()))
                        };
                        for frame in opening {
                            send(&frame);
                        }
                        store.update(Update::Connected);
                    }
                    Some(Some(Heard::Message(text))) => {
                        if let Some(frame) = machine.received(&text, now()) {
                            let welcome = matches!(
                                frame,
                                aidaemon_client_protocol::ServerFrame::Welcome { .. }
                            );
                            // Signed in over iroh: keep the daemon's address,
                            // also when a link found this browser paired
                            // already (a key kept from before) and it never
                            // paired anew.
                            if welcome && iroh {
                                keep_remote_gateway(&remote_address());
                            }
                            // Welcomed with the key it has: a pairing link
                            // still in the address has nothing left to do.
                            if welcome && offered_pairing().is_some() {
                                forget_pairing_link();
                            }
                            store.update(Update::Frame(frame));
                        }
                    }
                    Some(Some(Heard::Refused(why))) => {
                        break machine.failed(DropKind::Unauthorized, why);
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
            // The daemon refused the socket's own sign-in (an older daemon,
            // a clock too far off, an endpoint not yet this device's): sign
            // in with a challenge now, which says what is wrong if anything
            // is. A socket that only failed to connect waits and tries the
            // same way again.
            if sign_in.is_some() && !opened && disconnect.kind == DropKind::Unauthorized {
                by_challenge = true;
                continue;
            }
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

/// The sign-in a socket's upgrade carries over iroh (protocol 1.48), signed
/// with the page's key; `None` when it cannot be made (the iroh module or
/// the key not there), so the page signs in with a challenge.
async fn socket_sign_in(device: &str) -> Option<String> {
    let endpoint = text_of(remote_endpoint()).await.ok()?;
    signin::socket_sign_in(&remote_address(), &endpoint, device, now(), &PageKey)
        .await
        .ok()
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

/// Open a conversation shell's socket on the page's own WebSocket
/// (protocol 1.42): its JSON messages come back as text.
pub fn open_shell(store: Store, epoch: u64, url: String, ticket: String) {
    use aidaemon_client_core::shells::{self, ShellEvent};
    let (events_tx, mut events) = mpsc::unbounded::<String>();
    let on_event = Closure::<dyn FnMut(String)>::new(move |text: String| {
        let _ = events_tx.unbounded_send(text);
    });
    let socket = std::rc::Rc::new(ShellSocket::new(&url, &ticket, &on_event));
    let keep = std::rc::Rc::new(std::cell::RefCell::new(Some(on_event)));
    store.shell_attached(
        epoch,
        crate::store::ShellLink {
            send: Box::new({
                let socket = socket.clone();
                move |text: String| socket.send_shell(&text)
            }),
            close: Box::new(move || {
                socket.close_shell();
                keep.borrow_mut().take();
            }),
        },
    );
    spawn(async move {
        while let Some(text) = events.next().await {
            let Some(event) = shells::read(&text) else {
                continue;
            };
            let ended = matches!(event, ShellEvent::Closed(_));
            store.shell_event(epoch, event);
            if ended {
                return;
            }
        }
    });
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
                    Ok(MediaHeader::Focus {
                        what,
                        label,
                        rect,
                        typing,
                    }) => aidaemon_client_core::surfaces::focus_event(what, label, rect, typing),
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
