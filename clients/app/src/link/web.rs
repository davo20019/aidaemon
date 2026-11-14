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

    type Socket;
    #[wasm_bindgen(constructor)]
    fn new(url: &str, event: &Closure<dyn FnMut(String, String)>) -> Socket;
    #[wasm_bindgen(method)]
    fn send(this: &Socket, text: &str);
    #[wasm_bindgen(method)]
    fn close(this: &Socket);
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
    spawn(async move {
        let mut device = paired_device(store).await;
        let mut machine = LinkMachine::new(
            ClientIdentity {
                kind: ClientKind::Web,
                name: "aidaemon-app".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            },
            None,
            vec![Capability::ViewActions],
        );
        let (commands_tx, mut commands) = mpsc::unbounded::<(String, Command)>();
        store.attach(Box::new(move |id, command| {
            let _ = commands_tx.unbounded_send((id, command));
        }));
        let base = origin();
        let url = aidaemon_client_core::machine::socket_url(&base);
        let mut grant = GrantCache::default();
        loop {
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
                    timer(wait).await;
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
                    command = commands.next() => {
                        if let Some((id, command)) = command
                            && let Some(frame) = machine.command(id, command)
                        {
                            send(&frame);
                        }
                        None
                    }
                    _ = futures::FutureExt::fuse(&mut tick) => {
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
            timer(wait).await;
        }
    });
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
