//! The web app's iroh connection, for a page on a site of its own (the
//! project's web app): it reaches the daemon by key through a relay, since a
//! browser cannot open iroh's direct path. `js/browser.js` loads this module
//! only then, so a page the daemon serves itself downloads none of it.
//!
//! [`Daemon`] is one daemon dialed with this browser's endpoint key: plain
//! requests ([`Daemon::fetch`], which the page's service worker uses for
//! everything the app addresses by URL) and WebSockets ([`Daemon::socket`]).

use std::fmt::Display;

use aidaemon_client_core::remote::{self, Dialer, Message, RemoteUrl};
use aidaemon_client_core::signin::SignInError;
use bytes::Bytes;
use futures::channel::mpsc;
use futures::{SinkExt as _, StreamExt as _};
use http_body_util::Full;
use wasm_bindgen::prelude::*;

fn error(why: impl Display) -> JsValue {
    js_sys::Error::new(&why.to_string()).into()
}

/// 32 bytes from 64 hex digits.
fn key_bytes(hex: &str) -> Option<[u8; 32]> {
    let mut bytes = [0u8; 32];
    if hex.len() != 64 {
        return None;
    }
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(bytes)
}

/// Header pairs from a flat `[name, value, name, value, …]` list.
fn header_map(flat: &[String]) -> Result<http::HeaderMap, JsValue> {
    let mut headers = http::HeaderMap::new();
    for pair in flat.chunks(2) {
        let [name, value] = pair else {
            return Err(error("headers come in name, value pairs"));
        };
        headers.append(
            http::HeaderName::try_from(name.as_str()).map_err(error)?,
            http::HeaderValue::try_from(value.as_str()).map_err(error)?,
        );
    }
    Ok(headers)
}

/// The daemon at an `iroh://` address, dialed with this browser's endpoint
/// key (kept by the page, apart from its signing key).
#[wasm_bindgen]
pub struct Daemon {
    url: String,
    dialer: Dialer,
}

#[wasm_bindgen]
impl Daemon {
    /// `url`: the daemon's `iroh://<key>@<relay>` address; `secret`: this
    /// browser's endpoint key, 64 hex digits.
    #[wasm_bindgen(constructor)]
    pub fn new(url: String, secret: String) -> Result<Daemon, JsValue> {
        RemoteUrl::parse(&url).ok_or_else(|| error(format!("not a daemon address: {url}")))?;
        let secret = key_bytes(&secret).ok_or_else(|| error("the endpoint key is damaged"))?;
        Ok(Self {
            url,
            dialer: Dialer::with_key(secret),
        })
    }

    /// The endpoint key (hex) this browser dials with.
    pub fn endpoint(&self) -> String {
        self.dialer.endpoint()
    }

    /// Reach the daemon now (the relay, then the connection), ahead of the
    /// app's first request, which then takes this connection. Rejects when
    /// the daemon cannot be reached; the request reports that itself.
    pub fn dial(&self) -> js_sys::Promise {
        let (url, dialer) = (self.url.clone(), self.dialer.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            remote::dial(&url, &dialer).await.map_err(error)?;
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Start over: close this browser's connection to the daemon and its
    /// relay link, failing a dial still in flight at once, so the next
    /// request or socket dials afresh. For a link that dropped, or a page
    /// that woke or came back online while still dialing.
    pub fn reset(&self) -> js_sys::Promise {
        let (url, dialer) = (self.url.clone(), self.dialer.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            remote::reset(&url, &dialer).await.map_err(error)?;
            Ok(JsValue::UNDEFINED)
        })
    }

    /// One request to the daemon: `path` with its query, `headers` as flat
    /// pairs. Resolves to a [`Reply`], whatever its status; rejects when the
    /// daemon cannot be reached.
    pub fn fetch(
        &self,
        method: String,
        path: String,
        headers: Vec<String>,
        body: Vec<u8>,
    ) -> js_sys::Promise {
        let (url, dialer) = (self.url.clone(), self.dialer.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let mut request = http::Request::builder()
                .method(http::Method::from_bytes(method.as_bytes()).map_err(error)?)
                .uri(path)
                .body(Full::new(Bytes::from(body)))
                .map_err(error)?;
            *request.headers_mut() = header_map(&headers)?;
            let response = remote::send(&url, &dialer, request).await.map_err(error)?;
            let headers = response
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    Some([name.as_str().to_owned(), value.to_str().ok()?.to_owned()])
                })
                .flatten()
                .collect();
            Ok(Reply {
                status: response.status().as_u16(),
                headers,
                body: response.into_body().to_vec(),
            }
            .into())
        })
    }

    /// A WebSocket at `path` with `headers` (flat pairs). `event(kind, data)`
    /// hears `"open"`, `"text"` (a string), `"binary"` (a `Uint8Array`),
    /// `"close"` (why) and `"refused"` (the daemon refused the token).
    /// Frames sent before it opens wait for it.
    pub fn socket(&self, path: String, headers: Vec<String>, event: js_sys::Function) -> Socket {
        let (url, dialer) = (
            format!("{}{path}", self.url.trim_end_matches('/')),
            self.dialer.clone(),
        );
        let (out, mut frames) = mpsc::unbounded::<Message>();
        let say = move |kind: &str, data: JsValue| {
            let _ = event.call2(&JsValue::NULL, &JsValue::from_str(kind), &data);
        };
        wasm_bindgen_futures::spawn_local(async move {
            let opened = match header_map(&headers) {
                Ok(headers) => remote::websocket_with(&url, &headers, &dialer).await,
                Err(why) => Err(SignInError::Failed(
                    why.as_string().unwrap_or_else(|| "bad headers".into()),
                )),
            };
            let socket = match opened {
                Ok(socket) => socket,
                Err(SignInError::Refused(why)) => return say("refused", JsValue::from_str(&why)),
                Err(why) => return say("close", JsValue::from_str(&why.to_string())),
            };
            say("open", JsValue::NULL);
            let (mut sink, stream) = socket.split();
            let mut stream = stream.fuse();
            loop {
                futures::select! {
                    frame = frames.next() => match frame {
                        Some(frame) => {
                            if sink.send(frame).await.is_err() {
                                break;
                            }
                        }
                        None => {
                            let _ = sink.close().await;
                            return;
                        }
                    },
                    message = stream.next() => match message {
                        Some(Ok(Message::Text(text))) => say("text", JsValue::from_str(&text)),
                        Some(Ok(Message::Binary(bytes))) => {
                            say("binary", js_sys::Uint8Array::from(bytes.as_ref()).into());
                        }
                        Some(Ok(Message::Close(frame))) => {
                            let why = frame.map_or_else(String::new, |f| f.reason.to_string());
                            return say("close", JsValue::from_str(&why));
                        }
                        Some(Ok(_)) => {}
                        Some(Err(why)) => return say("close", JsValue::from_str(&why.to_string())),
                        None => break,
                    },
                }
            }
            say("close", JsValue::from_str(""));
        });
        Socket { out }
    }
}

/// A daemon's reply to [`Daemon::fetch`].
#[wasm_bindgen]
pub struct Reply {
    status: u16,
    headers: Vec<String>,
    body: Vec<u8>,
}

#[wasm_bindgen]
impl Reply {
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Flat `[name, value, …]` pairs.
    #[wasm_bindgen(getter)]
    pub fn headers(&self) -> Vec<String> {
        self.headers.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn body(&self) -> Vec<u8> {
        self.body.clone()
    }
}

/// A WebSocket over iroh ([`Daemon::socket`]).
#[wasm_bindgen]
pub struct Socket {
    out: mpsc::UnboundedSender<Message>,
}

#[wasm_bindgen]
impl Socket {
    pub fn send(&self, text: String) {
        let _ = self.out.unbounded_send(Message::text(text));
    }

    pub fn close(&self) {
        self.out.close_channel();
    }
}
