//! Discord Gateway v10 (JSON encoding, no compression).

use std::collections::HashSet;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aidaemon_channels::{ChannelError, ChannelEvent};
use futures::{SinkExt, StreamExt};
use reqwest::Method;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

use crate::mapping::{map_interaction, map_message, THREAD_CHANNEL_TYPES};
use crate::DiscordChannel;

const OP_DISPATCH: u64 = 0;
const OP_HEARTBEAT: u64 = 1;
const OP_IDENTIFY: u64 = 2;
const OP_RESUME: u64 = 6;
const OP_RECONNECT: u64 = 7;
const OP_INVALID_SESSION: u64 = 9;
const OP_HELLO: u64 = 10;
const OP_HEARTBEAT_ACK: u64 = 11;

const HELLO_TIMEOUT: Duration = Duration::from_secs(30);

/// Session state shared between connections and visible to the channel.
#[derive(Debug, Default)]
pub(crate) struct GatewayState {
    pub(crate) bot_user_id: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) resume_url: Option<String>,
    pub(crate) seq: Option<u64>,
    pub(crate) threads: HashSet<String>,
}

impl GatewayState {
    fn can_resume(&self) -> bool {
        self.session_id.is_some() && self.resume_url.is_some()
    }

    fn clear_session(&mut self) {
        self.session_id = None;
        self.resume_url = None;
        self.seq = None;
    }
}

/// Why a connection ended.
enum Outcome {
    Shutdown,
    /// Reconnect; `delay` selects the wait before it.
    Reconnect {
        delay: Delay,
    },
    Fatal(ChannelError),
}

enum Delay {
    None,
    Backoff,
    InvalidSession,
}

/// A pseudo-random fraction in [0, 1); good enough for jitter.
fn jitter() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let mut x = u64::from(nanos) ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x % 10_000) as f64 / 10_000.0
}

pub(crate) fn connect_url(base: &str) -> String {
    if base.contains('?') {
        base.to_string()
    } else {
        format!("{}/?v=10&encoding=json", base.trim_end_matches('/'))
    }
}

/// Sleep unless shutdown fires first; returns false on shutdown.
async fn sleep_or_shutdown(d: Duration, shutdown: &CancellationToken) -> bool {
    tokio::select! {
        _ = shutdown.cancelled() => false,
        _ = tokio::time::sleep(d) => true,
    }
}

impl DiscordChannel {
    async fn gateway_base(&self) -> Result<String, ChannelError> {
        if let Some(url) = self.gateway_url.lock().unwrap().clone() {
            return Ok(url);
        }
        let body = self.rest.request(Method::GET, "/gateway/bot", None).await?;
        let url = body
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| ChannelError::Other("GET /gateway/bot returned no url".into()))?
            .to_string();
        *self.gateway_url.lock().unwrap() = Some(url.clone());
        Ok(url)
    }

    pub(crate) async fn run_gateway(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        let mut backoff = self.backoff_initial;
        loop {
            if shutdown.is_cancelled() || events.is_closed() {
                return Ok(());
            }
            let outcome = self.connect_once(&events, &shutdown, &mut backoff).await;
            let delay = match outcome {
                Outcome::Shutdown => return Ok(()),
                Outcome::Fatal(e) => return Err(e),
                Outcome::Reconnect { delay } => delay,
            };
            let wait = match delay {
                Delay::None => Duration::ZERO,
                Delay::Backoff => {
                    let d = backoff.mul_f64(1.0 + jitter() * 0.25);
                    backoff = (backoff * 2).min(self.backoff_max);
                    d
                }
                // Discord asks for a random 1-5 s wait (scaled to the
                // configured initial backoff, which defaults to 1 s).
                Delay::InvalidSession => self.backoff_initial.mul_f64(1.0 + jitter() * 4.0),
            };
            if !wait.is_zero() && !sleep_or_shutdown(wait, &shutdown).await {
                return Ok(());
            }
        }
    }

    async fn connect_once(
        &self,
        events: &mpsc::Sender<ChannelEvent>,
        shutdown: &CancellationToken,
        backoff: &mut Duration,
    ) -> Outcome {
        let (resume, base) = {
            let st = self.state.lock().unwrap();
            if st.can_resume() {
                (true, st.resume_url.clone())
            } else {
                (false, None)
            }
        };
        let base = match base {
            Some(b) => b,
            None => match self.gateway_base().await {
                Ok(b) => b,
                Err(ChannelError::Auth(e)) => return Outcome::Fatal(ChannelError::Auth(e)),
                Err(e) => {
                    tracing::warn!(error = %e, "discord: gateway url lookup failed");
                    return Outcome::Reconnect {
                        delay: Delay::Backoff,
                    };
                }
            },
        };
        let url = connect_url(&base);
        tracing::debug!(%url, resume, "discord: connecting to gateway");

        let connect = tokio::select! {
            _ = shutdown.cancelled() => return Outcome::Shutdown,
            r = tokio_tungstenite::connect_async(url.as_str()) => r,
        };
        let ws = match connect {
            Ok((ws, _)) => ws,
            Err(e) => {
                tracing::warn!(error = %e, "discord: gateway connect failed");
                return Outcome::Reconnect {
                    delay: Delay::Backoff,
                };
            }
        };
        let (mut sink, mut stream) = ws.split();

        // Hello.
        let hello = tokio::select! {
            _ = shutdown.cancelled() => return Outcome::Shutdown,
            r = tokio::time::timeout(HELLO_TIMEOUT, next_json(&mut stream)) => r,
        };
        let interval = match hello {
            Ok(Ok(v)) if v.get("op").and_then(Value::as_u64) == Some(OP_HELLO) => v
                .get("d")
                .and_then(|d| d.get("heartbeat_interval"))
                .and_then(Value::as_u64)
                .unwrap_or(41_250),
            Ok(Err(StreamEnd::Closed(code))) => return self.on_close(code),
            other => {
                tracing::warn!(ok = other.is_ok(), "discord: no Hello from gateway");
                return Outcome::Reconnect {
                    delay: Delay::Backoff,
                };
            }
        };
        let interval = Duration::from_millis(interval.max(1));

        let handshake = if resume {
            let st = self.state.lock().unwrap();
            json!({"op": OP_RESUME, "d": {
                "token": self.rest.token,
                "session_id": st.session_id,
                "seq": st.seq,
            }})
        } else {
            json!({"op": OP_IDENTIFY, "d": {
                "token": self.rest.token,
                "intents": self.intents,
                "properties": {
                    "os": std::env::consts::OS,
                    "browser": "aidaemon",
                    "device": "aidaemon",
                },
            }})
        };
        if sink
            .send(Message::Text(handshake.to_string()))
            .await
            .is_err()
        {
            return Outcome::Reconnect {
                delay: Delay::Backoff,
            };
        }

        let mut acked = true;
        let hb = tokio::time::sleep_until(Instant::now() + interval.mul_f64(jitter()));
        tokio::pin!(hb);

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    let _ = sink.send(Message::Close(Some(CloseFrame {
                        code: CloseCode::Normal,
                        reason: "shutdown".into(),
                    }))).await;
                    return Outcome::Shutdown;
                }
                _ = &mut hb => {
                    if !acked {
                        tracing::warn!("discord: heartbeat not acknowledged; reconnecting");
                        let _ = sink.send(Message::Close(Some(CloseFrame {
                            code: CloseCode::Library(4000),
                            reason: "zombied".into(),
                        }))).await;
                        return Outcome::Reconnect { delay: Delay::None };
                    }
                    acked = false;
                    if sink.send(Message::Text(self.heartbeat_payload())).await.is_err() {
                        return Outcome::Reconnect { delay: Delay::Backoff };
                    }
                    hb.as_mut().reset(Instant::now() + interval);
                }
                frame = next_json(&mut stream) => {
                    let payload = match frame {
                        Ok(v) => v,
                        Err(StreamEnd::Closed(code)) => return self.on_close(code),
                    };
                    let op = payload.get("op").and_then(Value::as_u64).unwrap_or(u64::MAX);
                    match op {
                        OP_DISPATCH => {
                            if let Some(s) = payload.get("s").and_then(Value::as_u64) {
                                self.state.lock().unwrap().seq = Some(s);
                            }
                            let t = payload.get("t").and_then(Value::as_str).unwrap_or("");
                            let d = payload.get("d").cloned().unwrap_or(Value::Null);
                            match self.on_dispatch(t, &d, backoff) {
                                Some(ev) => {
                                    if events.send(ev).await.is_err() {
                                        let _ = sink.send(Message::Close(None)).await;
                                        return Outcome::Shutdown;
                                    }
                                }
                                None => continue,
                            }
                        }
                        OP_HEARTBEAT => {
                            if sink.send(Message::Text(self.heartbeat_payload())).await.is_err() {
                                return Outcome::Reconnect { delay: Delay::Backoff };
                            }
                        }
                        OP_HEARTBEAT_ACK => acked = true,
                        OP_RECONNECT => {
                            tracing::debug!("discord: gateway requested reconnect");
                            let _ = sink.send(Message::Close(Some(CloseFrame {
                                code: CloseCode::Library(4000),
                                reason: "reconnect".into(),
                            }))).await;
                            return Outcome::Reconnect { delay: Delay::None };
                        }
                        OP_INVALID_SESSION => {
                            let resumable = payload.get("d").and_then(Value::as_bool).unwrap_or(false);
                            tracing::debug!(resumable, "discord: invalid session");
                            if !resumable {
                                self.state.lock().unwrap().clear_session();
                            }
                            let _ = sink.send(Message::Close(Some(CloseFrame {
                                code: CloseCode::Library(4000),
                                reason: "invalid session".into(),
                            }))).await;
                            return Outcome::Reconnect { delay: Delay::InvalidSession };
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn heartbeat_payload(&self) -> String {
        let seq = self.state.lock().unwrap().seq;
        json!({"op": OP_HEARTBEAT, "d": seq}).to_string()
    }

    fn on_close(&self, code: Option<u16>) -> Outcome {
        match code {
            Some(4004) => Outcome::Fatal(ChannelError::Auth(
                "discord gateway closed with 4004 (authentication failed)".into(),
            )),
            Some(c @ 4010..=4014) => {
                let why = match c {
                    4010 => "invalid shard",
                    4011 => "sharding required",
                    4012 => "invalid API version",
                    4013 => "invalid intents",
                    _ => "disallowed intents (enable them in the developer portal)",
                };
                Outcome::Fatal(ChannelError::Other(format!(
                    "discord gateway closed with {c} ({why})"
                )))
            }
            Some(c @ (4007 | 4009)) => {
                tracing::debug!(code = c, "discord: session expired; re-identifying");
                self.state.lock().unwrap().clear_session();
                Outcome::Reconnect {
                    delay: Delay::Backoff,
                }
            }
            other => {
                tracing::debug!(code = ?other, "discord: gateway closed; reconnecting");
                Outcome::Reconnect {
                    delay: Delay::Backoff,
                }
            }
        }
    }

    fn on_dispatch(&self, t: &str, d: &Value, backoff: &mut Duration) -> Option<ChannelEvent> {
        match t {
            "READY" => {
                let mut st = self.state.lock().unwrap();
                st.bot_user_id = d
                    .get("user")
                    .and_then(|u| u.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                st.session_id = d
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                st.resume_url = d
                    .get("resume_gateway_url")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                *backoff = self.backoff_initial;
                tracing::info!(bot = ?st.bot_user_id, "discord: gateway ready");
                None
            }
            "RESUMED" => {
                *backoff = self.backoff_initial;
                None
            }
            "MESSAGE_CREATE" => {
                let st = self.state.lock().unwrap();
                map_message(d, st.bot_user_id.as_deref(), &st.threads).map(ChannelEvent::Message)
            }
            "INTERACTION_CREATE" => {
                let answer = {
                    let st = self.state.lock().unwrap();
                    map_interaction(d, &st.threads)
                }?;
                if let (Some(id), Some(token)) = (
                    d.get("id").and_then(Value::as_str),
                    d.get("token").and_then(Value::as_str),
                ) {
                    // Acknowledge with a deferred update (type 6) so the
                    // button does not show "interaction failed". Spawned so
                    // a slow REST call never stalls heartbeats.
                    let rest = self.rest.clone();
                    let path = format!("/interactions/{id}/{token}/callback");
                    tokio::spawn(async move {
                        if let Err(e) = rest
                            .request(Method::POST, &path, Some(&json!({"type": 6})))
                            .await
                        {
                            tracing::warn!(error = %e, "discord: interaction ack failed");
                        }
                    });
                }
                Some(ChannelEvent::Approval(answer))
            }
            "THREAD_CREATE" | "THREAD_UPDATE" => {
                self.note_thread(d);
                None
            }
            "THREAD_DELETE" => {
                if let Some(id) = d.get("id").and_then(Value::as_str) {
                    self.state.lock().unwrap().threads.remove(id);
                }
                None
            }
            "GUILD_CREATE" | "THREAD_LIST_SYNC" => {
                if let Some(threads) = d.get("threads").and_then(Value::as_array) {
                    for th in threads {
                        self.note_thread(th);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn note_thread(&self, channel: &Value) {
        let is_thread = channel
            .get("type")
            .and_then(Value::as_u64)
            .is_some_and(|t| THREAD_CHANNEL_TYPES.contains(&t));
        if let (true, Some(id)) = (is_thread, channel.get("id").and_then(Value::as_str)) {
            self.state.lock().unwrap().threads.insert(id.to_string());
        }
    }
}

enum StreamEnd {
    Closed(Option<u16>),
}

/// Next JSON text frame; skips pings and unparsable frames.
async fn next_json<S>(stream: &mut S) -> Result<Value, StreamEnd>
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(t))) => match serde_json::from_str::<Value>(&t) {
                Ok(v) => return Ok(v),
                Err(e) => tracing::debug!(error = %e, "discord: unparsable gateway frame"),
            },
            Some(Ok(Message::Close(frame))) => {
                return Err(StreamEnd::Closed(frame.map(|f| u16::from(f.code))))
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => {
                tracing::debug!(error = %e, "discord: gateway read error");
                return Err(StreamEnd::Closed(None));
            }
            None => return Err(StreamEnd::Closed(None)),
        }
    }
}
