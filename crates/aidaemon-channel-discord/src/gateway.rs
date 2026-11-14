//! Discord Gateway v10 (JSON encoding): one connection per shard, optional
//! `zlib-stream` transport compression.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aidaemon_channels::{ChannelError, ChannelEvent};
use futures::{SinkExt, StreamExt};
use reqwest::Method;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_util::sync::CancellationToken;

use crate::DiscordChannel;
use crate::mapping::{THREAD_CHANNEL_TYPES, map_command, map_interaction, map_message};

const OP_DISPATCH: u64 = 0;
const OP_HEARTBEAT: u64 = 1;
const OP_IDENTIFY: u64 = 2;
const OP_RESUME: u64 = 6;
const OP_RECONNECT: u64 = 7;
const OP_INVALID_SESSION: u64 = 9;
const OP_HELLO: u64 = 10;
const OP_HEARTBEAT_ACK: u64 = 11;

const HELLO_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a spent `session_start_limit` is taken to last when refilled.
const SESSION_LIMIT_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);

/// How many connections to open.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ShardMode {
    /// One, identified without a `shard` field (the default).
    Single,
    Fixed(u32),
    /// As many as `GET /gateway/bot` recommends.
    Recommended,
}

/// One shard's session.
#[derive(Debug, Default)]
pub(crate) struct ShardSession {
    session_id: Option<String>,
    resume_url: Option<String>,
    seq: Option<u64>,
}

impl ShardSession {
    fn can_resume(&self) -> bool {
        self.session_id.is_some() && self.resume_url.is_some()
    }

    fn clear(&mut self) {
        self.session_id = None;
        self.resume_url = None;
        self.seq = None;
    }
}

/// What `GET /gateway/bot` says about starting sessions.
#[derive(Debug, Default, Clone, Copy)]
struct SessionStartLimit {
    /// Sessions allowed per window; 0 when not reported.
    total: u32,
    remaining: u32,
    reset_after: Duration,
    max_concurrency: u32,
}

impl SessionStartLimit {
    fn parse(body: &Value) -> Self {
        let limit = body.get("session_start_limit");
        let field = |key: &str| limit.and_then(|l| l.get(key)).and_then(Value::as_u64);
        let number = |key: &str| field(key).and_then(|n| u32::try_from(n).ok()).unwrap_or(0);
        Self {
            total: number("total"),
            remaining: number("remaining"),
            reset_after: Duration::from_millis(field("reset_after").unwrap_or(0)),
            max_concurrency: number("max_concurrency"),
        }
    }
}

/// Paces identifies: per bucket (`shard % max_concurrency`) one per
/// interval, and none past the day's `session_start_limit`.
#[derive(Debug)]
struct IdentifyGate {
    interval: Duration,
    state: Mutex<GateState>,
}

#[derive(Debug)]
struct GateState {
    /// When each bucket may next identify.
    next: Vec<Instant>,
    total: u32,
    remaining: u32,
    reset_at: Instant,
}

impl IdentifyGate {
    fn new(limit: SessionStartLimit, interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            interval,
            state: Mutex::new(GateState {
                next: vec![now; limit.max_concurrency.max(1) as usize],
                total: limit.total,
                remaining: limit.remaining,
                reset_at: now + limit.reset_after,
            }),
        }
    }

    /// Take the next free slot for this shard and say when it is.
    fn reserve(&self, shard: u32) -> Instant {
        let mut state = self.state.lock().unwrap();
        let bucket = shard as usize % state.next.len();
        let mut slot = state.next[bucket].max(Instant::now());
        if state.total > 0 {
            if slot >= state.reset_at {
                state.remaining = state.total;
                state.reset_at = slot + SESSION_LIMIT_WINDOW;
            }
            if state.remaining == 0 {
                slot = state.reset_at;
                tracing::warn!("discord: session start limit reached; waiting for it to reset");
                state.remaining = state.total;
                state.reset_at = slot + SESSION_LIMIT_WINDOW;
            }
            state.remaining -= 1;
        }
        state.next[bucket] = slot + self.interval;
        slot
    }
}

/// The shards one `run` will open.
struct ShardPlan {
    total: u32,
    /// Say which shard this is when identifying.
    announce: bool,
    gate: Option<IdentifyGate>,
}

/// Session state shared between connections and visible to the channel.
#[derive(Debug, Default)]
pub(crate) struct GatewayState {
    pub(crate) bot_user_id: Option<String>,
    /// The bot's username, from READY.
    pub(crate) bot_username: Option<String>,
    /// Register commands in this guild instead of globally.
    pub(crate) command_guild: Option<String>,
    /// The application id, from READY (needed to register commands).
    pub(crate) application_id: Option<String>,
    /// The command menu to (re-)register when the gateway becomes ready.
    pub(crate) commands: Option<Vec<aidaemon_channels::CommandSpec>>,
    /// One per shard, indexed by shard id. Kept across `run`s for resuming.
    pub(crate) shards: Vec<ShardSession>,
    pub(crate) threads: HashSet<String>,
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

pub(crate) fn connect_url(base: &str, zlib: bool) -> String {
    let mut url = if base.contains('?') {
        base.to_string()
    } else {
        format!("{}/?v=10&encoding=json", base.trim_end_matches('/'))
    };
    if zlib && !url.contains("compress=") {
        url.push_str("&compress=zlib-stream");
    }
    url
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

    fn zlib_enabled(&self) -> bool {
        #[cfg(feature = "zlib")]
        {
            self.zlib
        }
        #[cfg(not(feature = "zlib"))]
        {
            false
        }
    }

    /// Decide how many shards to open. Only a sharded bot asks
    /// `GET /gateway/bot` for this, so the default start is unchanged.
    /// `None` when shutdown came first.
    async fn plan_shards(
        &self,
        shutdown: &CancellationToken,
    ) -> Result<Option<ShardPlan>, ChannelError> {
        let fixed = match self.shards {
            ShardMode::Single => {
                return Ok(Some(ShardPlan {
                    total: 1,
                    announce: false,
                    gate: None,
                }));
            }
            ShardMode::Fixed(count) => Some(count),
            ShardMode::Recommended => None,
        };
        let mut delay = self.backoff_initial;
        let body = loop {
            let result = tokio::select! {
                _ = shutdown.cancelled() => return Ok(None),
                r = self.rest.request(Method::GET, "/gateway/bot", None) => r,
            };
            match result {
                Ok(body) => break Some(body),
                Err(ChannelError::Auth(e)) => return Err(ChannelError::Auth(e)),
                // A fixed count can start without the limits, paced as the
                // strictest bot; a recommendation has nothing to go on.
                Err(e) if fixed.is_some() => {
                    tracing::warn!(error = %e, "discord: no session limits; pacing identifies");
                    break None;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "discord: shard lookup failed; retrying");
                    if !sleep_or_shutdown(delay, shutdown).await {
                        return Ok(None);
                    }
                    delay = (delay * 2).min(self.backoff_max);
                }
            }
        };
        let limit = body
            .as_ref()
            .map(SessionStartLimit::parse)
            .unwrap_or_default();
        let recommended = body
            .as_ref()
            .and_then(|b| b.get("shards"))
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok());
        if let Some(url) = body
            .as_ref()
            .and_then(|b| b.get("url"))
            .and_then(Value::as_str)
        {
            let mut known = self.gateway_url.lock().unwrap();
            if known.is_none() {
                *known = Some(url.to_string());
            }
        }
        Ok(Some(ShardPlan {
            total: fixed.or(recommended).unwrap_or(1).max(1),
            announce: true,
            gate: Some(IdentifyGate::new(limit, self.identify_interval)),
        }))
    }

    pub(crate) async fn run_gateway(
        &self,
        events: mpsc::Sender<ChannelEvent>,
        shutdown: CancellationToken,
    ) -> Result<(), ChannelError> {
        let Some(plan) = self.plan_shards(&shutdown).await? else {
            return Ok(());
        };
        tracing::info!(shards = plan.total, "discord: starting gateway");
        {
            let mut st = self.state.lock().unwrap();
            // Sessions belong to one shard layout; a new layout starts over.
            if st.shards.len() != plan.total as usize {
                st.shards = (0..plan.total).map(|_| ShardSession::default()).collect();
            }
        }
        if plan.total == 1 {
            return self.run_shard(0, &plan, &events, &shutdown).await;
        }
        // Any shard ending ends them all: a fatal error, or nobody listening.
        let stop = shutdown.child_token();
        let shards = (0..plan.total).map(|shard| {
            let (plan, events, stop) = (&plan, &events, &stop);
            async move {
                let result = self.run_shard(shard, plan, events, stop).await;
                stop.cancel();
                result
            }
        });
        futures::future::join_all(shards)
            .await
            .into_iter()
            .collect()
    }

    /// One shard's connection, reconnected until it should stop.
    async fn run_shard(
        &self,
        shard: u32,
        plan: &ShardPlan,
        events: &mpsc::Sender<ChannelEvent>,
        shutdown: &CancellationToken,
    ) -> Result<(), ChannelError> {
        let mut backoff = self.backoff_initial;
        loop {
            if shutdown.is_cancelled() || events.is_closed() {
                return Ok(());
            }
            let outcome = self
                .connect_once(shard, plan, events, shutdown, &mut backoff)
                .await;
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
            if !wait.is_zero() && !sleep_or_shutdown(wait, shutdown).await {
                return Ok(());
            }
        }
    }

    async fn connect_once(
        &self,
        shard: u32,
        plan: &ShardPlan,
        events: &mpsc::Sender<ChannelEvent>,
        shutdown: &CancellationToken,
        backoff: &mut Duration,
    ) -> Outcome {
        let (resume, base) = {
            let st = self.state.lock().unwrap();
            let session = &st.shards[shard as usize];
            if session.can_resume() {
                (true, session.resume_url.clone())
            } else {
                (false, None)
            }
        };
        // Before connecting, so the handshake is not left waiting on Hello's
        // heartbeat. A resume costs no identify.
        if let (false, Some(gate)) = (resume, &plan.gate) {
            let slot = gate.reserve(shard);
            tokio::select! {
                _ = shutdown.cancelled() => return Outcome::Shutdown,
                _ = tokio::time::sleep_until(slot) => {}
            }
        }
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
        let url = connect_url(&base, self.zlib_enabled());
        tracing::debug!(%url, resume, shard, "discord: connecting to gateway");

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
        // The zlib context belongs to one connection.
        let mut frames = FrameDecoder::new(self.zlib_enabled());

        // Hello.
        let hello = tokio::select! {
            _ = shutdown.cancelled() => return Outcome::Shutdown,
            r = tokio::time::timeout(HELLO_TIMEOUT, next_json(&mut stream, &mut frames)) => r,
        };
        let interval = match hello {
            Ok(Ok(v)) if v.get("op").and_then(Value::as_u64) == Some(OP_HELLO) => v
                .get("d")
                .and_then(|d| d.get("heartbeat_interval"))
                .and_then(Value::as_u64)
                .unwrap_or(41_250),
            Ok(Err(StreamEnd::Closed(code))) => return self.on_close(shard, code),
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
            let session = &st.shards[shard as usize];
            json!({"op": OP_RESUME, "d": {
                "token": self.rest.token,
                "session_id": session.session_id,
                "seq": session.seq,
            }})
        } else {
            let mut identify = json!({"op": OP_IDENTIFY, "d": {
                "token": self.rest.token,
                "intents": self.intents,
                "properties": {
                    "os": std::env::consts::OS,
                    "browser": "aidaemon",
                    "device": "aidaemon",
                },
            }});
            if plan.announce {
                identify["d"]["shard"] = json!([shard, plan.total]);
            }
            identify
        };
        if sink
            .send(Message::Text(handshake.to_string().into()))
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
                    if sink.send(Message::Text(self.heartbeat_payload(shard).into())).await.is_err() {
                        return Outcome::Reconnect { delay: Delay::Backoff };
                    }
                    hb.as_mut().reset(Instant::now() + interval);
                }
                frame = next_json(&mut stream, &mut frames) => {
                    let payload = match frame {
                        Ok(v) => v,
                        Err(StreamEnd::Closed(code)) => return self.on_close(shard, code),
                    };
                    let op = payload.get("op").and_then(Value::as_u64).unwrap_or(u64::MAX);
                    match op {
                        OP_DISPATCH => {
                            if let Some(s) = payload.get("s").and_then(Value::as_u64) {
                                self.state.lock().unwrap().shards[shard as usize].seq = Some(s);
                            }
                            let t = payload.get("t").and_then(Value::as_str).unwrap_or("");
                            let d = payload.get("d").cloned().unwrap_or(Value::Null);
                            match self.on_dispatch(shard, t, &d, backoff) {
                                Some(mut ev) => {
                                    self.names
                                        .prepare_event(&mut ev, |id| async move {
                                            self.fetch_name(&id).await
                                        })
                                        .await;
                                    if events.send(ev).await.is_err() {
                                        let _ = sink.send(Message::Close(None)).await;
                                        return Outcome::Shutdown;
                                    }
                                }
                                None => continue,
                            }
                        }
                        // The guard sends the heartbeat; a successful send
                        // falls through to the no-op arm.
                        OP_HEARTBEAT
                            if sink.send(Message::Text(self.heartbeat_payload(shard).into())).await.is_err() =>
                        {
                            return Outcome::Reconnect { delay: Delay::Backoff };
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
                                self.state.lock().unwrap().shards[shard as usize].clear();
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

    fn heartbeat_payload(&self, shard: u32) -> String {
        let seq = self.state.lock().unwrap().shards[shard as usize].seq;
        json!({"op": OP_HEARTBEAT, "d": seq}).to_string()
    }

    fn on_close(&self, shard: u32, code: Option<u16>) -> Outcome {
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
                self.state.lock().unwrap().shards[shard as usize].clear();
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

    fn on_dispatch(
        &self,
        shard: u32,
        t: &str,
        d: &Value,
        backoff: &mut Duration,
    ) -> Option<ChannelEvent> {
        match t {
            "READY" => {
                let mut st = self.state.lock().unwrap();
                st.bot_user_id = d
                    .get("user")
                    .and_then(|u| u.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                st.shards[shard as usize].session_id = d
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                st.shards[shard as usize].resume_url = d
                    .get("resume_gateway_url")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                st.bot_username = d
                    .get("user")
                    .and_then(|u| u.get("username"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                st.application_id = d
                    .get("application")
                    .and_then(|a| a.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| st.bot_user_id.clone());
                *backoff = self.backoff_initial;
                tracing::info!(bot = ?st.bot_user_id, shard, "discord: gateway ready");
                // A command menu asked for before the gateway was ready. It
                // belongs to the application, so one shard registers it.
                if let (0, Some(app), Some(commands)) =
                    (shard, st.application_id.clone(), st.commands.clone())
                {
                    let rest = self.rest.clone();
                    let guild = st.command_guild.clone();
                    tokio::spawn(async move {
                        if let Err(e) =
                            crate::put_commands(&rest, &app, guild.as_deref(), &commands).await
                        {
                            tracing::warn!(error = %e, "discord: command registration failed");
                        }
                    });
                }
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
            "INTERACTION_CREATE" if d.get("type").and_then(Value::as_u64) == Some(2) => {
                let command = {
                    let st = self.state.lock().unwrap();
                    map_command(d, &st.threads)
                }?;
                if let (Some(id), Some(token)) = (
                    d.get("id").and_then(Value::as_str),
                    d.get("token").and_then(Value::as_str),
                ) {
                    // Acknowledge at once with a deferred reply (type 5):
                    // Discord fails an invocation not answered within 3 s;
                    // `answer_command` later fills in the reply.
                    let rest = self.rest.clone();
                    let path = format!("/interactions/{id}/{token}/callback");
                    tokio::spawn(async move {
                        if let Err(e) = rest
                            .request(Method::POST, &path, Some(&json!({"type": 5})))
                            .await
                        {
                            tracing::warn!(error = %e, "discord: command ack failed");
                        }
                    });
                }
                Some(ChannelEvent::Command(command))
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

/// Turns binary frames into payloads, when compression is on.
struct FrameDecoder {
    #[cfg(feature = "zlib")]
    zlib: Option<crate::zlib::ZlibStream>,
}

impl FrameDecoder {
    #[cfg_attr(not(feature = "zlib"), allow(unused_variables))]
    fn new(zlib: bool) -> Self {
        Self {
            #[cfg(feature = "zlib")]
            zlib: zlib.then(crate::zlib::ZlibStream::new),
        }
    }

    /// A complete payload if this frame finished one. An error means the
    /// stream cannot be read any further.
    #[cfg_attr(not(feature = "zlib"), allow(unused_variables))]
    fn binary(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, String> {
        #[cfg(feature = "zlib")]
        if let Some(zlib) = &mut self.zlib {
            return zlib.push(frame);
        }
        Ok(None)
    }
}

/// Next JSON frame; skips pings and unparsable frames.
async fn next_json<S>(stream: &mut S, frames: &mut FrameDecoder) -> Result<Value, StreamEnd>
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(t))) => match serde_json::from_str::<Value>(&t) {
                Ok(v) => return Ok(v),
                Err(e) => tracing::debug!(error = %e, "discord: unparsable gateway frame"),
            },
            Some(Ok(Message::Binary(frame))) => match frames.binary(&frame) {
                Ok(None) => {}
                Ok(Some(payload)) => match serde_json::from_slice::<Value>(&payload) {
                    Ok(v) => return Ok(v),
                    Err(e) => tracing::debug!(error = %e, "discord: unparsable gateway frame"),
                },
                Err(e) => {
                    tracing::warn!(error = %e, "discord: compressed stream unreadable");
                    return Err(StreamEnd::Closed(None));
                }
            },
            Some(Ok(Message::Close(frame))) => {
                return Err(StreamEnd::Closed(frame.map(|f| u16::from(f.code))));
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
