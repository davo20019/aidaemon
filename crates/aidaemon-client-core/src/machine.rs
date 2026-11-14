//! The link's rules with no I/O: what to send when a socket opens, what a
//! frame changes, when to ping, when silence means the socket is dead, and
//! how long to wait before the next try. A transport (tokio's in
//! [`crate::link`], or a UI framework's own socket and timers) feeds it what
//! happened and does what it says, so every client keeps the same contract:
//! resume from the last durable `seq`, resend unanswered commands under
//! their own `request_id`, and tell "the daemon is not running" from "this
//! device was refused".
//!
//! Time is the caller's monotonic clock in milliseconds, so the machine runs
//! the same on a desktop, in a browser and in a test.

use std::collections::BTreeMap;
use std::time::Duration;

use aidaemon_client_protocol::{
    Capability, ClientFrame, ClientInfo, ClientKind, Command, ErrorCode, ProtocolVersion,
    ServerFrame,
};

/// How often a quiet socket is pinged.
pub const PING_EVERY: Duration = Duration::from_secs(25);
/// Two missed pongs: the socket is treated as dead.
pub const SILENT_LIMIT: Duration = Duration::from_secs(60);
/// The first wait before trying again.
pub const RETRY_MIN: Duration = Duration::from_millis(500);
/// The longest wait between tries.
pub const RETRY_MAX: Duration = Duration::from_secs(10);

/// What the client says it is in `hello`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientIdentity {
    pub kind: ClientKind,
    /// The program's name (`aidaemon-tui`).
    pub name: String,
    pub version: Option<String>,
}

/// Why a connection ended, as far as a client should tell the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropKind {
    /// Nothing answered at the address: the daemon (or its gateway) is not
    /// running.
    DaemonNotRunning,
    /// The token or device was refused; retrying alone will not fix it.
    Unauthorized,
    /// The daemon speaks another major protocol version.
    VersionUnsupported,
    /// The connection was lost; it comes back by itself.
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnect {
    pub kind: DropKind,
    pub reason: String,
    /// When the link tries again.
    pub retry_in: Duration,
}

/// What the link tells the app.
#[allow(clippy::large_enum_variant)] // Frames dominate; boxing buys nothing here.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    /// Signing in and opening the socket.
    Connecting,
    /// The socket is open and `hello` is sent.
    Connected,
    Disconnected(Disconnect),
    Frame(ServerFrame),
}

/// What a timer tick asks the transport to do.
#[derive(Debug, Clone, PartialEq)]
pub enum Tick {
    /// Nothing yet.
    Wait,
    /// Send this ping.
    Ping(ClientFrame),
    /// The gateway has been silent too long: close the socket and report
    /// this.
    Dead(Disconnect),
}

/// The link's state between and across connections.
#[derive(Debug, Clone)]
pub struct LinkMachine {
    client: ClientIdentity,
    device: Option<String>,
    capabilities: Vec<Capability>,
    /// The last durable `seq` delivered to the app.
    cursor: Option<u64>,
    /// Commands sent but not answered, by request id.
    pending: BTreeMap<String, Command>,
    /// The gateway refused the token on this socket.
    refused: Option<String>,
    /// The socket is open and `hello` was sent.
    open: bool,
    last_heard_ms: u64,
    last_ping_ms: u64,
    /// The wait before the next try.
    wait: Duration,
}

impl LinkMachine {
    pub fn new(
        client: ClientIdentity,
        device: Option<String>,
        capabilities: Vec<Capability>,
    ) -> Self {
        Self {
            client,
            device,
            capabilities,
            cursor: None,
            pending: BTreeMap::new(),
            refused: None,
            open: false,
            last_heard_ms: 0,
            last_ping_ms: 0,
            wait: RETRY_MIN,
        }
    }

    /// The last durable `seq` delivered.
    pub fn cursor(&self) -> Option<u64> {
        self.cursor
    }

    /// Whether a socket is open now.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Commands sent and not yet answered.
    pub fn pending(&self) -> impl Iterator<Item = (&String, &Command)> {
        self.pending.iter()
    }

    /// The socket opened at `now_ms`: send these frames, in order (`hello`,
    /// then every unanswered command again).
    pub fn opened(&mut self, now_ms: u64) -> Vec<ClientFrame> {
        self.opened_with_token(now_ms, None)
    }

    /// [`LinkMachine::opened`] for a socket that could not carry the token
    /// in a header (a browser's): `hello` carries it.
    pub fn opened_with_token(&mut self, now_ms: u64, token: Option<String>) -> Vec<ClientFrame> {
        self.open = true;
        self.refused = None;
        self.last_heard_ms = now_ms;
        self.last_ping_ms = now_ms;
        let mut frames = vec![ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: self.client.kind,
                name: self.client.name.clone(),
                version: self.client.version.clone(),
                device: self.device.clone(),
            },
            capabilities: self.capabilities.clone(),
            resume_from: self.cursor,
            token,
        }];
        frames.extend(
            self.pending
                .iter()
                .map(|(request_id, command)| ClientFrame::Command {
                    request_id: request_id.clone(),
                    command: command.clone(),
                }),
        );
        frames
    }

    /// The app sends `command`. It is kept until answered; the frame to send
    /// now comes back while a socket is open (otherwise `opened` sends it).
    pub fn command(&mut self, request_id: String, command: Command) -> Option<ClientFrame> {
        self.pending.insert(request_id.clone(), command.clone());
        self.open.then_some(ClientFrame::Command {
            request_id,
            command,
        })
    }

    /// A text message arrived at `now_ms`: the frame for the app, if it is
    /// one.
    pub fn received(&mut self, text: &str, now_ms: u64) -> Option<ServerFrame> {
        self.last_heard_ms = now_ms;
        let frame = serde_json::from_str::<ServerFrame>(text).ok()?;
        self.note(&frame);
        Some(frame)
    }

    /// Anything else arrived (a pong, a binary message): the socket is
    /// alive.
    pub fn heard(&mut self, now_ms: u64) {
        self.last_heard_ms = now_ms;
    }

    fn note(&mut self, frame: &ServerFrame) {
        match frame {
            ServerFrame::Event { seq: Some(seq), .. } | ServerFrame::Snapshot { seq, .. } => {
                self.cursor = Some(*seq);
            }
            ServerFrame::Ack { request_id, .. }
            | ServerFrame::Error {
                request_id: Some(request_id),
                ..
            } => {
                self.pending.remove(request_id);
            }
            ServerFrame::Error {
                request_id: None,
                error,
            } if matches!(
                error.code,
                ErrorCode::Unauthorized | ErrorCode::VersionUnsupported
            ) =>
            {
                self.refused = Some(error.message.clone());
            }
            _ => {}
        }
    }

    /// A timer fired at `now_ms`, with a socket open.
    pub fn tick(&mut self, now_ms: u64) -> Tick {
        if !self.open {
            return Tick::Wait;
        }
        if now_ms.saturating_sub(self.last_heard_ms) > millis(SILENT_LIMIT) {
            return Tick::Dead(self.closed("no answer from the gateway"));
        }
        if now_ms.saturating_sub(self.last_ping_ms) >= millis(PING_EVERY) {
            self.last_ping_ms = now_ms;
            return Tick::Ping(ClientFrame::Ping);
        }
        Tick::Wait
    }

    /// The open socket closed (or failed) for `reason`. A connection that
    /// worked starts the backoff over.
    pub fn closed(&mut self, reason: impl Into<String>) -> Disconnect {
        self.open = false;
        let (kind, reason) = match self.refused.take() {
            Some(why) => (DropKind::Unauthorized, why),
            None => (DropKind::Dropped, reason.into()),
        };
        if kind == DropKind::Dropped {
            self.wait = RETRY_MIN;
        }
        self.disconnect(kind, reason)
    }

    /// Signing in or opening the socket failed: the backoff keeps growing.
    pub fn failed(&mut self, kind: DropKind, reason: impl Into<String>) -> Disconnect {
        self.open = false;
        self.disconnect(kind, reason.into())
    }

    fn disconnect(&mut self, kind: DropKind, reason: String) -> Disconnect {
        let retry_in = self.wait;
        self.wait = (self.wait * 2).min(RETRY_MAX);
        Disconnect {
            kind,
            reason,
            retry_in,
        }
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// `ws://host:port/client/v1/ws` as `http://host:port`.
pub fn http_base(url: &str) -> String {
    let (scheme, rest) = match url.split_once("://") {
        Some(("wss", rest)) => ("https", rest),
        Some(("ws", rest)) => ("http", rest),
        Some((other, rest)) => (other, rest),
        None => ("http", url),
    };
    let host = rest.split('/').next().unwrap_or(rest);
    format!("{scheme}://{host}")
}

/// `http://host:port` (or a socket address) as the gateway's socket URL.
pub fn socket_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    let (scheme, rest) = match base.split_once("://") {
        Some(("https", rest)) => ("wss", rest),
        Some(("http", rest)) => ("ws", rest),
        Some((other, rest)) => (other, rest),
        None => ("ws", base),
    };
    let host = rest.split('/').next().unwrap_or(rest);
    format!("{scheme}://{host}/client/v1/ws")
}

/// A new request id.
pub fn request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_client_protocol::ProtocolError;

    fn machine() -> LinkMachine {
        LinkMachine::new(
            ClientIdentity {
                kind: ClientKind::Desktop,
                name: "test".into(),
                version: None,
            },
            None,
            Vec::new(),
        )
    }

    fn stop(session: &str) -> Command {
        Command::StopTurn {
            session_id: session.into(),
        }
    }

    #[test]
    fn the_url_forms_follow_each_other() {
        assert_eq!(
            http_base("ws://127.0.0.1:8788/client/v1/ws"),
            "http://127.0.0.1:8788"
        );
        assert_eq!(
            http_base("wss://daemon.example.invalid/client/v1/ws"),
            "https://daemon.example.invalid"
        );
        assert_eq!(
            socket_url("http://127.0.0.1:8788/"),
            "ws://127.0.0.1:8788/client/v1/ws"
        );
        assert_eq!(
            socket_url("https://daemon.example.invalid"),
            "wss://daemon.example.invalid/client/v1/ws"
        );
    }

    #[test]
    fn a_command_waits_for_a_socket_and_is_resent_until_answered() {
        let mut m = machine();
        assert_eq!(m.command("r1".into(), stop("s1")), None);
        let frames = m.opened(0);
        assert!(matches!(
            frames[0],
            ClientFrame::Hello {
                resume_from: None,
                ..
            }
        ));
        assert!(
            matches!(&frames[1], ClientFrame::Command { request_id, .. } if request_id == "r1")
        );
        assert!(m.command("r2".into(), stop("s2")).is_some());

        m.closed("gone");
        assert_eq!(m.opened(10).len(), 3, "hello and both unanswered commands");
        let ack = r#"{"type":"ack","request_id":"r1","result":{"type":"stopped","report":"ok"}}"#;
        assert!(m.received(ack, 11).is_some());
        assert_eq!(m.pending().count(), 1);
    }

    #[test]
    fn the_next_hello_resumes_from_the_last_durable_seq() {
        let mut m = machine();
        m.opened(0);
        let event = r#"{"type":"event","seq":42,"at_ms":1,"event":{"type":"assistant_message","session_id":"s1","text":"hi"}}"#;
        assert!(m.received(event, 1).is_some());
        m.closed("gone");
        assert!(matches!(
            m.opened(2)[0],
            ClientFrame::Hello {
                resume_from: Some(42),
                ..
            }
        ));
    }

    #[test]
    fn a_refusal_closes_as_unauthorized_and_failures_back_off() {
        let mut m = machine();
        m.opened(0);
        let refused = ServerFrame::Error {
            request_id: None,
            error: ProtocolError::new(ErrorCode::Unauthorized, "revoked"),
        };
        m.received(&serde_json::to_string(&refused).unwrap(), 1);
        let d = m.closed("closed");
        assert_eq!(
            (d.kind, d.reason.as_str()),
            (DropKind::Unauthorized, "revoked")
        );

        let a = m.failed(DropKind::DaemonNotRunning, "refused");
        let b = m.failed(DropKind::DaemonNotRunning, "refused");
        assert!(b.retry_in > a.retry_in);
        assert!(m.failed(DropKind::Dropped, "x").retry_in <= RETRY_MAX);

        m.opened(5);
        assert_eq!(
            m.closed("lost").retry_in,
            RETRY_MIN,
            "a working connection resets the wait"
        );
    }

    #[test]
    fn a_quiet_socket_is_pinged_and_a_silent_one_is_dead() {
        let mut m = machine();
        assert_eq!(m.tick(0), Tick::Wait, "no socket, nothing to do");
        m.opened(0);
        assert_eq!(m.tick(1_000), Tick::Wait);
        assert_eq!(m.tick(millis(PING_EVERY)), Tick::Ping(ClientFrame::Ping));
        m.heard(millis(PING_EVERY) + 10);
        assert_eq!(m.tick(millis(PING_EVERY) + 20), Tick::Wait);
        let dead = m.tick(millis(PING_EVERY) + millis(SILENT_LIMIT) + 11);
        assert!(matches!(
            dead,
            Tick::Dead(Disconnect {
                kind: DropKind::Dropped,
                ..
            })
        ));
        assert!(!m.is_open());
    }
}
