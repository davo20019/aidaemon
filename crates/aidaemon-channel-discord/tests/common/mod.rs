//! Local test doubles: a scripted HTTP server and a WebSocket gateway.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channel_discord::DiscordChannel;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

pub const TOKEN: &str = "synthetic.discord.token";
pub const BOT_ID: &str = "100000000000000009";
pub const CHANNEL_ID: &str = "100000000000000200";
pub const GUILD_ID: &str = "100000000000000300";

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Value,
    /// The body as sent (multipart uploads are not JSON).
    pub raw: Vec<u8>,
}

type Scripted = (u16, String);

/// Minimal HTTP/1.1 server. Responses are served from a FIFO script; when
/// it is empty, a message object with a fresh snowflake id is returned.
pub struct MockHttp {
    pub base: String,
    pub requests: Arc<Mutex<Vec<Recorded>>>,
    script: Arc<Mutex<VecDeque<Scripted>>>,
}

impl MockHttp {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let script = Arc::new(Mutex::new(VecDeque::new()));
        let counter = Arc::new(Mutex::new(0u64));
        let (req2, script2) = (requests.clone(), script.clone());
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let (r, s, c) = (req2.clone(), script2.clone(), counter.clone());
                tokio::spawn(serve_conn(sock, r, s, c));
            }
        });
        Self {
            base,
            requests,
            script,
        }
    }

    pub fn push(&self, status: u16, body: Value) {
        self.script
            .lock()
            .unwrap()
            .push_back((status, body.to_string()));
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    pub async fn wait_for(&self, n: usize) -> Vec<Recorded> {
        for _ in 0..200 {
            if self.requests.lock().unwrap().len() >= n {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.requests()
    }
}

async fn serve_conn(
    sock: TcpStream,
    requests: Arc<Mutex<Vec<Recorded>>>,
    script: Arc<Mutex<VecDeque<Scripted>>>,
    counter: Arc<Mutex<u64>>,
) {
    let mut reader = BufReader::new(sock);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
            return;
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut headers = HashMap::new();
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).await.unwrap_or(0) == 0 {
                return;
            }
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }
        let len: usize = headers
            .get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut body = vec![0u8; len];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }
        let raw = body;
        let body = serde_json::from_slice(&raw).unwrap_or(Value::Null);
        requests.lock().unwrap().push(Recorded {
            method,
            path,
            headers,
            body,
            raw,
        });
        let (status, resp) = script.lock().unwrap().pop_front().unwrap_or_else(|| {
            let mut c = counter.lock().unwrap();
            *c += 1;
            (
                200,
                json!({"id": format!("{}", 100000000000001000u64 + *c), "channel_id": CHANNEL_ID})
                    .to_string(),
            )
        });
        let out = format!(
            "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{resp}",
            resp.len()
        );
        if reader.get_mut().write_all(out.as_bytes()).await.is_err() {
            return;
        }
    }
}

/// A local gateway endpoint.
pub struct MockGateway {
    listener: TcpListener,
    pub url: String,
}

pub struct GwConn {
    pub ws: WebSocketStream<TcpStream>,
}

impl MockGateway {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        Self { listener, url }
    }

    pub async fn accept(&self) -> GwConn {
        let (sock, _) = tokio::time::timeout(Duration::from_secs(5), self.listener.accept())
            .await
            .expect("client did not connect")
            .unwrap();
        GwConn {
            ws: tokio_tungstenite::accept_async(sock).await.unwrap(),
        }
    }
}

impl MockGateway {
    /// Like [`Self::accept`], also returning the connection's query string.
    pub async fn accept_with_query(&self) -> (GwConn, String) {
        use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};

        let (sock, _) = tokio::time::timeout(Duration::from_secs(5), self.listener.accept())
            .await
            .expect("client did not connect")
            .unwrap();
        let query = Arc::new(Mutex::new(String::new()));
        let seen = query.clone();
        #[allow(clippy::result_large_err)]
        let callback =
            move |request: &Request, response: Response| -> Result<Response, ErrorResponse> {
                *seen.lock().unwrap() = request.uri().query().unwrap_or("").to_string();
                Ok(response)
            };
        let ws = tokio_tungstenite::accept_hdr_async(sock, callback)
            .await
            .unwrap();
        let query = query.lock().unwrap().clone();
        (GwConn { ws }, query)
    }
}

impl GwConn {
    pub async fn send(&mut self, v: Value) {
        self.ws.send(Message::text(v.to_string())).await.unwrap();
    }

    pub async fn hello(&mut self, interval_ms: u64) {
        self.send(json!({"op": 10, "d": {"heartbeat_interval": interval_ms}}))
            .await;
    }

    pub async fn ready(&mut self, session_id: &str, resume_url: &str) {
        self.send(json!({"op": 0, "s": 1, "t": "READY", "d": {
            "v": 10,
            "user": {"id": BOT_ID, "username": "aidaemon", "bot": true},
            "session_id": session_id,
            "resume_gateway_url": resume_url,
            "guilds": [],
        }}))
        .await;
    }

    pub async fn dispatch(&mut self, seq: u64, t: &str, d: Value) {
        self.send(json!({"op": 0, "s": seq, "t": t, "d": d})).await;
    }

    /// Next JSON frame from the client, including heartbeats.
    pub async fn recv_any(&mut self) -> Value {
        loop {
            let msg = tokio::time::timeout(Duration::from_secs(5), self.ws.next())
                .await
                .expect("timed out waiting for client frame")
                .expect("stream ended")
                .expect("ws error");
            if let Message::Text(t) = msg {
                return serde_json::from_str(&t).unwrap();
            }
        }
    }

    /// Next non-heartbeat frame.
    pub async fn recv(&mut self) -> Value {
        loop {
            let v = self.recv_any().await;
            if v["op"] != 1 {
                return v;
            }
        }
    }

    pub async fn close(&mut self, code: u16) {
        self.ws
            .close(Some(CloseFrame {
                code: CloseCode::from(code),
                reason: "test".into(),
            }))
            .await
            .unwrap();
    }
}

pub fn channel(http: &MockHttp) -> DiscordChannel {
    DiscordChannel::new(TOKEN)
        .with_api_base(http.base.clone())
        .with_http_client(reqwest::Client::builder().no_proxy().build().unwrap())
        .with_download_client(reqwest::Client::builder().no_proxy())
        .with_reconnect_backoff(Duration::from_millis(10), Duration::from_millis(50))
}
