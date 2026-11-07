//! Local stand-ins for the Slack Web API (a hand-rolled HTTP/1.1 server) and
//! the Socket Mode WebSocket. Nothing here touches the network.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;

pub const BOT_TOKEN: &str = "xoxb-synthetic";
pub const APP_TOKEN: &str = "xapp-synthetic";
pub const BOT_USER: &str = "U000BOT";

#[derive(Debug, Clone)]
pub struct Request {
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Web API method name (`chat.postMessage`).
    pub fn method(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or("")
    }

    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Response {
    pub fn json(body: Value) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body: body.to_string(),
        }
    }

    pub fn status(mut self, status: u16) -> Self {
        self.status = status;
        self
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

type Handler = dyn Fn(&Request, usize) -> Response + Send + Sync;

/// Minimal Web API mock. The handler receives the request and how many
/// earlier calls hit the same method.
pub struct MockApi {
    pub base: String,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl MockApi {
    pub async fn start<F>(handler: F) -> Self
    where
        F: Fn(&Request, usize) -> Response + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests: Arc<Mutex<Vec<Request>>> = Arc::default();
        let handler: Arc<Handler> = Arc::new(handler);
        let recorded = requests.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let handler = handler.clone();
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    let _ = serve_http(stream, handler, recorded).await;
                });
            }
        });
        Self {
            base: format!("http://{addr}/api"),
            requests,
        }
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }

    pub fn calls(&self, method: &str) -> Vec<Request> {
        self.requests()
            .into_iter()
            .filter(|r| r.method() == method)
            .collect()
    }
}

async fn serve_http(
    mut stream: TcpStream,
    handler: Arc<Handler>,
    recorded: Arc<Mutex<Vec<Request>>>,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let path = request_line.split(' ').nth(1).unwrap_or("").to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let content_length = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < header_end + content_length {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = String::from_utf8_lossy(&buf[header_end..]).to_string();
    let request = Request {
        path,
        headers,
        body,
    };

    let prior = {
        let mut guard = recorded.lock().unwrap();
        let prior = guard
            .iter()
            .filter(|r| r.method() == request.method())
            .count();
        guard.push(request.clone());
        prior
    };
    let response = handler(&request, prior);

    let mut out = format!(
        "HTTP/1.1 {} MOCK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status,
        response.body.len()
    );
    for (k, v) in &response.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(&response.body);
    stream.write_all(out.as_bytes()).await?;
    stream.shutdown().await
}

/// Socket Mode mock. Connection `i` sends `scripts[i]` frames in order, then
/// records whatever the client sends until it disconnects.
pub struct MockSocket {
    pub url: String,
    received: Arc<Mutex<Vec<String>>>,
    connections: Arc<AtomicUsize>,
}

impl MockSocket {
    pub async fn start(scripts: Vec<Vec<Value>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let received: Arc<Mutex<Vec<String>>> = Arc::default();
        let connections = Arc::new(AtomicUsize::new(0));
        let (rec, conns) = (received.clone(), connections.clone());
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let index = conns.fetch_add(1, Ordering::SeqCst);
                let script = scripts.get(index).cloned().unwrap_or_default();
                let rec = rec.clone();
                tokio::spawn(async move {
                    let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                        return;
                    };
                    let (mut tx, mut rx) = ws.split();
                    for frame in script {
                        if tx.send(Message::Text(frame.to_string())).await.is_err() {
                            return;
                        }
                    }
                    while let Some(Ok(msg)) = rx.next().await {
                        match msg {
                            Message::Text(text) => rec.lock().unwrap().push(text),
                            Message::Close(_) => break,
                            _ => {}
                        }
                    }
                });
            }
        });
        Self {
            url: format!("ws://{addr}/socket"),
            received,
            connections,
        }
    }

    pub fn received(&self) -> Vec<String> {
        self.received.lock().unwrap().clone()
    }

    pub fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    /// Envelope ids the client acked, in order.
    pub fn acks(&self) -> Vec<String> {
        self.received()
            .iter()
            .filter_map(|t| serde_json::from_str::<Value>(t).ok())
            .filter_map(|v| v.get("envelope_id")?.as_str().map(str::to_string))
            .collect()
    }

    /// Wait until `n` acks arrived (or panic after 5s).
    pub async fn wait_for_acks(&self, n: usize) -> Vec<String> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let acks = self.acks();
            if acks.len() >= n {
                return acks;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "only {} of {n} acks arrived: {acks:?}",
                acks.len()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// A Web API mock that answers `auth.test` and `apps.connections.open`
/// (pointing at `socket_url`) and `ok` with a ts for everything else.
pub async fn standard_api(socket_url: String) -> MockApi {
    MockApi::start(move |req, _| match req.method() {
        "auth.test" => Response::json(serde_json::json!({
            "ok": true, "user_id": BOT_USER, "team_id": "T000SYNTH", "user": "aidaemon"
        })),
        "apps.connections.open" => {
            Response::json(serde_json::json!({ "ok": true, "url": socket_url }))
        }
        _ => Response::json(serde_json::json!({ "ok": true, "ts": "1700000000.999999" })),
    })
    .await
}
