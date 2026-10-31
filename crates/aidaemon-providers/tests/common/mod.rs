//! A tiny scripted HTTP/1.1 server for provider tests. Each accepted
//! connection consumes the next scripted response and records the request.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{ProviderError, StreamEvent};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
pub struct MockResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub chunks: Vec<Vec<u8>>,
    pub delay_before: Duration,
    pub chunk_delay: Duration,
}

impl MockResponse {
    /// A 200 SSE response whose body is split into `chunk_size`-byte writes,
    /// so events and JSON arrive fragmented.
    pub fn sse(body: impl Into<String>, chunk_size: usize) -> Self {
        let body: String = body.into();
        Self {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            chunks: body
                .as_bytes()
                .chunks(chunk_size.max(1))
                .map(<[u8]>::to_vec)
                .collect(),
            delay_before: Duration::ZERO,
            chunk_delay: Duration::from_millis(1),
        }
    }

    pub fn status(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            chunks: vec![body.as_bytes().to_vec()],
            delay_before: Duration::ZERO,
            chunk_delay: Duration::ZERO,
        }
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn delay(mut self, delay: Duration) -> Self {
        self.delay_before = delay;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    /// Lower-cased header names.
    pub headers: HashMap<String, String>,
    pub body: Value,
}

pub struct MockServer {
    pub base: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

impl MockServer {
    pub async fn start(responses: Vec<MockResponse>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let recorded = requests.clone();
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                let queue = queue.clone();
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    let _ = handle(socket, queue, recorded).await;
                });
            }
        });
        Self { base, requests }
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

async fn handle(
    mut socket: TcpStream,
    queue: Arc<Mutex<VecDeque<MockResponse>>>,
    recorded: Arc<Mutex<Vec<Recorded>>>,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let headers: HashMap<String, String> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buf.len() < header_end + length {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = serde_json::from_slice(&buf[header_end..]).unwrap_or(Value::Null);
    recorded.lock().unwrap().push(Recorded {
        method,
        path,
        headers,
        body,
    });

    let response = queue
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or_else(|| MockResponse::status(500, r#"{"error":{"message":"unscripted"}}"#));
    tokio::time::sleep(response.delay_before).await;
    let mut head = format!("HTTP/1.1 {} Mock\r\n", response.status);
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("connection: close\r\n\r\n");
    socket.write_all(head.as_bytes()).await?;
    socket.flush().await?;
    for chunk in &response.chunks {
        socket.write_all(chunk).await?;
        socket.flush().await?;
        tokio::time::sleep(response.chunk_delay).await;
    }
    socket.shutdown().await
}

/// Format JSON values as `data:` SSE events (OpenAI style).
pub fn data_events(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .collect::<String>()
}

/// Format `(event, data)` pairs as named SSE events (Anthropic style).
pub fn named_events(events: &[(&str, Value)]) -> String {
    events
        .iter()
        .map(|(name, data)| format!("event: {name}\ndata: {data}\n\n"))
        .collect()
}

/// Drain a provider stream into (deltas, done responses, first error).
pub async fn drain(
    mut stream: BoxStream<'_, Result<StreamEvent, ProviderError>>,
) -> (
    Vec<String>,
    Vec<aidaemon_core::ModelResponse>,
    Option<ProviderError>,
) {
    let mut deltas = Vec::new();
    let mut done = Vec::new();
    let mut error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::TextDelta(d)) => deltas.push(d),
            Ok(StreamEvent::Done(r)) => done.push(r),
            Err(e) => {
                error = Some(e);
                break;
            }
        }
    }
    (deltas, done, error)
}
