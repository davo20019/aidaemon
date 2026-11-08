//! A local HTTP server standing in for a chat platform's API in tests:
//! it records every request and answers each with JSON chosen by a
//! responder.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

#[derive(Debug, Clone)]
pub(crate) struct Recorded {
    pub method: String,
    /// The path with its query string.
    pub path: String,
    pub body: String,
}

impl Recorded {
    /// The body as JSON (`Null` if it is not JSON).
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

type Responder = dyn Fn(&Recorded) -> Value + Send + Sync;

pub(crate) struct MockApi {
    pub base: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

impl MockApi {
    /// Serve on a free local port, answering each request with `respond`.
    pub async fn start(respond: impl Fn(&Recorded) -> Value + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests: Arc<Mutex<Vec<Recorded>>> = Arc::default();
        let respond: Arc<Responder> = Arc::new(respond);
        let recorded = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(serve(socket, recorded.clone(), respond.clone()));
            }
        });
        Self { base, requests }
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    /// Wait (up to five seconds) for a request matching `pred`.
    pub async fn wait_for(&self, pred: impl Fn(&Recorded) -> bool) -> Recorded {
        for _ in 0..500 {
            if let Some(found) = self.requests().into_iter().find(|r| pred(r)) {
                return found;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("no matching request; saw {:?}", self.requests());
    }

    /// An HTTP client that reaches the mock (never through a proxy).
    pub fn client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }
}

/// Serve one connection's requests in order.
async fn serve(
    socket: tokio::net::TcpStream,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    respond: Arc<Responder>,
) {
    let mut reader = BufReader::new(socket);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
            return;
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let mut length = 0usize;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).await.unwrap_or(0) == 0 {
                return;
            }
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((key, value)) = header.split_once(':') {
                if key.eq_ignore_ascii_case("content-length") {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
        }
        let mut body = vec![0u8; length];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }
        let request = Recorded {
            method,
            path,
            body: String::from_utf8_lossy(&body).into_owned(),
        };
        let text = respond(&request).to_string();
        recorded.lock().unwrap().push(request);
        let out = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{text}",
            text.len()
        );
        if reader.get_mut().write_all(out.as_bytes()).await.is_err() {
            return;
        }
    }
}
