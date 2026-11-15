//! A one-route HTTP server for the HTTP backends' unit tests.

use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug, Clone)]
pub(crate) struct Seen {
    pub head: String,
    pub body: Value,
}

pub(crate) type Handler = Arc<dyn Fn(&Value) -> (u16, String) + Send + Sync>;

/// A one-route HTTP server: reads each request, answers with `handler`.
pub(crate) async fn serve(handler: Handler) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let handler = handler.clone();
            let log = log.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head, body_start) = loop {
                    let n = socket.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break (String::from_utf8_lossy(&buf[..at]).to_string(), at + 4);
                    }
                };
                let length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                while buf.len() < body_start + length {
                    let n = socket.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                let body: Value = serde_json::from_slice(&buf[body_start..body_start + length])
                    .unwrap_or(Value::Null);
                let (status, reply) = handler(&body);
                log.lock().unwrap().push(Seen { head, body });
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (format!("http://{addr}"), seen)
}

pub(crate) fn fixed(status: u16, body: &str) -> Handler {
    let body = body.to_string();
    Arc::new(move |_| (status, body.clone()))
}
