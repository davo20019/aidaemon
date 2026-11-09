//! The shared downloader against a local HTTP server. No network access.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_channels::download::{DownloadClient, DownloadRequest};
use aidaemon_channels::ChannelError;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const TOKEN: &str = "xoxb-synthetic-secret";

/// Answers each connection with the next canned response (the last repeats)
/// and records the request heads.
struct Server {
    base: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    async fn start(responses: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests: Arc<Mutex<Vec<String>>> = Arc::default();
        let recorded = requests.clone();
        tokio::spawn(async move {
            let mut next = 0;
            while let Ok((mut stream, _)) = listener.accept().await {
                let response = responses[next.min(responses.len() - 1)].clone();
                next += 1;
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut chunk = [0u8; 1024];
                    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => head.extend_from_slice(&chunk[..n]),
                        }
                    }
                    recorded
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&head).to_lowercase());
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        Self { base, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn redirect(location: &str) -> String {
    format!(
        "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
}

fn client() -> DownloadClient {
    DownloadClient::from_builder(reqwest::Client::builder().no_proxy())
}

/// Fetch `url` with the bearer token, allowing only `allowed_base`.
async fn fetch(url: &str, allowed_base: &str, max_bytes: u64) -> Result<Vec<u8>, ChannelError> {
    let origin = reqwest::Url::parse(allowed_base).unwrap().origin();
    client()
        .fetch(DownloadRequest {
            url,
            bearer_token: Some(TOKEN),
            allowed: &|target| target.origin() == origin,
            max_bytes,
            timeout: Duration::from_secs(5),
            user_agent: None,
        })
        .await
        .map(|downloaded| downloaded.bytes)
}

#[tokio::test]
async fn a_file_is_fetched_with_the_token() {
    let server = Server::start(vec![ok("file bytes")]).await;
    let bytes = fetch(&format!("{}/f", server.base), &server.base, 100)
        .await
        .unwrap();
    assert_eq!(bytes, b"file bytes");
    assert!(server.requests()[0].contains(&format!("authorization: bearer {TOKEN}")));
}

#[tokio::test]
async fn a_declared_size_over_the_limit_is_refused() {
    let server = Server::start(vec![ok("0123456789")]).await;
    let err = fetch(&format!("{}/f", server.base), &server.base, 5)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChannelError::TooLarge { limit_bytes: 5 }),
        "{err}"
    );
}

#[tokio::test]
async fn a_body_with_no_declared_size_is_cut_off_at_the_limit() {
    let response = format!(
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}",
        "x".repeat(4096)
    );
    let server = Server::start(vec![response]).await;
    let err = fetch(&format!("{}/f", server.base), &server.base, 10)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChannelError::TooLarge { limit_bytes: 10 }),
        "{err}"
    );
}

#[tokio::test]
async fn exactly_the_limit_passes() {
    let server = Server::start(vec![ok("12345")]).await;
    let bytes = fetch(&format!("{}/f", server.base), &server.base, 5)
        .await
        .unwrap();
    assert_eq!(bytes.len(), 5);
}

#[tokio::test]
async fn a_redirect_on_the_allowed_host_keeps_the_token_and_sends_no_referer() {
    let server = Server::start(vec![redirect("/next"), ok("moved bytes")]).await;
    let bytes = fetch(&format!("{}/f", server.base), &server.base, 100)
        .await
        .unwrap();
    assert_eq!(bytes, b"moved bytes");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].starts_with("get /next "));
    assert!(requests[1].contains("authorization: bearer"));
    assert!(!requests[1].contains("referer"), "{}", requests[1]);
}

#[tokio::test]
async fn a_redirect_to_another_host_is_refused_and_never_contacted() {
    let other = Server::start(vec![ok("stolen")]).await;
    let server = Server::start(vec![redirect(&format!("{}/steal", other.base))]).await;
    let err = fetch(&format!("{}/f", server.base), &server.base, 100)
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)), "{err}");
    assert!(other.requests().is_empty(), "the token left the host");
    assert!(!err.to_string().contains(&other.base), "{err}");
}

#[tokio::test]
async fn a_redirect_loop_stops() {
    let server = Server::start(vec![redirect("/again")]).await;
    let err = fetch(&format!("{}/f", server.base), &server.base, 100)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("too many redirects"), "{err}");
    assert_eq!(server.requests().len(), 4);
}

#[tokio::test]
async fn a_url_off_the_allowed_hosts_is_never_requested() {
    let server = Server::start(vec![ok("x")]).await;
    let err = fetch(
        &format!("{}/f", server.base),
        "https://files.example.test",
        100,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, ChannelError::Other(_)), "{err}");
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn statuses_map_to_typed_errors() {
    let status = |code: &str, extra: &str| {
        format!("HTTP/1.1 {code} X\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n")
    };
    let server = Server::start(vec![
        status("401", ""),
        status("403", ""),
        status("429", "Retry-After: 7\r\n"),
        status("503", ""),
        status("404", ""),
    ])
    .await;
    let url = format!("{}/f", server.base);
    assert!(matches!(
        fetch(&url, &server.base, 100).await,
        Err(ChannelError::Auth(_))
    ));
    assert!(matches!(
        fetch(&url, &server.base, 100).await,
        Err(ChannelError::Auth(_))
    ));
    assert!(matches!(
        fetch(&url, &server.base, 100).await,
        Err(ChannelError::RateLimited {
            retry_after_secs: Some(7)
        })
    ));
    assert!(matches!(
        fetch(&url, &server.base, 100).await,
        Err(ChannelError::Transport(_))
    ));
    assert!(matches!(
        fetch(&url, &server.base, 100).await,
        Err(ChannelError::Other(_))
    ));
}

#[tokio::test]
async fn a_token_in_the_url_stays_out_of_transport_errors() {
    // Bound, then released: nothing listens there.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let url = format!("{base}/file/bot{TOKEN}/documents/a.txt");
    let err = fetch(&url, &base, 100).await.unwrap_err();
    assert!(matches!(err, ChannelError::Transport(_)), "{err}");
    let shown = format!("{err} {err:?}");
    assert!(!shown.contains(TOKEN), "{shown}");
    assert!(!shown.contains("/file/bot"), "{shown}");
}

#[tokio::test]
async fn an_invalid_url_is_reported_without_echoing_it() {
    let err = fetch(&format!("not a url {TOKEN}"), "http://127.0.0.1", 100)
        .await
        .unwrap_err();
    assert!(!err.to_string().contains(TOKEN), "{err}");
}
