//! `web_fetch`: GET a URL and return it as markdown.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::redirect::Policy;
use reqwest::Url;
use serde::Deserialize;
use serde_json::{json, Value};

use aidaemon_core::{
    AccessManifest, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult, ToolSpec,
};

use crate::util::{fail, parse_args, truncate_chars};

/// Fetch a web page over http(s) and convert HTML to markdown.
///
/// Redirects are followed only within the same host, so the host declared
/// in [`Tool::access`] is the only host contacted. A cross-host redirect is
/// returned to the model as an error naming the new location, which it can
/// fetch explicitly (and policies can check) if appropriate.
#[derive(Debug, Clone)]
pub struct WebFetch {
    max_chars: usize,
    max_body_bytes: usize,
    timeout: Duration,
    use_proxy: bool,
    user_agent: String,
}

impl Default for WebFetch {
    fn default() -> Self {
        Self {
            max_chars: 20_000,
            max_body_bytes: 5 * 1024 * 1024,
            timeout: Duration::from_secs(30),
            use_proxy: true,
            user_agent: concat!("aidaemon-tools/", env!("CARGO_PKG_VERSION")).into(),
        }
    }
}

impl WebFetch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Characters of converted text returned to the model.
    pub fn max_chars(mut self, chars: usize) -> Self {
        self.max_chars = chars.max(100);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Ignore `HTTP(S)_PROXY` environment variables.
    pub fn without_proxy(mut self) -> Self {
        self.use_proxy = false;
        self
    }

    pub fn user_agent(mut self, agent: impl Into<String>) -> Self {
        self.user_agent = agent.into();
        self
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
    url: String,
}

fn parse_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw.trim()).map_err(|e| {
        format!("invalid URL `{raw}`: {e}. Give a full URL such as https://example.com/page")
    })?;
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(format!("unsupported scheme `{other}`; only http and https")),
    }
    if url.host_str().is_none() {
        return Err(format!("URL `{raw}` has no host"));
    }
    Ok(url)
}

fn html_to_markdown(html: &str) -> String {
    let converter = htmd::HtmlToMarkdown::builder()
        .skip_tags(vec![
            "script", "style", "noscript", "head", "svg", "iframe", "template",
        ])
        .build();
    converter.convert(html).unwrap_or_else(|_| html.to_string())
}

#[async_trait]
impl Tool for WebFetch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch".into(),
            description: format!(
                "Fetch a URL with HTTP GET (http/https only) and return the page as markdown \
                 (HTML is converted; plain text and JSON are returned as is). Output is capped \
                 at {} characters. Redirects to a different host are not followed; the new \
                 location is reported instead. HTTP errors are reported with their status.",
                self.max_chars
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."}
                },
                "required": ["url"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    fn validate(&self, arguments: &Value) -> Result<(), String> {
        let url = arguments
            .get("url")
            .and_then(Value::as_str)
            .ok_or("`url` is required and must be a string")?;
        parse_url(url).map(|_| ())
    }

    fn access(&self, arguments: &Value) -> AccessManifest {
        let host = arguments
            .get("url")
            .and_then(Value::as_str)
            .and_then(|u| parse_url(u).ok())
            .and_then(|u| u.host_str().map(str::to_string));
        AccessManifest {
            hosts: host.into_iter().collect(),
            ..Default::default()
        }
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let args: FetchArgs = match parse_args("web_fetch", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        let url = match parse_url(&args.url) {
            Ok(u) => u,
            Err(e) => return Ok(fail(e)),
        };
        let host = url.host_str().unwrap_or_default().to_string();
        let same_host = host.clone();
        let mut builder = reqwest::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .redirect(Policy::custom(move |attempt| {
                if attempt.previous().len() >= 10 {
                    attempt.error("too many redirects")
                } else if attempt.url().host_str() == Some(same_host.as_str())
                    && matches!(attempt.url().scheme(), "http" | "https")
                {
                    attempt.follow()
                } else {
                    attempt.stop()
                }
            }));
        if !self.use_proxy {
            builder = builder.no_proxy();
        }
        let client = builder
            .build()
            .map_err(|e| format!("cannot build HTTP client: {e}"))?;

        let request = client.get(url.clone()).send();
        let mut response = tokio::select! {
            r = request => match r {
                Ok(r) => r,
                Err(e) => return Ok(fail(format!("request to {url} failed: {e}"))
                    .with_details(json!({"url": url.as_str(), "host": host}))),
            },
            _ = ctx.cancel.cancelled() => return Ok(fail("cancelled")),
        };
        let status = response.status();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        let mut body: Vec<u8> = Vec::new();
        let mut body_truncated = false;
        loop {
            let chunk = tokio::select! {
                c = response.chunk() => c,
                _ = ctx.cancel.cancelled() => return Ok(fail("cancelled")),
            };
            match chunk {
                Ok(Some(bytes)) => {
                    let room = self.max_body_bytes - body.len();
                    if bytes.len() > room {
                        body.extend_from_slice(&bytes[..room]);
                        body_truncated = true;
                        break;
                    }
                    body.extend_from_slice(&bytes);
                }
                Ok(None) => break,
                Err(e) => return Ok(fail(format!("reading the body of {final_url} failed: {e}"))),
            }
        }

        let mut details = json!({
            "url": url.as_str(),
            "final_url": final_url,
            "host": host,
            "status": status.as_u16(),
            "content_type": content_type,
            "bytes": body.len(),
        });

        if status.is_redirection() {
            let target = location.unwrap_or_else(|| "(no Location header)".into());
            details["location"] = json!(target);
            return Ok(fail(format!(
                "HTTP {status}: {final_url} redirects to {target}, a different host. \
                 Fetch that URL explicitly if it is the page you want."
            ))
            .with_details(details));
        }

        let lower = content_type.to_ascii_lowercase();
        let is_html = lower.contains("html") || (lower.is_empty() && looks_like_html(&body));
        let is_text = is_html
            || lower.is_empty()
            || lower.starts_with("text/")
            || lower.contains("json")
            || lower.contains("xml")
            || lower.contains("javascript");
        if !is_text {
            return Ok(fail(format!(
                "HTTP {status}: {final_url} returned `{content_type}` ({} bytes), which is not \
                 text and cannot be shown.",
                body.len()
            ))
            .with_details(details));
        }
        let raw = String::from_utf8_lossy(&body);
        let text = if is_html {
            html_to_markdown(&raw)
        } else {
            raw.into_owned()
        };

        if !status.is_success() {
            let (snippet, _) = truncate_chars(text.trim(), 2000);
            return Ok(
                fail(format!("HTTP {status} from {final_url}.\n\n{snippet}")).with_details(details),
            );
        }

        let (mut text, cut) = truncate_chars(text.trim(), self.max_chars);
        if cut || body_truncated {
            text.push_str(&format!(
                "\n\n[Content truncated to {} characters.]",
                self.max_chars
            ));
        }
        details["truncated"] = json!(cut || body_truncated);
        Ok(ToolOutput::text(format!(
            "URL: {final_url}\nStatus: {}\nContent-Type: {content_type}\n\n{text}",
            status.as_u16()
        ))
        .with_details(details))
    }
}

fn looks_like_html(body: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&body[..body.len().min(512)]).to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::ctx;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serve canned responses on a local port: path -> (status line, content type, body).
    async fn serve(routes: Vec<(&'static str, &'static str, &'static str, String)>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let routes = routes.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let (status, ctype, body) = routes
                        .iter()
                        .find(|(p, ..)| *p == path)
                        .map(|(_, s, c, b)| (*s, *c, b.clone()))
                        .unwrap_or(("404 Not Found", "text/plain", "missing".to_string()));
                    let extra = if status.starts_with("302") {
                        format!("Location: {body}\r\n")
                    } else {
                        String::new()
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    #[tokio::test]
    async fn fetches_html_as_markdown() {
        let base = serve(vec![
            (
                "/page",
                "200 OK",
                "text/html; charset=utf-8",
                "<html><head><title>T</title><script>evil()</script></head><body><h1>Synthetic Title</h1><p>Hello <b>world</b>.</p></body></html>".into(),
            ),
            ("/big", "200 OK", "text/plain", "x".repeat(5000)),
            ("/gone", "404 Not Found", "text/html", "<p>no such page</p>".into()),
            ("/away", "302 Found", "text/plain", "http://elsewhere.invalid/".into()),
            ("/img", "200 OK", "image/png", "\u{1}PNG".into()),
        ])
        .await;
        let tool = WebFetch::new().without_proxy().max_chars(1000);

        let url = format!("{base}/page");
        let out = tool.call(json!({"url": url}), &ctx()).await.unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("# Synthetic Title"), "{}", out.content);
        assert!(out.content.contains("Hello **world**."));
        assert!(!out.content.contains("evil"));
        assert_eq!(out.details["status"], 200);

        let out = tool
            .call(json!({"url": format!("{base}/big")}), &ctx())
            .await
            .unwrap();
        assert!(out
            .content
            .contains("[Content truncated to 1000 characters.]"));
        assert_eq!(out.details["truncated"], true);

        let out = tool
            .call(json!({"url": format!("{base}/gone")}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error);
        assert_eq!(out.details["status"], 404);
        assert!(out.content.contains("404") && out.content.contains("no such page"));

        let out = tool
            .call(json!({"url": format!("{base}/away")}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("elsewhere.invalid"));
        assert_eq!(out.details["status"], 302);

        let out = tool
            .call(json!({"url": format!("{base}/img")}), &ctx())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("image/png"));
    }

    #[test]
    fn validates_scheme_and_declares_host() {
        let tool = WebFetch::new();
        assert!(tool
            .validate(&json!({"url": "file:///etc/passwd"}))
            .is_err());
        assert!(tool.validate(&json!({"url": "not a url"})).is_err());
        assert!(tool
            .validate(&json!({"url": "https://example.com/x"}))
            .is_ok());
        assert_eq!(
            tool.access(&json!({"url": "https://docs.example.com/a?b=c"}))
                .hosts,
            vec!["docs.example.com"]
        );
        assert!(tool.annotations().read_only);
    }
}
