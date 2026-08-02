use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::traits::{
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolExecutionContext, ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::StatusUpdate;

const DEFAULT_MAX_CHARS: usize = 20_000;
const MAX_MAX_CHARS: usize = 50_000;

/// Hard cap on response-body bytes buffered from a web fetch/search. The
/// model-facing output is already bounded (extracted-text char limit / result
/// count); this only bounds *transient memory* against a pathologically large
/// or mislabeled response that `.text()`/`.json()` would otherwise buffer whole.
pub(crate) const MAX_FETCH_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Append `chunk` to `buf` without exceeding `max_bytes`. Returns true if the
/// chunk was (partially) dropped because the cap was reached. Truncation is on a
/// raw byte boundary, so callers must decode with `from_utf8_lossy`.
pub(crate) fn append_capped(buf: &mut Vec<u8>, chunk: &[u8], max_bytes: usize) -> bool {
    let remaining = max_bytes.saturating_sub(buf.len());
    if chunk.len() > remaining {
        buf.extend_from_slice(&chunk[..remaining]);
        true
    } else {
        buf.extend_from_slice(chunk);
        false
    }
}

/// Stream a response body into memory, capping at `max_bytes` to avoid unbounded
/// allocation. Returns the collected bytes and whether the body was truncated.
pub(crate) async fn read_body_capped(
    resp: reqwest::Response,
    max_bytes: usize,
) -> reqwest::Result<(Vec<u8>, bool)> {
    let mut resp = resp;
    let mut buf: Vec<u8> = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = resp.chunk().await? {
        if append_capped(&mut buf, &chunk, max_bytes) {
            truncated = true;
            break;
        }
    }
    Ok((buf, truncated))
}

/// Validates a URL for SSRF vulnerabilities.
/// Returns Ok(()) if the URL is safe to fetch, Err with a message otherwise.
pub fn validate_url_for_ssrf(url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;

    // 1. Only allow http and https schemes
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(format!(
                "Blocked scheme '{}': only http/https allowed",
                scheme
            ))
        }
    }

    // 2. Must have a host
    let host = parsed
        .host_str()
        .ok_or_else(|| "URL must have a host".to_string())?;

    // 3. Block known dangerous hostnames
    let host_lower = host.to_lowercase();
    const BLOCKED_HOSTS: &[&str] = &[
        "localhost",
        "127.0.0.1",
        "::1",
        "[::1]",
        "0.0.0.0",
        "metadata.google.internal",
        "metadata.goog",
        "169.254.169.254",
    ];
    for blocked in BLOCKED_HOSTS {
        if host_lower == *blocked {
            return Err(format!("Blocked host: {}", host));
        }
    }

    // 4. Block hosts that look like internal addresses
    if host_lower.ends_with(".internal")
        || host_lower.ends_with(".local")
        || host_lower.ends_with(".localhost")
    {
        return Err(format!("Blocked internal hostname: {}", host));
    }

    // 5. Resolve the hostname and check all IP addresses
    let port = parsed.port().unwrap_or(match parsed.scheme() {
        "https" => 443,
        _ => 80,
    });

    // Try to resolve the hostname
    let socket_addr = format!("{}:{}", host, port);
    match socket_addr.to_socket_addrs() {
        Ok(addrs) => {
            for addr in addrs {
                if is_blocked_ip(addr.ip()) {
                    return Err(format!(
                        "Blocked IP address {} (resolved from {})",
                        addr.ip(),
                        host
                    ));
                }
            }
        }
        Err(_) => {
            // If we can't resolve, it might be a raw IP - try parsing it
            if let Ok(ip) = host.parse::<IpAddr>() {
                if is_blocked_ip(ip) {
                    return Err(format!("Blocked IP address: {}", ip));
                }
            }
            // If resolution fails and it's not an IP, let the request fail naturally
        }
    }

    Ok(())
}

/// Check if an IP address is in a blocked range (private, loopback, link-local, etc.)
fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => is_blocked_ipv4(ipv4),
        IpAddr::V6(ipv6) => is_blocked_ipv6(ipv6),
    }
}

fn is_blocked_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();

    // Loopback: 127.0.0.0/8
    if octets[0] == 127 {
        return true;
    }

    // Private: 10.0.0.0/8
    if octets[0] == 10 {
        return true;
    }

    // Private: 172.16.0.0/12 (172.16.0.0 - 172.31.255.255)
    if octets[0] == 172 && (16..=31).contains(&octets[1]) {
        return true;
    }

    // Private: 192.168.0.0/16
    if octets[0] == 192 && octets[1] == 168 {
        return true;
    }

    // Link-local: 169.254.0.0/16 (includes cloud metadata at 169.254.169.254)
    if octets[0] == 169 && octets[1] == 254 {
        return true;
    }

    // Broadcast: 255.255.255.255
    if ip == Ipv4Addr::BROADCAST {
        return true;
    }

    // Unspecified: 0.0.0.0
    if ip == Ipv4Addr::UNSPECIFIED {
        return true;
    }

    // Documentation ranges (TEST-NET): 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24
    if (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
    {
        return true;
    }

    // Shared address space (CGNAT): 100.64.0.0/10
    if octets[0] == 100 && (64..=127).contains(&octets[1]) {
        return true;
    }

    false
}

fn is_blocked_ipv6(ip: Ipv6Addr) -> bool {
    // Loopback: ::1
    if ip.is_loopback() {
        return true;
    }

    // Unspecified: ::
    if ip.is_unspecified() {
        return true;
    }

    // IPv4-mapped addresses: check the embedded IPv4
    if let Some(ipv4) = ip.to_ipv4_mapped() {
        return is_blocked_ipv4(ipv4);
    }

    // Link-local: fe80::/10
    let segments = ip.segments();
    if (segments[0] & 0xffc0) == 0xfe80 {
        return true;
    }

    // Unique local addresses (private): fc00::/7
    if (segments[0] & 0xfe00) == 0xfc00 {
        return true;
    }

    false
}

/// A coarse classification of *why* a host is blocked by the private-network
/// policy. The whole point of this enum is **secret safety**: it carries only a
/// fixed, low-cardinality category — never the URL, host, path, query, or
/// credentials. Surfacing `class.label()` in an error message therefore cannot
/// leak any caller-supplied data.
///
/// Shared by `web_fetch` and the `browser` tool so both name a blocked request
/// the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockedHostClass {
    /// Loopback (`127.0.0.0/8`, `::1`, `localhost`, `0.0.0.0`).
    Loopback,
    /// RFC1918 / unique-local private ranges (`10/8`, `172.16/12`, `192.168/16`,
    /// `fc00::/7`) and CGNAT shared space (`100.64.0.0/10`).
    PrivateNetwork,
    /// Link-local, including the cloud metadata endpoint (`169.254.0.0/16`,
    /// notably `169.254.169.254`; `fe80::/10`; `*.internal`/metadata hostnames).
    LinkLocalMetadata,
    /// A non-http(s) scheme (e.g. `file:`, `ftp:`, `data:`).
    DisallowedScheme,
    /// Malformed URL or a URL with no host.
    Malformed,
    /// Other reserved/blocked address (broadcast, documentation/TEST-NET ranges,
    /// `.local` hostnames) that isn't one of the more specific classes above.
    OtherReserved,
}

impl BlockedHostClass {
    /// A short, human-readable label naming ONLY the host class. Safe to embed
    /// in a user/LLM-facing error: it contains no caller data.
    pub fn label(self) -> &'static str {
        match self {
            BlockedHostClass::Loopback => "loopback address",
            BlockedHostClass::PrivateNetwork => "private network",
            BlockedHostClass::LinkLocalMetadata => "link-local/metadata address",
            BlockedHostClass::DisallowedScheme => "disallowed scheme",
            BlockedHostClass::Malformed => "malformed URL",
            BlockedHostClass::OtherReserved => "reserved/blocked address",
        }
    }
}

/// Classify a blocked IPv4 address into a [`BlockedHostClass`]. Mirrors
/// [`is_blocked_ipv4`] exactly (same ranges, same order) but reports the
/// category instead of a bool. Returns `None` for a public address.
fn classify_blocked_ipv4(ip: Ipv4Addr) -> Option<BlockedHostClass> {
    let octets = ip.octets();

    // Loopback: 127.0.0.0/8
    if octets[0] == 127 {
        return Some(BlockedHostClass::Loopback);
    }
    // Unspecified: 0.0.0.0 — treated as loopback-class (targets the local host).
    if ip == Ipv4Addr::UNSPECIFIED {
        return Some(BlockedHostClass::Loopback);
    }
    // Private: 10.0.0.0/8
    if octets[0] == 10 {
        return Some(BlockedHostClass::PrivateNetwork);
    }
    // Private: 172.16.0.0/12
    if octets[0] == 172 && (16..=31).contains(&octets[1]) {
        return Some(BlockedHostClass::PrivateNetwork);
    }
    // Private: 192.168.0.0/16
    if octets[0] == 192 && octets[1] == 168 {
        return Some(BlockedHostClass::PrivateNetwork);
    }
    // Shared address space (CGNAT): 100.64.0.0/10
    if octets[0] == 100 && (64..=127).contains(&octets[1]) {
        return Some(BlockedHostClass::PrivateNetwork);
    }
    // Link-local: 169.254.0.0/16 (includes cloud metadata 169.254.169.254)
    if octets[0] == 169 && octets[1] == 254 {
        return Some(BlockedHostClass::LinkLocalMetadata);
    }
    // Broadcast + documentation/TEST-NET ranges.
    if ip == Ipv4Addr::BROADCAST
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
    {
        return Some(BlockedHostClass::OtherReserved);
    }
    None
}

/// Classify a blocked IPv6 address into a [`BlockedHostClass`]. Mirrors
/// [`is_blocked_ipv6`] (IPv4-mapped addresses are unwrapped and classified as
/// their embedded v4 class). Returns `None` for a public address.
fn classify_blocked_ipv6(ip: Ipv6Addr) -> Option<BlockedHostClass> {
    if ip.is_loopback() || ip.is_unspecified() {
        return Some(BlockedHostClass::Loopback);
    }
    // IPv4-mapped (::ffff:a.b.c.d) — classify by the embedded IPv4 address.
    if let Some(ipv4) = ip.to_ipv4_mapped() {
        return classify_blocked_ipv4(ipv4);
    }
    let segments = ip.segments();
    // Link-local: fe80::/10
    if (segments[0] & 0xffc0) == 0xfe80 {
        return Some(BlockedHostClass::LinkLocalMetadata);
    }
    // Unique local (private): fc00::/7
    if (segments[0] & 0xfe00) == 0xfc00 {
        return Some(BlockedHostClass::PrivateNetwork);
    }
    None
}

fn classify_blocked_ip(ip: IpAddr) -> Option<BlockedHostClass> {
    match ip {
        IpAddr::V4(v4) => classify_blocked_ipv4(v4),
        IpAddr::V6(v6) => classify_blocked_ipv6(v6),
    }
}

/// Classify a URL against the private-network policy and, if blocked, return the
/// host CLASS — never the URL itself.
///
/// This is the shared, secret-safe entry point used by the `browser` tool to
/// build request-level block errors. It applies the SAME policy as
/// [`validate_url_for_ssrf`] (same scheme allow-list, same blocked-host names,
/// same IP ranges, same DNS resolution), so the two tools never diverge.
///
/// Returns `None` when the URL is allowed (public http/https).
pub fn classify_blocked_host(url: &str) -> Option<BlockedHostClass> {
    let parsed = match reqwest::Url::parse(url) {
        Ok(u) => u,
        Err(_) => return Some(BlockedHostClass::Malformed),
    };

    match parsed.scheme() {
        "http" | "https" => {}
        _ => return Some(BlockedHostClass::DisallowedScheme),
    }

    let host = match parsed.host_str() {
        Some(h) => h,
        None => return Some(BlockedHostClass::Malformed),
    };
    let host_lower = host.to_lowercase();

    // Named loopback / metadata hosts (mirrors BLOCKED_HOSTS in validate_url_for_ssrf).
    match host_lower.as_str() {
        "localhost" | "127.0.0.1" | "::1" | "[::1]" | "0.0.0.0" => {
            return Some(BlockedHostClass::Loopback)
        }
        "metadata.google.internal" | "metadata.goog" | "169.254.169.254" => {
            return Some(BlockedHostClass::LinkLocalMetadata)
        }
        _ => {}
    }
    if host_lower.ends_with(".internal") {
        return Some(BlockedHostClass::LinkLocalMetadata);
    }
    if host_lower.ends_with(".local") || host_lower.ends_with(".localhost") {
        // `.localhost` resolves to loopback; `.local` is mDNS link-local-ish but
        // historically grouped with internal names — report as reserved.
        if host_lower.ends_with(".localhost") {
            return Some(BlockedHostClass::Loopback);
        }
        return Some(BlockedHostClass::OtherReserved);
    }

    // Resolve and classify every address the host maps to.
    let port = parsed.port().unwrap_or(match parsed.scheme() {
        "https" => 443,
        _ => 80,
    });
    let socket_addr = format!("{}:{}", host, port);
    match socket_addr.to_socket_addrs() {
        Ok(addrs) => {
            for addr in addrs {
                if let Some(class) = classify_blocked_ip(addr.ip()) {
                    return Some(class);
                }
            }
        }
        Err(_) => {
            // Couldn't resolve via DNS — try parsing the host as a raw IP.
            if let Ok(ip) = host.parse::<IpAddr>() {
                if let Some(class) = classify_blocked_ip(ip) {
                    return Some(class);
                }
            }
            // Unresolvable non-IP host: let the request fail naturally (matches
            // validate_url_for_ssrf). Not classified as blocked here.
        }
    }

    None
}

/// Build an HTTP client with browser-like headers.
/// Shared by WebFetchTool and DuckDuckGo search backend.
fn redirects_allowed_for_execution(mandate_execution: bool) -> bool {
    !mandate_execution
}

fn build_browser_client_with_redirects(follow_redirects: bool) -> Client {
    let redirect_policy = if follow_redirects {
        reqwest::redirect::Policy::custom(|attempt| {
            // Re-validate each redirect hop against SSRF rules
            let url = attempt.url().to_string();
            if let Err(_reason) = validate_url_for_ssrf(&url) {
                attempt.stop()
            } else if attempt.previous().len() >= 10 {
                attempt.stop()
            } else {
                attempt.follow()
            }
        })
    } else {
        reqwest::redirect::Policy::none()
    };

    Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(redirect_policy)
        .user_agent(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:142.0) Gecko/20100101 Firefox/142.0",
        )
        .default_headers({
            let mut h = reqwest::header::HeaderMap::new();
            h.insert(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    .parse()
                    .unwrap(),
            );
            h.insert("Accept-Language", "en-US,en;q=0.5".parse().unwrap());
            h.insert("Accept-Encoding", "gzip, deflate, br".parse().unwrap());
            h.insert("DNT", "1".parse().unwrap());
            h.insert("Upgrade-Insecure-Requests", "1".parse().unwrap());
            h.insert("Sec-Fetch-Dest", "document".parse().unwrap());
            h.insert("Sec-Fetch-Mode", "navigate".parse().unwrap());
            h.insert("Sec-Fetch-Site", "none".parse().unwrap());
            h.insert("Sec-Fetch-User", "?1".parse().unwrap());
            h.insert("Sec-GPC", "1".parse().unwrap());
            h
        })
        .build()
        .expect("failed to build browser HTTP client")
}

pub fn build_browser_client() -> Client {
    build_browser_client_with_redirects(true)
}

pub struct WebFetchTool {
    client: Client,
    mandate_client: Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: build_browser_client(),
            mandate_client: build_browser_client_with_redirects(false),
        }
    }

    /// Core implementation shared by `call()` and `call_with_status_outcome()`.
    /// Returns the formatted output string and truncation info (if the
    /// extracted text was truncated to `max_chars`).
    async fn execute(
        &self,
        arguments: &str,
        mandate_execution: bool,
    ) -> anyhow::Result<(String, Option<crate::traits::TruncationInfo>)> {
        let args: Value = serde_json::from_str(arguments)?;
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: url"))?;
        let max_chars = args["max_chars"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_CHARS)
            .clamp(1, MAX_MAX_CHARS);

        // SSRF protection: validate URL before fetching
        if let Err(reason) = validate_url_for_ssrf(url) {
            return Ok((format!("Request blocked: {}", reason), None));
        }

        let client = if redirects_allowed_for_execution(mandate_execution) {
            &self.client
        } else {
            &self.mandate_client
        };
        let resp = client.get(url).send().await?;
        if !resp.status().is_success() {
            return Ok((
                format!("Error fetching {}: HTTP {}", url, resp.status()),
                None,
            ));
        }
        let (body, _truncated) = read_body_capped(resp, MAX_FETCH_BODY_BYTES).await?;
        let html = String::from_utf8_lossy(&body).into_owned();

        Ok(build_fetch_reply(url, &html, max_chars))
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch a readable web page and extract its content; not for REST/JSON API endpoints"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "web_fetch",
            "description": "Fetch a readable web page and extract its content. Strips ads/navigation. Do NOT use for REST/JSON API endpoints or machine-readable responses; use http_request for APIs. For login-required sites, use browser instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 20000, max 50000)"
                    }
                },
                "required": ["url"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: true,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let url = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| {
                args.get("url")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();

        ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_target_hint(ToolTargetHintKind::Url, url)
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.execute(arguments, false)
            .await
            .map(|(output, _)| output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let _ = status_tx;
        let (output, truncation) = self.execute(arguments, false).await?;
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                truncation,
                ..Default::default()
            },
        })
    }

    async fn call_with_execution_context(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        exec_ctx: ToolExecutionContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let _ = status_tx;
        let (output, truncation) = self.execute(arguments, exec_ctx.mandate_execution).await?;
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                truncation,
                ..Default::default()
            },
        })
    }
}

/// Extracted text shorter than this (on a large page) means extraction
/// failed, not that the page is thin.
const MIN_EXTRACTED_CHARS: usize = 200;
/// Only pages at least this large trigger the extraction-failure check —
/// genuinely tiny pages legitimately yield little text.
const MIN_HTML_FOR_EXTRACTION_CHECK: usize = 10_000;

/// Full fetch-reply pipeline: strip non-visible blocks, extract readable
/// text, detect failed extraction, format with truncation honesty.
fn build_fetch_reply(
    url: &str,
    html: &str,
    max_chars: usize,
) -> (String, Option<crate::traits::TruncationInfo>) {
    let text = extract_page_text(html, url);
    let trimmed = text.trim();
    if trimmed.chars().count() < MIN_EXTRACTED_CHARS && html.len() >= MIN_HTML_FOR_EXTRACTION_CHECK
    {
        // A large page that yields almost no readable text is the signature
        // of a JavaScript-rendered (or bot-walled) page. Without this notice
        // the model treats the empty result as the page's actual content —
        // or worse, re-fetches the same URL expecting a different outcome.
        // This is a distinct extraction-failure signal, not truncation, so it
        // stays an embedded instructional marker (no TruncationInfo applies).
        return (
            format!(
                "Content from {}:\n\n{}\n\n[⚠ EXTRACTION FAILED — this {} KB page yielded only {} \
                 characters of readable text; it is likely JavaScript-rendered or blocking \
                 non-browser clients. Do NOT treat this as the page's content, and do NOT re-fetch \
                 this same URL — the result will not change. Fetch a different source URL from your \
                 search results, or use a browser-based tool on this URL if one is available.]",
                url,
                trimmed,
                html.len() / 1024,
                trimmed.chars().count(),
            ),
            None,
        );
    }
    format_fetch_result(url, &text, max_chars)
}

/// Extract readable page text: readability first, raw-HTML-to-markdown
/// fallback. `<script>`/`<style>`/`<noscript>` blocks are stripped FIRST —
/// both paths can leak inline script bodies into the "text" (observed:
/// Wikipedia fetches returning JS soup that consumed the whole max_chars
/// window before any article content).
fn extract_page_text(html: &str, url: &str) -> String {
    let cleaned = strip_nonvisible_blocks(html);
    let parsed_url = reqwest::Url::parse(url)
        .unwrap_or_else(|_| reqwest::Url::parse("http://example.com").unwrap());
    let mut cursor = Cursor::new(cleaned.as_bytes());
    match llm_readability::extractor::extract(&mut cursor, &parsed_url) {
        Ok(product) if !product.text.trim().is_empty() => product.text,
        _ => htmd::convert(&cleaned).unwrap_or_else(|_| cleaned.clone()),
    }
}

/// Remove `<script>`, `<style>`, and `<noscript>` blocks (case-insensitive,
/// unclosed blocks dropped to end-of-input).
fn strip_nonvisible_blocks(html: &str) -> String {
    let mut out = html.to_string();
    for tag in ["script", "style", "noscript"] {
        out = strip_tag_blocks(&out, tag);
    }
    out
}

fn strip_tag_blocks(html: &str, tag: &str) -> String {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let bytes = html.as_bytes();
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    while let Some(start) = find_ascii_ci(bytes, open.as_bytes(), pos) {
        // The byte after "<tag" must terminate the tag name, else "<scriptx"
        // or a "<style-like" element would match.
        let after = bytes.get(start + open.len());
        if !matches!(
            after,
            Some(b' ') | Some(b'>') | Some(b'\t') | Some(b'\n') | Some(b'/')
        ) {
            out.push_str(&html[pos..start + open.len()]);
            pos = start + open.len();
            continue;
        }
        out.push_str(&html[pos..start]);
        match find_ascii_ci(bytes, close.as_bytes(), start) {
            Some(end) => pos = end + close.len(),
            None => {
                pos = html.len();
                break;
            }
        }
    }
    out.push_str(&html[pos..]);
    out
}

/// ASCII-case-insensitive substring search from `from`. Byte positions are
/// safe to slice with because the needles are pure ASCII.
fn find_ascii_ci(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= haystack.len() || needle.is_empty() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w.eq_ignore_ascii_case(needle))
        .map(|i| i + from)
}

/// Format extracted page text, truncating to `max_chars` (bytes, floored to a
/// char boundary). Truncation is reported via the returned `TruncationInfo`,
/// not embedded in the text — the agent loop renders the instructional
/// notice from that metadata (see `crate::utils::render_truncation_notice`),
/// so text-scanning consumers (classifiers, summaries) always see clean
/// output. A passive "[Truncated]" marker baked into the text is routinely
/// ignored and the model fabricates the omitted content (e.g. inventing the
/// rest of a roster) — this is why the notice must reach the model through
/// the loop's instructional rendering rather than raw tool output.
fn format_fetch_result(
    url: &str,
    text: &str,
    max_chars: usize,
) -> (String, Option<crate::traits::TruncationInfo>) {
    let mut result = format!("Content from {}:\n\n", url);
    if text.len() > max_chars {
        // Find a valid UTF-8 char boundary at or before max_chars
        let mut end = max_chars;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        result.push_str(&text[..end]);
        let truncation = crate::traits::TruncationInfo {
            shown_chars: text[..end].chars().count(),
            total_chars: text.chars().count(),
            remediation_hint: Some(
                "If the answer may be in the omitted part, re-fetch with a larger max_chars \
                 (up to 50000) or fetch a more specific page; if the full content is still \
                 not visible, tell the user the page was longer than you could read."
                    .to_string(),
            ),
        };
        (result, Some(truncation))
    } else {
        result.push_str(text);
        (result, None)
    }
}

#[cfg(test)]
mod ssrf_policy_tests {
    use super::*;

    /// Every URL the shared classifier marks blocked must ALSO be rejected by
    /// `validate_url_for_ssrf`, and the class must match the expected category.
    /// This is the single-source-of-truth invariant: the browser tool's
    /// host-class classifier and the web_fetch validator never diverge.
    fn assert_blocked(url: &str, expected: BlockedHostClass) {
        let class = classify_blocked_host(url);
        assert_eq!(
            class,
            Some(expected),
            "expected {url} blocked as {expected:?}, got {class:?}"
        );
        assert!(
            validate_url_for_ssrf(url).is_err(),
            "validate_url_for_ssrf must also reject {url}"
        );
        // Secret-safety: the label must never echo the URL.
        let label = expected.label();
        assert!(
            !url.contains(label) || label.is_empty(),
            "label must be a fixed class string, not derived from the url"
        );
    }

    fn assert_allowed(url: &str) {
        assert_eq!(
            classify_blocked_host(url),
            None,
            "{url} should be allowed (public)"
        );
    }

    #[test]
    fn loopback_is_blocked() {
        assert_blocked("http://127.0.0.1/", BlockedHostClass::Loopback);
        assert_blocked("http://127.0.0.1:8080/admin", BlockedHostClass::Loopback);
        assert_blocked("http://127.5.6.7/", BlockedHostClass::Loopback); // 127/8
        assert_blocked("http://localhost/", BlockedHostClass::Loopback);
        assert_blocked("http://[::1]/", BlockedHostClass::Loopback);
        assert_blocked("http://0.0.0.0/", BlockedHostClass::Loopback);
    }

    #[test]
    fn rfc1918_private_ranges_are_blocked() {
        assert_blocked("http://10.0.0.1/", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://10.255.255.255/", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://172.16.0.1/", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://172.31.255.255/", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://192.168.1.1/", BlockedHostClass::PrivateNetwork);
        // CGNAT shared space.
        assert_blocked("http://100.64.0.1/", BlockedHostClass::PrivateNetwork);
    }

    #[test]
    fn link_local_and_cloud_metadata_are_blocked() {
        assert_blocked(
            "http://169.254.169.254/latest/meta-data/",
            BlockedHostClass::LinkLocalMetadata,
        );
        assert_blocked("http://169.254.0.1/", BlockedHostClass::LinkLocalMetadata);
        assert_blocked(
            "http://metadata.google.internal/",
            BlockedHostClass::LinkLocalMetadata,
        );
        assert_blocked(
            "http://anything.internal/",
            BlockedHostClass::LinkLocalMetadata,
        );
    }

    #[test]
    fn ipv4_mapped_ipv6_is_blocked_by_embedded_class() {
        // ::ffff:127.0.0.1 → loopback
        assert_blocked("http://[::ffff:127.0.0.1]/", BlockedHostClass::Loopback);
        // ::ffff:10.0.0.1 → private network
        assert_blocked(
            "http://[::ffff:10.0.0.1]/",
            BlockedHostClass::PrivateNetwork,
        );
        // ::ffff:169.254.169.254 → link-local/metadata
        assert_blocked(
            "http://[::ffff:169.254.169.254]/",
            BlockedHostClass::LinkLocalMetadata,
        );
    }

    #[test]
    fn ipv6_unique_local_and_link_local_are_blocked() {
        assert_blocked("http://[fc00::1]/", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://[fd12:3456::1]/", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://[fe80::1]/", BlockedHostClass::LinkLocalMetadata);
    }

    #[test]
    fn disallowed_schemes_are_blocked() {
        assert_blocked("file:///etc/passwd", BlockedHostClass::DisallowedScheme);
        assert_blocked("ftp://example.com/", BlockedHostClass::DisallowedScheme);
        assert_blocked(
            "data:text/html,<h1>hi</h1>",
            BlockedHostClass::DisallowedScheme,
        );
    }

    #[test]
    fn malformed_urls_are_blocked() {
        assert_eq!(
            classify_blocked_host("not a url"),
            Some(BlockedHostClass::Malformed)
        );
        assert_eq!(
            classify_blocked_host("http://"),
            Some(BlockedHostClass::Malformed)
        );
    }

    #[test]
    fn resolve_path_classifies_private_targets() {
        // A raw-IP host with an explicit port exercises the resolve→classify
        // path (to_socket_addrs succeeds and yields the embedded IP). This is
        // the same code path a DNS-rebinding host would hit when it resolves to
        // a private address — deterministic in CI because no real DNS is needed.
        assert_blocked("http://10.1.2.3:8443/x", BlockedHostClass::PrivateNetwork);
        assert_blocked("http://127.0.0.1:9000/x", BlockedHostClass::Loopback);
        // An unresolvable non-IP host is NOT falsely reported as blocked here —
        // the validator lets such a request fail naturally at connect time. This
        // documents that pure DNS-rebinding (public name → private A-record at
        // request time) is only catchable at request time (see browser
        // request-interception deferral), not by this pre-flight check.
        assert_eq!(
            classify_blocked_host("http://nonexistent-host.invalid/"),
            None,
            "unresolvable host is not pre-flight blocked (request-time only)"
        );
    }

    #[test]
    fn public_urls_are_allowed() {
        assert_allowed("https://example.com/");
        assert_allowed("https://www.rust-lang.org/learn");
        assert_allowed("http://93.184.216.34/"); // example.com public IP
        assert_allowed("https://8.8.8.8/");
    }

    #[test]
    fn labels_are_fixed_and_leak_no_data() {
        // Every label is a fixed class string with no caller data.
        for class in [
            BlockedHostClass::Loopback,
            BlockedHostClass::PrivateNetwork,
            BlockedHostClass::LinkLocalMetadata,
            BlockedHostClass::DisallowedScheme,
            BlockedHostClass::Malformed,
            BlockedHostClass::OtherReserved,
        ] {
            let label = class.label();
            assert!(!label.is_empty());
            // No URL syntax, no scheme, no query, no credentials.
            for forbidden in ["://", "?", "=", "@", "127.", "169.254", "secret"] {
                assert!(
                    !label.contains(forbidden),
                    "label {label:?} must not contain {forbidden:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    #[test]
    fn mandate_web_fetch_never_follows_redirects() {
        assert!(redirects_allowed_for_execution(false));
        assert!(!redirects_allowed_for_execution(true));
    }

    #[test]
    fn truncated_fetch_carries_instructional_notice() {
        let text = "a".repeat(120);
        let (out, truncation) = format_fetch_result("https://example.com/page", &text, 50);
        assert!(out.contains("Content from https://example.com/page"));
        // The notice is metadata, not embedded text — text-scanning consumers
        // (classifiers, summaries) must see clean output.
        assert!(!out.contains("OUTPUT TRUNCATED"));
        assert!(!out.contains("[Truncated]"));
        let info = truncation.expect("truncation info");
        assert_eq!(info.total_chars, 120);
        assert_eq!(info.shown_chars, 50);
        // Remediation hint must be fetch-flavored, not terminal-flavored, and
        // preserved verbatim.
        let hint = info.remediation_hint.expect("remediation hint");
        assert!(hint.contains("max_chars"));
        assert!(!hint.contains("wc -l"));
    }

    #[test]
    fn untruncated_fetch_has_no_notice() {
        let (out, truncation) = format_fetch_result("https://example.com", "short content", 1000);
        assert!(out.contains("short content"));
        assert!(!out.contains("OUTPUT TRUNCATED"));
        assert!(truncation.is_none());
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        // 4-byte emoji straddling the cap must not panic.
        let text = format!("{}🦀🦀🦀", "x".repeat(49));
        let (_out, truncation) = format_fetch_result("https://example.com", &text, 51);
        assert!(truncation.is_some());
    }

    #[test]
    fn fetch_result_truncation_is_metadata_not_text() {
        let long = "y".repeat(9000);
        let (text, truncation) = format_fetch_result("https://example.com", &long, 4000);
        assert!(!text.contains("OUTPUT TRUNCATED"));
        let info = truncation.expect("truncation info");
        assert_eq!(info.total_chars, 9000);
        assert!(info
            .remediation_hint
            .as_deref()
            .unwrap_or("")
            .contains("max_chars"));
    }
}

#[cfg(test)]
mod extraction_tests {
    use super::*;

    /// Mimics the observed Wikipedia failure: a large inline <script> block
    /// ahead of the real content leaked into the "extracted" text and
    /// consumed the whole max_chars window.
    fn script_heavy_page() -> String {
        format!(
            "<html><head><title>Ecuador national football team - Wikipedia</title>\
             <script>(function(){{var className=\"client-js vector-feature-language-{}\";}})();</script>\
             <style>.mw-parser{{display:none}}{}</style></head>\
             <body><h1>Squad</h1><p>{}</p>\
             <table><tr><td>Willian Pacho</td></tr><tr><td>Moises Caicedo</td></tr></table>\
             </body></html>",
            "x".repeat(8_000),
            "y".repeat(8_000),
            "The current squad was announced ahead of the tournament. ".repeat(20),
        )
    }

    #[test]
    fn script_and_style_blocks_never_reach_the_model() {
        let (reply, _truncation) = build_fetch_reply(
            "https://en.wikipedia.org/wiki/X",
            &script_heavy_page(),
            2_000,
        );
        assert!(
            !reply.contains("var className"),
            "script body leaked: {}",
            reply
        );
        assert!(
            !reply.contains("display:none"),
            "style body leaked: {}",
            reply
        );
        assert!(reply.contains("Willian Pacho") || reply.contains("current squad"));
    }

    #[test]
    fn near_empty_extraction_gets_instructional_notice() {
        // Big page, no readable text — the JS-rendered-page signature.
        let html = format!(
            "<html><head><script>{}</script></head><body><div id=\"root\"></div></body></html>",
            "z".repeat(20_000)
        );
        let (reply, truncation) = build_fetch_reply("https://spa.example.com", &html, 20_000);
        assert!(
            reply.contains("EXTRACTION FAILED"),
            "missing notice: {}",
            reply
        );
        assert!(reply.contains("different source"));
        // Must not pretend the page was read.
        assert!(!reply.contains("OUTPUT TRUNCATED"));
        assert!(truncation.is_none());
    }

    #[test]
    fn normal_page_has_no_extraction_notice() {
        let (reply, _truncation) =
            build_fetch_reply("https://example.com", &script_heavy_page(), 20_000);
        assert!(!reply.contains("EXTRACTION FAILED"));
    }

    #[test]
    fn small_thin_pages_are_not_flagged() {
        // A genuinely tiny page is not an extraction failure.
        let (reply, _truncation) = build_fetch_reply(
            "https://example.com",
            "<html><body><p>hi</p></body></html>",
            20_000,
        );
        assert!(!reply.contains("EXTRACTION FAILED"));
    }

    #[test]
    fn append_capped_respects_byte_ceiling() {
        // Under the cap: whole chunk appended, not truncated.
        let mut buf = Vec::new();
        assert!(!append_capped(&mut buf, b"hello", 100));
        assert_eq!(buf, b"hello");

        // Exactly at the cap: still appended fully, not truncated.
        let mut buf = vec![0u8; 8];
        assert!(!append_capped(&mut buf, &[1u8, 2u8], 10));
        assert_eq!(buf.len(), 10);

        // Over the cap: partial copy up to the ceiling, truncated reported.
        let mut buf = vec![0u8; 8];
        assert!(append_capped(&mut buf, &[1, 2, 3, 4], 10));
        assert_eq!(buf.len(), 10);

        // Already full: nothing copied, truncation reported.
        let mut buf = vec![0u8; 10];
        assert!(append_capped(&mut buf, b"x", 10));
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn append_capped_truncation_is_utf8_safe() {
        // "€" is 3 bytes (E2 82 AC); capping at 3 keeps "ab" + 1 byte of it.
        // Decoding the capped bytes with from_utf8_lossy must not panic.
        let mut buf = Vec::new();
        assert!(append_capped(&mut buf, "ab€".as_bytes(), 3));
        assert_eq!(buf.len(), 3);
        let s = String::from_utf8_lossy(&buf);
        assert!(s.starts_with("ab"));
    }
}
