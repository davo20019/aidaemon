use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolTargetHintKind, ToolVerificationMode,
};

const DEFAULT_MAX_CHARS: usize = 20_000;
const MAX_MAX_CHARS: usize = 50_000;

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
pub fn build_browser_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            // Re-validate each redirect hop against SSRF rules
            let url = attempt.url().to_string();
            if let Err(_reason) = validate_url_for_ssrf(&url) {
                attempt.stop()
            } else if attempt.previous().len() >= 10 {
                attempt.stop()
            } else {
                attempt.follow()
            }
        }))
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

pub struct WebFetchTool {
    client: Client,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: build_browser_client(),
        }
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
            return Ok(format!("Request blocked: {}", reason));
        }

        let resp = self.client.get(url).send().await?;
        if !resp.status().is_success() {
            return Ok(format!("Error fetching {}: HTTP {}", url, resp.status()));
        }
        let html = resp.text().await?;

        // Try readability extraction first
        let parsed_url = reqwest::Url::parse(url)
            .unwrap_or_else(|_| reqwest::Url::parse("http://example.com").unwrap());
        let text = {
            let mut cursor = Cursor::new(html.as_bytes());
            match llm_readability::extractor::extract(&mut cursor, &parsed_url) {
                Ok(product) if !product.text.trim().is_empty() => product.text,
                _ => {
                    // Fallback: convert raw HTML to markdown
                    htmd::convert(&html).unwrap_or_else(|_| html.clone())
                }
            }
        };

        let mut result = format!("Content from {}:\n\n", url);
        if text.len() > max_chars {
            // Find a valid UTF-8 char boundary at or before max_chars
            let mut end = max_chars;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            result.push_str(&text[..end]);
            result.push_str("\n\n[Truncated]");
        } else {
            result.push_str(&text);
        }

        Ok(result)
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
