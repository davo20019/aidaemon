use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities};

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
        "Fetch a URL and extract its readable content"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "web_fetch",
            "description": "Fetch a URL and extract its readable content. Strips ads/navigation. For login-required sites, use browser instead.",
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
