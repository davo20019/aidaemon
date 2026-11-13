//! `web_fetch`: GET a page and return its readable text.
//!
//! Every request, and every redirect hop, is checked against the
//! private-network policy ([`validate_url_for_ssrf`]). By default redirects
//! stay on the requested host, so the host declared in [`Tool::access`] is
//! the only host contacted; [`WebFetch::follow_public_redirects`] allows
//! other public hosts.

use std::collections::HashMap;
use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::redirect::Policy;
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use serde_json::{json, Value};

use aidaemon_core::{
    AccessManifest, CallEffects, EffectClass, EffectKey, Hook, HookContext, Tool, ToolAnnotations,
    ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult,
};

use crate::fs::ToolText;
use crate::outcome::{spec_of, OutcomeTool};
use crate::semantics::{
    NetworkReach, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolTargetHintKind, ToolVerificationMode,
};
use crate::util::{fail, parse_args, truncate_chars};

const DEFAULT_MAX_CHARS: usize = 20_000;
const MAX_MAX_CHARS: usize = 50_000;

/// Hard cap on response-body bytes buffered from a web fetch/search. The
/// model-facing output is already bounded (extracted-text char limit / result
/// count); this only bounds *transient memory* against a pathologically large
/// or mislabeled response that `.text()`/`.json()` would otherwise buffer whole.
pub const MAX_FETCH_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Append `chunk` to `buf` without exceeding `max_bytes`. Returns true if the
/// chunk was (partially) dropped because the cap was reached. Truncation is on a
/// raw byte boundary, so callers must decode with `from_utf8_lossy`.
pub fn append_capped(buf: &mut Vec<u8>, chunk: &[u8], max_bytes: usize) -> bool {
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
pub async fn read_body_capped(
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

/// Refuses at connection time any name that resolves to a blocked address.
/// The check before a request and the connection look a name up separately,
/// so a name could answer with a public address first and a private one next.
struct PublicOnlyResolver;

impl reqwest::dns::Resolve for PublicOnlyResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<std::net::SocketAddr> =
                tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            if let Some(blocked) = addrs.iter().find(|addr| is_blocked_ip(addr.ip())) {
                let reason = format!("{host} resolves to blocked address {}", blocked.ip());
                return Err(
                    std::io::Error::new(std::io::ErrorKind::PermissionDenied, reason).into(),
                );
            }
            Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs)
        })
    }
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

/// What the owner approves to reach a site on this computer, or `None` when
/// `url` is not loopback-only (public, private network, metadata and malformed
/// targets are never approvable). A dev site's name is its own key;
/// `localhost` and IP literals are shared by every local service, so their key
/// includes the port.
pub fn local_site_key(url: &str) -> Option<String> {
    if !resolves_only_to_loopback(url) {
        return None;
    }
    let parsed = reqwest::Url::parse(url).ok()?;
    let host = parsed.host_str()?.to_lowercase();
    let shared_name = host == "localhost"
        || host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok();
    if shared_name {
        Some(format!("{host}:{}", parsed.port_or_known_default()?))
    } else {
        Some(host)
    }
}

/// Whether `url` targets this computer and nothing else: blocked as loopback,
/// with every address its host resolves to on loopback. A name that also maps
/// to a private or metadata address is not loopback-only.
fn resolves_only_to_loopback(url: &str) -> bool {
    if classify_blocked_host(url) != Some(BlockedHostClass::Loopback) {
        return false;
    }
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = parsed.host_str().map(str::to_lowercase) else {
        return false;
    };
    if matches!(
        host.as_str(),
        "localhost" | "127.0.0.1" | "::1" | "[::1]" | "0.0.0.0"
    ) || host.ends_with(".localhost")
    {
        return true;
    }
    let Some(port) = parsed.port_or_known_default() else {
        return false;
    };
    match format!("{host}:{port}").to_socket_addrs() {
        Ok(addrs) => {
            let mut addrs = addrs.peekable();
            addrs.peek().is_some()
                && addrs
                    .all(|addr| classify_blocked_ip(addr.ip()) == Some(BlockedHostClass::Loopback))
        }
        // A literal IP in a form the resolver refused was classified above.
        Err(_) => host
            .parse::<IpAddr>()
            .is_ok_and(|ip| classify_blocked_ip(ip) == Some(BlockedHostClass::Loopback)),
    }
}

/// Whether `host` is a cloud metadata endpoint: by name, by address, or by
/// what it resolves to. Narrower than [`BlockedHostClass::LinkLocalMetadata`],
/// which also covers ordinary `*.internal` names.
fn is_metadata_endpoint(host: &str, port: u16) -> bool {
    let name = host.trim_end_matches('.').to_ascii_lowercase();
    if matches!(name.as_str(), "metadata.google.internal" | "metadata.goog") {
        return true;
    }
    let is_metadata_ip = |ip: IpAddr| match ip {
        IpAddr::V4(v4) => v4.octets()[..2] == [169, 254],
        IpAddr::V6(v6) => {
            (v6.segments()[0] & 0xffc0) == 0xfe80
                || v6 == Ipv6Addr::new(0xfd00, 0xec2, 0, 0, 0, 0, 0, 0x254)
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| v4.octets()[..2] == [169, 254])
        }
    };
    if let Ok(ip) = name.parse::<IpAddr>() {
        return is_metadata_ip(ip);
    }
    (name.as_str(), port)
        .to_socket_addrs()
        .is_ok_and(|mut addrs| addrs.any(|addr| is_metadata_ip(addr.ip())))
}

/// Why a call to `target` (`host:port`) is refused under `reach`. Names only
/// the class of address, never the target.
fn network_refusal(reach: NetworkReach, target: &str) -> Option<&'static str> {
    let (host, port) = target.rsplit_once(':')?;
    let port: u16 = port.parse().ok()?;
    if is_metadata_endpoint(host, port) {
        return Some("cloud metadata endpoint");
    }
    if reach == NetworkReach::Any {
        return None;
    }
    let authority = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    let class = classify_blocked_host(&format!("http://{authority}/"))?;
    (reach == NetworkReach::PublicOnly || class != BlockedHostClass::Loopback)
        .then(|| class.label())
}

/// Refuses calls whose declared egress names an address the tool may not
/// reach, before anyone is asked. It sees the first request of a call: a
/// redirect exists only once that request runs, so the tools recheck each
/// hop themselves.
#[derive(Debug, Clone, Default)]
pub struct NetworkGuard {
    reach: HashMap<String, NetworkReach>,
}

impl NetworkGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// State how far `tool` may reach. A tool not named reaches
    /// [`NetworkReach::Any`].
    pub fn reach(mut self, tool: impl Into<String>, reach: NetworkReach) -> Self {
        self.reach.insert(tool.into(), reach);
        self
    }
}

#[async_trait]
impl Hook for NetworkGuard {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        let reach = self.reach.get(&call.name).copied().unwrap_or_default();
        let targets: Vec<String> = tool
            .effects
            .keys
            .iter()
            .filter(|key| key.class == EffectClass::Egress)
            .map(|key| key.target.clone())
            .collect();
        if targets.is_empty() {
            return ToolDecision::Allow;
        }
        // Classifying can resolve a name.
        let refusal = tokio::task::spawn_blocking(move || {
            targets
                .iter()
                .find_map(|target| network_refusal(reach, target))
        })
        .await
        .unwrap_or(Some("address that could not be checked"));
        match refusal {
            Some(label) => ToolDecision::Deny {
                reason: format!("Request blocked: target is a {label}"),
            },
            None => ToolDecision::Allow,
        }
    }
}

const BROWSER_USER_AGENT: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:142.0) Gecko/20100101 Firefox/142.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Redirects {
    SameHost,
    Public,
}

/// Which redirects one client follows.
enum Hops {
    None,
    SameHost(String),
    Public,
}

/// Fetch a web page over http(s) and return its readable text.
#[derive(Debug, Clone)]
pub struct WebFetch {
    max_chars: usize,
    timeout: Duration,
    use_proxy: bool,
    user_agent: String,
    browser_headers: bool,
    allow_private: bool,
    redirects: Redirects,
    text: ToolText,
    public_client: OnceLock<Result<Client, String>>,
    no_redirect_client: OnceLock<Result<Client, String>>,
}

impl Default for WebFetch {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_MAX_CHARS,
            timeout: Duration::from_secs(30),
            use_proxy: true,
            user_agent: concat!("aidaemon-tools/", env!("CARGO_PKG_VERSION")).into(),
            browser_headers: false,
            allow_private: false,
            redirects: Redirects::SameHost,
            text: ToolText {
                short: "Fetch a web page and return its readable text".into(),
                schema: "Fetch a URL with HTTP GET (http/https only) and return the readable text \
                         of the page; plain text and JSON are returned as is. Private and \
                         internal addresses are refused. Redirects to a different host are not \
                         followed; the new location is reported instead."
                    .into(),
            },
            public_client: OnceLock::new(),
            no_redirect_client: OnceLock::new(),
        }
    }
}

impl WebFetch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Default characters of text returned to the model.
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

    /// Send a browser's user agent and headers, for sites that refuse other
    /// clients.
    pub fn browser_identity(mut self) -> Self {
        self.user_agent = BROWSER_USER_AGENT.into();
        self.browser_headers = true;
        self
    }

    /// Follow redirects to other public hosts as well. Each hop is still
    /// checked against the private-network policy.
    pub fn follow_public_redirects(mut self) -> Self {
        self.redirects = Redirects::Public;
        self
    }

    /// Turn off the private-network policy, for a tool that is meant to reach
    /// local services.
    pub fn allow_private_hosts(mut self) -> Self {
        self.allow_private = true;
        self
    }

    /// Replace the model-facing descriptions.
    pub fn with_text(mut self, text: ToolText) -> Self {
        self.text = text;
        self
    }

    fn build_client(&self, hops: Hops) -> Result<Client, String> {
        let guarded = !self.allow_private;
        let allowed = move |url: &str| !guarded || validate_url_for_ssrf(url).is_ok();
        let policy = match hops {
            Hops::None => Policy::none(),
            Hops::Public => Policy::custom(move |attempt| {
                if !allowed(attempt.url().as_str()) || attempt.previous().len() >= 10 {
                    attempt.stop()
                } else {
                    attempt.follow()
                }
            }),
            Hops::SameHost(host) => Policy::custom(move |attempt| {
                if attempt.previous().len() >= 10 {
                    attempt.error("too many redirects")
                } else if attempt.url().host_str() == Some(host.as_str())
                    && matches!(attempt.url().scheme(), "http" | "https")
                    && allowed(attempt.url().as_str())
                {
                    attempt.follow()
                } else {
                    attempt.stop()
                }
            }),
        };
        let mut builder = Client::builder()
            .timeout(self.timeout)
            .redirect(policy)
            .user_agent(&self.user_agent);
        if guarded {
            builder = builder.dns_resolver(std::sync::Arc::new(PublicOnlyResolver));
        }
        if self.browser_headers {
            builder = builder.default_headers(browser_headers());
        }
        if !self.use_proxy {
            builder = builder.no_proxy();
        }
        builder
            .build()
            .map_err(|error| format!("cannot build HTTP client: {error}"))
    }

    fn client_for(&self, url: &str, follow: bool) -> anyhow::Result<Client> {
        let cached = |cell: &OnceLock<Result<Client, String>>, hops| {
            cell.get_or_init(|| self.build_client(hops)).clone()
        };
        let client = match (follow, self.redirects) {
            (false, _) => cached(&self.no_redirect_client, Hops::None),
            (true, Redirects::Public) => cached(&self.public_client, Hops::Public),
            (true, Redirects::SameHost) => {
                let host = Url::parse(url)?.host_str().unwrap_or_default().to_string();
                self.build_client(Hops::SameHost(host))
            }
        };
        client.map_err(anyhow::Error::msg)
    }

    /// Check the policy, send the request and read the capped body.
    async fn fetch(&self, url: &str, follow: bool, read_error_body: bool) -> anyhow::Result<Fetch> {
        if !self.allow_private {
            if let Err(reason) = validate_url_for_ssrf(url) {
                return Ok(Fetch::Blocked(reason));
            }
        }
        let response = self.client_for(url, follow)?.get(url).send().await?;
        let status = response.status();
        let final_url = response.url().to_string();
        let header = |name| {
            response
                .headers()
                .get(name)
                .and_then(|value: &reqwest::header::HeaderValue| value.to_str().ok())
                .map(str::to_string)
        };
        let content_type = header(reqwest::header::CONTENT_TYPE).unwrap_or_default();
        let location = header(reqwest::header::LOCATION);
        let (body, body_truncated) = if status.is_success() || read_error_body {
            read_body_capped(response, MAX_FETCH_BODY_BYTES).await?
        } else {
            (Vec::new(), false)
        };
        Ok(Fetch::Response(Fetched {
            status,
            final_url,
            content_type,
            location,
            body,
            body_truncated,
        }))
    }
}

enum Fetch {
    Blocked(String),
    Response(Fetched),
}

struct Fetched {
    status: StatusCode,
    final_url: String,
    content_type: String,
    location: Option<String>,
    body: Vec<u8>,
    body_truncated: bool,
}

fn browser_headers() -> reqwest::header::HeaderMap {
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
}

/// An HTTP client with browser-like headers whose redirects are re-validated
/// against the private-network policy.
pub fn build_browser_client() -> Client {
    WebFetch::new()
        .browser_identity()
        .build_client(Hops::Public)
        .expect("failed to build browser HTTP client")
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

/// The one effect of fetching `url`.
fn url_effect(url: &Url) -> Option<EffectKey> {
    // `Url` brackets an IPv6 host; the effect target does not.
    let host = url
        .host_str()?
        .trim_start_matches('[')
        .trim_end_matches(']');
    Some(EffectKey::egress(url.scheme(), host, url.port()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
    url: String,
    max_chars: Option<usize>,
}

#[async_trait]
impl OutcomeTool for WebFetch {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        &self.text.short
    }

    fn schema(&self) -> Value {
        json!({
            "name": "web_fetch",
            "description": self.text.schema,
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

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Value = serde_json::from_str(arguments)?;
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: url"))?;
        let max_chars = args["max_chars"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(self.max_chars)
            .clamp(1, MAX_MAX_CHARS);

        let fetched = match self.fetch(url, !facts.mandate_execution, false).await? {
            Fetch::Blocked(reason) => {
                return Ok(ToolCallOutcome::from_output(format!(
                    "Request blocked: {reason}"
                )))
            }
            Fetch::Response(fetched) => fetched,
        };
        if !fetched.status.is_success() {
            return Ok(ToolCallOutcome::from_output(format!(
                "Error fetching {}: HTTP {}",
                url, fetched.status
            )));
        }
        let html = String::from_utf8_lossy(&fetched.body).into_owned();
        let (output, truncation) = build_fetch_reply(url, &html, max_chars);
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                truncation,
                ..Default::default()
            },
            finish: None,
        })
    }
}

#[async_trait]
impl Tool for WebFetch {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec_of(self)
    }

    /// A read, but the request (its URL, query included) leaves the
    /// machine, so outbound policies such as an egress secret scan see it.
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            egress: true,
            ..ToolAnnotations::READ_ONLY
        }
    }

    fn outbound(&self, arguments: &Value) -> aidaemon_core::Outbound {
        outbound_of_url(arguments)
    }

    /// The host contacted. An unparsable URL is refused by `validate` before
    /// any policy runs; its effects are unknown.
    fn effects(&self, arguments: &Value) -> CallEffects {
        arguments
            .get("url")
            .and_then(Value::as_str)
            .and_then(|url| parse_url(url).ok())
            .and_then(|url| url_effect(&url))
            .map_or_else(CallEffects::unknown, |key| CallEffects {
                keys: vec![key],
                complete: true,
                understood: false,
            })
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
        let max_chars = args
            .max_chars
            .unwrap_or(self.max_chars)
            .clamp(1, MAX_MAX_CHARS.max(self.max_chars));
        let host = url.host_str().unwrap_or_default().to_string();
        let result = tokio::select! {
            result = self.fetch(url.as_str(), true, true) => result,
            _ = ctx.cancel.cancelled() => return Ok(fail("cancelled")),
        };
        let fetched = match result {
            Ok(Fetch::Blocked(reason)) => {
                return Ok(fail(format!("Request blocked: {reason}"))
                    .with_details(json!({"url": url.as_str(), "host": host})))
            }
            Ok(Fetch::Response(fetched)) => fetched,
            Err(e) => {
                return Ok(fail(format!("request to {url} failed: {e}"))
                    .with_details(json!({"url": url.as_str(), "host": host})))
            }
        };
        let Fetched {
            status,
            final_url,
            content_type,
            location,
            body,
            body_truncated,
        } = fetched;

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
                "HTTP {status}: {final_url} redirects to {target}, which was not followed. \
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
            extract_page_text(&raw, &final_url)
        } else {
            raw.into_owned()
        };

        if !status.is_success() {
            let (snippet, _) = truncate_chars(text.trim(), 2000);
            return Ok(
                fail(format!("HTTP {status} from {final_url}.\n\n{snippet}")).with_details(details),
            );
        }

        let (mut text, cut) = truncate_chars(text.trim(), max_chars);
        if cut || body_truncated {
            text.push_str(&format!(
                "\n\n[Content truncated to {max_chars} characters.]"
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

/// A fetch sends its URL and nothing else.
fn outbound_of_url(arguments: &Value) -> aidaemon_core::Outbound {
    match arguments.get("url").and_then(Value::as_str) {
        Some(url) => aidaemon_core::Outbound::Exact(vec![url.to_string()]),
        None => aidaemon_core::Outbound::Composed,
    }
}

impl crate::semantics::ToolSemantics for WebFetch {
    fn call_outbound(&self, arguments: &str) -> aidaemon_core::Outbound {
        serde_json::from_str::<Value>(arguments).map_or(aidaemon_core::Outbound::Composed, |args| {
            outbound_of_url(&args)
        })
    }

    fn dispatch_facts(&self) -> crate::semantics::ToolDispatchFacts {
        crate::semantics::ToolDispatchFacts {
            always_offered: true,
            network_reach: crate::semantics::NetworkReach::PublicOnly,
            same_tool_streak: crate::semantics::SameToolStreak::Versatile,
            ..Default::default()
        }
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
            .with_operation(crate::semantics::ToolCallOperation::Get)
            .with_verification_mode(ToolVerificationMode::ResultContent)
            .with_target_hint(ToolTargetHintKind::Url, url)
    }

    /// The page's host.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| url_effect(&parse_url(args.get("url")?.as_str()?).ok()?))
            .map_or_else(aidaemon_core::CallEffects::unknown, |key| {
                aidaemon_core::CallEffects {
                    keys: vec![key],
                    complete: true,
                    understood: false,
                }
            })
    }
}

fn looks_like_html(body: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&body[..body.len().min(512)]).to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html")
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
) -> (String, Option<crate::semantics::TruncationInfo>) {
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
    let full_page = || htmd::convert(&cleaned).unwrap_or_else(|_| cleaned.clone());
    match llm_readability::extractor::extract(&mut cursor, &parsed_url) {
        Ok(product) if !product.text.trim().is_empty() => {
            // Readability keeps one block. On an article that is the content;
            // on a listing page (downloads, release lists, tables) it is a
            // side section, and the answer is in what it dropped. A short
            // extract cut from a much larger page gets the whole page.
            let article_chars = product.text.trim().chars().count();
            if article_chars >= MIN_ARTICLE_CHARS {
                return product.text;
            }
            let full = full_page();
            let full_chars = full.trim().chars().count();
            if (article_chars as f64) >= MIN_ARTICLE_COVERAGE * full_chars as f64 {
                product.text
            } else {
                full
            }
        }
        _ => full_page(),
    }
}

/// Readability output at least this long is an article in its own right.
const MIN_ARTICLE_CHARS: usize = 8_000;
/// A shorter extract must cover at least this share of the page's text to
/// count as its main content.
const MIN_ARTICLE_COVERAGE: f64 = 0.35;

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
) -> (String, Option<crate::semantics::TruncationInfo>) {
    let mut result = format!("Content from {}:\n\n", url);
    if text.len() > max_chars {
        // Find a valid UTF-8 char boundary at or before max_chars
        let mut end = max_chars;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        result.push_str(&text[..end]);
        let truncation = crate::semantics::TruncationInfo {
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_name_resolving_to_a_blocked_address_is_refused_at_connection() {
        use reqwest::dns::Resolve;
        use std::str::FromStr;
        let name = reqwest::dns::Name::from_str("localhost").unwrap();
        let error = match PublicOnlyResolver.resolve(name).await {
            Ok(_) => panic!("localhost must not resolve"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("blocked address"), "{error}");
    }
    use crate::testutil::ctx;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn web_fetch_declares_its_egress_host() {
        let tool = WebFetch::new();
        assert!(tool.annotations().egress);
        assert!(tool.annotations().read_only);
        let effects = tool.effects(&json!({"url": "https://Docs.Example.Invalid/guide?q=1"}));
        assert!(effects.complete);
        assert_eq!(
            effects.keys,
            vec![EffectKey::egress("https", "docs.example.invalid", None)]
        );
        let local = tool.effects(&json!({"url": "http://[::1]:8765/status"}));
        assert_eq!(
            local.keys,
            vec![EffectKey::egress("http", "::1", Some(8765))]
        );
        assert!(!tool.effects(&json!({"url": "not a url"})).complete);
    }

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
        let tool = WebFetch::new()
            .without_proxy()
            .allow_private_hosts()
            .max_chars(1000);

        let url = format!("{base}/page");
        let out = tool.call(json!({"url": url}), &ctx()).await.unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert!(out.content.contains("Hello"), "{}", out.content);
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

    #[tokio::test]
    async fn private_hosts_are_blocked_by_default() {
        let base = serve(vec![("/page", "200 OK", "text/plain", "private".into())]).await;
        let tool = WebFetch::new().without_proxy();
        let out = tool
            .call(json!({"url": format!("{base}/page")}), &ctx())
            .await
            .unwrap();
        assert!(
            out.is_error && out.content.contains("Request blocked"),
            "{}",
            out.content
        );
        assert!(!out.content.contains("private\n"));
    }

    #[tokio::test]
    async fn a_cross_host_redirect_is_reported_not_followed() {
        let base = serve(vec![(
            "/away",
            "302 Found",
            "text/plain",
            "http://localhost:9/landing".into(),
        )])
        .await;
        let tool = WebFetch::new().without_proxy().allow_private_hosts();
        let out = tool
            .call(json!({"url": format!("{base}/away")}), &ctx())
            .await
            .unwrap();
        assert!(
            out.is_error && out.content.contains("localhost:9"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn mandate_execution_never_follows_a_redirect() {
        let base = serve(vec![
            ("/away", "302 Found", "text/plain", "/landing".into()),
            ("/landing", "200 OK", "text/html", "<p>arrived</p>".into()),
        ])
        .await;
        let tool = WebFetch::new()
            .without_proxy()
            .allow_private_hosts()
            .follow_public_redirects();
        let arguments = json!({"url": format!("{base}/away")}).to_string();

        let followed = tool
            .execute(&arguments, &ToolCallContext::default())
            .await
            .unwrap();
        assert!(followed.output.contains("arrived"), "{}", followed.output);

        let facts = ToolCallContext {
            mandate_execution: true,
            ..ToolCallContext::default()
        };
        let stopped = tool.execute(&arguments, &facts).await.unwrap();
        assert!(stopped.output.contains("HTTP 302"), "{}", stopped.output);
    }

    #[test]
    fn metadata_endpoints_are_refused_for_every_reach() {
        for target in [
            "169.254.169.254:80",
            "metadata.google.internal:80",
            "metadata.goog:443",
            "fe80::1:80",
            "fd00:ec2::254:80",
        ] {
            for reach in [
                NetworkReach::Any,
                NetworkReach::PublicAndLoopback,
                NetworkReach::PublicOnly,
            ] {
                assert!(
                    network_refusal(reach, target).is_some(),
                    "{target} {reach:?}"
                );
            }
        }
    }

    #[test]
    fn reach_decides_private_and_loopback() {
        use NetworkReach::*;
        // A shell still reaches the owner's own network and internal names.
        for target in ["10.0.0.5:22", "localhost:3000", "db.internal:5432"] {
            assert_eq!(network_refusal(Any, target), None, "{target}");
        }
        assert_eq!(network_refusal(PublicAndLoopback, "localhost:3000"), None);
        assert_eq!(
            network_refusal(PublicAndLoopback, "shop.localhost:80"),
            None
        );
        assert!(network_refusal(PublicAndLoopback, "10.0.0.5:80").is_some());
        assert!(network_refusal(PublicOnly, "localhost:3000").is_some());
        assert!(network_refusal(PublicOnly, "::1:80").is_some());
        assert_eq!(network_refusal(PublicOnly, "8.8.8.8:443"), None);
    }

    #[test]
    fn a_refusal_names_the_class_not_the_target() {
        let label = network_refusal(NetworkReach::PublicOnly, "10.1.2.3:8443").unwrap();
        assert!(!label.contains("10.1.2.3"));
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

    /// A synthetic download page: a short release list, one long prose
    /// section and a lot of navigation, shaped like real listing pages.
    fn listing_page() -> String {
        let nav: String = (0..400)
            .map(|i| format!("<li><a href=\"/nav/{i}\">Section {i} overview</a></li>"))
            .collect();
        let prose: String = (0..12)
            .map(|i| {
                format!(
                    "<p>Paragraph {i} explains how release artifacts are signed and how a \
                     careful user can check that a downloaded installer is genuine before \
                     running it on their machine, step by step.</p>"
                )
            })
            .collect();
        format!(
            "<html><body><nav><ul>{nav}</ul></nav>\
             <div class=\"releases\"><h2>Downloads</h2><ul>\
             <li><a href=\"/r/9.4.2\">Download Acme 9.4.2</a></li>\
             <li><a href=\"/r/9.4.1\">Acme 9.4.1</a></li></ul></div>\
             <div class=\"article\"><h2>Verifying downloads</h2>{prose}</div>\
             </body></html>"
        )
    }

    #[test]
    fn listing_page_keeps_content_readability_would_drop() {
        let text = extract_page_text(&listing_page(), "https://acme.example/downloads/");
        assert!(
            text.contains("Download Acme 9.4.2"),
            "{}",
            &text[..text.len().min(400)]
        );
    }

    #[test]
    fn long_article_still_uses_readability() {
        let nav: String = (0..200)
            .map(|i| format!("<li><a href=\"/nav/{i}\">Menu entry {i}</a></li>"))
            .collect();
        let body: String = (0..120)
            .map(|i| {
                format!(
                    "<p>Paragraph {i} of a long synthetic article about mountain cities, \
                     their climate, history and the people who live there today.</p>"
                )
            })
            .collect();
        let html = format!(
            "<html><body><nav><ul>{nav}</ul></nav><article><h1>Mountain cities</h1>{body}</article></body></html>"
        );
        let text = extract_page_text(&html, "https://wiki.example/mountain");
        assert!(text.contains("Paragraph 119"));
        assert!(
            !text.contains("Menu entry 150"),
            "navigation leaked into an article"
        );
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
    use crate::semantics::ToolSemantics;

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

    #[test]
    fn web_fetch_declares_egress_host() {
        let tool = WebFetch::new();
        let effects =
            tool.call_effects(&json!({"url": "https://Docs.Example.Invalid/guide"}).to_string());
        assert!(effects.complete);
        assert_eq!(
            effects.keys,
            vec![aidaemon_core::EffectKey::egress(
                "https",
                "docs.example.invalid",
                None
            )]
        );
        let bad = tool.call_effects(&json!({"url": "::"}).to_string());
        assert!(!bad.complete);
    }
}
