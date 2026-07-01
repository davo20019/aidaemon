use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::config::{SearchBackendKind, SearchConfig};
use crate::traits::{Tool, ToolCapabilities};

use super::web_fetch::{build_browser_client, read_body_capped, MAX_FETCH_BODY_BYTES};

const DEFAULT_MAX_RESULTS: usize = 8;
const MAX_MAX_RESULTS: usize = 20;
/// Per-backend wall-clock budget during parallel fan-out; a slow backend is
/// dropped rather than stalling the whole search.
const BACKEND_TIMEOUT: Duration = Duration::from_secs(12);
/// Reciprocal-rank-fusion constant (standard value from the RRF literature).
const RRF_K: f64 = 60.0;

// ---------------------------------------------------------------------------
// SearchBackend trait + result type
// ---------------------------------------------------------------------------

pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    /// Publication age/date when the backend provides one (e.g. "2 days ago").
    pub age: Option<String>,
}

/// Recency window for time-sensitive queries, mapped to each backend's
/// native freshness parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    Day,
    Week,
    Month,
    Year,
}

impl Freshness {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "day" | "d" => Some(Self::Day),
            "week" | "w" => Some(Self::Week),
            "month" | "m" => Some(Self::Month),
            "year" | "y" => Some(Self::Year),
            _ => None,
        }
    }

    fn ddg_param(self) -> &'static str {
        match self {
            Self::Day => "d",
            Self::Week => "w",
            Self::Month => "m",
            Self::Year => "y",
        }
    }

    fn brave_param(self) -> &'static str {
        match self {
            Self::Day => "pd",
            Self::Week => "pw",
            Self::Month => "pm",
            Self::Year => "py",
        }
    }

    fn searxng_param(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Week => "week",
            Self::Month => "month",
            Self::Year => "year",
        }
    }
}

#[async_trait]
pub trait SearchBackend: Send + Sync {
    fn name(&self) -> &'static str;
    async fn search(
        &self,
        query: &str,
        max_results: usize,
        freshness: Option<Freshness>,
    ) -> anyhow::Result<Vec<SearchResult>>;
}

/// Build a single backend of the given kind from config.
pub fn build_backend(kind: &SearchBackendKind, config: &SearchConfig) -> Box<dyn SearchBackend> {
    match kind {
        SearchBackendKind::Brave => Box::new(BraveBackend::new(&config.api_key)),
        SearchBackendKind::DuckDuckGo => Box::new(DuckDuckGoBackend::new()),
        SearchBackendKind::Searxng => Box::new(SearxngBackend::new(&config.searxng_url)),
    }
}

/// All backends that should serve a search: the configured primary first,
/// then (when `parallel` is enabled) every other backend that has the
/// credentials/endpoint it needs. DuckDuckGo needs nothing, so it is always
/// available as a supplementary source.
fn configured_backends(config: &SearchConfig) -> Vec<Box<dyn SearchBackend>> {
    let mut kinds = vec![config.backend.clone()];
    if config.parallel {
        for kind in [
            SearchBackendKind::Brave,
            SearchBackendKind::Searxng,
            SearchBackendKind::DuckDuckGo,
        ] {
            if kinds.contains(&kind) {
                continue;
            }
            let available = match kind {
                SearchBackendKind::Brave => !config.api_key.is_empty(),
                SearchBackendKind::Searxng => !config.searxng_url.is_empty(),
                SearchBackendKind::DuckDuckGo => true,
            };
            if available {
                kinds.push(kind);
            }
        }
    }
    kinds.iter().map(|k| build_backend(k, config)).collect()
}

// ---------------------------------------------------------------------------
// DuckDuckGo backend (default, no API key)
// ---------------------------------------------------------------------------

pub struct DuckDuckGoBackend {
    client: Client,
}

impl DuckDuckGoBackend {
    pub fn new() -> Self {
        Self {
            client: build_browser_client(),
        }
    }
}

#[async_trait]
impl SearchBackend for DuckDuckGoBackend {
    fn name(&self) -> &'static str {
        "duckduckgo"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        freshness: Option<Freshness>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let mut params = vec![("q", query.to_string())];
        if let Some(f) = freshness {
            params.push(("df", f.ddg_param().to_string()));
        }
        let url = reqwest::Url::parse_with_params(
            "https://lite.duckduckgo.com/lite/",
            params.iter().map(|(k, v)| (*k, v.as_str())),
        )?
        .to_string();
        let resp = self.client.get(&url).send().await?;
        let status = resp.status();
        let (body, _truncated) = read_body_capped(resp, MAX_FETCH_BODY_BYTES).await?;
        let html = String::from_utf8_lossy(&body).into_owned();

        if !status.is_success() {
            anyhow::bail!("DuckDuckGo returned HTTP {}", status);
        }
        if is_ddg_challenge(&html) {
            anyhow::bail!("DuckDuckGo blocked the request with a bot challenge");
        }

        Ok(parse_ddg_lite_html(&html, max_results))
    }
}

/// Detect DuckDuckGo's anti-bot challenge page, which returns HTTP 200 with
/// no results. Without this check a block is indistinguishable from a query
/// that genuinely has no matches.
fn is_ddg_challenge(html: &str) -> bool {
    html.contains("anomaly-modal")
        || html.contains("challenge-form")
        || html.contains("Unfortunately, bots use DuckDuckGo too")
}

/// Parse the DuckDuckGo lite HTML page into results.
///
/// DuckDuckGo lite returns a simple HTML page with results in a table.
/// Each result has: a link (<a> tag with href) and snippet text.
/// We parse it with simple string scanning since it's a minimal page.
fn parse_ddg_lite_html(html: &str, max_results: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    // Find all result links: <a rel="nofollow" href="..." class="result-link">Title</a>
    // and their snippets in <td class="result-snippet">...</td>
    let mut pos = 0;
    while results.len() < max_results {
        // Find next result link
        let link_start = match html[pos..].find("class=\"result-link\"") {
            Some(p) => pos + p,
            None => break,
        };

        // Extract href from the <a> tag — scan backward for href="
        let tag_start = html[..link_start].rfind("<a ").unwrap_or(link_start);
        let href = extract_attr(&html[tag_start..], "href").unwrap_or_default();

        // Extract title (text between > and </a>)
        let title_start = match html[link_start..].find('>') {
            Some(p) => link_start + p + 1,
            None => {
                pos = link_start + 20;
                continue;
            }
        };
        let title_end = match html[title_start..].find("</a>") {
            Some(p) => title_start + p,
            None => {
                pos = title_start;
                continue;
            }
        };
        let title = strip_tags(&html[title_start..title_end]);

        // Find snippet after this link
        let snippet = if let Some(sn_pos) = html[title_end..].find("class=\"result-snippet\"") {
            let sn_start = title_end + sn_pos;
            let sn_content_start = match html[sn_start..].find('>') {
                Some(p) => sn_start + p + 1,
                None => sn_start,
            };
            let sn_end = match html[sn_content_start..].find("</td>") {
                Some(p) => sn_content_start + p,
                None => sn_content_start,
            };
            strip_tags(&html[sn_content_start..sn_end])
                .trim()
                .to_string()
        } else {
            String::new()
        };

        if !href.is_empty() && !title.is_empty() {
            results.push(SearchResult {
                title,
                url: decode_ddg_redirect(&href),
                snippet,
                age: None,
            });
        }

        pos = title_end + 1;
    }

    results
}

/// DuckDuckGo lite wraps result URLs in a tracking redirect:
/// `//duckduckgo.com/l/?uddg=<percent-encoded target>&rut=...`.
/// Unwrap it to the real destination so the LLM (and web_fetch) get a
/// usable URL instead of a protocol-relative redirect.
fn decode_ddg_redirect(href: &str) -> String {
    let absolute = if let Some(rest) = href.strip_prefix("//") {
        format!("https://{}", rest)
    } else if href.starts_with('/') {
        format!("https://duckduckgo.com{}", href)
    } else {
        href.to_string()
    };

    if let Ok(url) = reqwest::Url::parse(&absolute) {
        let is_ddg_redirect = url
            .host_str()
            .is_some_and(|h| h == "duckduckgo.com" || h.ends_with(".duckduckgo.com"))
            && url.path() == "/l/";
        if is_ddg_redirect {
            if let Some((_, target)) = url.query_pairs().find(|(k, _)| k == "uddg") {
                return target.into_owned();
            }
        }
        return absolute;
    }

    href.to_string()
}

/// Extract an attribute value from an HTML tag fragment.
fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr);
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(html_decode(&tag[start..end]))
}

/// Strip HTML tags from a string.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    html_decode(&out)
}

/// Decode common HTML entities.
fn html_decode(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

// ---------------------------------------------------------------------------
// Brave backend (API key required)
// ---------------------------------------------------------------------------

pub struct BraveBackend {
    client: Client,
    api_key: String,
}

impl BraveBackend {
    pub fn new(api_key: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl SearchBackend for BraveBackend {
    fn name(&self) -> &'static str {
        "brave"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        freshness: Option<Freshness>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let mut params = vec![("q", query.to_string()), ("count", max_results.to_string())];
        if let Some(f) = freshness {
            params.push(("freshness", f.brave_param().to_string()));
        }
        let url = reqwest::Url::parse_with_params(
            "https://api.search.brave.com/res/v1/web/search",
            params.iter().map(|(k, v)| (*k, v.as_str())),
        )?
        .to_string();

        // Retry with exponential backoff for rate limiting (429)
        let max_retries = 3;
        let mut last_status = reqwest::StatusCode::OK;

        for attempt in 0..max_retries {
            let resp = self
                .client
                .get(&url)
                .header("X-Subscription-Token", &self.api_key)
                .header("Accept", "application/json")
                .send()
                .await?;

            last_status = resp.status();

            if resp.status().is_success() {
                let (body, _truncated) = read_body_capped(resp, MAX_FETCH_BODY_BYTES).await?;
                let data: Value = serde_json::from_slice(&body)?;
                let empty = vec![];
                let web_results = data["web"]["results"].as_array().unwrap_or(&empty);

                let results = web_results
                    .iter()
                    .take(max_results)
                    .filter_map(|r| {
                        let mut snippet = r["description"].as_str().unwrap_or("").to_string();
                        // Paid plans return extra_snippets with more page context;
                        // fold a couple in so the LLM sees more than one sentence.
                        if let Some(extra) = r["extra_snippets"].as_array() {
                            for s in extra.iter().take(2).filter_map(|s| s.as_str()) {
                                if !snippet.is_empty() {
                                    snippet.push_str(" … ");
                                }
                                snippet.push_str(s);
                            }
                        }
                        Some(SearchResult {
                            title: r["title"].as_str()?.to_string(),
                            url: r["url"].as_str()?.to_string(),
                            snippet,
                            age: r["age"]
                                .as_str()
                                .or_else(|| r["page_age"].as_str())
                                .map(|s| s.to_string()),
                        })
                    })
                    .collect();

                return Ok(results);
            }

            // Retry on 429 (rate limited) with exponential backoff
            if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && attempt < max_retries - 1
            {
                let delay = crate::backoff::exponential_backoff(
                    Duration::from_secs(1),
                    attempt as u32,
                    Duration::MAX,
                ); // 1s, 2s, 4s
                tokio::time::sleep(delay).await;
                continue;
            }

            // Non-retryable error or exhausted retries
            break;
        }

        anyhow::bail!("Brave search API error: HTTP {}", last_status)
    }
}

// ---------------------------------------------------------------------------
// SearxNG backend (self-hosted metasearch, no API key)
// ---------------------------------------------------------------------------

pub struct SearxngBackend {
    client: Client,
    base_url: String,
}

impl SearxngBackend {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }
}

#[async_trait]
impl SearchBackend for SearxngBackend {
    fn name(&self) -> &'static str {
        "searxng"
    }

    async fn search(
        &self,
        query: &str,
        max_results: usize,
        freshness: Option<Freshness>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        anyhow::ensure!(
            !self.base_url.is_empty(),
            "SearxNG backend selected but search.searxng_url is not configured"
        );
        let mut params = vec![("q", query.to_string()), ("format", "json".to_string())];
        if let Some(f) = freshness {
            params.push(("time_range", f.searxng_param().to_string()));
        }
        let url = reqwest::Url::parse_with_params(
            &format!("{}/search", self.base_url),
            params.iter().map(|(k, v)| (*k, v.as_str())),
        )?;

        let resp = self.client.get(url).send().await?;
        anyhow::ensure!(
            resp.status().is_success(),
            "SearxNG returned HTTP {} (ensure 'json' is in search.formats in the SearxNG settings)",
            resp.status()
        );
        let (body, _truncated) = read_body_capped(resp, MAX_FETCH_BODY_BYTES).await?;
        let data: Value = serde_json::from_slice(&body)?;
        let empty = vec![];
        let results = data["results"]
            .as_array()
            .unwrap_or(&empty)
            .iter()
            .take(max_results)
            .filter_map(|r| {
                Some(SearchResult {
                    title: r["title"].as_str()?.to_string(),
                    url: r["url"].as_str()?.to_string(),
                    snippet: r["content"].as_str().unwrap_or("").to_string(),
                    age: r["publishedDate"]
                        .as_str()
                        .and_then(|s| s.split('T').next())
                        .map(|s| s.to_string()),
                })
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Result merging (reciprocal rank fusion across backends)
// ---------------------------------------------------------------------------

struct MergedResult {
    title: String,
    url: String,
    snippet: String,
    age: Option<String>,
    sources: Vec<&'static str>,
    score: f64,
}

/// Normalize a URL for cross-backend deduplication: scheme-insensitive,
/// fragment stripped, trailing slash stripped.
fn normalize_url(url: &str) -> String {
    if let Ok(mut u) = reqwest::Url::parse(url) {
        u.set_fragment(None);
        let s = u.to_string();
        let s = s
            .strip_prefix("https://")
            .or_else(|| s.strip_prefix("http://"))
            .unwrap_or(&s);
        return s.trim_end_matches('/').to_ascii_lowercase();
    }
    url.trim_end_matches('/').to_ascii_lowercase()
}

/// Merge ranked result lists from multiple backends using reciprocal rank
/// fusion: each result scores sum(1 / (k + rank)) across the backends that
/// returned it, so results confirmed by several sources rank highest. Ties
/// keep primary-backend order (stable sort, primary list is inserted first).
fn merge_ranked(
    lists: &[(&'static str, Vec<SearchResult>)],
    max_results: usize,
) -> Vec<MergedResult> {
    let mut map: HashMap<String, MergedResult> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for (backend, results) in lists {
        for (rank, r) in results.iter().enumerate() {
            let key = normalize_url(&r.url);
            let entry = map.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                MergedResult {
                    title: r.title.clone(),
                    url: r.url.clone(),
                    snippet: String::new(),
                    age: None,
                    sources: Vec::new(),
                    score: 0.0,
                }
            });
            entry.score += 1.0 / (RRF_K + rank as f64 + 1.0);
            if r.snippet.len() > entry.snippet.len() {
                entry.snippet = r.snippet.clone();
            }
            if entry.age.is_none() {
                entry.age = r.age.clone();
            }
            if !entry.sources.contains(backend) {
                entry.sources.push(backend);
            }
        }
    }

    let mut merged: Vec<MergedResult> = order
        .into_iter()
        .filter_map(|key| map.remove(&key))
        .collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(max_results);
    merged
}

// ---------------------------------------------------------------------------
// WebSearchTool
// ---------------------------------------------------------------------------

pub struct WebSearchTool {
    backends: Vec<Box<dyn SearchBackend>>,
}

impl WebSearchTool {
    pub fn new(config: &SearchConfig) -> Self {
        Self {
            backends: configured_backends(config),
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web and return titles, URLs, and snippets"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "web_search",
            "description": "Search the web. Queries all configured search backends in parallel and returns merged, deduplicated results (titles, URLs, snippets, publication age when known). Use to find current information, research topics, check facts. One focused search is almost always enough; for single-fact lookups do NOT re-search with rephrased queries — synthesize promptly. Results are snippet previews, NOT full pages: never assemble complete lists, rosters, or enumerations from snippets — fetch a source page with web_fetch and answer from its text. Set 'freshness' for time-sensitive queries (news, prices, releases). If results are consistently empty, the search backend may be blocked — suggest the user set up Brave Search via manage_config (search.backend = 'brave' + search.api_key) or a self-hosted SearxNG instance (search.searxng_url).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of merged results (default 8, max 20)"
                    },
                    "freshness": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Only return results published within this window. Use for time-sensitive queries."
                    }
                },
                "required": ["query"],
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
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;
        let max_results = args["max_results"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, MAX_MAX_RESULTS);
        let freshness = args["freshness"].as_str().and_then(Freshness::parse);

        // Fan out to every backend concurrently; tolerate individual failures.
        let searches = self.backends.iter().map(|backend| async move {
            let name = backend.name();
            match tokio::time::timeout(
                BACKEND_TIMEOUT,
                backend.search(query, max_results, freshness),
            )
            .await
            {
                Ok(Ok(results)) => (name, Ok(results)),
                Ok(Err(e)) => (name, Err(e.to_string())),
                Err(_) => (
                    name,
                    Err(format!("timed out after {}s", BACKEND_TIMEOUT.as_secs())),
                ),
            }
        });
        let outcomes = futures::future::join_all(searches).await;

        let mut successes: Vec<(&'static str, Vec<SearchResult>)> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        for (name, outcome) in outcomes {
            match outcome {
                Ok(results) => successes.push((name, results)),
                Err(err) => failures.push(format!("{}: {}", name, err)),
            }
        }

        if successes.is_empty() {
            anyhow::bail!("All search backends failed — {}", failures.join("; "));
        }

        let merged = merge_ranked(&successes, max_results);

        if merged.is_empty() {
            let mut msg = format!("No results found for: {}", query);
            if !failures.is_empty() {
                msg.push_str(&format!(
                    "\n(Note: some backends were unavailable — {})",
                    failures.join("; ")
                ));
            }
            return Ok(msg);
        }

        let multi_source = self.backends.len() > 1;
        Ok(format_search_output(&merged, multi_source, &failures))
    }
}

/// Appended to every non-empty result set. Snippets are previews — without an
/// explicit guard the model assembles "complete" enumerations (rosters, member
/// lists, counts) out of 8 partial snippets, inventing the entries it cannot
/// see. Same failure mode as passive truncation markers (see
/// `utils::truncation_notice`).
const SNIPPET_GUARD_FOOTER: &str = "[⚠ SNIPPETS ONLY — the entries above are search-result \
     previews, not full page content. Do NOT enumerate lists, rosters, members, or counts \
     assembled from snippets, and do NOT present a \"complete\" answer built from them — \
     inventing entries that are not literally visible is an error. For enumeration or \
     list-type questions, fetch a source page with web_fetch and answer only from text you \
     can see; corroborate factual answers with a second independent source when possible, \
     and name the sources you used. If a fetch fails, try the next result URL — do not \
     retry the same one. If no fetched page contains the full answer, say it is incomplete \
     instead of filling the gaps.]";

fn format_search_output(
    merged: &[MergedResult],
    multi_source: bool,
    failures: &[String],
) -> String {
    let formatted: Vec<String> = merged
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut entry = format!("{}. [{}]({})", i + 1, r.title, r.url);
            if !r.snippet.is_empty() {
                entry.push_str(&format!("\n   {}", r.snippet));
            }
            let mut meta = Vec::new();
            if let Some(age) = &r.age {
                meta.push(age.clone());
            }
            if multi_source {
                meta.push(format!("via {}", r.sources.join(", ")));
            }
            if !meta.is_empty() {
                entry.push_str(&format!("\n   ({})", meta.join(" — ")));
            }
            entry
        })
        .collect();

    let mut output = formatted.join("\n\n");
    if !failures.is_empty() {
        output.push_str(&format!(
            "\n\n(Note: some backends were unavailable — {})",
            failures.join("; ")
        ));
    }
    output.push_str("\n\n");
    output.push_str(SNIPPET_GUARD_FOOTER);

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_protocol_relative_ddg_redirect() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Ftokio%2Flatest%2Ftokio%2Fruntime%2F&rut=904d86ad457b";
        assert_eq!(
            decode_ddg_redirect(href),
            "https://docs.rs/tokio/latest/tokio/runtime/"
        );
    }

    #[test]
    fn decodes_relative_ddg_redirect() {
        let href = "/l/?uddg=https%3A%2F%2Ftokio.rs%2F&rut=abc";
        assert_eq!(decode_ddg_redirect(href), "https://tokio.rs/");
    }

    #[test]
    fn passes_through_direct_urls() {
        assert_eq!(
            decode_ddg_redirect("https://example.com/page"),
            "https://example.com/page"
        );
        // Protocol-relative non-DDG URLs become absolute.
        assert_eq!(
            decode_ddg_redirect("//example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn parses_ddg_lite_fixture_and_unwraps_redirects() {
        let html = r#"
        <table>
        <tr><td>1.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftokio.rs%2F&amp;rut=abc" class="result-link">Tokio - An asynchronous Rust runtime</a></td></tr>
        <tr><td>&nbsp;</td><td class="result-snippet">Tokio is an asynchronous runtime for the Rust programming language.</td></tr>
        <tr><td>2.&nbsp;</td><td><a rel="nofollow" href="https://docs.rs/tokio" class="result-link">tokio - Rust</a></td></tr>
        <tr><td>&nbsp;</td><td class="result-snippet">API documentation for the tokio crate.</td></tr>
        </table>
        "#;
        let results = parse_ddg_lite_html(html, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].url, "https://tokio.rs/");
        assert_eq!(results[0].title, "Tokio - An asynchronous Rust runtime");
        assert!(results[0].snippet.contains("asynchronous runtime"));
        assert_eq!(results[1].url, "https://docs.rs/tokio");
    }

    #[test]
    fn detects_ddg_challenge_page() {
        assert!(is_ddg_challenge(
            "<html><div class=\"anomaly-modal\">...</div></html>"
        ));
        assert!(is_ddg_challenge(
            "<p>Unfortunately, bots use DuckDuckGo too.</p>"
        ));
        assert!(!is_ddg_challenge("<td class=\"result-snippet\">ok</td>"));
    }

    #[test]
    fn normalizes_urls_for_dedup() {
        assert_eq!(
            normalize_url("https://Example.com/Page/"),
            normalize_url("http://example.com/page")
        );
        assert_eq!(
            normalize_url("https://example.com/page#section"),
            normalize_url("https://example.com/page")
        );
        assert_ne!(
            normalize_url("https://example.com/page?a=1"),
            normalize_url("https://example.com/page")
        );
    }

    #[test]
    fn merge_dedups_and_boosts_multi_source_results() {
        let lists = vec![
            (
                "brave",
                vec![
                    SearchResult {
                        title: "Only Brave".into(),
                        url: "https://a.example/".into(),
                        snippet: "short".into(),
                        age: Some("2 days ago".into()),
                    },
                    SearchResult {
                        title: "Both".into(),
                        url: "https://b.example/".into(),
                        snippet: "brave snippet".into(),
                        age: None,
                    },
                ],
            ),
            (
                "duckduckgo",
                vec![SearchResult {
                    title: "Both (ddg)".into(),
                    url: "http://b.example".into(),
                    snippet: "a much longer duckduckgo snippet".into(),
                    age: None,
                }],
            ),
        ];
        let merged = merge_ranked(&lists, 10);
        assert_eq!(merged.len(), 2);
        // b.example appears in both backends → ranks above a.example despite
        // being rank 2 in the brave list.
        assert_eq!(merged[0].url, "https://b.example/");
        assert_eq!(merged[0].sources, vec!["brave", "duckduckgo"]);
        // Longest snippet wins.
        assert_eq!(merged[0].snippet, "a much longer duckduckgo snippet");
        assert_eq!(merged[1].sources, vec!["brave"]);
        assert_eq!(merged[1].age.as_deref(), Some("2 days ago"));
    }

    #[test]
    fn merge_respects_max_results() {
        let lists = vec![(
            "brave",
            (0..5)
                .map(|i| SearchResult {
                    title: format!("r{}", i),
                    url: format!("https://example.com/{}", i),
                    snippet: String::new(),
                    age: None,
                })
                .collect::<Vec<_>>(),
        )];
        assert_eq!(merge_ranked(&lists, 3).len(), 3);
    }

    #[test]
    fn freshness_parses_and_maps() {
        assert_eq!(Freshness::parse("day"), Some(Freshness::Day));
        assert_eq!(Freshness::parse("WEEK"), Some(Freshness::Week));
        assert_eq!(Freshness::parse("m"), Some(Freshness::Month));
        assert_eq!(Freshness::parse("bogus"), None);
        assert_eq!(Freshness::Day.brave_param(), "pd");
        assert_eq!(Freshness::Week.ddg_param(), "w");
        assert_eq!(Freshness::Year.searxng_param(), "year");
    }

    #[test]
    fn backend_selection_respects_config() {
        // DDG only (no key, no searxng).
        let cfg = SearchConfig::default();
        let names: Vec<&str> = configured_backends(&cfg).iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["duckduckgo"]);

        // Brave primary + key → brave first, ddg supplements.
        let cfg = SearchConfig {
            backend: SearchBackendKind::Brave,
            api_key: "k".into(),
            ..Default::default()
        };
        let names: Vec<&str> = configured_backends(&cfg).iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["brave", "duckduckgo"]);

        // parallel=false → primary only.
        let cfg = SearchConfig {
            backend: SearchBackendKind::Brave,
            api_key: "k".into(),
            parallel: false,
            ..Default::default()
        };
        let names: Vec<&str> = configured_backends(&cfg).iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["brave"]);

        // Everything configured → primary first, then brave, searxng, ddg order.
        let cfg = SearchConfig {
            backend: SearchBackendKind::DuckDuckGo,
            api_key: "k".into(),
            searxng_url: "http://localhost:8888".into(),
            parallel: true,
        };
        let names: Vec<&str> = configured_backends(&cfg).iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["duckduckgo", "brave", "searxng"]);
    }
}

#[cfg(test)]
mod format_output_tests {
    use super::*;

    fn merged(title: &str, snippet: &str) -> MergedResult {
        MergedResult {
            title: title.to_string(),
            url: format!("https://example.com/{}", title),
            snippet: snippet.to_string(),
            age: None,
            sources: vec!["duckduckgo"],
            score: 1.0,
        }
    }

    #[test]
    fn search_output_carries_snippet_honesty_guard() {
        let results = vec![merged("squad", "Ecuador announced its 26-man roster...")];
        let out = format_search_output(&results, false, &[]);
        assert!(out.contains("Ecuador announced"));
        // Snippets are previews — the model must not assemble "complete"
        // enumerations (rosters, lists, counts) out of them.
        assert!(out.contains("SNIPPETS ONLY"));
        assert!(out.contains("Do NOT enumerate"));
        assert!(out.contains("web_fetch"));
        // Factual answers should be corroborated, not single-sourced.
        assert!(out.contains("second independent source"));
        assert!(out.contains("name the sources"));
    }

    #[test]
    fn backend_failures_still_noted_in_output() {
        let results = vec![merged("a", "snippet a")];
        let out = format_search_output(&results, false, &["brave: timeout".to_string()]);
        assert!(out.contains("some backends were unavailable"));
        assert!(out.contains("brave: timeout"));
    }

    #[test]
    fn schema_description_scopes_synthesis_to_non_enumeration() {
        let tool = WebSearchTool { backends: vec![] };
        let schema = tool.schema();
        let desc = schema["description"].as_str().unwrap();
        // "synthesize promptly" must not invite assembling full lists from snippets.
        assert!(desc.contains("previews"));
        assert!(desc.contains("web_fetch"));
    }
}
