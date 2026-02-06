use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::config::{SearchBackendKind, SearchConfig};
use crate::traits::Tool;

use super::web_fetch::build_browser_client;

// ---------------------------------------------------------------------------
// SearchBackend trait + result type
// ---------------------------------------------------------------------------

pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[async_trait]
pub trait SearchBackend: Send + Sync {
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>>;
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
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            "https://lite.duckduckgo.com/lite/",
            &[("q", query)],
        )?
        .to_string();
        let resp = self.client.get(&url).send().await?;
        let html = resp.text().await?;

        // DuckDuckGo lite returns a simple HTML page with results in a table.
        // Each result has: a link (<a> tag with href) and snippet text.
        // We parse it with simple string scanning since it's a minimal page.
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
                None => { pos = link_start + 20; continue; }
            };
            let title_end = match html[title_start..].find("</a>") {
                Some(p) => title_start + p,
                None => { pos = title_start; continue; }
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
                strip_tags(&html[sn_content_start..sn_end]).trim().to_string()
            } else {
                String::new()
            };

            if !href.is_empty() && !title.is_empty() {
                results.push(SearchResult {
                    title,
                    url: href,
                    snippet,
                });
            }

            pos = title_end + 1;
        }

        Ok(results)
    }
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
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        let url = reqwest::Url::parse_with_params(
            "https://api.search.brave.com/res/v1/web/search",
            &[("q", query), ("count", &max_results.to_string())],
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
                let data: Value = resp.json().await?;
                let empty = vec![];
                let web_results = data["web"]["results"].as_array().unwrap_or(&empty);

                let results = web_results
                    .iter()
                    .take(max_results)
                    .filter_map(|r| {
                        Some(SearchResult {
                            title: r["title"].as_str()?.to_string(),
                            url: r["url"].as_str()?.to_string(),
                            snippet: r["description"].as_str().unwrap_or("").to_string(),
                        })
                    })
                    .collect();

                return Ok(results);
            }

            // Retry on 429 (rate limited) with exponential backoff
            if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && attempt < max_retries - 1 {
                let delay_secs = 2u64.pow(attempt as u32); // 1s, 2s, 4s
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                continue;
            }

            // Non-retryable error or exhausted retries
            break;
        }

        anyhow::bail!("Brave search API error: HTTP {}", last_status)
    }
}

// ---------------------------------------------------------------------------
// WebSearchTool
// ---------------------------------------------------------------------------

pub struct WebSearchTool {
    backend: Box<dyn SearchBackend>,
}

impl WebSearchTool {
    pub fn new(config: &SearchConfig) -> Self {
        let backend: Box<dyn SearchBackend> = match config.backend {
            SearchBackendKind::Brave => Box::new(BraveBackend::new(&config.api_key)),
            SearchBackendKind::DuckDuckGo => Box::new(DuckDuckGoBackend::new()),
        };
        Self { backend }
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
            "description": "Search the web. Returns titles, URLs, and snippets for your query. Use to find current information, research topics, check facts. If results are consistently empty, the search backend may be blocked — suggest the user set up Brave Search via manage_config (search.backend = 'brave' + search.api_key).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5)"
                    }
                },
                "required": ["query"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;
        let max_results = args["max_results"].as_u64().unwrap_or(5) as usize;

        let results = self.backend.search(query, max_results).await?;

        if results.is_empty() {
            return Ok(format!("No results found for: {}", query));
        }

        let formatted: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if r.snippet.is_empty() {
                    format!("{}. [{}]({})", i + 1, r.title, r.url)
                } else {
                    format!("{}. [{}]({})\n   {}", i + 1, r.title, r.url, r.snippet)
                }
            })
            .collect();

        Ok(formatted.join("\n\n"))
    }
}
