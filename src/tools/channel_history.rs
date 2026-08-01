use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::traits::{Tool, ToolCapabilities};

/// Tool that reads Slack channel conversation history via the Slack API.
/// Provides rich message data (text, sender, timestamps, threads, reactions, mentions)
/// for the LLM to analyze and summarize.
pub struct ReadChannelHistoryTool {
    http: Client,
    slack_tokens: Vec<String>,
    base_url: String,
    /// token index + user ID → display name cache
    user_cache: RwLock<HashMap<String, String>>,
    /// token index → Slack workspace URL resolved on first use (for permalinks)
    workspace_url: RwLock<HashMap<usize, String>>,
}

impl ReadChannelHistoryTool {
    pub fn new(slack_tokens: Vec<String>) -> Self {
        Self::with_base_url(slack_tokens, "https://slack.com")
    }

    fn with_base_url(slack_tokens: Vec<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("failed to build HTTP client"),
            slack_tokens,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            user_cache: RwLock::new(HashMap::new()),
            workspace_url: RwLock::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    fn new_with_base_url_for_tests(slack_tokens: Vec<String>, base_url: String) -> Self {
        Self::with_base_url(slack_tokens, base_url)
    }

    fn api_url(&self, method: &str) -> String {
        format!("{}/api/{}", self.base_url, method)
    }

    fn token_at(&self, token_index: usize) -> Option<&str> {
        self.slack_tokens.get(token_index).map(|s| s.as_str())
    }

    fn slack_error_hint(err: &str) -> &'static str {
        match err {
            "channel_not_found" => {
                "The channel was not found. The bot may not be a member of this channel."
            }
            "not_in_channel" => {
                "The bot is not a member of this channel. Invite the bot first with /invite @aidaemon."
            }
            "missing_scope" => {
                "The Slack app is missing the 'channels:history' OAuth scope. The workspace admin needs to add it in the Slack app settings."
            }
            "invalid_auth" | "token_revoked" | "account_inactive" => {
                "The Slack bot token is invalid or revoked. Check the bot_token in config.toml."
            }
            "ratelimited" => "Rate limited by Slack API. Try again in a few seconds.",
            _ => "An unexpected Slack API error occurred.",
        }
    }

    /// Resolve a Slack user ID to a display name, with caching.
    async fn resolve_user(&self, token_index: usize, user_id: &str) -> String {
        let cache_key = format!("{}:{}", token_index, user_id);
        // Check cache first
        {
            let cache = self.user_cache.read().await;
            if let Some(name) = cache.get(&cache_key) {
                return name.clone();
            }
        }

        let token = match self.token_at(token_index) {
            Some(t) => t,
            None => return user_id.to_string(),
        };

        let resp = self
            .http
            .get(self.api_url("users.info"))
            .bearer_auth(token)
            .query(&[("user", user_id)])
            .send()
            .await;

        let name = match resp {
            Ok(r) => {
                if let Ok(body) = r.json::<Value>().await {
                    if body["ok"].as_bool() == Some(true) {
                        body["user"]["profile"]["display_name"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .or_else(|| body["user"]["profile"]["real_name"].as_str())
                            .or_else(|| body["user"]["name"].as_str())
                            .unwrap_or(user_id)
                            .to_string()
                    } else {
                        user_id.to_string()
                    }
                } else {
                    user_id.to_string()
                }
            }
            Err(_) => user_id.to_string(),
        };

        // Cache the result
        {
            let mut cache = self.user_cache.write().await;
            cache.insert(cache_key, name.clone());
        }
        name
    }

    /// Resolve the workspace URL via auth.test (cached).
    async fn get_workspace_url(&self, token_index: usize) -> Option<String> {
        {
            let cached = self.workspace_url.read().await;
            if let Some(url) = cached.get(&token_index) {
                return Some(url.clone());
            }
        }

        let token = self.token_at(token_index)?;
        let resp = self
            .http
            .post(self.api_url("auth.test"))
            .bearer_auth(token)
            .send()
            .await
            .ok()?;

        let body = resp.json::<Value>().await.ok()?;
        if body["ok"].as_bool() == Some(true) {
            if let Some(url) = body["url"].as_str() {
                let url = url.trim_end_matches('/').to_string();
                let mut cached = self.workspace_url.write().await;
                cached.insert(token_index, url.clone());
                return Some(url);
            }
        }
        None
    }

    /// Build a permalink for a message.
    fn build_permalink(workspace_url: &str, channel_id: &str, ts: &str) -> String {
        // Slack permalink format: ts "1705312200.123456" → "p1705312200123456" (remove dot)
        let p_ts = format!("p{}", ts.replace('.', ""));
        format!("{}/archives/{}/{}", workspace_url, channel_id, p_ts)
    }

    /// Replace <@U123> mentions in text with resolved names.
    async fn resolve_mentions(&self, token_index: usize, text: &str) -> String {
        let re = regex::Regex::new(r"<@(U[A-Z0-9]+)>").unwrap();
        let mut result = text.to_string();

        // Collect all unique user IDs from mentions
        let user_ids: Vec<String> = re
            .captures_iter(text)
            .map(|c| c[1].to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Resolve them all
        for uid in user_ids {
            let name = self.resolve_user(token_index, &uid).await;
            result = result.replace(&format!("<@{}>", uid), &format!("@{}", name));
        }
        result
    }

    /// Resolve channel name from channel ID.
    async fn resolve_channel_name(&self, token_index: usize, channel_id: &str) -> Option<String> {
        let token = self.token_at(token_index)?;
        let resp = self
            .http
            .get(self.api_url("conversations.info"))
            .bearer_auth(token)
            .query(&[("channel", channel_id)])
            .send()
            .await
            .ok()?;

        let body = resp.json::<Value>().await.ok()?;
        if body["ok"].as_bool() == Some(true) {
            body["channel"]["name"].as_str().map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Fetch conversations.history from Slack API.
    async fn fetch_history(
        &self,
        channel_id: &str,
        limit: u64,
        oldest: Option<&str>,
        latest: Option<&str>,
    ) -> anyhow::Result<(usize, Vec<Value>)> {
        if self.slack_tokens.is_empty() {
            anyhow::bail!("No Slack bot token configured");
        }

        let mut params: Vec<(&str, String)> = vec![
            ("channel", channel_id.to_string()),
            ("limit", limit.to_string()),
        ];
        if let Some(o) = oldest {
            params.push(("oldest", o.to_string()));
        }
        if let Some(l) = latest {
            params.push(("latest", l.to_string()));
        }

        let mut failures = Vec::new();
        for (token_index, token) in self.slack_tokens.iter().enumerate() {
            let resp = match self
                .http
                .get(self.api_url("conversations.history"))
                .bearer_auth(token)
                .query(&params)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    failures.push(format!("token #{} transport error: {}", token_index + 1, e));
                    continue;
                }
            };

            let body = match resp.json::<Value>().await {
                Ok(body) => body,
                Err(e) => {
                    failures.push(format!("token #{} invalid JSON: {}", token_index + 1, e));
                    continue;
                }
            };

            if body["ok"].as_bool() == Some(true) {
                return Ok((
                    token_index,
                    body["messages"].as_array().cloned().unwrap_or_default(),
                ));
            }

            let err = body["error"].as_str().unwrap_or("unknown error");
            warn!(
                channel_id,
                token_index,
                error = err,
                "Slack conversations.history API error"
            );
            failures.push(format!(
                "token #{}: {} ({})",
                token_index + 1,
                err,
                Self::slack_error_hint(err)
            ));
        }

        anyhow::bail!(
            "Slack API error: no configured Slack bot token could read channel {}. {}",
            channel_id,
            failures.join("; ")
        );
    }

    /// Format a single message for output.
    async fn format_message(
        &self,
        token_index: usize,
        msg: &Value,
        channel_id: &str,
        workspace_url: Option<&str>,
        bot_owner_id: Option<&str>,
    ) -> String {
        let ts = msg["ts"].as_str().unwrap_or("0");
        let user_id = msg["user"].as_str().unwrap_or("unknown");
        let raw_text = msg["text"].as_str().unwrap_or("");

        // Resolve user name and mentions
        let user_name = self.resolve_user(token_index, user_id).await;
        let text = self.resolve_mentions(token_index, raw_text).await;

        // Format timestamp
        let timestamp = ts
            .split('.')
            .next()
            .and_then(|s| s.parse::<i64>().ok())
            .and_then(|secs| DateTime::from_timestamp(secs, 0).map(|dt| dt.naive_utc()))
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| ts.to_string());

        let mut line = format!("[{}] {}: {}", timestamp, user_name, text);

        // Metadata line
        let mut meta_parts: Vec<String> = Vec::new();

        // Check if bot owner was mentioned
        if let Some(owner_id) = bot_owner_id {
            if raw_text.contains(&format!("<@{}>", owner_id)) {
                meta_parts.push("⚠️ You were mentioned".to_string());
            }
        }

        // Thread reply count
        if let Some(reply_count) = msg["reply_count"].as_u64() {
            if reply_count > 0 {
                meta_parts.push(format!("💬 {} replies", reply_count));
            }
        }

        // Reactions
        if let Some(reactions) = msg["reactions"].as_array() {
            let reaction_strs: Vec<String> = reactions
                .iter()
                .filter_map(|r| {
                    let name = r["name"].as_str()?;
                    let count = r["count"].as_u64().unwrap_or(1);
                    Some(format!(":{}:{}", name, count))
                })
                .collect();
            if !reaction_strs.is_empty() {
                meta_parts.push(reaction_strs.join(" "));
            }
        }

        // Permalink
        if let Some(ws_url) = workspace_url {
            meta_parts.push(Self::build_permalink(ws_url, channel_id, ts));
        }

        if !meta_parts.is_empty() {
            line.push_str(&format!("\n  {}", meta_parts.join(" | ")));
        }

        line
    }
}

#[async_trait]
impl Tool for ReadChannelHistoryTool {
    fn name(&self) -> &str {
        "read_channel_history"
    }

    fn description(&self) -> &str {
        "Read recent messages from a Slack channel's conversation history. Use this when the user asks about what was discussed, what needs attention, key topics, or any question about actual channel conversations. Returns messages with metadata (reactions, threads, mentions) for analysis."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_channel_history",
            "description": "Read recent messages from a Slack channel's conversation history. ALWAYS use this tool when the user asks about what was discussed, takeaways, what happened, key topics, or any question about channel conversations. This reads ALL messages in the channel, not just messages sent to you. The channel_id is auto-detected — you can call this with no arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to fetch (default 50, max 200)",
                        "default": 50
                    },
                    "oldest": {
                        "type": "string",
                        "description": "Start of time range. Accepts relative like '2d', '1w', '3h', ISO 8601, or unix timestamp."
                    },
                    "latest": {
                        "type": "string",
                        "description": "End of time range. Same formats as oldest."
                    },
                    "channel_id": {
                        "type": "string",
                        "description": "Slack channel ID to read from. Defaults to current channel."
                    }
                },
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

        // Determine channel ID: explicit arg > _channel_id injection
        let channel_id = if let Some(cid) = args["channel_id"].as_str() {
            info!(
                channel_id = cid,
                "read_channel_history: using explicit channel_id"
            );
            cid.to_string()
        } else if let Some(injected) = args["_channel_id"].as_str() {
            // _channel_id is formatted as "slack:C12345" — extract the raw ID
            if let Some(raw) = injected.strip_prefix("slack:") {
                info!(
                    channel_id = raw,
                    injected, "read_channel_history: using injected _channel_id"
                );
                raw.to_string()
            } else {
                warn!(
                    injected,
                    "read_channel_history: _channel_id not a slack channel"
                );
                return Ok(
                    "This tool only works in Slack channels. The current channel is not a Slack channel."
                        .to_string(),
                );
            }
        } else {
            warn!("read_channel_history: no channel_id available (not injected, not provided)");
            return Ok(
                "No channel_id provided and not in a Slack channel. Use this tool from a Slack channel, or provide a channel_id parameter."
                    .to_string(),
            );
        };

        let has_token = !self.slack_tokens.is_empty();
        info!(channel_id = %channel_id, has_token, "read_channel_history: fetching history");

        let limit = args["limit"].as_u64().unwrap_or(50).min(200);

        let oldest = args["oldest"].as_str().and_then(parse_time_param);
        let latest = args["latest"].as_str().and_then(parse_time_param);

        // Fetch messages
        let (token_index, messages) = match self
            .fetch_history(&channel_id, limit, oldest.as_deref(), latest.as_deref())
            .await
        {
            Ok((token_index, msgs)) => {
                info!(channel_id = %channel_id, count = msgs.len(), "read_channel_history: fetched messages");
                (token_index, msgs)
            }
            Err(e) => {
                warn!(channel_id = %channel_id, error = %e, "read_channel_history: fetch_history failed");
                // Return the error as a user-friendly message instead of propagating
                // so the LLM gets actionable info rather than a generic "Error:" prefix
                return Ok(format!(
                    "Failed to read channel history: {}. \
                     Do NOT try to work around this with curl or terminal commands. \
                     Tell the user about this error so they can fix the configuration.",
                    e
                ));
            }
        };

        if messages.is_empty() {
            return Ok("No messages found in the specified time range.".to_string());
        }

        // Get workspace URL for permalinks
        let workspace_url = self.get_workspace_url(token_index).await;

        // Resolve channel name
        let channel_name = self
            .resolve_channel_name(token_index, &channel_id)
            .await
            .map(|n| format!("#{}", n))
            .unwrap_or_else(|| channel_id.clone());

        // Try to detect the bot owner from _session_id context — not reliable,
        // but user mentions are still highlighted via the ⚠️ marker when the
        // owner's user ID is known. We'll extract from _channel_id context if available.
        let bot_owner_id: Option<String> = None;

        // Messages come newest-first from Slack; reverse for chronological order
        let mut formatted: Vec<String> = Vec::with_capacity(messages.len());
        for msg in messages.iter().rev() {
            // Skip bot messages and system messages (join/leave/etc.)
            if msg["subtype"].as_str().is_some() && msg["subtype"].as_str() != Some("bot_message") {
                continue;
            }

            let line = self
                .format_message(
                    token_index,
                    msg,
                    &channel_id,
                    workspace_url.as_deref(),
                    bot_owner_id.as_deref(),
                )
                .await;
            formatted.push(line);
        }

        // Build header with date range
        let date_range = if formatted.len() >= 2 {
            // Extract dates from first and last formatted messages
            let first_ts = messages
                .last() // oldest (reversed)
                .and_then(|m| m["ts"].as_str())
                .and_then(|ts| ts.split('.').next()?.parse::<i64>().ok())
                .and_then(|s| DateTime::from_timestamp(s, 0).map(|dt| dt.naive_utc()))
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_default();
            let last_ts = messages
                .first() // newest (reversed)
                .and_then(|m| m["ts"].as_str())
                .and_then(|ts| ts.split('.').next()?.parse::<i64>().ok())
                .and_then(|s| DateTime::from_timestamp(s, 0).map(|dt| dt.naive_utc()))
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_default();
            if first_ts == last_ts {
                first_ts
            } else {
                format!("{} to {}", first_ts, last_ts)
            }
        } else {
            String::new()
        };

        let header = format!(
            "Channel history ({}, {} messages{}):\n",
            channel_name,
            formatted.len(),
            if date_range.is_empty() {
                String::new()
            } else {
                format!(", {}", date_range)
            },
        );

        Ok(format!("{}\n{}", header, formatted.join("\n\n")))
    }
}

/// Parse a time parameter that can be:
/// - Relative: "1h", "2d", "1w", "3m" (hours, days, weeks, months)
/// - Unix timestamp: "1705312800"
/// - ISO 8601: "2024-01-15T10:00:00Z"
///
/// Returns a unix timestamp string suitable for the Slack API.
pub fn parse_time_param(input: &str) -> Option<String> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    // Try relative time: "2d", "1w", "3h", "1m"
    if let Some(relative) = parse_relative_time(input) {
        return Some(relative.to_string());
    }

    // Try pure numeric (unix timestamp passthrough)
    if input.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return Some(input.to_string());
    }

    // Try ISO 8601
    if let Ok(dt) = DateTime::parse_from_rfc3339(input) {
        return Some(dt.timestamp().to_string());
    }

    // Try ISO 8601 without timezone (assume UTC)
    if let Ok(dt) = NaiveDateTime::parse_from_str(input, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.and_utc().timestamp().to_string());
    }

    // Try date-only
    if let Ok(dt) = chrono::NaiveDate::parse_from_str(input, "%Y-%m-%d") {
        return dt
            .and_hms_opt(0, 0, 0)
            .map(|dt| dt.and_utc().timestamp().to_string());
    }

    warn!("Could not parse time parameter: {}", input);
    None
}

/// Parse relative time like "2d", "1w", "3h" into a unix timestamp.
fn parse_relative_time(input: &str) -> Option<i64> {
    let re = regex::Regex::new(r"^(\d+)\s*([hdwm])$").ok()?;
    let caps = re.captures(input)?;
    let amount: i64 = caps[1].parse().ok()?;
    let unit = &caps[2];

    let seconds = match unit {
        "h" => amount * 3600,
        "d" => amount * 86400,
        "w" => amount * 7 * 86400,
        "m" => amount * 30 * 86400,
        _ => return None,
    };

    let now = Utc::now().timestamp();
    Some(now - seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn test_parse_time_relative_hours() {
        let result = parse_time_param("3h");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        let now = Utc::now().timestamp();
        // Should be roughly 3 hours ago (within 5 seconds tolerance)
        assert!((now - ts - 3 * 3600).abs() < 5);
    }

    #[test]
    fn test_parse_time_relative_days() {
        let result = parse_time_param("2d");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        let now = Utc::now().timestamp();
        assert!((now - ts - 2 * 86400).abs() < 5);
    }

    #[test]
    fn test_parse_time_relative_weeks() {
        let result = parse_time_param("1w");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        let now = Utc::now().timestamp();
        assert!((now - ts - 7 * 86400).abs() < 5);
    }

    #[test]
    fn test_parse_time_relative_months() {
        let result = parse_time_param("3m");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        let now = Utc::now().timestamp();
        assert!((now - ts - 90 * 86400).abs() < 5);
    }

    #[test]
    fn test_parse_time_unix_timestamp() {
        let result = parse_time_param("1705312800");
        assert_eq!(result, Some("1705312800".to_string()));
    }

    #[test]
    fn test_parse_time_unix_timestamp_with_dot() {
        let result = parse_time_param("1705312800.123456");
        assert_eq!(result, Some("1705312800.123456".to_string()));
    }

    #[test]
    fn test_parse_time_iso8601() {
        let result = parse_time_param("2024-01-15T10:00:00Z");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        assert_eq!(ts, 1705312800);
    }

    #[test]
    fn test_parse_time_iso8601_no_tz() {
        let result = parse_time_param("2024-01-15T10:00:00");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        assert_eq!(ts, 1705312800);
    }

    #[test]
    fn test_parse_time_date_only() {
        let result = parse_time_param("2024-01-15");
        assert!(result.is_some());
        let ts: i64 = result.unwrap().parse().unwrap();
        // 2024-01-15 00:00:00 UTC
        assert_eq!(ts, 1705276800);
    }

    #[test]
    fn test_parse_time_empty() {
        assert!(parse_time_param("").is_none());
    }

    #[test]
    fn test_parse_time_invalid() {
        assert!(parse_time_param("not-a-time").is_none());
    }

    #[test]
    fn test_permalink_construction() {
        let url = ReadChannelHistoryTool::build_permalink(
            "https://myworkspace.slack.com",
            "C12345",
            "1705312200.123456",
        );
        assert_eq!(
            url,
            "https://myworkspace.slack.com/archives/C12345/p1705312200123456"
        );
    }

    #[test]
    fn test_permalink_no_dot_in_ts() {
        let url =
            ReadChannelHistoryTool::build_permalink("https://team.slack.com", "C999", "1705312200");
        assert_eq!(url, "https://team.slack.com/archives/C999/p1705312200");
    }

    #[test]
    fn test_tool_schema() {
        let tool = ReadChannelHistoryTool::new(vec!["xoxb-test".to_string()]);
        let schema = tool.schema();
        assert_eq!(schema["name"], "read_channel_history");
        assert!(schema["description"].as_str().unwrap().len() > 10);
        let params = &schema["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["limit"].is_object());
        assert!(params["properties"]["oldest"].is_object());
        assert!(params["properties"]["latest"].is_object());
        assert!(params["properties"]["channel_id"].is_object());
    }

    #[tokio::test]
    async fn test_missing_channel_id() {
        let tool = ReadChannelHistoryTool::new(vec!["xoxb-test".to_string()]);
        let result = tool.call("{}").await.unwrap();
        assert!(result.contains("No channel_id"));
    }

    #[tokio::test]
    async fn test_non_slack_channel() {
        let tool = ReadChannelHistoryTool::new(vec!["xoxb-test".to_string()]);
        let result = tool
            .call(r#"{"_channel_id": "telegram:12345"}"#)
            .await
            .unwrap();
        assert!(result.contains("only works in Slack"));
    }

    #[tokio::test]
    async fn test_fetch_history_tries_later_slack_tokens() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0; 4096];
                let n = socket.read(&mut buf).await.unwrap();
                let req = String::from_utf8_lossy(&buf[..n]);
                let req_lower = req.to_ascii_lowercase();
                let body = if req_lower.contains("authorization: bearer xoxb-first") {
                    r#"{"ok":false,"error":"not_in_channel"}"#
                } else if req_lower.contains("authorization: bearer xoxb-second") {
                    r#"{"ok":true,"messages":[{"ts":"1705312800.000100","user":"U1","text":"from second token"}]}"#
                } else {
                    r#"{"ok":false,"error":"invalid_auth"}"#
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
            }
        });

        let tool = ReadChannelHistoryTool::new_with_base_url_for_tests(
            vec!["xoxb-first".to_string(), "xoxb-second".to_string()],
            base_url,
        );
        let (token_index, messages) = tool.fetch_history("C123", 10, None, None).await.unwrap();

        assert_eq!(token_index, 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["text"].as_str(), Some("from second token"));
        server.await.unwrap();
    }
}
