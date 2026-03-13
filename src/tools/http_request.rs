use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{mpsc, RwLock};
use tracing::warn;

use crate::config::HttpAuthProfile;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::sanitize::{sanitize_external_content, sanitize_output, wrap_untrusted_output};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::web_fetch::validate_url_for_ssrf;
use crate::traits::{
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolTargetHintKind, ToolVerificationMode,
};
use crate::types::{ApprovalResponse, StatusUpdate};

/// Timeout for approval requests (5 minutes).
const APPROVAL_TIMEOUT_SECS: u64 = 300;

/// Maximum number of redirect hops to follow.
const MAX_REDIRECTS: usize = 5;

/// Default maximum response size (1 MB).
const DEFAULT_MAX_RESPONSE_BYTES: u64 = 1_048_576;

/// Absolute maximum response size (5 MB).
const ABSOLUTE_MAX_RESPONSE_BYTES: u64 = 5_242_880;

/// Default timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum timeout in seconds.
const MAX_TIMEOUT_SECS: u64 = 120;

pub struct HttpRequestTool {
    profiles: Arc<RwLock<HashMap<String, HttpAuthProfile>>>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
    session_approvals: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    oauth_gateway: Arc<RwLock<Option<crate::oauth::OAuthGateway>>>,
}

impl HttpRequestTool {
    pub fn new(
        profiles: Arc<RwLock<HashMap<String, HttpAuthProfile>>>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            profiles,
            approval_tx,
            session_approvals: Arc::new(RwLock::new(HashMap::new())),
            oauth_gateway: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_oauth_gateway(mut self, gateway: crate::oauth::OAuthGateway) -> Self {
        self.oauth_gateway = Arc::new(RwLock::new(Some(gateway)));
        self
    }

    pub async fn set_oauth_gateway(&self, gateway: crate::oauth::OAuthGateway) {
        let mut guard = self.oauth_gateway.write().await;
        *guard = Some(gateway);
    }

    /// Check if a request domain is allowed by the profile's allowed_domains.
    /// Supports exact match and subdomain match (api.twitter.com matches twitter.com)
    /// but NOT suffix tricks (evil-twitter.com does NOT match twitter.com).
    pub(crate) fn domain_matches(request_domain: &str, allowed: &str) -> bool {
        let req = request_domain.to_lowercase();
        let allow = allowed.to_lowercase();
        if req == allow {
            return true;
        }
        req.ends_with(&format!(".{}", allow))
    }

    /// Normalize a header name for case-insensitive comparison.
    fn normalize_header_name(name: &str) -> String {
        name.chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .map(|ch| ch.to_ascii_lowercase())
            .collect()
    }

    /// Detect custom headers that are likely carrying credentials.
    /// Those must go through auth profiles so approval and policy checks apply.
    fn header_name_looks_like_auth(name: &str) -> bool {
        let normalized = Self::normalize_header_name(name);
        match normalized.as_str() {
            "authorization" | "proxyauthorization" | "cookie" | "setcookie" | "apikey"
            | "xapikey" | "authtoken" | "xauthtoken" | "accesstoken" | "xaccesstoken"
            | "sessiontoken" | "xsessiontoken" | "privatetoken" | "xprivatetoken" => true,
            _ => {
                normalized.ends_with("apikey")
                    || normalized.ends_with("authtoken")
                    || normalized.ends_with("accesstoken")
                    || normalized.ends_with("sessiontoken")
            }
        }
    }

    fn header_value_looks_like_auth(value: &str) -> bool {
        let trimmed = value.trim_start();
        ["bearer ", "basic ", "digest ", "token "]
            .iter()
            .any(|prefix| {
                trimmed.len() > prefix.len() && trimmed[..prefix.len()].eq_ignore_ascii_case(prefix)
            })
    }

    fn query_value_looks_like_embedded_json(value: &str) -> bool {
        let trimmed = value.trim();
        trimmed.starts_with('{') || trimmed.starts_with('[')
    }

    fn rebuild_query_pairs(url: &mut reqwest::Url, retained_pairs: &[(String, String)]) {
        url.set_query(None);
        if retained_pairs.is_empty() {
            return;
        }
        {
            let mut pairs = url.query_pairs_mut();
            for (key, value) in retained_pairs {
                pairs.append_pair(key, value);
            }
        }
    }

    fn recover_embedded_tool_params_from_url(
        args: &mut Value,
        url: &mut reqwest::Url,
    ) -> anyhow::Result<Vec<String>> {
        let Some(map) = args.as_object_mut() else {
            return Ok(Vec::new());
        };

        let mut retained_pairs: Vec<(String, String)> = Vec::new();
        let mut recovered: Vec<String> = Vec::new();
        let mut stripped_any = false;

        for (key, value) in url.query_pairs() {
            let key_owned = key.into_owned();
            let value_owned = value.into_owned();
            let normalized = key_owned.to_ascii_lowercase();
            let mut strip_from_url = false;

            match normalized.as_str() {
                "auth_profile" | "content_type" => {
                    strip_from_url = true;
                    if !map.contains_key(normalized.as_str()) {
                        map.insert(normalized.clone(), Value::String(value_owned.clone()));
                        recovered.push(normalized);
                    }
                }
                "follow_redirects" => {
                    strip_from_url = true;
                    if !map.contains_key("follow_redirects") {
                        if let Ok(parsed) = value_owned.parse::<bool>() {
                            map.insert("follow_redirects".to_string(), Value::Bool(parsed));
                            recovered.push("follow_redirects".to_string());
                        }
                    }
                }
                "timeout_secs" | "max_response_bytes" => {
                    strip_from_url = true;
                    if !map.contains_key(normalized.as_str()) {
                        if let Ok(parsed) = value_owned.parse::<u64>() {
                            map.insert(normalized.clone(), Value::Number(parsed.into()));
                            recovered.push(normalized);
                        }
                    }
                }
                "headers" => {
                    let looks_embedded = Self::query_value_looks_like_embedded_json(&value_owned);
                    if looks_embedded {
                        strip_from_url = true;
                        if !map.contains_key("headers") {
                            let parsed = serde_json::from_str::<Value>(&value_owned)?;
                            if parsed.is_object() {
                                map.insert("headers".to_string(), parsed);
                                recovered.push("headers".to_string());
                            }
                        }
                    }
                }
                "body" => {
                    let looks_embedded = Self::query_value_looks_like_embedded_json(&value_owned);
                    if looks_embedded {
                        strip_from_url = true;
                        if !map.contains_key("body") {
                            map.insert("body".to_string(), Value::String(value_owned.clone()));
                            recovered.push("body".to_string());
                        }
                    }
                }
                "_session_id"
                | "_task_id"
                | "_goal_id"
                | "_channel_visibility"
                | "_trusted_session"
                | "_user_role" => {
                    strip_from_url = true;
                }
                _ => {}
            }

            if !strip_from_url {
                retained_pairs.push((key_owned, value_owned));
            } else {
                stripped_any = true;
            }
        }

        if stripped_any {
            recovered.sort();
            recovered.dedup();
            Self::rebuild_query_pairs(url, &retained_pairs);
        }

        Ok(recovered)
    }

    fn embedded_tool_params_in_url(url: &reqwest::Url) -> Vec<String> {
        let mut leaked: Vec<String> = Vec::new();

        for (key, value) in url.query_pairs() {
            let normalized = key.to_ascii_lowercase();
            let is_reserved_tool_param = matches!(
                normalized.as_str(),
                "auth_profile"
                    | "content_type"
                    | "follow_redirects"
                    | "timeout_secs"
                    | "max_response_bytes"
                    | "_session_id"
                    | "_task_id"
                    | "_goal_id"
                    | "_channel_visibility"
                    | "_trusted_session"
                    | "_user_role"
            );
            let looks_like_serialized_tool_payload =
                matches!(normalized.as_str(), "headers" | "body")
                    && Self::query_value_looks_like_embedded_json(&value);

            if is_reserved_tool_param || looks_like_serialized_tool_payload {
                leaked.push(key.into_owned());
            }
        }

        let lowered_path = url.path().to_ascii_lowercase();
        for needle in [
            "auth_profile=",
            "content_type=",
            "follow_redirects=",
            "timeout_secs=",
            "max_response_bytes=",
            "_session_id=",
            "_task_id=",
            "_goal_id=",
            "_channel_visibility=",
            "_trusted_session=",
            "_user_role=",
            "headers=",
            "body=",
        ] {
            if lowered_path.contains(needle) {
                leaked.push(needle.trim_end_matches('=').to_string());
            }
        }

        leaked.sort();
        leaked.dedup();
        leaked
    }

    fn approval_scope_key(
        method: &str,
        url: &reqwest::Url,
        auth_profile_name: Option<&str>,
        content_type: Option<&str>,
    ) -> String {
        let host = url.host_str().unwrap_or("").to_ascii_lowercase();
        let port = url
            .port_or_known_default()
            .map(|p| format!(":{}", p))
            .unwrap_or_default();
        let auth = auth_profile_name.unwrap_or("-");
        let content_type = content_type.unwrap_or("-");
        format!(
            "{} {}://{}{}{} [auth:{}] [content-type:{}]",
            method,
            url.scheme().to_ascii_lowercase(),
            host,
            port,
            url.path(),
            auth,
            content_type
        )
    }

    fn requires_runtime_approval(
        risk: RiskLevel,
        is_session_approved: bool,
        is_trusted_session: bool,
    ) -> bool {
        risk != RiskLevel::Safe && !is_session_approved && !is_trusted_session
    }

    async fn is_session_approved(&self, session_id: &str, approval_key: &str) -> bool {
        self.session_approvals
            .read()
            .await
            .get(session_id)
            .is_some_and(|approved| approved.contains(approval_key))
    }

    async fn remember_session_approval(&self, session_id: &str, approval_key: &str) {
        self.session_approvals
            .write()
            .await
            .entry(session_id.to_string())
            .or_default()
            .insert(approval_key.to_string());
    }

    fn blocked_manual_auth_headers(
        headers: Option<&serde_json::Map<String, Value>>,
    ) -> Vec<String> {
        let mut blocked: Vec<String> = headers
            .into_iter()
            .flat_map(|map| map.iter())
            .filter_map(|(name, value)| {
                let looks_like_auth = Self::header_name_looks_like_auth(name)
                    || value
                        .as_str()
                        .is_some_and(Self::header_value_looks_like_auth);
                looks_like_auth.then(|| name.to_string())
            })
            .collect();
        blocked.sort();
        blocked.dedup();
        blocked
    }

    fn same_origin(original: &reqwest::Url, candidate: &reqwest::Url) -> bool {
        original.scheme().eq_ignore_ascii_case(candidate.scheme())
            && original
                .host_str()
                .unwrap_or("")
                .eq_ignore_ascii_case(candidate.host_str().unwrap_or(""))
            && original.port_or_known_default() == candidate.port_or_known_default()
    }

    fn redirect_behavior(method: &str, status_code: u16, has_body: bool) -> (String, bool) {
        match status_code {
            301..=303 => ("GET".to_string(), false),
            307 | 308 => (method.to_string(), has_body),
            _ => (method.to_string(), false),
        }
    }

    fn append_chunk_with_limit(
        collected: &mut Vec<u8>,
        observed_bytes: &mut u64,
        chunk: &[u8],
        max_response_bytes: u64,
    ) -> bool {
        *observed_bytes = observed_bytes.saturating_add(chunk.len() as u64);

        let remaining = (max_response_bytes as usize).saturating_sub(collected.len());
        if remaining > 0 {
            let to_copy = remaining.min(chunk.len());
            collected.extend_from_slice(&chunk[..to_copy]);
        }

        *observed_bytes > max_response_bytes
    }

    fn is_binary_content_type(content_type: &str) -> bool {
        content_type.starts_with("image/")
            || content_type.starts_with("audio/")
            || content_type.starts_with("video/")
            || content_type.contains("octet-stream")
    }

    fn append_truncation_notice(text: &str, max_response_bytes: u64) -> String {
        if text.is_empty() {
            format!(
                "[Truncated: response exceeded {} bytes limit]",
                max_response_bytes
            )
        } else {
            format!(
                "{}\n\n[Truncated: response exceeded {} bytes limit]",
                text, max_response_bytes
            )
        }
    }

    fn content_type_is_json(content_type: &str) -> bool {
        let normalized = content_type
            .split(';')
            .next()
            .unwrap_or(content_type)
            .trim()
            .to_ascii_lowercase();
        normalized == "application/json"
            || normalized.ends_with("+json")
            || normalized == "text/json"
    }

    fn summarize_json_value(value: &Value) -> Option<String> {
        match value {
            Value::Object(map) => {
                let mut lines = Vec::new();
                let mut top_keys: Vec<&str> = map.keys().map(|key| key.as_str()).collect();
                top_keys.sort_unstable();
                if !top_keys.is_empty() {
                    let key_list = top_keys
                        .iter()
                        .take(10)
                        .copied()
                        .collect::<Vec<_>>()
                        .join(", ");
                    let suffix = if top_keys.len() > 10 { ", ..." } else { "" };
                    lines.push(format!("Top-level keys: {}{}", key_list, suffix));
                }

                let mut array_keys: Vec<&str> = map
                    .iter()
                    .filter_map(|(key, value)| value.as_array().map(|_| key.as_str()))
                    .collect();
                array_keys.sort_unstable();
                for key in array_keys.into_iter().take(3) {
                    if let Some(items) = map.get(key).and_then(|value| value.as_array()) {
                        lines.push(format!("{}: array({} item(s))", key, items.len()));
                        if let Some(first_obj) = items.first().and_then(|value| value.as_object()) {
                            let mut item_keys: Vec<&str> =
                                first_obj.keys().map(|item_key| item_key.as_str()).collect();
                            item_keys.sort_unstable();
                            if !item_keys.is_empty() {
                                let sample_keys = item_keys
                                    .iter()
                                    .take(8)
                                    .copied()
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                let suffix = if item_keys.len() > 8 { ", ..." } else { "" };
                                lines.push(format!("{}[0] keys: {}{}", key, sample_keys, suffix));
                            }
                        }
                    }
                }

                if lines.is_empty() {
                    None
                } else {
                    Some(lines.join("\n"))
                }
            }
            Value::Array(items) => {
                Some(format!("Top-level JSON array with {} item(s)", items.len()))
            }
            _ => None,
        }
    }

    fn format_json_response_body(
        text: &str,
        content_type: &str,
        max_response_bytes: u64,
        truncated: bool,
    ) -> Option<String> {
        let trimmed = text.trim_start();
        if !(Self::content_type_is_json(content_type)
            || trimmed.starts_with('{')
            || trimmed.starts_with('['))
        {
            return None;
        }

        let value = serde_json::from_str::<Value>(text).ok()?;
        let pretty = serde_json::to_string_pretty(&value).ok()?;
        let body = if let Some(summary) = Self::summarize_json_value(&value) {
            format!("JSON summary:\n{}\n\n{}", summary, pretty)
        } else {
            pretty
        };

        if truncated {
            Some(Self::append_truncation_notice(&body, max_response_bytes))
        } else {
            Some(body)
        }
    }

    fn format_response_body(
        bytes: &[u8],
        content_type: &str,
        observed_bytes: u64,
        max_response_bytes: u64,
        truncated: bool,
    ) -> String {
        if Self::is_binary_content_type(content_type) {
            if truncated {
                return format!(
                    "[Binary response truncated at {} bytes limit, content-type: {}]",
                    max_response_bytes, content_type
                );
            }
            return format!(
                "[Binary response: {} bytes, content-type: {}]",
                observed_bytes, content_type
            );
        }

        match String::from_utf8(bytes.to_vec()) {
            Ok(text) => {
                if let Some(json_text) = Self::format_json_response_body(
                    &text,
                    content_type,
                    max_response_bytes,
                    truncated,
                ) {
                    return json_text;
                }
                if truncated {
                    Self::append_truncation_notice(&text, max_response_bytes)
                } else {
                    text
                }
            }
            Err(err) => {
                if truncated && err.utf8_error().error_len().is_none() {
                    let valid_up_to = err.utf8_error().valid_up_to();
                    let valid_text = std::str::from_utf8(&bytes[..valid_up_to]).unwrap_or("");
                    return Self::append_truncation_notice(valid_text, max_response_bytes);
                }

                if truncated {
                    format!(
                        "[Non-UTF8 response truncated at {} bytes limit, content-type: {}]",
                        max_response_bytes, content_type
                    )
                } else {
                    format!(
                        "[Non-UTF8 response: {} bytes, content-type: {}]",
                        observed_bytes, content_type
                    )
                }
            }
        }
    }

    fn build_oauth_retry_note(
        profile_name: &str,
        refresh_message: &str,
        retry_status: reqwest::StatusCode,
    ) -> String {
        if retry_status == reqwest::StatusCode::UNAUTHORIZED {
            format!(
                "aidaemon refreshed OAuth profile '{}' and retried the request once, but the remote API still returned HTTP 401 Unauthorized. {}",
                profile_name, refresh_message
            )
        } else {
            format!(
                "aidaemon refreshed OAuth profile '{}' and retried the request once. {}",
                profile_name, refresh_message
            )
        }
    }

    fn oauth_diagnostic_line(
        status_code: u16,
        auth_profile_name: Option<&str>,
        oauth_retry_note: Option<&str>,
    ) -> Option<String> {
        if let Some(note) = oauth_retry_note {
            return Some(format!("OAuth diagnostic: {}", note));
        }

        if status_code == 401 {
            if let Some(profile_name) = auth_profile_name {
                return Some(format!(
                    "OAuth diagnostic: HTTP 401 Unauthorized while using auth_profile='{}'. This response alone does not prove the token is expired. It may also indicate revoked credentials, missing scopes, or app permission mismatch.",
                    profile_name
                ));
            }
        }

        None
    }

    /// Build an OAuth 1.0a Authorization header value (RFC 5849).
    fn build_oauth1a_header(
        method: &str,
        url: &str,
        profile: &HttpAuthProfile,
        body: Option<&str>,
        content_type: Option<&str>,
    ) -> anyhow::Result<String> {
        use base64::Engine;
        use hmac::Mac;

        let api_key = profile
            .api_key
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("OAuth 1.0a requires api_key"))?;
        let api_secret = profile
            .api_secret
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("OAuth 1.0a requires api_secret"))?;
        let access_token = profile
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("OAuth 1.0a requires access_token"))?;
        let access_token_secret = profile
            .access_token_secret
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("OAuth 1.0a requires access_token_secret"))?;

        // Generate nonce and timestamp
        let nonce: String = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..32)
                .map(|_| {
                    let idx = rng.gen_range(0..36);
                    if idx < 10 {
                        (b'0' + idx) as char
                    } else {
                        (b'a' + idx - 10) as char
                    }
                })
                .collect()
        };
        let timestamp = chrono::Utc::now().timestamp().to_string();

        // Collect OAuth parameters
        let mut params: Vec<(String, String)> = vec![
            ("oauth_consumer_key".into(), api_key.into()),
            ("oauth_nonce".into(), nonce.clone()),
            ("oauth_signature_method".into(), "HMAC-SHA1".into()),
            ("oauth_timestamp".into(), timestamp.clone()),
            ("oauth_token".into(), access_token.into()),
            ("oauth_version".into(), "1.0".into()),
        ];

        // Parse URL to extract query params
        let parsed_url =
            reqwest::Url::parse(url).map_err(|e| anyhow::anyhow!("Invalid URL: {}", e))?;
        for (k, v) in parsed_url.query_pairs() {
            params.push((k.into_owned(), v.into_owned()));
        }

        // Include form-encoded body params in signature (NOT JSON body)
        let is_form =
            content_type.is_some_and(|ct| ct.contains("application/x-www-form-urlencoded"));
        if is_form {
            if let Some(body_str) = body {
                for pair in body_str.split('&') {
                    if let Some((k, v)) = pair.split_once('=') {
                        params.push((percent_decode(k), percent_decode(v)));
                    }
                }
            }
        }

        // Sort parameters
        params.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Build parameter string
        let param_string: String = params
            .iter()
            .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        // Build base URL (without query string)
        let base_url = format!(
            "{}://{}{}",
            parsed_url.scheme(),
            parsed_url.host_str().unwrap_or(""),
            parsed_url.path()
        );

        // Build signature base string
        let base_string = format!(
            "{}&{}&{}",
            method.to_uppercase(),
            percent_encode(&base_url),
            percent_encode(&param_string)
        );

        // Build signing key
        let signing_key = format!(
            "{}&{}",
            percent_encode(api_secret),
            percent_encode(access_token_secret)
        );

        // HMAC-SHA1 signature
        type HmacSha1 = hmac::Hmac<sha1::Sha1>;
        let mut mac = HmacSha1::new_from_slice(signing_key.as_bytes())
            .map_err(|e| anyhow::anyhow!("HMAC key error: {}", e))?;
        mac.update(base_string.as_bytes());
        let signature =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());

        // Build Authorization header value
        Ok(format!(
            "OAuth oauth_consumer_key=\"{}\", oauth_nonce=\"{}\", oauth_signature=\"{}\", oauth_signature_method=\"HMAC-SHA1\", oauth_timestamp=\"{}\", oauth_token=\"{}\", oauth_version=\"1.0\"",
            percent_encode(api_key),
            percent_encode(&nonce),
            percent_encode(&signature),
            percent_encode(&timestamp),
            percent_encode(access_token),
        ))
    }

    /// Apply auth to a request builder based on profile auth_type.
    pub(crate) fn apply_auth(
        builder: reqwest::RequestBuilder,
        profile: &HttpAuthProfile,
        method: &str,
        url: &str,
        body: Option<&str>,
        content_type: Option<&str>,
    ) -> anyhow::Result<reqwest::RequestBuilder> {
        use crate::config::HttpAuthType;
        match profile.auth_type {
            HttpAuthType::Oauth1a => {
                let auth_header =
                    Self::build_oauth1a_header(method, url, profile, body, content_type)?;
                Ok(builder.header("Authorization", auth_header))
            }
            HttpAuthType::Bearer => {
                let token = profile
                    .token
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Bearer auth requires token"))?;
                Ok(builder.bearer_auth(token))
            }
            HttpAuthType::Header => {
                let name = profile
                    .header_name
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Header auth requires header_name"))?;
                let value = profile
                    .header_value
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Header auth requires header_value"))?;
                Ok(builder.header(name, value))
            }
            HttpAuthType::Basic => {
                let username = profile
                    .username
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Basic auth requires username"))?;
                let password = profile
                    .password
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("Basic auth requires password"))?;
                Ok(builder.basic_auth(username, Some(password)))
            }
        }
    }

    /// Classify request risk level.
    fn classify_risk(method: &str, has_auth: bool) -> RiskLevel {
        match method {
            "GET" | "HEAD" if !has_auth => RiskLevel::Safe,
            "GET" | "HEAD" => RiskLevel::Medium,
            _ => RiskLevel::High,
        }
    }

    /// Auto-detect content type from body.
    fn detect_content_type(body: &str) -> &'static str {
        let trimmed = body.trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            "application/json"
        } else if trimmed.contains('=') && !trimmed.contains('<') {
            "application/x-www-form-urlencoded"
        } else {
            "text/plain"
        }
    }

    /// Strip credential values from error messages (sync version using pre-locked profiles).
    fn strip_credentials_from_error_with(
        profiles: &HashMap<String, HttpAuthProfile>,
        error: &str,
    ) -> String {
        let mut result = error.to_string();
        for profile in profiles.values() {
            for cred in profile.credential_values() {
                if cred.len() >= 4 {
                    result = result.replace(cred, "[REDACTED]");
                }
            }
        }
        result
    }

    /// Request approval from the user.
    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
        risk_level: RiskLevel,
        warnings: Vec<String>,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level,
                warnings,
                permission_mode: PermissionMode::Cautious,
                response_tx,
                kind: Default::default(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(Duration::from_secs(APPROVAL_TIMEOUT_SECS), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                warn!("Approval response channel closed for http_request");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                warn!(
                    "Approval request timed out for http_request ({}s)",
                    APPROVAL_TIMEOUT_SECS
                );
                Ok(ApprovalResponse::Deny)
            }
        }
    }

    async fn try_refresh_oauth_profile(
        &self,
        profile_name: &str,
    ) -> anyhow::Result<Option<String>> {
        let gateway = self.oauth_gateway.read().await.clone();
        let Some(gateway) = gateway else {
            return Ok(None);
        };
        if gateway.get_provider(profile_name).await.is_none() {
            return Ok(None);
        }
        let result = gateway.refresh_token(profile_name).await?;
        Ok(Some(result))
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_request(
        &self,
        client: &reqwest::Client,
        method: &str,
        url: &str,
        parsed_url: &reqwest::Url,
        body: Option<&str>,
        content_type: Option<&str>,
        custom_headers: Option<&serde_json::Map<String, Value>>,
        profile: Option<&HttpAuthProfile>,
        profiles_snapshot: &HashMap<String, HttpAuthProfile>,
        follow_redirects: bool,
    ) -> anyhow::Result<(reqwest::Response, usize)> {
        let mut current_url = url.to_string();
        let mut current_method = method.to_string();
        let original_url = parsed_url.clone();
        let mut redirect_count = 0;
        let mut resend_body = false;

        let response = loop {
            let mut builder = match current_method.as_str() {
                "GET" => client.get(&current_url),
                "POST" => client.post(&current_url),
                "PUT" => client.put(&current_url),
                "PATCH" => client.patch(&current_url),
                "DELETE" => client.delete(&current_url),
                _ => return Err(anyhow::anyhow!("Unsupported method: {}", current_method)),
            };

            if let Some(ct) = content_type {
                builder = builder.header("Content-Type", ct);
            }
            if let Some(payload) = body {
                if redirect_count == 0 || resend_body {
                    builder = builder.body(payload.to_string());
                }
            }

            if let Some(headers) = custom_headers {
                for (k, v) in headers {
                    if let Some(val) = v.as_str() {
                        builder = builder.header(k.as_str(), val);
                    }
                }
            }

            let current_parsed = reqwest::Url::parse(&current_url).unwrap_or(parsed_url.clone());
            let same_origin = Self::same_origin(&original_url, &current_parsed);

            if let Some(profile) = profile {
                if same_origin {
                    builder = Self::apply_auth(
                        builder,
                        profile,
                        &current_method,
                        &current_url,
                        if redirect_count == 0 || resend_body {
                            body
                        } else {
                            None
                        },
                        content_type,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "{}",
                            Self::strip_credentials_from_error_with(
                                profiles_snapshot,
                                &e.to_string()
                            )
                        )
                    })?;
                } else {
                    warn!(
                        "Auth stripped on cross-origin redirect: {} -> {}",
                        original_url, current_parsed
                    );
                }
            }

            let resp = builder.send().await.map_err(|e| {
                anyhow::anyhow!(
                    "{}",
                    Self::strip_credentials_from_error_with(profiles_snapshot, &e.to_string())
                )
            })?;

            if follow_redirects && resp.status().is_redirection() {
                redirect_count += 1;
                if redirect_count > MAX_REDIRECTS {
                    return Err(anyhow::anyhow!(
                        "Request stopped: exceeded maximum {} redirects",
                        MAX_REDIRECTS
                    ));
                }

                if let Some(location) = resp.headers().get("location") {
                    let location_str = location.to_str().unwrap_or("");
                    let next_url = if location_str.starts_with("http") {
                        location_str.to_string()
                    } else {
                        let base = reqwest::Url::parse(&current_url).unwrap_or(parsed_url.clone());
                        base.join(location_str)
                            .map(|u| u.to_string())
                            .unwrap_or(location_str.to_string())
                    };

                    if let Err(reason) = validate_url_for_ssrf(&next_url) {
                        return Err(anyhow::anyhow!(
                            "Redirect blocked (hop {}): {}",
                            redirect_count,
                            reason
                        ));
                    }

                    if let Ok(next_parsed) = reqwest::Url::parse(&next_url) {
                        if next_parsed.scheme() != "https" {
                            return Err(anyhow::anyhow!(
                                "Redirect blocked: redirected to non-HTTPS URL"
                            ));
                        }
                    }

                    current_url = next_url;
                    let (next_method, next_resend_body) = Self::redirect_behavior(
                        &current_method,
                        resp.status().as_u16(),
                        body.is_some(),
                    );
                    current_method = next_method;
                    resend_body = next_resend_body;
                    continue;
                }

                return Err(anyhow::anyhow!(
                    "Redirect response (HTTP {}) missing Location header",
                    resp.status()
                ));
            }

            break resp;
        };

        Ok((response, redirect_count))
    }

    /// Core implementation shared by `call()` and `call_with_status_outcome()`.
    /// Returns the formatted output string and the HTTP status code (if one was observed).
    async fn execute(&self, arguments: &str) -> anyhow::Result<(String, Option<u16>)> {
        let mut args: Value = serde_json::from_str(arguments)?;

        // Parse parameters
        let method = args["method"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: method"))?
            .to_uppercase();
        let url_str = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: url"))?;

        // Build URL with query params
        let mut parsed_url =
            reqwest::Url::parse(url_str).map_err(|e| anyhow::anyhow!("Invalid URL: {}", e))?;
        let recovered_tool_params =
            Self::recover_embedded_tool_params_from_url(&mut args, &mut parsed_url)?;
        if let Some(qp) = args["query_params"].as_object() {
            for (k, v) in qp {
                let val_owned = v
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| v.to_string());
                parsed_url.query_pairs_mut().append_pair(k, &val_owned);
            }
        }
        let url = parsed_url.to_string();
        let auth_profile_name = args["auth_profile"].as_str();
        let body = args["body"].as_str();
        let content_type_param = args["content_type"].as_str();
        let follow_redirects = args["follow_redirects"].as_bool().unwrap_or(true);
        let timeout_secs = args["timeout_secs"]
            .as_u64()
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);
        let max_response_bytes = args["max_response_bytes"]
            .as_u64()
            .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES)
            .min(ABSOLUTE_MAX_RESPONSE_BYTES);
        let custom_headers = args["headers"].as_object();

        // Step 1: HTTPS enforcement
        if parsed_url.scheme() != "https" {
            return Ok((
                "Request blocked: only HTTPS URLs are allowed".to_string(),
                None,
            ));
        }

        // Step 2: SSRF validation
        if let Err(reason) = validate_url_for_ssrf(&url) {
            return Ok((format!("Request blocked: {}", reason), None));
        }

        // Step 3: Catch malformed tool calls where tool-only args were stuffed into the URL.
        let leaked_tool_params = Self::embedded_tool_params_in_url(&parsed_url);
        if !leaked_tool_params.is_empty() {
            return Ok((format!(
                "Request blocked: tool-only parameters were embedded in the URL ({}). Put them in the top-level `http_request` arguments instead of `url`.",
                leaked_tool_params.join(", ")
            ), None));
        }
        if !recovered_tool_params.is_empty() {
            warn!(
                recovered = ?recovered_tool_params,
                endpoint = %parsed_url,
                "Recovered embedded http_request tool parameters from URL"
            );
        }

        // Step 4: Block manual credential headers so auth always flows through profiles.
        let blocked_headers = Self::blocked_manual_auth_headers(custom_headers);
        if !blocked_headers.is_empty() {
            return Ok((format!(
                "Request blocked: credential-bearing headers are not allowed in `headers` ({}). Configure an auth_profile instead.",
                blocked_headers.join(", ")
            ), None));
        }

        // Step 5: Resolve auth profile and check domain
        let profiles_snapshot = self.profiles.read().await.clone();
        let profile = if let Some(name) = auth_profile_name {
            let p = profiles_snapshot
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Unknown auth profile: '{}'", name))?;

            let request_host = parsed_url.host_str().unwrap_or("");
            let domain_ok = p
                .allowed_domains
                .iter()
                .any(|d| Self::domain_matches(request_host, d));
            if !domain_ok {
                return Ok((
                    format!(
                    "Request blocked: domain '{}' is not in the allowed domains for profile '{}'",
                    request_host, name
                ),
                    None,
                ));
            }
            Some(p)
        } else {
            None
        };

        // Step 6: Auto-detect content type
        let content_type = content_type_param
            .map(|s| s.to_string())
            .or_else(|| body.map(|b| Self::detect_content_type(b).to_string()));

        // Step 7: Scan for secrets in outbound data
        let check_parts = format!(
            "{} {} {}",
            url,
            body.unwrap_or(""),
            custom_headers
                .map(|h| serde_json::to_string(h).unwrap_or_default())
                .unwrap_or_default()
        );
        let (_, has_secrets) = sanitize_output(&check_parts);
        if has_secrets {
            return Ok((
                "Request blocked: outbound data appears to contain secrets or credentials. \
                 Review the URL, body, and headers for leaked API keys or tokens."
                    .to_string(),
                None,
            ));
        }

        // Step 8: Classify risk and request approval
        let risk = Self::classify_risk(&method, profile.is_some());
        let session_id = args["_session_id"].as_str().unwrap_or("unknown");
        let is_trusted_session = args["_trusted_session"].as_bool().unwrap_or(false);
        let approval_key = Self::approval_scope_key(
            &method,
            &parsed_url,
            auth_profile_name,
            content_type.as_deref(),
        );

        if Self::requires_runtime_approval(
            risk,
            self.is_session_approved(session_id, &approval_key).await,
            is_trusted_session,
        ) {
            let mut desc = format!("{} {}", method, url);
            if let Some(name) = auth_profile_name {
                desc.push_str(&format!(" [auth: {}]", name));
            }
            if let Some(b) = body {
                let boundary = crate::utils::floor_char_boundary(b, 200);
                let snippet = if b.len() > 200 { &b[..boundary] } else { b };
                desc.push_str(&format!("\nBody: {}", snippet));
            }

            let mut warnings = vec![format!("HTTP {} request to external API", method)];
            if profile.is_some() {
                warnings.push("Authenticated request — credentials will be sent".to_string());
            }
            if matches!(method.as_str(), "POST" | "PUT" | "PATCH" | "DELETE") {
                warnings.push("Write operation — may modify remote data".to_string());
            }

            match self
                .request_approval(session_id, &desc, risk, warnings)
                .await?
            {
                ApprovalResponse::AllowOnce => {}
                ApprovalResponse::AllowSession | ApprovalResponse::AllowAlways => {
                    self.remember_session_approval(session_id, &approval_key)
                        .await;
                }
                ApprovalResponse::Deny => {
                    return Ok(("Request denied by user".to_string(), None));
                }
            }
        }

        // Step 9: Execute request, optionally refreshing OAuth-backed bearer auth once on 401.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?;
        if !matches!(method.as_str(), "GET" | "POST" | "PUT" | "PATCH" | "DELETE") {
            return Ok((format!("Unsupported method: {}", method), None));
        }

        let (mut response, mut redirect_count) = self
            .execute_request(
                &client,
                &method,
                &url,
                &parsed_url,
                body,
                content_type.as_deref(),
                custom_headers,
                profile.as_ref(),
                &profiles_snapshot,
                follow_redirects,
            )
            .await?;

        let mut oauth_retry_note: Option<String> = None;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            if let Some(profile_name) = auth_profile_name {
                let is_bearer_profile = profile.as_ref().is_some_and(|resolved| {
                    matches!(resolved.auth_type, crate::config::HttpAuthType::Bearer)
                });
                if is_bearer_profile {
                    match self.try_refresh_oauth_profile(profile_name).await {
                        Ok(Some(refresh_message)) => {
                            let refreshed_profiles = self.profiles.read().await.clone();
                            let refreshed_profile = refreshed_profiles.get(profile_name).cloned();
                            if let Some(refreshed_profile_value) = refreshed_profile {
                                let (retry_response, retry_redirects) = self
                                    .execute_request(
                                        &client,
                                        &method,
                                        &url,
                                        &parsed_url,
                                        body,
                                        content_type.as_deref(),
                                        custom_headers,
                                        Some(&refreshed_profile_value),
                                        &refreshed_profiles,
                                        follow_redirects,
                                    )
                                    .await?;
                                let retry_status = retry_response.status();
                                response = retry_response;
                                redirect_count = retry_redirects;
                                oauth_retry_note = Some(Self::build_oauth_retry_note(
                                    profile_name,
                                    &refresh_message,
                                    retry_status,
                                ));
                            } else {
                                oauth_retry_note = Some(format!(
                                    "aidaemon refreshed OAuth provider state for '{}' but could not rebuild the HTTP auth profile, so it did not retry the request. {}",
                                    profile_name, refresh_message
                                ));
                            }
                        }
                        Ok(None) => {
                            oauth_retry_note = Some(format!(
                                "HTTP 401 Unauthorized from the remote API while using auth_profile='{}'. \
aidaemon did not attempt an OAuth refresh because this profile is not managed by the OAuth gateway.",
                                profile_name
                            ));
                        }
                        Err(err) => {
                            oauth_retry_note = Some(format!(
                                "HTTP 401 Unauthorized from the remote API while using auth_profile='{}'. \
aidaemon attempted an OAuth refresh once, but it failed: {}",
                                profile_name, err
                            ));
                        }
                    }
                } else {
                    oauth_retry_note = Some(format!(
                        "HTTP 401 Unauthorized from the remote API while using auth_profile='{}'. \
aidaemon did not attempt an OAuth refresh because this profile is not bearer-token OAuth-managed.",
                        profile_name
                    ));
                }
            }
        }

        // Step 9: Format response
        let status = response.status();
        let resp_headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let content_type_resp = resp_headers
            .get("content-type")
            .cloned()
            .unwrap_or_default();

        let mut collected_body = Vec::new();
        let mut observed_bytes = 0u64;
        let mut truncated = false;
        let response_profiles = self.profiles.read().await.clone();
        while let Some(chunk) = response.chunk().await.map_err(|e| {
            anyhow::anyhow!(
                "{}",
                Self::strip_credentials_from_error_with(&response_profiles, &e.to_string())
            )
        })? {
            truncated = Self::append_chunk_with_limit(
                &mut collected_body,
                &mut observed_bytes,
                &chunk,
                max_response_bytes,
            );
            if truncated {
                break;
            }
        }

        let body_str = Self::format_response_body(
            &collected_body,
            &content_type_resp,
            observed_bytes,
            max_response_bytes,
            truncated,
        );

        // Sanitize response content (strip prompt injection)
        let sanitized_body = sanitize_external_content(&body_str);

        // Build result with response details
        let status_code = status.as_u16();
        let captured_status = Some(status_code);
        let mut result = format!(
            "HTTP {} {}\n",
            status_code,
            status.canonical_reason().unwrap_or("")
        );

        // Include relevant headers
        for key in [
            "content-type",
            "x-rate-limit-remaining",
            "x-rate-limit-reset",
            "retry-after",
        ] {
            if let Some(val) = resp_headers.get(key) {
                result.push_str(&format!("{}: {}\n", key, val));
            }
        }
        if redirect_count > 0 {
            result.push_str(&format!("(followed {} redirect(s))\n", redirect_count));
        }

        // For error responses (4xx/5xx), add structured diagnostic section
        if status_code >= 400 {
            result.push_str("\n--- API ERROR ---\n");
            result.push_str(&format!("Status: {}\n", status_code));
            if let Some(diagnostic_line) = Self::oauth_diagnostic_line(
                status_code,
                auth_profile_name,
                oauth_retry_note.as_deref(),
            ) {
                result.push_str(&diagnostic_line);
                result.push('\n');
            }

            // Try to parse common API error formats (JSON with detail/message/error fields)
            if let Ok(error_json) = serde_json::from_str::<Value>(&body_str) {
                // Extract error details from common API patterns
                let detail = error_json["detail"]
                    .as_str()
                    .or_else(|| error_json["message"].as_str())
                    .or_else(|| error_json["error"]["message"].as_str())
                    .or_else(|| error_json["error"].as_str())
                    .or_else(|| error_json["error_description"].as_str());
                let title = error_json["title"]
                    .as_str()
                    .or_else(|| error_json["error"]["type"].as_str())
                    .or_else(|| error_json["error_code"].as_str());
                let error_type = error_json["type"].as_str();

                if let Some(t) = title {
                    result.push_str(&format!("Error: {}\n", t));
                }
                if let Some(d) = detail {
                    result.push_str(&format!("Detail: {}\n", d));
                }
                if let Some(et) = error_type {
                    result.push_str(&format!("Type: {}\n", et));
                }

                // Include any nested errors array
                if let Some(errors) = error_json["errors"].as_array() {
                    for (i, err) in errors.iter().enumerate().take(3) {
                        let msg = err["message"]
                            .as_str()
                            .or_else(|| err["detail"].as_str())
                            .unwrap_or("unknown");
                        result.push_str(&format!("  Error {}: {}\n", i + 1, msg));
                    }
                }
            }

            result.push_str("--- END ERROR ---\n");
            result.push_str("\nIMPORTANT: Report the FULL error details above to the user. ");
            result.push_str("Do NOT retry the same request — diagnose the root cause first.\n");
        }

        result.push('\n');
        result.push_str(&sanitized_body);

        // Wrap as untrusted external data
        Ok((
            wrap_untrusted_output("http_request", &result),
            captured_status,
        ))
    }
}

/// Simple percent-decoding for form-encoded body params.
fn percent_decode(s: &str) -> String {
    let mut result = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                result.push(byte);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            result.push(b' ');
        } else {
            result.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8(result).unwrap_or_else(|_| s.to_string())
}

/// RFC 3986 percent-encoding for OAuth signature base string.
fn percent_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                result.push(byte as char);
            }
            _ => {
                result.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    result
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Make authenticated HTTP requests to external APIs"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "http_request",
            "description": "Make HTTP requests to external APIs with pre-configured auth profiles. Supports OAuth 1.0a, Bearer, Header, and Basic auth. HTTPS only. Pass only the real endpoint URL in `url`; keep `auth_profile`, `headers`, `body`, `content_type`, `query_params`, and other request options as top-level arguments, never inside the URL. Write operations require user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        "description": "HTTP method"
                    },
                    "url": {
                        "type": "string",
                        "description": "Full HTTPS endpoint URL only. Do NOT append tool arguments like auth_profile, headers, body, content_type, timeout_secs, follow_redirects, or max_response_bytes to this URL."
                    },
                    "auth_profile": {
                        "type": "string",
                        "description": "Top-level auth profile name (e.g. 'twitter', 'stripe'). Do not embed this in the URL."
                    },
                    "headers": {
                        "type": "object",
                        "description": "Top-level additional non-auth request headers"
                    },
                    "body": {
                        "type": "string",
                        "description": "Top-level request body (JSON string, form data, etc.). Do not embed this in the URL."
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Top-level Content-Type header (auto-detected if omitted)"
                    },
                    "query_params": {
                        "type": "object",
                        "description": "Actual remote query parameters appended to the URL. Do not place tool control args here."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default 30, max 120)"
                    },
                    "follow_redirects": {
                        "type": "boolean",
                        "description": "Follow redirects (default true, max 5 hops)"
                    },
                    "max_response_bytes": {
                        "type": "integer",
                        "description": "Maximum response size in bytes (default 1MB, max 5MB)"
                    }
                },
                "required": ["method", "url"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.execute(arguments).await.map(|(output, _)| output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let _ = status_tx;
        let (output, http_status) = self.execute(arguments).await?;
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                http_status,
                ..Default::default()
            },
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let args = serde_json::from_str::<Value>(arguments).ok();
        let method = args
            .as_ref()
            .and_then(|value| value.get("method"))
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_ascii_uppercase())
            .unwrap_or_default();
        let url = args
            .as_ref()
            .and_then(|value| value.get("url"))
            .and_then(|value| value.as_str())
            .unwrap_or_default();

        match method.as_str() {
            "GET" | "HEAD" | "OPTIONS" => ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent)
                .with_target_hint(ToolTargetHintKind::Url, url),
            "POST" | "PUT" | "PATCH" | "DELETE" => {
                ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Url, url)
            }
            _ => ToolCallSemantics::mutation().with_target_hint(ToolTargetHintKind::Url, url),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HttpAuthProfile, HttpAuthType};

    fn make_oauth_profile() -> HttpAuthProfile {
        HttpAuthProfile {
            auth_type: HttpAuthType::Oauth1a,
            allowed_domains: vec!["twitter.com".to_string(), "api.x.com".to_string()],
            api_key: Some("consumer_key_123".to_string()),
            api_secret: Some("consumer_secret_456".to_string()),
            access_token: Some("access_token_789".to_string()),
            access_token_secret: Some("access_secret_abc".to_string()),
            user_id: Some("12345".to_string()),
            token: None,
            header_name: None,
            header_value: None,
            username: None,
            password: None,
        }
    }

    fn make_bearer_profile() -> HttpAuthProfile {
        HttpAuthProfile {
            auth_type: HttpAuthType::Bearer,
            allowed_domains: vec!["api.stripe.com".to_string()],
            api_key: None,
            api_secret: None,
            access_token: None,
            access_token_secret: None,
            user_id: None,
            token: Some("sk_test_secret_token_value".to_string()),
            header_name: None,
            header_value: None,
            username: None,
            password: None,
        }
    }

    fn make_header_profile() -> HttpAuthProfile {
        HttpAuthProfile {
            auth_type: HttpAuthType::Header,
            allowed_domains: vec!["api.example.com".to_string()],
            api_key: None,
            api_secret: None,
            access_token: None,
            access_token_secret: None,
            user_id: None,
            token: None,
            header_name: Some("X-API-Key".to_string()),
            header_value: Some("my_secret_api_key".to_string()),
            username: None,
            password: None,
        }
    }

    #[test]
    fn test_schema_has_required_fields() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let schema = tool.schema();
        assert_eq!(schema["name"], "http_request");
        assert!(schema["description"].as_str().unwrap().len() > 10);
        assert!(schema["description"]
            .as_str()
            .unwrap()
            .contains("never inside the URL"));
        assert!(schema["parameters"]["properties"]["method"].is_object());
        assert!(schema["parameters"]["properties"]["url"].is_object());
        assert!(schema["parameters"]["properties"]["url"]["description"]
            .as_str()
            .unwrap()
            .contains("Do NOT append tool arguments"));
        let required = schema["parameters"]["required"].as_array().unwrap();
        assert!(required.contains(&json!("method")));
        assert!(required.contains(&json!("url")));
    }

    #[test]
    fn test_domain_matching_exact() {
        assert!(HttpRequestTool::domain_matches(
            "twitter.com",
            "twitter.com"
        ));
        assert!(HttpRequestTool::domain_matches(
            "api.stripe.com",
            "api.stripe.com"
        ));
    }

    #[test]
    fn test_domain_matching_subdomain() {
        assert!(HttpRequestTool::domain_matches(
            "api.twitter.com",
            "twitter.com"
        ));
        assert!(HttpRequestTool::domain_matches(
            "upload.api.twitter.com",
            "twitter.com"
        ));
    }

    #[test]
    fn test_domain_matching_rejects_suffix_attacks() {
        assert!(!HttpRequestTool::domain_matches(
            "evil-twitter.com",
            "twitter.com"
        ));
        assert!(!HttpRequestTool::domain_matches(
            "nottwitter.com",
            "twitter.com"
        ));
        assert!(!HttpRequestTool::domain_matches(
            "twitter.com.evil.com",
            "twitter.com"
        ));
    }

    #[test]
    fn test_domain_matching_case_insensitive() {
        assert!(HttpRequestTool::domain_matches(
            "API.Twitter.COM",
            "twitter.com"
        ));
        assert!(HttpRequestTool::domain_matches(
            "twitter.com",
            "Twitter.COM"
        ));
    }

    #[test]
    fn test_trusted_session_skips_runtime_approval() {
        assert!(!HttpRequestTool::requires_runtime_approval(
            RiskLevel::High,
            false,
            true
        ));
        assert!(!HttpRequestTool::requires_runtime_approval(
            RiskLevel::Critical,
            false,
            true
        ));
    }

    #[test]
    fn test_untrusted_write_request_still_requires_approval() {
        assert!(HttpRequestTool::requires_runtime_approval(
            RiskLevel::High,
            false,
            false
        ));
        assert!(!HttpRequestTool::requires_runtime_approval(
            RiskLevel::Safe,
            false,
            false
        ));
        assert!(!HttpRequestTool::requires_runtime_approval(
            RiskLevel::High,
            true,
            false
        ));
    }

    #[test]
    fn test_auth_header_detection_by_name() {
        assert!(HttpRequestTool::header_name_looks_like_auth(
            "Authorization"
        ));
        assert!(HttpRequestTool::header_name_looks_like_auth("X-API-Key"));
        assert!(HttpRequestTool::header_name_looks_like_auth("X-Auth-Token"));
        assert!(!HttpRequestTool::header_name_looks_like_auth("Accept"));
        assert!(!HttpRequestTool::header_name_looks_like_auth(
            "X-API-Version"
        ));
    }

    #[test]
    fn test_auth_header_detection_by_value() {
        assert!(HttpRequestTool::header_value_looks_like_auth(
            "Bearer secret-token"
        ));
        assert!(HttpRequestTool::header_value_looks_like_auth(
            "Basic dXNlcjpwYXNz"
        ));
        assert!(!HttpRequestTool::header_value_looks_like_auth(
            "application/json"
        ));
    }

    #[test]
    fn test_same_origin_requires_matching_port() {
        let original = reqwest::Url::parse("https://api.example.com/resource").unwrap();
        let default_https = reqwest::Url::parse("https://api.example.com:443/other").unwrap();
        let different_port = reqwest::Url::parse("https://api.example.com:8443/other").unwrap();

        assert!(HttpRequestTool::same_origin(&original, &default_https));
        assert!(!HttpRequestTool::same_origin(&original, &different_port));
    }

    #[test]
    fn test_redirect_behavior_drops_body_for_302() {
        let (method, resend_body) = HttpRequestTool::redirect_behavior("POST", 302, true);
        assert_eq!(method, "GET");
        assert!(!resend_body);
    }

    #[test]
    fn test_redirect_behavior_preserves_body_for_307_and_308() {
        let (method_307, resend_307) = HttpRequestTool::redirect_behavior("POST", 307, true);
        let (method_308, resend_308) = HttpRequestTool::redirect_behavior("PATCH", 308, true);

        assert_eq!(method_307, "POST");
        assert!(resend_307);
        assert_eq!(method_308, "PATCH");
        assert!(resend_308);
    }

    #[test]
    fn test_append_chunk_with_limit_stops_at_configured_size() {
        let mut collected = Vec::new();
        let mut observed = 0;

        let first =
            HttpRequestTool::append_chunk_with_limit(&mut collected, &mut observed, b"hello", 7);
        assert!(!first);
        assert_eq!(collected, b"hello");
        assert_eq!(observed, 5);

        let second =
            HttpRequestTool::append_chunk_with_limit(&mut collected, &mut observed, b"world", 7);
        assert!(second);
        assert_eq!(collected, b"hellowo");
        assert_eq!(observed, 10);
    }

    #[test]
    fn test_format_response_body_handles_truncated_utf8_prefix() {
        let bytes = vec![0xC3, 0xA9, 0xC3];
        let formatted = HttpRequestTool::format_response_body(&bytes, "text/plain", 4, 3, true);
        assert!(formatted.starts_with("é"));
        assert!(formatted.contains("response exceeded 3 bytes limit"));
        assert!(!formatted.contains('\u{FFFD}'));
    }

    #[test]
    fn test_format_response_body_reports_binary_truncation() {
        let formatted = HttpRequestTool::format_response_body(b"\x89PNG", "image/png", 10, 4, true);
        assert!(formatted.contains("Binary response truncated"));
        assert!(formatted.contains("image/png"));
    }

    #[test]
    fn test_format_response_body_pretty_prints_json_with_summary() {
        let raw = br#"{"studies":[{"protocolSection":{"identificationModule":{"briefTitle":"Skin Trial"}}}],"nextPageToken":"abc"}"#;
        let formatted = HttpRequestTool::format_response_body(
            raw,
            "application/json",
            raw.len() as u64,
            4096,
            false,
        );
        assert!(formatted.contains("JSON summary:"));
        assert!(formatted.contains("studies: array(1 item(s))"));
        assert!(formatted.contains("\"briefTitle\": \"Skin Trial\""));
        assert!(formatted.contains('\n'));
    }

    #[test]
    fn test_build_oauth_retry_note_mentions_persistent_401() {
        let note = HttpRequestTool::build_oauth_retry_note(
            "twitter",
            "Token refreshed for twitter",
            reqwest::StatusCode::UNAUTHORIZED,
        );
        assert!(note.contains("still returned HTTP 401 Unauthorized"));
        assert!(note.contains("Token refreshed for twitter"));
    }

    #[test]
    fn test_oauth_diagnostic_line_avoids_expiry_guessing() {
        let diagnostic =
            HttpRequestTool::oauth_diagnostic_line(401, Some("twitter"), None).unwrap();
        assert!(diagnostic.contains("does not prove the token is expired"));
        assert!(diagnostic.contains("missing scopes"));
        assert!(diagnostic.contains("app permission mismatch"));
    }

    #[test]
    fn test_oauth_diagnostic_line_prefers_concrete_retry_note() {
        let diagnostic = HttpRequestTool::oauth_diagnostic_line(
            401,
            Some("twitter"),
            Some("retried the request once"),
        )
        .unwrap();
        assert_eq!(
            diagnostic,
            "OAuth diagnostic: retried the request once".to_string()
        );
    }

    #[test]
    fn test_oauth1a_signing_produces_valid_header() {
        let profile = make_oauth_profile();
        let result = HttpRequestTool::build_oauth1a_header(
            "POST",
            "https://api.twitter.com/2/tweets",
            &profile,
            Some("{\"text\": \"hello\"}"),
            Some("application/json"),
        );
        assert!(result.is_ok());
        let header = result.unwrap();
        assert!(header.starts_with("OAuth "));
        assert!(header.contains("oauth_consumer_key="));
        assert!(header.contains("oauth_signature="));
        assert!(header.contains("oauth_nonce="));
        assert!(header.contains("oauth_timestamp="));
        assert!(header.contains("oauth_token="));
        assert!(header.contains("oauth_signature_method=\"HMAC-SHA1\""));
        assert!(header.contains("oauth_version=\"1.0\""));
    }

    #[test]
    fn test_oauth1a_signing_fails_without_required_fields() {
        let mut profile = make_oauth_profile();
        profile.api_key = None;
        let result = HttpRequestTool::build_oauth1a_header(
            "GET",
            "https://api.twitter.com/2/tweets",
            &profile,
            None,
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("api_key"));
    }

    #[test]
    fn test_content_type_auto_detection() {
        assert_eq!(
            HttpRequestTool::detect_content_type("{\"text\": \"hello\"}"),
            "application/json"
        );
        assert_eq!(
            HttpRequestTool::detect_content_type("[1, 2, 3]"),
            "application/json"
        );
        assert_eq!(
            HttpRequestTool::detect_content_type("key=value&other=data"),
            "application/x-www-form-urlencoded"
        );
        assert_eq!(
            HttpRequestTool::detect_content_type("just some text"),
            "text/plain"
        );
    }

    #[test]
    fn test_embedded_tool_params_detects_reserved_fields() {
        let url = reqwest::Url::parse(
            "https://api.twitter.com/2/tweets?auth_profile=twitter&headers=%7B%22Content-Type%22:%22application/json%22%7D",
        )
        .unwrap();
        let leaked = HttpRequestTool::embedded_tool_params_in_url(&url);
        assert!(leaked.contains(&"auth_profile".to_string()));
        assert!(leaked.contains(&"headers".to_string()));
    }

    #[test]
    fn test_recover_embedded_tool_params_moves_fields_out_of_url() {
        let mut args = json!({
            "method": "POST",
            "url": "https://api.twitter.com/2/tweets?auth_profile=twitter&body=%7B%22text%22%3A%22hello%22%7D&content_type=application%2Fjson&keep=1"
        });
        let mut url = reqwest::Url::parse(args["url"].as_str().unwrap()).unwrap();

        let recovered =
            HttpRequestTool::recover_embedded_tool_params_from_url(&mut args, &mut url).unwrap();

        assert_eq!(args["auth_profile"], "twitter");
        assert_eq!(args["body"], "{\"text\":\"hello\"}");
        assert_eq!(args["content_type"], "application/json");
        assert!(recovered.contains(&"auth_profile".to_string()));
        assert!(recovered.contains(&"body".to_string()));
        assert!(recovered.contains(&"content_type".to_string()));
        assert_eq!(url.as_str(), "https://api.twitter.com/2/tweets?keep=1");
    }

    #[tokio::test]
    async fn test_session_approval_scope_ignores_query_and_body_values() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let first_url = reqwest::Url::parse("https://api.twitter.com/2/tweets?text=first").unwrap();
        let second_url =
            reqwest::Url::parse("https://api.twitter.com/2/tweets?text=second").unwrap();
        let first_key = HttpRequestTool::approval_scope_key(
            "POST",
            &first_url,
            Some("twitter"),
            Some("application/json"),
        );
        let second_key = HttpRequestTool::approval_scope_key(
            "POST",
            &second_url,
            Some("twitter"),
            Some("application/json"),
        );

        assert_eq!(first_key, second_key);
        assert!(!tool.is_session_approved("telegram:test", &first_key).await);

        tool.remember_session_approval("telegram:test", &first_key)
            .await;

        assert!(tool.is_session_approved("telegram:test", &second_key).await);
    }

    #[test]
    fn test_risk_classification() {
        assert_eq!(
            HttpRequestTool::classify_risk("GET", false),
            RiskLevel::Safe
        );
        assert_eq!(
            HttpRequestTool::classify_risk("HEAD", false),
            RiskLevel::Safe
        );
        assert_eq!(
            HttpRequestTool::classify_risk("GET", true),
            RiskLevel::Medium
        );
        assert_eq!(
            HttpRequestTool::classify_risk("POST", false),
            RiskLevel::High
        );
        assert_eq!(
            HttpRequestTool::classify_risk("POST", true),
            RiskLevel::High
        );
        assert_eq!(
            HttpRequestTool::classify_risk("DELETE", true),
            RiskLevel::High
        );
    }

    #[test]
    fn test_credential_stripping_in_errors() {
        let mut profiles = HashMap::new();
        profiles.insert("twitter".to_string(), make_oauth_profile());
        profiles.insert("stripe".to_string(), make_bearer_profile());

        let error = "Connection failed to api.twitter.com with token access_token_789 and secret consumer_secret_456";
        let stripped = HttpRequestTool::strip_credentials_from_error_with(&profiles, error);
        assert!(!stripped.contains("access_token_789"));
        assert!(!stripped.contains("consumer_secret_456"));
        assert!(stripped.contains("[REDACTED]"));
        assert!(stripped.contains("Connection failed"));
    }

    #[test]
    fn test_credential_stripping_preserves_short_values() {
        let mut profiles = HashMap::new();
        let mut profile = make_oauth_profile();
        profile.api_key = Some("ab".to_string()); // too short to strip
        profiles.insert("test".to_string(), profile);

        let error = "error with ab inside";
        let stripped = HttpRequestTool::strip_credentials_from_error_with(&profiles, error);
        // Short values (< 4 chars) should not be stripped to avoid false positives
        assert!(stripped.contains("ab"));
    }

    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("hello"), "hello");
        assert_eq!(percent_encode("hello world"), "hello%20world");
        assert_eq!(percent_encode("a=b&c=d"), "a%3Db%26c%3Dd");
        assert_eq!(percent_encode("test-value_ok.here~"), "test-value_ok.here~");
        assert_eq!(percent_encode("100%"), "100%25");
    }

    #[test]
    fn test_oauth1a_form_body_included_in_signature() {
        // When content-type is form-urlencoded, body params should be in the signature
        let profile = make_oauth_profile();
        let result = HttpRequestTool::build_oauth1a_header(
            "POST",
            "https://api.twitter.com/oauth/request_token",
            &profile,
            Some("oauth_callback=oob"),
            Some("application/x-www-form-urlencoded"),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_credential_values_collection() {
        let profile = make_oauth_profile();
        let creds = profile.credential_values();
        assert!(creds.contains(&"consumer_key_123"));
        assert!(creds.contains(&"consumer_secret_456"));
        assert!(creds.contains(&"access_token_789"));
        assert!(creds.contains(&"access_secret_abc"));
        assert_eq!(creds.len(), 4);
    }

    #[test]
    fn test_credential_values_bearer() {
        let profile = make_bearer_profile();
        let creds = profile.credential_values();
        assert!(creds.contains(&"sk_test_secret_token_value"));
        assert_eq!(creds.len(), 1);
    }

    #[test]
    fn test_credential_values_header() {
        let profile = make_header_profile();
        let creds = profile.credential_values();
        assert!(creds.contains(&"my_secret_api_key"));
        assert_eq!(creds.len(), 1);
    }

    #[tokio::test]
    async fn test_https_enforcement() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let result = tool
            .call(r#"{"method": "GET", "url": "http://example.com/api"}"#)
            .await
            .unwrap();
        assert!(result.contains("only HTTPS"));
    }

    #[tokio::test]
    async fn test_domain_not_in_allowed_list() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let mut profiles = HashMap::new();
        profiles.insert("twitter".to_string(), make_oauth_profile());
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(profiles)), tx);
        let result = tool
            .call(
                r#"{"method": "GET", "url": "https://api.evil.com/steal", "auth_profile": "twitter"}"#,
            )
            .await
            .unwrap();
        assert!(result.contains("not in the allowed domains"));
    }

    #[tokio::test]
    async fn test_manual_authorization_header_is_blocked() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let result = tool
            .call(
                r#"{
                    "method": "GET",
                    "url": "https://example.com/api",
                    "headers": {
                        "Authorization": "Basic dXNlcjpwYXNz"
                    }
                }"#,
            )
            .await
            .unwrap();
        assert!(result.contains("credential-bearing headers are not allowed"));
        assert!(result.contains("Authorization"));
    }

    #[tokio::test]
    async fn test_embedded_tool_params_are_recovered_before_profile_resolution() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let err = tool
            .call(
                r#"{
                    "method": "POST",
                    "url": "https://api.twitter.com/2/tweets?auth_profile=twitter&headers=%7B%22Content-Type%22%3A%22application/json%22%7D",
                    "body": "{\"text\":\"hello\"}",
                    "_session_id": "telegram:test"
                }"#,
            )
            .await
            .expect_err("recovered call should reach normal auth-profile validation");

        assert!(err.to_string().contains("Unknown auth profile: 'twitter'"));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn test_manual_api_key_header_is_blocked() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let result = tool
            .call(
                r#"{
                    "method": "GET",
                    "url": "https://example.com/api",
                    "headers": {
                        "X-API-Key": "super-secret-value"
                    }
                }"#,
            )
            .await
            .unwrap();
        assert!(result.contains("credential-bearing headers are not allowed"));
        assert!(result.contains("X-API-Key"));
    }

    #[tokio::test]
    async fn test_non_auth_headers_are_not_blocked_by_manual_auth_guard() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let result = tool
            .call(
                r#"{
                    "method": "GET",
                    "url": "http://example.com/api",
                    "headers": {
                        "Accept": "application/json"
                    }
                }"#,
            )
            .await
            .unwrap();
        assert!(result.contains("only HTTPS"));
    }

    #[tokio::test]
    async fn test_unknown_auth_profile() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let result = tool
            .call(
                r#"{"method": "GET", "url": "https://api.example.com/test", "auth_profile": "nonexistent"}"#,
            )
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown auth profile"));
    }

    #[tokio::test]
    async fn call_with_status_outcome_returns_none_for_validation_failure() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tool = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx);
        let outcome = tool
            .call_with_status_outcome(
                r#"{"method": "GET", "url": "http://example.com/api"}"#,
                None,
            )
            .await
            .unwrap();
        assert!(outcome.metadata.http_status.is_none());
        assert!(outcome.output.contains("HTTPS") || outcome.output.contains("http"));
    }

    #[tokio::test]
    async fn call_and_call_with_status_outcome_produce_same_output() {
        let (tx1, _rx1) = tokio::sync::mpsc::channel(1);
        let tool1 = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx1);
        let (tx2, _rx2) = tokio::sync::mpsc::channel(1);
        let tool2 = HttpRequestTool::new(Arc::new(RwLock::new(HashMap::new())), tx2);

        let args = r#"{"method": "GET", "url": "http://example.com/api"}"#;
        let call_result = tool1.call(args).await.unwrap();
        let outcome_result = tool2.call_with_status_outcome(args, None).await.unwrap();
        assert_eq!(call_result, outcome_result.output);
    }
}
