use std::collections::HashMap;
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
use crate::traits::{Tool, ToolCapabilities};
use crate::types::ApprovalResponse;

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
}

impl HttpRequestTool {
    pub fn new(
        profiles: Arc<RwLock<HashMap<String, HttpAuthProfile>>>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            profiles,
            approval_tx,
        }
    }

    /// Check if a request domain is allowed by the profile's allowed_domains.
    /// Supports exact match and subdomain match (api.twitter.com matches twitter.com)
    /// but NOT suffix tricks (evil-twitter.com does NOT match twitter.com).
    fn domain_matches(request_domain: &str, allowed: &str) -> bool {
        let req = request_domain.to_lowercase();
        let allow = allowed.to_lowercase();
        if req == allow {
            return true;
        }
        req.ends_with(&format!(".{}", allow))
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
    fn apply_auth(
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
            "description": "Make HTTP requests to external APIs with pre-configured auth profiles. Supports OAuth 1.0a, Bearer, Header, and Basic auth. HTTPS only. Write operations require user approval.",
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
                        "description": "Full HTTPS URL"
                    },
                    "auth_profile": {
                        "type": "string",
                        "description": "Name of configured auth profile (e.g. 'twitter', 'stripe')"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional request headers"
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body (JSON string, form data, etc.)"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Content-Type header (auto-detected if omitted)"
                    },
                    "query_params": {
                        "type": "object",
                        "description": "Query parameters appended to URL"
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

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;

        // Parse parameters
        let method = args["method"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: method"))?
            .to_uppercase();
        let url_str = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: url"))?;
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

        // Build URL with query params
        let mut parsed_url =
            reqwest::Url::parse(url_str).map_err(|e| anyhow::anyhow!("Invalid URL: {}", e))?;
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

        // Step 1: HTTPS enforcement
        if parsed_url.scheme() != "https" {
            return Ok("Request blocked: only HTTPS URLs are allowed".to_string());
        }

        // Step 2: SSRF validation
        if let Err(reason) = validate_url_for_ssrf(&url) {
            return Ok(format!("Request blocked: {}", reason));
        }

        // Step 3: Resolve auth profile and check domain
        let profiles_guard = self.profiles.read().await;
        let profile = if let Some(name) = auth_profile_name {
            let p = profiles_guard
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Unknown auth profile: '{}'", name))?;

            // Verify request domain is in allowed_domains
            let request_host = parsed_url.host_str().unwrap_or("");
            let domain_ok = p
                .allowed_domains
                .iter()
                .any(|d| Self::domain_matches(request_host, d));
            if !domain_ok {
                return Ok(format!(
                    "Request blocked: domain '{}' is not in the allowed domains for profile '{}'",
                    request_host, name
                ));
            }
            Some(p)
        } else {
            None
        };

        // Step 4: Auto-detect content type
        let content_type = content_type_param
            .map(|s| s.to_string())
            .or_else(|| body.map(|b| Self::detect_content_type(b).to_string()));

        // Step 5: Scan for secrets in outbound data
        let check_parts = format!(
            "{} {} {}",
            url,
            body.unwrap_or(""),
            args["headers"]
                .as_object()
                .map(|h| serde_json::to_string(h).unwrap_or_default())
                .unwrap_or_default()
        );
        let (_, has_secrets) = sanitize_output(&check_parts);
        if has_secrets {
            return Ok(
                "Request blocked: outbound data appears to contain secrets or credentials. \
                 Review the URL, body, and headers for leaked API keys or tokens."
                    .to_string(),
            );
        }

        // Step 6: Classify risk and request approval
        let risk = Self::classify_risk(&method, profile.is_some());
        let session_id = args["_session_id"].as_str().unwrap_or("unknown");

        if risk != RiskLevel::Safe {
            let mut desc = format!("{} {}", method, url);
            if let Some(name) = auth_profile_name {
                desc.push_str(&format!(" [auth: {}]", name));
            }
            if let Some(b) = body {
                let snippet = if b.len() > 200 { &b[..200] } else { b };
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
                ApprovalResponse::AllowOnce | ApprovalResponse::AllowSession => {}
                // For http_request, we treat AllowAlways same as AllowOnce (Cautious mode)
                ApprovalResponse::AllowAlways => {}
                ApprovalResponse::Deny => {
                    return Ok("Request denied by user".to_string());
                }
            }
        }

        // Step 7: Execute request with manual redirect loop
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?;

        let mut current_url = url.clone();
        let mut current_method = method.clone();
        let original_host = parsed_url.host_str().unwrap_or("").to_string();
        let mut redirect_count = 0;

        let response = loop {
            let mut builder = match current_method.as_str() {
                "GET" => client.get(&current_url),
                "POST" => client.post(&current_url),
                "PUT" => client.put(&current_url),
                "PATCH" => client.patch(&current_url),
                "DELETE" => client.delete(&current_url),
                _ => return Ok(format!("Unsupported method: {}", current_method)),
            };

            // Set content-type and body
            if let Some(ref ct) = content_type {
                builder = builder.header("Content-Type", ct.as_str());
            }
            if let Some(b) = body {
                // Only send body on first request (not redirects)
                if redirect_count == 0 {
                    builder = builder.body(b.to_string());
                }
            }

            // Add custom headers
            if let Some(headers) = args["headers"].as_object() {
                for (k, v) in headers {
                    if let Some(val) = v.as_str() {
                        builder = builder.header(k.as_str(), val);
                    }
                }
            }

            // Apply auth (only if same origin)
            let current_parsed = reqwest::Url::parse(&current_url).unwrap_or(parsed_url.clone());
            let current_host = current_parsed.host_str().unwrap_or("").to_string();
            let same_origin = current_host == original_host;

            if let Some(p) = profile {
                if same_origin {
                    builder = Self::apply_auth(
                        builder,
                        p,
                        &current_method,
                        &current_url,
                        if redirect_count == 0 { body } else { None },
                        content_type.as_deref(),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "{}",
                            Self::strip_credentials_from_error_with(
                                &profiles_guard,
                                &e.to_string()
                            )
                        )
                    })?;
                } else {
                    warn!(
                        "Auth stripped on cross-origin redirect: {} -> {}",
                        original_host, current_host
                    );
                }
            }

            let resp = builder.send().await.map_err(|e| {
                anyhow::anyhow!(
                    "{}",
                    Self::strip_credentials_from_error_with(&profiles_guard, &e.to_string())
                )
            })?;

            // Handle redirects manually
            if follow_redirects && resp.status().is_redirection() {
                redirect_count += 1;
                if redirect_count > MAX_REDIRECTS {
                    return Ok(format!(
                        "Request stopped: exceeded maximum {} redirects",
                        MAX_REDIRECTS
                    ));
                }

                if let Some(location) = resp.headers().get("location") {
                    let location_str = location.to_str().unwrap_or("");
                    let next_url = if location_str.starts_with("http") {
                        location_str.to_string()
                    } else {
                        // Relative redirect
                        let base = reqwest::Url::parse(&current_url).unwrap_or(parsed_url.clone());
                        base.join(location_str)
                            .map(|u| u.to_string())
                            .unwrap_or(location_str.to_string())
                    };

                    // SSRF check on each redirect hop
                    if let Err(reason) = validate_url_for_ssrf(&next_url) {
                        return Ok(format!(
                            "Redirect blocked (hop {}): {}",
                            redirect_count, reason
                        ));
                    }

                    // HTTPS enforcement on redirects
                    if let Ok(next_parsed) = reqwest::Url::parse(&next_url) {
                        if next_parsed.scheme() != "https" {
                            return Ok("Redirect blocked: redirected to non-HTTPS URL".to_string());
                        }
                    }

                    current_url = next_url;
                    // 301/302/303 redirects change method to GET
                    if matches!(resp.status().as_u16(), 301..=303) {
                        current_method = "GET".to_string();
                    }
                    continue;
                } else {
                    return Ok(format!(
                        "Redirect response (HTTP {}) missing Location header",
                        resp.status()
                    ));
                }
            }

            break resp;
        };

        // Step 8: Format response
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

        let bytes = response.bytes().await.map_err(|e| {
            anyhow::anyhow!(
                "{}",
                Self::strip_credentials_from_error_with(&profiles_guard, &e.to_string())
            )
        })?;

        // Binary detection: non-UTF8 returns metadata only
        let body_str = if content_type_resp.starts_with("image/")
            || content_type_resp.starts_with("audio/")
            || content_type_resp.starts_with("video/")
            || content_type_resp.contains("octet-stream")
        {
            format!(
                "[Binary response: {} bytes, content-type: {}]",
                bytes.len(),
                content_type_resp
            )
        } else {
            match String::from_utf8(bytes.to_vec()) {
                Ok(text) => {
                    if text.len() > max_response_bytes as usize {
                        let mut end = max_response_bytes as usize;
                        while end > 0 && !text.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!(
                            "{}\n\n[Truncated: {} bytes total]",
                            &text[..end],
                            text.len()
                        )
                    } else {
                        text
                    }
                }
                Err(_) => {
                    format!(
                        "[Non-UTF8 response: {} bytes, content-type: {}]",
                        bytes.len(),
                        content_type_resp
                    )
                }
            }
        };

        // Sanitize response content (strip prompt injection)
        let sanitized_body = sanitize_external_content(&body_str);

        // Build result with response details
        let status_code = status.as_u16();
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
        Ok(wrap_untrusted_output("http_request", &result))
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
        assert!(schema["parameters"]["properties"]["method"].is_object());
        assert!(schema["parameters"]["properties"]["url"].is_object());
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
}
