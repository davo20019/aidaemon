use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::config::{SearchBackendKind, SearchConfig};
use crate::oauth::{OAuthGateway, SharedHttpProfiles};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::web_search::{BraveBackend, DuckDuckGoBackend, SearchBackend, SearchResult};
use crate::tools::{HttpRequestTool, ManageHttpAuthTool, ManageOAuthTool, ManageSkillsTool};
use crate::traits::{StateStore, Tool, ToolCapabilities};
use crate::types::StatusUpdate;

use super::manage_skills::{ApiSafeProbe, LearnedApiArtifact};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApiAuthMode {
    Existing,
    Oauth2Pkce,
    Oauth2AuthorizationCode,
    Oauth2ClientCredentials,
    Bearer,
    Header,
    Basic,
    Oauth1a,
}

#[derive(Deserialize, Default)]
struct ManageApiArgs {
    action: String,
    service: Option<String>,
    auth_mode: Option<String>,
    allowed_domains: Option<Vec<String>>,
    header_name: Option<String>,
    username: Option<String>,
    user_id: Option<String>,
    display_name: Option<String>,
    authorize_url: Option<String>,
    token_url: Option<String>,
    scopes: Option<Vec<String>>,
    client_id: Option<String>,
    client_secret: Option<String>,
    docs_url: Option<String>,
    openapi_url: Option<String>,
    learn_url: Option<String>,
    learn_kind: Option<String>,
    verify_url: Option<String>,
    verify_method: Option<String>,
    timeout_secs: Option<u64>,
    connect: Option<bool>,
    #[serde(default)]
    _session_id: String,
}

pub struct ManageApiTool {
    manage_http_auth: ManageHttpAuthTool,
    manage_oauth: ManageOAuthTool,
    manage_skills: Option<ManageSkillsTool>,
    http_request: HttpRequestTool,
    search_backend: Box<dyn SearchBackend>,
}

impl ManageApiTool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config_path: PathBuf,
        skills_dir: Option<PathBuf>,
        skill_registry_urls: Vec<String>,
        search_config: SearchConfig,
        profiles: SharedHttpProfiles,
        approval_tx: mpsc::Sender<ApprovalRequest>,
        state_store: Arc<dyn StateStore>,
        oauth_gateway: OAuthGateway,
    ) -> Self {
        let manage_skills = skills_dir.map(|dir| {
            ManageSkillsTool::new(dir, state_store.clone(), approval_tx.clone())
                .with_http_profiles(profiles.clone())
                .with_registries(skill_registry_urls)
        });
        let search_backend: Box<dyn SearchBackend> = match search_config.backend {
            SearchBackendKind::Brave => Box::new(BraveBackend::new(&search_config.api_key)),
            SearchBackendKind::DuckDuckGo => Box::new(DuckDuckGoBackend::new()),
        };

        Self {
            manage_http_auth: ManageHttpAuthTool::new(
                config_path.clone(),
                profiles.clone(),
                approval_tx.clone(),
                state_store.clone(),
            ),
            manage_oauth: ManageOAuthTool::new(
                oauth_gateway,
                state_store,
                config_path,
                approval_tx.clone(),
            ),
            manage_skills,
            http_request: HttpRequestTool::new(profiles, approval_tx),
            search_backend,
        }
    }

    fn validate_service_name(raw: &str) -> anyhow::Result<String> {
        let trimmed = raw.trim();
        anyhow::ensure!(!trimmed.is_empty(), "Service name must not be empty");
        anyhow::ensure!(
            trimmed
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')),
            "Service name '{}' is invalid. Use only letters, numbers, '_' or '-'.",
            trimmed
        );
        Ok(trimmed.to_ascii_lowercase())
    }

    fn parse_auth_mode(raw: Option<&str>) -> anyhow::Result<ApiAuthMode> {
        match raw
            .unwrap_or("existing")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "existing" | "" => Ok(ApiAuthMode::Existing),
            "oauth2_pkce" | "oauth2" | "pkce" => Ok(ApiAuthMode::Oauth2Pkce),
            "oauth2_authorization_code" | "authorization_code" | "auth_code" => {
                Ok(ApiAuthMode::Oauth2AuthorizationCode)
            }
            "oauth2_client_credentials" | "client_credentials" => {
                Ok(ApiAuthMode::Oauth2ClientCredentials)
            }
            "bearer" => Ok(ApiAuthMode::Bearer),
            "header" => Ok(ApiAuthMode::Header),
            "basic" => Ok(ApiAuthMode::Basic),
            "oauth1a" | "oauth_1a" | "oauth-1a" => Ok(ApiAuthMode::Oauth1a),
            other => anyhow::bail!(
                "Unknown auth_mode '{}'. Use one of: existing, oauth2_pkce, oauth2_authorization_code, oauth2_client_credentials, bearer, header, basic, oauth1a.",
                other
            ),
        }
    }

    fn auth_mode_label(auth_mode: ApiAuthMode) -> &'static str {
        match auth_mode {
            ApiAuthMode::Existing => "existing",
            ApiAuthMode::Oauth2Pkce => "oauth2_pkce",
            ApiAuthMode::Oauth2AuthorizationCode => "oauth2_authorization_code",
            ApiAuthMode::Oauth2ClientCredentials => "oauth2_client_credentials",
            ApiAuthMode::Bearer => "bearer",
            ApiAuthMode::Header => "header",
            ApiAuthMode::Basic => "basic",
            ApiAuthMode::Oauth1a => "oauth1a",
        }
    }

    fn verify_method(raw: Option<&str>) -> anyhow::Result<String> {
        let method = raw.unwrap_or("GET").trim().to_ascii_uppercase();
        anyhow::ensure!(
            matches!(method.as_str(), "GET" | "HEAD"),
            "verify_method must be GET or HEAD for safe deterministic probes."
        );
        Ok(method)
    }

    fn session_id(args: &ManageApiArgs) -> &str {
        if args._session_id.is_empty() {
            "unknown"
        } else {
            args._session_id.as_str()
        }
    }

    async fn send_stage(status_tx: Option<&mpsc::Sender<StatusUpdate>>, chunk: String) {
        if let Some(tx) = status_tx {
            let _ = tx
                .send(StatusUpdate::ToolProgress {
                    name: "manage_api".to_string(),
                    chunk,
                })
                .await;
        }
    }

    fn format_live_probe_result(reason: &str, result: &str) -> String {
        let primary = crate::traits::extract_primary_message_content(result, &[]);
        let excerpt = primary.trim();
        if excerpt.is_empty() {
            format!("Live probe result ({}):", reason)
        } else {
            format!("Live probe result ({}):\n{}", reason, excerpt)
        }
    }

    fn learning_target(args: &ManageApiArgs) -> Option<(String, Option<String>)> {
        if let Some(url) = args
            .openapi_url
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            return Some((url.to_string(), Some("openapi".to_string())));
        }
        if let Some(url) = args
            .docs_url
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            return Some((url.to_string(), args.learn_kind.clone()));
        }
        args.learn_url
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(|url| (url.to_string(), args.learn_kind.clone()))
    }

    fn url_looks_like_direct_learning_source(url: &str) -> bool {
        let lower = url.trim().to_ascii_lowercase();
        if lower.ends_with(".json")
            || lower.ends_with(".yaml")
            || lower.ends_with(".yml")
            || lower.contains("/openapi")
            || lower.contains("/swagger")
            || lower.contains("/api-docs")
            || lower.contains("/reference")
            || lower.contains("/docs")
            || lower.contains("/graphql")
        {
            return true;
        }

        let Ok(parsed) = reqwest::Url::parse(url) else {
            return false;
        };
        let path = parsed.path().trim_matches('/');
        path.split('/').count() > 2
    }

    fn host_root(host: &str) -> String {
        let parts: Vec<&str> = host.split('.').collect();
        if parts.len() >= 2 {
            format!("{}.{}", parts[parts.len() - 2], parts[parts.len() - 1])
        } else {
            host.to_string()
        }
    }

    fn discovery_candidate_urls(raw_url: &str) -> anyhow::Result<Vec<(String, Option<String>)>> {
        let parsed = reqwest::Url::parse(raw_url)
            .map_err(|err| anyhow::anyhow!("Invalid API URL '{}': {}", raw_url, err))?;
        let mut roots = Vec::new();

        let mut current = parsed.clone();
        current.set_query(None);
        current.set_fragment(None);
        roots.push(current.clone());

        if current.path() != "/" {
            let mut origin = current.clone();
            origin.set_path("/");
            origin.set_query(None);
            origin.set_fragment(None);
            roots.push(origin);
        }

        let suffixes: &[(&str, Option<&str>)] = &[
            ("openapi.json", Some("openapi")),
            ("swagger.json", Some("openapi")),
            ("swagger.yaml", Some("openapi")),
            ("swagger.yml", Some("openapi")),
            ("api-docs", Some("docs")),
            ("v3/api-docs", Some("openapi")),
            ("reference", Some("docs")),
            ("docs", Some("docs")),
            ("graphql", Some("docs")),
        ];

        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut push = |url: reqwest::Url, kind: Option<&str>| {
            let key = url.to_string();
            if seen.insert(key.clone()) {
                out.push((key, kind.map(ToString::to_string)));
            }
        };

        push(current.clone(), None);
        for root in roots {
            for (suffix, kind) in suffixes {
                if let Ok(joined) = root.join(suffix) {
                    push(joined, *kind);
                }
            }
        }

        Ok(out)
    }

    fn result_host_matches_target(result_url: &str, target_host: &str) -> bool {
        let Ok(parsed) = reqwest::Url::parse(result_url) else {
            return false;
        };
        let Some(host) = parsed.host_str() else {
            return false;
        };
        let target_root = Self::host_root(target_host);
        let host_root = Self::host_root(host);
        host == target_host
            || host.ends_with(&format!(".{}", target_host))
            || host_root == target_root
    }

    async fn search_candidate_learning_urls(
        &self,
        service: &str,
        raw_url: &str,
    ) -> anyhow::Result<Vec<(String, Option<String>)>> {
        let parsed = reqwest::Url::parse(raw_url)
            .map_err(|err| anyhow::anyhow!("Invalid API URL '{}': {}", raw_url, err))?;
        let target_host = parsed
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("API URL '{}' is missing a host", raw_url))?;
        let query = format!(
            "site:{} {} API docs openapi swagger graphql reference",
            target_host, service
        );
        let results = self
            .search_backend
            .search(&query, 8)
            .await
            .unwrap_or_default();

        let mut candidates = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for SearchResult {
            url,
            title,
            snippet,
        } in results
        {
            if !Self::result_host_matches_target(&url, target_host) {
                continue;
            }
            let lower = format!("{} {}", title, snippet).to_ascii_lowercase();
            let inferred_kind = if lower.contains("openapi")
                || lower.contains("swagger")
                || url.to_ascii_lowercase().ends_with(".json")
                || url.to_ascii_lowercase().ends_with(".yaml")
                || url.to_ascii_lowercase().ends_with(".yml")
            {
                Some("openapi".to_string())
            } else if lower.contains("graphql")
                || url.to_ascii_lowercase().contains("/graphql")
                || lower.contains("docs")
                || lower.contains("reference")
            {
                Some("docs".to_string())
            } else {
                None
            };
            if seen.insert(url.clone()) {
                candidates.push((url, inferred_kind));
            }
        }

        Ok(candidates)
    }

    async fn discover_learning_target(
        &self,
        service: &str,
        raw_url: &str,
        requested_kind: Option<&str>,
        status_tx: Option<&mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<Vec<(String, Option<String>)>> {
        let direct_kind = requested_kind.map(ToString::to_string);
        let mut candidates = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut push = |url: String, kind: Option<String>| {
            if seen.insert(url.clone()) {
                candidates.push((url, kind));
            }
        };

        push(raw_url.to_string(), direct_kind.clone());

        if !Self::url_looks_like_direct_learning_source(raw_url) {
            Self::send_stage(
                status_tx,
                format!("Trying same-host API discovery from {}.", raw_url),
            )
            .await;
            for (candidate, kind) in Self::discovery_candidate_urls(raw_url)? {
                push(candidate, kind.or_else(|| direct_kind.clone()));
            }

            Self::send_stage(
                status_tx,
                format!(
                    "Same-host discovery exhausted for {}. Falling back to web search.",
                    raw_url
                ),
            )
            .await;
            for (candidate, kind) in self
                .search_candidate_learning_urls(service, raw_url)
                .await?
            {
                push(candidate, kind.or_else(|| direct_kind.clone()));
            }
        }

        Ok(candidates)
    }

    async fn run_oauth_stage(
        &self,
        service: &str,
        auth_mode: ApiAuthMode,
        args: &ManageApiArgs,
        status_tx: Option<&mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let mut sections = Vec::new();

        let should_register_custom = args.authorize_url.is_some()
            || args.token_url.is_some()
            || args.allowed_domains.is_some()
            || args.display_name.is_some()
            || args.scopes.is_some();
        if should_register_custom {
            Self::send_stage(
                status_tx,
                format!("Registering or updating OAuth provider `{}`.", service),
            )
            .await;
            let register_args = json!({
                "action": "register_provider",
                "service": service,
                "display_name": args.display_name.clone(),
                "auth_type": Self::auth_mode_label(auth_mode),
                "authorize_url": args.authorize_url.clone(),
                "token_url": args.token_url.clone(),
                "scopes": args.scopes.clone(),
                "allowed_domains": args.allowed_domains.clone(),
            });
            sections.push(self.manage_oauth.call(&register_args.to_string()).await?);
        }

        match (&args.client_id, &args.client_secret) {
            (Some(client_id), Some(client_secret)) => {
                Self::send_stage(
                    status_tx,
                    format!("Storing OAuth client credentials for `{}`.", service),
                )
                .await;
                let credential_args = json!({
                    "action": "set_credentials",
                    "service": service,
                    "client_id": client_id,
                    "client_secret": client_secret,
                });
                sections.push(self.manage_oauth.call(&credential_args.to_string()).await?);
            }
            (None, None) => {}
            _ => anyhow::bail!("Provide both client_id and client_secret together, or omit both."),
        }

        if args.connect.unwrap_or(true) {
            Self::send_stage(
                status_tx,
                format!("Starting OAuth connect flow for `{}`.", service),
            )
            .await;
            let connect_args = json!({
                "action": "connect",
                "service": service,
                "_session_id": Self::session_id(args),
            });
            sections.push(
                self.manage_oauth
                    .call_with_status(&connect_args.to_string(), status_tx.cloned())
                    .await?,
            );
        } else if sections.is_empty() {
            sections.push(format!(
                "OAuth onboarding for `{}` was skipped because connect=false and no provider or credential changes were requested.",
                service
            ));
        }

        Ok(sections.join("\n\n"))
    }

    async fn run_manual_auth_stage(
        &self,
        service: &str,
        auth_mode: ApiAuthMode,
        args: &ManageApiArgs,
        status_tx: Option<&mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<(String, bool)> {
        Self::send_stage(
            status_tx,
            format!(
                "Updating manual auth profile `{}` ({}) and refreshing runtime auth state.",
                service,
                Self::auth_mode_label(auth_mode)
            ),
        )
        .await;

        let upsert_args = json!({
            "action": "upsert",
            "profile": service,
            "auth_type": Self::auth_mode_label(auth_mode),
            "allowed_domains": args.allowed_domains.clone(),
            "header_name": args.header_name.clone(),
            "username": args.username.clone(),
            "user_id": args.user_id.clone(),
            "_session_id": Self::session_id(args),
        });
        let upsert_result = self.manage_http_auth.call(&upsert_args.to_string()).await?;

        let sync_args = json!({
            "action": "verify",
            "profile": service,
            "_session_id": Self::session_id(args),
        });
        let sync_result = self.manage_http_auth.call(&sync_args.to_string()).await?;
        let ready = !sync_result.contains("not ready for live API calls yet")
            && !sync_result.contains("missing secure credentials");

        Ok((format!("{}\n\n{}", upsert_result, sync_result), ready))
    }

    async fn run_learning_stage(
        &self,
        service: &str,
        args: &ManageApiArgs,
        status_tx: Option<&mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<Option<LearnedApiArtifact>> {
        let Some((url, learn_kind)) = Self::learning_target(args) else {
            return Ok(None);
        };
        let Some(manage_skills) = &self.manage_skills else {
            return Ok(Some(LearnedApiArtifact {
                output: "Learning skipped: the skills system is disabled for this daemon, so no reusable API guide could be saved."
                    .to_string(),
                safe_probe: None,
            }));
        };

        Self::send_stage(
            status_tx,
            format!("Learning API shape for `{}` from {}.", service, url),
        )
        .await;
        let candidates = self
            .discover_learning_target(service, &url, learn_kind.as_deref(), status_tx)
            .await?;
        let mut failures = Vec::new();

        for (candidate_url, candidate_kind) in candidates {
            let inferred_kind = candidate_kind
                .as_deref()
                .or(learn_kind.as_deref())
                .map(ToString::to_string);
            if candidate_url != url {
                Self::send_stage(
                    status_tx,
                    format!(
                        "Trying discovered API guide source for `{}`: {}.",
                        service, candidate_url
                    ),
                )
                .await;
            }

            match manage_skills
                .learn_api_and_persist(
                    Some(service),
                    Some(service),
                    &candidate_url,
                    inferred_kind.as_deref(),
                )
                .await
            {
                Ok(mut artifact) => {
                    if candidate_url != url {
                        artifact
                            .output
                            .push_str(&format!("\n- discovered source URL: {}", candidate_url));
                    }
                    return Ok(Some(artifact));
                }
                Err(err) => failures.push(format!("{} ({})", candidate_url, err)),
            }
        }

        anyhow::bail!(
            "Failed to learn API shape for '{}'. Tried {} candidate source(s): {}",
            service,
            failures.len(),
            failures.join(" | ")
        )
    }

    async fn run_verify_stage(
        &self,
        service: &str,
        ready_for_live_probe: bool,
        args: &ManageApiArgs,
        derived_probe: Option<ApiSafeProbe>,
        status_tx: Option<&mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        if !ready_for_live_probe {
            return Ok(
                "Live verification skipped: the auth profile is not ready for a real probe yet."
                    .to_string(),
            );
        }

        let probe = if let Some(url) = args
            .verify_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            ApiSafeProbe {
                method: Self::verify_method(args.verify_method.as_deref())?,
                url: url.to_string(),
                body: None,
                content_type: None,
                reason: "User-provided verify_url".to_string(),
            }
        } else if let Some(probe) = derived_probe {
            probe
        } else {
            return Ok(
                "Live verification skipped: no verify_url was provided and no safe probe could be derived from the learned API source."
                    .to_string(),
            );
        };

        Self::send_stage(
            status_tx,
            format!(
                "Running safe live probe for `{}` with {} {}.",
                service, probe.method, probe.url
            ),
        )
        .await;
        let verify_args = json!({
            "method": probe.method,
            "url": probe.url,
            "auth_profile": service,
            "body": probe.body,
            "content_type": probe.content_type,
            "timeout_secs": args.timeout_secs,
            "max_response_bytes": 4096,
            "_session_id": Self::session_id(args),
        });
        self.http_request
            .call(&verify_args.to_string())
            .await
            .map(|result| Self::format_live_probe_result(&probe.reason, &result))
    }

    async fn handle_onboard(
        &self,
        args: ManageApiArgs,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let service = Self::validate_service_name(
            args.service
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("action='onboard' requires 'service'"))?,
        )?;
        let auth_mode = Self::parse_auth_mode(args.auth_mode.as_deref())?;
        let mut sections = vec![format!("API onboarding for `{}`", service)];
        let mut ready_for_live_probe = !matches!(
            auth_mode,
            ApiAuthMode::Bearer | ApiAuthMode::Header | ApiAuthMode::Basic | ApiAuthMode::Oauth1a
        );

        match auth_mode {
            ApiAuthMode::Existing => {
                sections.push(
                    "Auth:\nUsing the existing connected service/profile state without changing auth configuration."
                        .to_string(),
                );
            }
            ApiAuthMode::Oauth2Pkce
            | ApiAuthMode::Oauth2AuthorizationCode
            | ApiAuthMode::Oauth2ClientCredentials => {
                let auth_result = self
                    .run_oauth_stage(&service, auth_mode, &args, status_tx.as_ref())
                    .await?;
                sections.push(format!("Auth:\n{}", auth_result));
            }
            ApiAuthMode::Bearer
            | ApiAuthMode::Header
            | ApiAuthMode::Basic
            | ApiAuthMode::Oauth1a => {
                let (auth_result, manual_ready) = self
                    .run_manual_auth_stage(&service, auth_mode, &args, status_tx.as_ref())
                    .await?;
                ready_for_live_probe = manual_ready;
                sections.push(format!("Auth:\n{}", auth_result));
            }
        }

        let mut derived_probe = None;
        if let Some(learning_artifact) = self
            .run_learning_stage(&service, &args, status_tx.as_ref())
            .await?
        {
            derived_probe = learning_artifact.safe_probe.clone();
            sections.push(format!("Learn:\n{}", learning_artifact.output));
        } else {
            sections.push(
                "Learn:\nLearning skipped: no docs/OpenAPI URL was provided for guide generation."
                    .to_string(),
            );
        }

        let verify_result = self
            .run_verify_stage(
                &service,
                ready_for_live_probe,
                &args,
                derived_probe,
                status_tx.as_ref(),
            )
            .await?;
        sections.push(format!("Verify:\n{}", verify_result));

        Ok(sections.join("\n\n"))
    }
}

#[async_trait]
impl Tool for ManageApiTool {
    fn name(&self) -> &str {
        "manage_api"
    }

    fn description(&self) -> &str {
        "Deterministically connect, learn, and verify external APIs by orchestrating auth, docs/spec ingestion, and a safe live probe"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_api",
            "description": "Deterministic API onboarding: connect auth, learn docs/specs into a skill, and verify with a safe probe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["onboard"],
                        "description": "Action"
                    },
                    "service": {
                        "type": "string",
                        "description": "Service/profile name"
                    },
                    "auth_mode": {
                        "type": "string",
                        "enum": ["existing", "oauth2_pkce", "oauth2_authorization_code", "oauth2_client_credentials", "bearer", "header", "basic", "oauth1a"],
                        "description": "existing, OAuth, or manual auth"
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Allowed API domains"
                    },
                    "header_name": {
                        "type": "string",
                        "description": "Required for header auth"
                    },
                    "username": {
                        "type": "string",
                        "description": "Required for basic auth"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional OAuth1a user/account id"
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Optional provider label"
                    },
                    "authorize_url": {
                        "type": "string",
                        "description": "OAuth authorize URL"
                    },
                    "token_url": {
                        "type": "string",
                        "description": "OAuth token URL"
                    },
                    "scopes": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional OAuth scopes"
                    },
                    "client_id": {
                        "type": "string",
                        "description": "Optional OAuth client_id"
                    },
                    "client_secret": {
                        "type": "string",
                        "description": "Optional OAuth client_secret"
                    },
                    "connect": {
                        "type": "boolean",
                        "description": "Run connect now (default true)"
                    },
                    "docs_url": {
                        "type": "string",
                        "description": "Official HTTPS docs URL or bare API URL"
                    },
                    "openapi_url": {
                        "type": "string",
                        "description": "Official HTTPS OpenAPI/Swagger URL"
                    },
                    "learn_url": {
                        "type": "string",
                        "description": "Generic HTTPS docs/spec/base URL"
                    },
                    "learn_kind": {
                        "type": "string",
                        "enum": ["auto", "openapi", "docs"],
                        "description": "Force learning mode"
                    },
                    "verify_url": {
                        "type": "string",
                        "description": "Optional safe read-only probe URL"
                    },
                    "verify_method": {
                        "type": "string",
                        "enum": ["GET", "HEAD"],
                        "description": "Safe probe method"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Probe timeout"
                    }
                },
                "required": ["action", "service"],
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
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        self.call_with_status(arguments, None).await
    }

    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        let args: ManageApiArgs = serde_json::from_str(arguments)?;
        match args.action.as_str() {
            "onboard" => self.handle_onboard(args, status_tx).await,
            other => Ok(format!("Unknown action '{}'. Use: onboard.", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_candidate_urls_expand_common_docs_and_spec_paths() {
        let candidates = ManageApiTool::discovery_candidate_urls("https://api.example.com")
            .expect("candidate discovery");
        let urls: Vec<String> = candidates.into_iter().map(|(url, _)| url).collect();
        assert!(urls.contains(&"https://api.example.com/openapi.json".to_string()));
        assert!(urls.contains(&"https://api.example.com/swagger.json".to_string()));
        assert!(urls.contains(&"https://api.example.com/docs".to_string()));
        assert!(urls.contains(&"https://api.example.com/graphql".to_string()));
    }

    #[test]
    fn direct_learning_source_detection_distinguishes_base_urls() {
        assert!(ManageApiTool::url_looks_like_direct_learning_source(
            "https://developers.linear.app/openapi.json"
        ));
        assert!(ManageApiTool::url_looks_like_direct_learning_source(
            "https://docs.example.com/reference"
        ));
        assert!(!ManageApiTool::url_looks_like_direct_learning_source(
            "https://api.example.com"
        ));
    }

    #[test]
    fn live_probe_result_omits_untrusted_wrapper_markers() {
        let wrapped = crate::tools::sanitize::wrap_untrusted_output(
            "http_request",
            "HTTP 200 OK\n\n{\"ok\":true}",
        );

        let formatted =
            ManageApiTool::format_live_probe_result("User-provided verify_url", &wrapped);

        assert!(formatted.contains("Live probe result (User-provided verify_url):"));
        assert!(formatted.contains("HTTP 200 OK"));
        assert!(formatted.contains("{\"ok\":true}"));
        assert!(!formatted.contains("UNTRUSTED EXTERNAL DATA"));
    }
}
