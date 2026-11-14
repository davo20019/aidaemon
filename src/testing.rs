//! Test infrastructure: MockProvider, TestChannel, and TestHarness.
//!
//! Provides a fully wired Agent with a mock LLM and in-memory state,
//! suitable for integration tests that exercise the real agent loop.

use crate::events::EventStoreForTests as _;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};

use crate::agent::Agent;
use crate::channels::{ChannelHub, SessionMap};
use crate::config::{IterationLimitConfig, ModelsConfig, ProviderKind};
use crate::events::EventStore;
use crate::llm_runtime::{SharedLlmRuntime, router_from_models};
use crate::memory::embeddings::EmbeddingService;
use crate::plans::PlanStore;
use crate::providers::ProviderError;
use crate::state::SqliteStateStore;
use crate::tools::RememberFactTool;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::track_requirements::TrackRequirementsTool;
use crate::tools::{SystemInfoTool, TerminalTool};
use crate::traits::{
    ChannelCapabilities, ChatOptions, HubChannel, ProviderResponse, TokenUsage, Tool, ToolCall,
    ToolRole,
};
use crate::types::{ApprovalResponse, MediaMessage};

// ---------------------------------------------------------------------------
// MockProvider
// ---------------------------------------------------------------------------

/// A recorded call to `MockProvider::chat()`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MockChatCall {
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub options: ChatOptions,
}

/// Mock LLM provider that returns scripted responses.
pub struct MockProvider {
    responses: Mutex<Vec<ProviderResponse>>,
    relationship_assessment_responses: Mutex<Vec<ProviderResponse>>,
    /// The request compiler's replies (the owner's units of work), kept
    /// apart from the main-loop queue. With none, it compiles nothing.
    owner_unit_responses: Mutex<Vec<ProviderResponse>>,
    response_delays: Mutex<Vec<Duration>>,
    pub call_log: Mutex<Vec<MockChatCall>>,
    reject_non_default_options: bool,
    strict_responses: bool,
    exhausted_response_calls: Mutex<u32>,
    /// Errors returned, in order, by the next main-loop calls before any
    /// scripted response is used.
    leading_errors: Mutex<Vec<String>>,
    /// Typed provider errors returned, in order, before `leading_errors`.
    leading_provider_errors: Mutex<Vec<ProviderError>>,
    /// Calls asking for one of these JSON-schema names fail (after being
    /// logged), whatever the script holds.
    failing_response_schemas: Vec<String>,
    /// When true (default), relationship-router and progress-evaluator LLM
    /// calls are silently intercepted (no call_log entry): a scripted
    /// relationship answers the former, else an empty response.
    pub skip_planning_calls: bool,
}

impl MockProvider {
    /// Create a provider that always returns "Mock response".
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(Vec::new()),
            relationship_assessment_responses: Mutex::new(Vec::new()),
            owner_unit_responses: Mutex::new(Vec::new()),
            response_delays: Mutex::new(Vec::new()),
            call_log: Mutex::new(Vec::new()),
            reject_non_default_options: false,
            strict_responses: false,
            exhausted_response_calls: Mutex::new(0),
            leading_errors: Mutex::new(Vec::new()),
            leading_provider_errors: Mutex::new(Vec::new()),
            skip_planning_calls: true,
            failing_response_schemas: Vec::new(),
        }
    }

    /// Create a provider with a FIFO queue of scripted responses.
    pub fn with_responses(responses: Vec<ProviderResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            relationship_assessment_responses: Mutex::new(Vec::new()),
            owner_unit_responses: Mutex::new(Vec::new()),
            response_delays: Mutex::new(Vec::new()),
            call_log: Mutex::new(Vec::new()),
            reject_non_default_options: false,
            strict_responses: false,
            exhausted_response_calls: Mutex::new(0),
            leading_errors: Mutex::new(Vec::new()),
            leading_provider_errors: Mutex::new(Vec::new()),
            skip_planning_calls: true,
            failing_response_schemas: Vec::new(),
        }
    }

    /// Create a provider with scripted responses and per-call delays.
    pub fn with_delayed_responses(
        responses: Vec<ProviderResponse>,
        response_delays: Vec<Duration>,
    ) -> Self {
        Self {
            responses: Mutex::new(responses),
            relationship_assessment_responses: Mutex::new(Vec::new()),
            owner_unit_responses: Mutex::new(Vec::new()),
            response_delays: Mutex::new(response_delays),
            call_log: Mutex::new(Vec::new()),
            reject_non_default_options: false,
            strict_responses: false,
            exhausted_response_calls: Mutex::new(0),
            leading_errors: Mutex::new(Vec::new()),
            leading_provider_errors: Mutex::new(Vec::new()),
            skip_planning_calls: true,
            failing_response_schemas: Vec::new(),
        }
    }

    /// Fail the next main-loop calls with these provider errors (as a 400)
    /// before the scripted responses start.
    pub fn with_leading_errors(self, errors: Vec<&str>) -> Self {
        *self.leading_errors.try_lock().expect("fresh mock") =
            errors.into_iter().map(str::to_string).collect();
        self
    }

    /// Fail the next main-loop calls with these typed provider errors
    /// before any `with_leading_errors` and scripted responses.
    pub fn with_leading_provider_errors(self, errors: Vec<ProviderError>) -> Self {
        *self.leading_provider_errors.try_lock().expect("fresh mock") = errors;
        self
    }

    pub fn with_strict_responses(mut self) -> Self {
        self.strict_responses = true;
        self
    }

    /// Fail every call that asks for the JSON schema `name` with a provider
    /// error (a background lane such as a verifier, mid-run).
    pub fn with_failing_response_schema(mut self, name: &str) -> Self {
        self.failing_response_schemas.push(name.to_string());
        self
    }

    /// Script the next turn after an earlier one finished; until then an
    /// empty script answers with the default text.
    pub async fn push_responses(&self, responses: Vec<ProviderResponse>) {
        self.responses.lock().await.extend(responses);
    }

    /// Check after a run as provider errors may be recovered by the agent loop.
    pub async fn assert_response_script_not_exhausted(&self) -> anyhow::Result<()> {
        let unexpected = *self.exhausted_response_calls.lock().await;
        anyhow::ensure!(
            unexpected == 0,
            "mock response script exhausted: {unexpected} unexpected execution-model call(s)"
        );
        Ok(())
    }

    /// Fail when scripted replies were never requested. A run that stops
    /// early (e.g. a new terminal gate) otherwise passes against a longer
    /// script, hiding the regression.
    pub async fn assert_response_script_consumed(&self) -> anyhow::Result<()> {
        let responses = self.responses.lock().await.len();
        let relationships = self.relationship_assessment_responses.lock().await.len();
        anyhow::ensure!(
            responses == 0 && relationships == 0,
            "mock script not fully consumed: {responses} execution, \
             {relationships} relationship response(s) left unused"
        );
        Ok(())
    }

    /// Simulate a provider/model that rejects advanced per-call options.
    /// Useful for testing compatibility fallback paths.
    pub fn rejecting_non_default_options(mut self) -> Self {
        self.reject_non_default_options = true;
        self
    }

    /// Supply the relationship router's dialogue-edge responses, kept apart
    /// from the main-loop queue. A fresh session has no candidate antecedent,
    /// so production does not call the router for a session's first turn.
    pub fn with_relationship_assessments(mut self, responses: Vec<ProviderResponse>) -> Self {
        self.relationship_assessment_responses = Mutex::new(responses);
        self
    }

    /// Supply the request compiler's replies: the owner's units of work for
    /// the next owner messages, in order.
    pub fn with_owner_units(mut self, responses: Vec<ProviderResponse>) -> Self {
        self.owner_unit_responses = Mutex::new(responses);
        self
    }

    /// The request compiler's reply naming `units` (what, asks for change).
    pub fn owner_units(units: &[(&str, bool)]) -> ProviderResponse {
        Self::text_response(
            &json!({
                "schema_version": 1,
                "units": units
                    .iter()
                    .map(|(what, asks_change)| json!({"what": what, "asks_change": asks_change}))
                    .collect::<Vec<_>>(),
            })
            .to_string(),
        )
    }

    /// One typed dialogue edge from the relationship router, naming no exact
    /// antecedent (the dialogue-state resolver binds it when unambiguous).
    pub fn relationship_assessment(request_relationship: &str) -> ProviderResponse {
        Self::text_response(
            &json!({
                "schema_version": 1,
                "confidence": "high",
                "request_relationship": request_relationship,
                "antecedent_user_message_id": Value::Null
            })
            .to_string(),
        )
    }

    /// The executing model declaring one open observation on its own
    /// checklist: the source of a run's stop-time expectations.
    pub fn observation_checklist() -> ProviderResponse {
        Self::tool_call_response(
            "track_requirements",
            r#"{"items":[{"text":"Observe the requested state","status":"pending","requires_observation":true}]}"#,
        )
    }

    /// Helper: build a text-only ProviderResponse.
    pub fn text_response(text: &str) -> ProviderResponse {
        ProviderResponse {
            content: Some(text.to_string()),
            tool_calls: vec![],
            usage: Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: None,
                cache_creation_input_tokens: None,
                model: "mock".to_string(),
                ..Default::default()
            }),
            thinking: None,
            response_note: None,
        }
    }

    /// Helper: build a tool-call ProviderResponse.
    pub fn tool_call_response(tool_name: &str, args: &str) -> ProviderResponse {
        ProviderResponse {
            content: None,
            tool_calls: vec![ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                name: tool_name.to_string(),
                arguments: args.to_string(),
                extra_content: None,
            }],
            usage: Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: None,
                cache_creation_input_tokens: None,
                model: "mock".to_string(),
                ..Default::default()
            }),
            thinking: None,
            response_note: None,
        }
    }

    /// How many times `chat()` was called.
    pub async fn call_count(&self) -> usize {
        self.call_log.lock().await.len()
    }
}

/// Make a test type an SDK provider. The type defines
/// `async fn respond(&self, model: &str, messages: &[Value], tools: &[Value],
/// options: &ChatOptions) -> anyhow::Result<ProviderResponse>`, working in
/// the daemon's chat JSON; errors are the daemon's classified
/// `ProviderError`s.
macro_rules! chat_json_provider {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl aidaemon_core::Provider for $ty {
            fn name(&self) -> &str {
                stringify!($ty)
            }

            async fn complete(
                &self,
                request: &aidaemon_core::ModelRequest,
            ) -> Result<aidaemon_core::ModelResponse, aidaemon_core::ProviderError> {
                use $crate::providers::sdk_backed::{
                    to_chat_messages, to_chat_options, to_chat_tools, to_sdk_error, to_sdk_response,
                };
                let messages = to_chat_messages(&request.messages);
                let tools = to_chat_tools(&request.tools);
                let options = to_chat_options(request);
                self.respond(&request.model, &messages, &tools, &options)
                    .await
                    .map(to_sdk_response)
                    .map_err(
                        |error| match error.downcast::<$crate::providers::ProviderError>() {
                            Ok(error) => to_sdk_error(&error),
                            Err(other) => aidaemon_core::ProviderError::Other(other.to_string()),
                        },
                    )
            }
        }
    };
}
pub(crate) use chat_json_provider;

/// The mock is an SDK provider like every real backend. It scripts and
/// records calls in the daemon's chat JSON, which is what the tests read.
#[async_trait]
impl aidaemon_core::Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn complete(
        &self,
        request: &aidaemon_core::ModelRequest,
    ) -> Result<aidaemon_core::ModelResponse, aidaemon_core::ProviderError> {
        use crate::providers::sdk_backed::{
            to_chat_messages, to_chat_options, to_chat_tools, to_sdk_error, to_sdk_response,
        };
        let messages = to_chat_messages(&request.messages);
        let tools = to_chat_tools(&request.tools);
        let options = to_chat_options(request);
        self.scripted(&request.model, &messages, &tools, &options)
            .await
            .map(to_sdk_response)
            .map_err(|error| match error.downcast::<ProviderError>() {
                Ok(error) => to_sdk_error(&error),
                Err(other) => aidaemon_core::ProviderError::Other(other.to_string()),
            })
    }
}

impl MockProvider {
    async fn scripted(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        // The request compiler runs beside every owner turn: answered from
        // its own queue, never recorded in call_log.
        if messages.iter().any(|m| {
            m.get("content").and_then(|c| c.as_str()).is_some_and(|s| {
                s.contains(aidaemon_runtime::turn::task_planning::REQUEST_UNITS_SYSTEM)
            })
        }) {
            let mut units = self.owner_unit_responses.lock().await;
            if !units.is_empty() {
                return Ok(units.remove(0));
            }
            return Ok(ProviderResponse {
                content: None,
                tool_calls: vec![],
                usage: None,
                thinking: None,
                response_note: None,
            });
        }
        // Silently intercept relationship-router and evaluator calls without
        // recording them in call_log.
        if self.skip_planning_calls {
            let is_relationship_call = messages.iter().any(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("task relationship router"))
            });
            let is_planning_call = is_relationship_call
                || messages.iter().any(|m| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.contains("progress evaluator"))
                });
            if is_planning_call {
                if is_relationship_call {
                    let mut relationships = self.relationship_assessment_responses.lock().await;
                    if !relationships.is_empty() {
                        return Ok(relationships.remove(0));
                    }
                }
                return Ok(ProviderResponse {
                    content: None,
                    tool_calls: vec![],
                    usage: None,
                    thinking: None,
                    response_note: None,
                });
            }
        }

        // Record the call
        self.call_log.lock().await.push(MockChatCall {
            model: model.to_string(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            options: options.clone(),
        });

        // `uncommitted_work` is an advisory recovery hint for the agent's own
        // retry ladder and `cache_key` only routes prompt-cache lookups; neither
        // is a provider option, so a provider that rejects non-default options
        // never sees them.
        let provider_facing = ChatOptions {
            uncommitted_work: false,
            cache_key: None,
            ..options.clone()
        };
        if let crate::traits::ResponseMode::JsonSchema { name, .. } = &options.response_mode
            && self.failing_response_schemas.contains(name)
        {
            return Err(ProviderError::from_status(503, "synthetic schema lane outage").into());
        }
        if self.reject_non_default_options && provider_facing != ChatOptions::default() {
            return Err(ProviderError::from_status(400, "unsupported chat options").into());
        }

        let delay = {
            let mut response_delays = self.response_delays.lock().await;
            if response_delays.is_empty() {
                None
            } else {
                Some(response_delays.remove(0))
            }
        };
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }

        {
            let mut typed = self.leading_provider_errors.lock().await;
            if !typed.is_empty() {
                return Err(typed.remove(0).into());
            }
            drop(typed);
            let mut errors = self.leading_errors.lock().await;
            if !errors.is_empty() {
                let error = errors.remove(0);
                return Err(ProviderError::from_status(400, &error).into());
            }
        }

        // Return next scripted response, or a default
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            if self.strict_responses {
                let mut unexpected = self.exhausted_response_calls.lock().await;
                *unexpected = unexpected.saturating_add(1);
                anyhow::bail!("mock response script exhausted");
            }
            Ok(MockProvider::text_response("Mock response"))
        } else {
            Ok(responses.remove(0))
        }
    }
}

// ---------------------------------------------------------------------------
// TestChannel
// ---------------------------------------------------------------------------

/// Captured message sent via the channel.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SentMessage {
    pub session_id: String,
    pub text: String,
}

/// A test channel that captures all outgoing messages.
pub struct TestChannel {
    pub messages: Mutex<Vec<SentMessage>>,
    pub default_approval: Mutex<ApprovalResponse>,
}

impl TestChannel {
    pub fn new() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            default_approval: Mutex::new(ApprovalResponse::AllowOnce),
        }
    }

    /// Get all messages sent to a specific session.
    #[allow(dead_code)]
    pub async fn messages_for(&self, session_id: &str) -> Vec<String> {
        self.messages
            .lock()
            .await
            .iter()
            .filter(|m| m.session_id == session_id)
            .map(|m| m.text.clone())
            .collect()
    }

    /// Total number of messages sent.
    #[allow(dead_code)]
    pub async fn message_count(&self) -> usize {
        self.messages.lock().await.len()
    }
}

#[async_trait]
impl HubChannel for TestChannel {
    fn name(&self) -> String {
        "test".to_string()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: false,
            media: false,
            max_message_len: 4096,
        }
    }

    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        self.messages.lock().await.push(SentMessage {
            session_id: session_id.to_string(),
            text: text.to_string(),
        });
        Ok(())
    }

    async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
        Ok(())
    }

    async fn request_approval(
        &self,
        _session_id: &str,
        _command: &str,
        _risk_level: RiskLevel,
        _warnings: &[String],
        _permission_mode: PermissionMode,
        _one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        Ok(self.default_approval.lock().await.clone())
    }
}

pub use aidaemon_channels::testing::CardChat;

// ---------------------------------------------------------------------------
// TestHarness
// ---------------------------------------------------------------------------

/// One owner-approval card a [`ScriptedApprover`] received.
#[derive(Debug, Clone)]
pub struct SeenCard {
    pub command: String,
    pub session_id: String,
    pub risk_level: RiskLevel,
    pub warnings: Vec<String>,
    /// The mode the card was sent with: it picks the buttons offered.
    pub permission_mode: crate::tools::command_risk::PermissionMode,
    /// The card's typed facts, when the request gave them.
    pub card: Option<aidaemon_core::ApprovalCard>,
}

/// Answers owner-approval cards from a script, in order: `Some(answer)`
/// answers at once; `None` (or a script that ran out) holds the card
/// unanswered, keeping its reply channel so a test can answer it late.
pub struct ScriptedApprover {
    pub cards: Arc<std::sync::Mutex<Vec<SeenCard>>>,
    pub held: Arc<std::sync::Mutex<Vec<tokio::sync::oneshot::Sender<ApprovalResponse>>>>,
}

impl ScriptedApprover {
    /// A broker whose cards this approver answers.
    pub fn start(script: Vec<Option<ApprovalResponse>>) -> (crate::tools::ApprovalBroker, Self) {
        let (tx, mut rx) = mpsc::channel::<crate::tools::terminal::ApprovalRequest>(8);
        let cards = Arc::new(std::sync::Mutex::new(Vec::new()));
        let held = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (seen, kept) = (cards.clone(), held.clone());
        tokio::spawn(async move {
            let mut script = script.into_iter();
            while let Some(request) = rx.recv().await {
                seen.lock().unwrap().push(SeenCard {
                    command: request.command.clone(),
                    session_id: request.session_id.clone(),
                    risk_level: request.risk_level,
                    warnings: request.warnings.clone(),
                    permission_mode: request.permission_mode,
                    card: request.card.as_deref().cloned(),
                });
                match script.next().flatten() {
                    Some(answer) => {
                        let _ = request.response_tx.send(answer);
                    }
                    None => kept.lock().unwrap().push(request.response_tx),
                }
            }
        });
        (crate::tools::ApprovalBroker::new(tx), Self { cards, held })
    }

    pub fn card_count(&self) -> usize {
        self.cards.lock().unwrap().len()
    }
}

/// Everything needed to run integration tests against the agent.
#[allow(dead_code)]
pub struct TestHarness {
    pub agent: Agent,
    pub state: Arc<SqliteStateStore>,
    pub provider: Arc<MockProvider>,
    pub channel: Arc<TestChannel>,
    /// Keep the temp file alive — DB is deleted when this drops.
    _db_file: tempfile::NamedTempFile,
    /// Keep the skills temp dir alive.
    _skills_dir: tempfile::TempDir,
}

/// What keeps a [`TestHarness`]'s database and skills directory alive after
/// its agent is taken out with [`TestHarness::into_agent`].
pub struct HarnessFiles {
    pub state: Arc<SqliteStateStore>,
    _db_file: tempfile::NamedTempFile,
    _skills_dir: tempfile::TempDir,
}

impl TestHarness {
    /// The agent, to hand to something that owns it, and the files that must
    /// outlive it.
    pub fn into_agent(self) -> (Agent, HarnessFiles) {
        (
            self.agent,
            HarnessFiles {
                state: self.state,
                _db_file: self._db_file,
                _skills_dir: self._skills_dir,
            },
        )
    }
}

/// Build a fully-wired agent with mock provider and temp-file SQLite DB.
///
/// Each call creates an isolated database, so tests can run in parallel.
pub use aidaemon_tools::testing::{CallWithFacts, call_facts, call_with_facts};

/// The `fake_mcp_server` test binary of `aidaemon-mcp`: built once per test
/// process into its own target directory (so it never waits on the build
/// lock of the running `cargo test`).
pub fn fake_mcp_server_path() -> PathBuf {
    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let exe = std::env::current_exe().expect("test binary path");
        // target/<profile>/deps/<test binary>
        let target = exe
            .ancestors()
            .nth(3)
            .expect("test binary under a target directory")
            .join("fake-mcp");
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        let status = std::process::Command::new(cargo)
            .args([
                "build",
                "--quiet",
                "-p",
                "aidaemon-mcp",
                "--features",
                "test-server",
                "--bin",
                "fake_mcp_server",
                "--target-dir",
            ])
            .arg(&target)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .status()
            .expect("run cargo build for fake_mcp_server");
        assert!(status.success(), "building fake_mcp_server failed");
        target
            .join("debug")
            .join(format!("fake_mcp_server{}", std::env::consts::EXE_SUFFIX))
    })
    .clone()
}

/// An MCP registry with the synthetic `fake` server (`fake_mcp_server`)
/// running: its tools are named `fake__<tool>` and offered on `mcp:fake`.
pub async fn fake_mcp_registry(state: Arc<SqliteStateStore>) -> crate::mcp::McpRegistry {
    let registry = crate::mcp::McpRegistry::new(state, crate::config::secret_store().clone());
    registry
        .add_server(
            "fake",
            crate::config::McpServerConfig {
                command: fake_mcp_server_path().to_string_lossy().into_owned(),
                args: Vec::new(),
                env: HashMap::new(),
            },
            false,
        )
        .await
        .expect("start the fake MCP server");
    registry
}

/// The attachment inbox of the agent [`setup_test_agent`] builds.
pub const TEST_INBOX_DIR: &str = "/tmp/aidaemon-test-attachments";

pub async fn setup_test_agent(provider: MockProvider) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(provider, vec![], None, true, true, false, None).await
}

/// Like [`setup_test_agent`], with every turn on the SDK kernel and the mock
/// provider behind the daemon's provider stack.
pub async fn setup_test_agent_sdk(provider: MockProvider) -> anyhow::Result<TestHarness> {
    setup_test_agent_sdk_with_llm_timeout(provider, None).await
}

/// Like [`setup_test_agent_sdk`], with a per-call model timeout.
pub async fn setup_test_agent_sdk_with_llm_timeout(
    provider: MockProvider,
    llm_call_timeout_secs: Option<u64>,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(
        provider,
        vec![],
        llm_call_timeout_secs,
        true,
        true,
        false,
        None,
    )
    .await
}

/// Build a chat-path harness with the real mandate management tool and a
/// deterministic one-time owner confirmation responder.
pub async fn setup_test_agent_with_mandates(provider: MockProvider) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(provider, vec![], None, true, true, true, None).await
}

/// Build a test agent with a custom policy config (e.g. trust_tier overrides)
/// and extra tools.
#[allow(dead_code)]
pub async fn setup_test_agent_with_policy(
    provider: MockProvider,
    policy_config: crate::config::PolicyConfig,
    extra_tools: Vec<Arc<dyn Tool>>,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(
        provider,
        extra_tools,
        None,
        true,
        true,
        false,
        Some(policy_config),
    )
    .await
}

/// Build a root-mode test agent without forcing executor-mode loop behavior.
#[allow(dead_code)]
pub async fn setup_test_agent_root(provider: MockProvider) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(provider, vec![], None, false, true, false, None).await
}

/// Root-mode test agent with an explicit policy configuration.
#[allow(dead_code)]
pub async fn setup_test_agent_root_with_policy(
    provider: MockProvider,
    policy_config: crate::config::PolicyConfig,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(
        provider,
        vec![],
        None,
        false,
        true,
        false,
        Some(policy_config),
    )
    .await
}

/// Build a root-mode test agent with extra tools and an optional per-LLM timeout.
#[allow(dead_code)]
pub async fn setup_test_agent_root_with_extra_tools_and_llm_timeout(
    provider: MockProvider,
    extra_tools: Vec<Arc<dyn Tool>>,
    llm_call_timeout_secs: Option<u64>,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(
        provider,
        extra_tools,
        llm_call_timeout_secs,
        false,
        true,
        false,
        None,
    )
    .await
}

/// Build a root-mode test agent with an exact toolset, without the default
/// SystemInfo/RememberFact tools. Useful for focused policy and tool-surface tests.
#[allow(dead_code)]
pub async fn setup_test_agent_root_with_only_tools_and_llm_timeout(
    provider: MockProvider,
    tools: Vec<Arc<dyn Tool>>,
    llm_call_timeout_secs: Option<u64>,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(
        provider,
        tools,
        llm_call_timeout_secs,
        false,
        false,
        false,
        None,
    )
    .await
}

/// Build a test agent with extra tools and an optional per-LLM-call timeout.
pub async fn setup_test_agent_with_extra_tools_and_llm_timeout(
    provider: MockProvider,
    extra_tools: Vec<Arc<dyn Tool>>,
    llm_call_timeout_secs: Option<u64>,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_internal(
        provider,
        extra_tools,
        llm_call_timeout_secs,
        true,
        true,
        false,
        None,
    )
    .await
}

async fn setup_test_agent_internal(
    provider: MockProvider,
    extra_tools: Vec<Arc<dyn Tool>>,
    llm_call_timeout_secs: Option<u64>,
    use_test_executor_mode: bool,
    include_default_tools: bool,
    include_manage_mandates: bool,
    policy_config: Option<crate::config::PolicyConfig>,
) -> anyhow::Result<TestHarness> {
    // Temp file for SQLite (pool needs a real file, not :memory:)
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();

    // Temp dir for skills
    let skills_dir = tempfile::TempDir::new()?;

    // Embedding service (downloads ~25MB model on first run, cached afterwards)
    let embedding_service = Arc::new(EmbeddingService::deterministic());

    // State store
    let state =
        Arc::new(SqliteStateStore::for_tests(&db_path, 100, None, embedding_service).await?);

    // Event store (reuse the same pool/options as the state store)
    let event_store = Arc::new(EventStore::new(state.pool()).await?);

    // Provider
    let provider = Arc::new(provider);

    // Tools — SystemInfoTool + RememberFactTool (no side effects, no approval)
    let mut tools: Vec<Arc<dyn Tool>> = if include_default_tools {
        vec![
            Arc::new(SystemInfoTool::new(
                crate::execution::active_execution_backend(),
            )),
            Arc::new(RememberFactTool::new(state.memory_stores())),
            Arc::new(TrackRequirementsTool::new(Arc::new(
                PlanStore::new(state.pool()).await?,
            ))),
        ]
    } else {
        Vec::new()
    };
    if include_manage_mandates {
        tools.push(Arc::new(
            crate::tools::ManageMandatesTool::new(
                state.clone() as Arc<dyn crate::traits::StateStore>
            )
            .with_strategy_skills(Some(
                crate::startup::tool_config::strategy_skills_dir(skills_dir.path().to_path_buf()),
            )),
        ));
    }
    tools.extend(extra_tools);

    // Models config (all tiers point to "mock-model")
    let models_config = ModelsConfig {
        default_model: "mock-model".to_string(),
        fallback_models: Vec::new(),
        primary: "mock-model".to_string(),
        fast: "mock-model".to_string(),
        smart: "mock-model".to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as crate::llm_runtime::ModelBackend,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );
    let goal_token_registry = crate::goal_tokens::GoalTokenRegistry::new();

    let mut agent = crate::agent::new_agent(crate::agent::AgentParts {
        dependencies: crate::agent::AgentDependencies {
            memory_store: state.memory_stores(),
            side_calls: crate::llm_runtime::gateway::SideCallGateway::new(
                llm_runtime.clone(),
                state.clone(),
                event_store.clone(),
            ),
            llm_runtime,
            state: state.clone() as Arc<dyn crate::traits::StateStore>,
            event_store,
            tools,
        },
        model: "mock-model".to_string(),
        system_prompt: "You are a helpful test assistant.".to_string(),
        config_path: PathBuf::from("config.toml"),
        daemon_paths: Default::default(),
        execution_backend: crate::execution::active_execution_backend(),
        skills_dir: skills_dir.path().to_path_buf(),
        max_depth: 3,
        max_iterations: 50,
        max_iterations_cap: 100,
        max_response_chars: 8000,
        timeout_secs: 30,
        max_facts: 20,
        iteration_config: IterationLimitConfig::Unlimited,
        task_token_budget: None,
        spawn_family: Default::default(),
        llm_call_timeout_secs,
        mcp_registry: None,
        goal_token_registry: Some(goal_token_registry),
        hub: None,
        record_decision_points: true,
        context_window_config: crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        policy_config: policy_config.unwrap_or_else(test_policy_config),
        path_aliases: crate::config::PathAliasConfig::default(),
        inbox_dir: Some(std::path::PathBuf::from(TEST_INBOX_DIR)),
        inherited_project_scope: None,
        specialists: Arc::new(crate::agent::specialists::SpecialistRegistry::load(None)),
        interactive_slot: None,
        vision_config: crate::config::VisionConfig::from_files(
            &crate::config::FilesConfig::default(),
        ),
        audio_config: crate::config::AudioConfig::from_files(&crate::config::FilesConfig::default()),
        stt_config: crate::config::SttConfig::from_files(&crate::config::FilesConfig::default()),
        harness_eval_config: (&crate::config::DiagnosticsHarnessEvalConfig::default()).into(),
    });

    if include_manage_mandates {
        // The owner confirms every mandate card the kernel sends (once);
        // any other card goes unanswered and is declined, as on a harness
        // with no owner.
        let (approval_tx, mut approval_rx) = mpsc::channel(8);
        agent
            .owner_approvals()
            .set_broker(crate::tools::ApprovalBroker::new(approval_tx));
        tokio::spawn(async move {
            while let Some(request) = approval_rx.recv().await {
                if matches!(
                    request.kind,
                    crate::types::ApprovalKind::GoalConfirmation
                        | crate::types::ApprovalKind::AutopilotConfirmation
                ) {
                    let _ = request.response_tx.send(ApprovalResponse::AllowOnce);
                }
            }
        });
    }

    if use_test_executor_mode {
        // Set executor mode so integration tests exercise the execution loop directly,
        // bypassing orchestrator routing and first-pass intent classification.
        agent.set_test_executor_mode();
    }

    // Channel (not wired to hub — tests call agent.handle_message directly)
    let channel = Arc::new(TestChannel::new());

    Ok(TestHarness {
        agent,
        state,
        provider,
        channel,
        _db_file: db_file,
        _skills_dir: skills_dir,
    })
}

/// Build a test agent with non-uniform model tiers.
///
/// The `primary_model` name is used as the agent's default model and for the
/// Primary router tier.  `smart_model` is used for the Smart tier (and Fast
/// tier).  Because `MockProvider` ignores model names (pops from its response
/// queue), the different names only affect model-routing behavior.
#[allow(dead_code)]
pub async fn setup_test_agent_with_models(
    provider: MockProvider,
    primary_model: &str,
    smart_model: &str,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_with_models_and_policy(
        provider,
        primary_model,
        smart_model,
        crate::testing::test_policy_config(),
    )
    .await
}

/// Like [`setup_test_agent_with_models`] but with a custom policy config
/// (e.g. trust_tier overrides).
#[allow(dead_code)]
pub async fn setup_test_agent_with_models_and_policy(
    provider: MockProvider,
    primary_model: &str,
    smart_model: &str,
    policy_config: crate::config::PolicyConfig,
) -> anyhow::Result<TestHarness> {
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state =
        Arc::new(SqliteStateStore::for_tests(&db_path, 100, None, embedding_service).await?);

    let event_store = Arc::new(EventStore::new(state.pool()).await?);

    let provider = Arc::new(provider);

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool::new(
            crate::execution::active_execution_backend(),
        )),
        Arc::new(RememberFactTool::new(state.memory_stores())),
        Arc::new(TrackRequirementsTool::new(Arc::new(
            PlanStore::new(state.pool()).await?,
        ))),
    ];

    // Configure default + fallback so router is enabled in tests.
    let models_config = ModelsConfig {
        default_model: primary_model.to_string(),
        fallback_models: vec![smart_model.to_string()],
        primary: primary_model.to_string(),
        fast: smart_model.to_string(),
        smart: smart_model.to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as crate::llm_runtime::ModelBackend,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );
    let goal_token_registry = crate::goal_tokens::GoalTokenRegistry::new();

    let agent = crate::agent::new_agent(crate::agent::AgentParts {
        dependencies: crate::agent::AgentDependencies {
            memory_store: state.memory_stores(),
            side_calls: crate::llm_runtime::gateway::SideCallGateway::new(
                llm_runtime.clone(),
                state.clone(),
                event_store.clone(),
            ),
            llm_runtime,
            state: state.clone() as Arc<dyn crate::traits::StateStore>,
            event_store,
            tools,
        },
        model: primary_model.to_string(),
        system_prompt: "You are a helpful test assistant.".to_string(),
        config_path: PathBuf::from("config.toml"),
        daemon_paths: Default::default(),
        execution_backend: crate::execution::active_execution_backend(),
        skills_dir: skills_dir.path().to_path_buf(),
        max_depth: 3,
        max_iterations: 50,
        max_iterations_cap: 100,
        max_response_chars: 8000,
        timeout_secs: 30,
        max_facts: 20,
        iteration_config: IterationLimitConfig::Unlimited,
        task_token_budget: None,
        spawn_family: Default::default(),
        llm_call_timeout_secs: None,
        mcp_registry: None,
        goal_token_registry: Some(goal_token_registry),
        hub: None,
        record_decision_points: true,
        context_window_config: crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        policy_config,
        path_aliases: crate::config::PathAliasConfig::default(),
        inbox_dir: None,
        inherited_project_scope: None,
        specialists: Arc::new(crate::agent::specialists::SpecialistRegistry::load(None)),
        interactive_slot: None,
        vision_config: crate::config::VisionConfig::from_files(
            &crate::config::FilesConfig::default(),
        ),
        audio_config: crate::config::AudioConfig::from_files(&crate::config::FilesConfig::default()),
        stt_config: crate::config::SttConfig::from_files(&crate::config::FilesConfig::default()),
        harness_eval_config: (&crate::config::DiagnosticsHarnessEvalConfig::default()).into(),
    });
    // Note: keeps orchestrator mode (depth=0) — used by orchestration tests

    let channel = Arc::new(TestChannel::new());

    Ok(TestHarness {
        agent,
        state,
        provider,
        channel,
        _db_file: db_file,
        _skills_dir: skills_dir,
    })
}

/// Build a test agent in orchestrator mode with non-uniform model tiers.
///
/// This enables model routing while retaining the full execution loop.
#[allow(dead_code)]
pub async fn setup_test_agent_orchestrator(provider: MockProvider) -> anyhow::Result<TestHarness> {
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state =
        Arc::new(SqliteStateStore::for_tests(&db_path, 100, None, embedding_service).await?);

    let event_store = Arc::new(EventStore::new(state.pool()).await?);

    let provider = Arc::new(provider);

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool::new(
            crate::execution::active_execution_backend(),
        )),
        Arc::new(RememberFactTool::new(state.memory_stores())),
        Arc::new(TrackRequirementsTool::new(Arc::new(
            PlanStore::new(state.pool()).await?,
        ))),
    ];

    // Configure default + fallback so router is enabled in tests.
    let models_config = ModelsConfig {
        default_model: "primary-model".to_string(),
        fallback_models: vec!["fast-model".to_string(), "smart-model".to_string()],
        primary: "primary-model".to_string(),
        fast: "fast-model".to_string(),
        smart: "smart-model".to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as crate::llm_runtime::ModelBackend,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );
    let goal_token_registry = crate::goal_tokens::GoalTokenRegistry::new();

    let agent = crate::agent::new_agent(crate::agent::AgentParts {
        dependencies: crate::agent::AgentDependencies {
            memory_store: state.memory_stores(),
            side_calls: crate::llm_runtime::gateway::SideCallGateway::new(
                llm_runtime.clone(),
                state.clone(),
                event_store.clone(),
            ),
            llm_runtime,
            state: state.clone() as Arc<dyn crate::traits::StateStore>,
            event_store,
            tools,
        },
        model: "primary-model".to_string(),
        system_prompt: "You are a helpful test assistant.".to_string(),
        config_path: PathBuf::from("config.toml"),
        daemon_paths: Default::default(),
        execution_backend: crate::execution::active_execution_backend(),
        skills_dir: skills_dir.path().to_path_buf(),
        max_depth: 3,
        max_iterations: 50,
        max_iterations_cap: 100,
        max_response_chars: 8000,
        timeout_secs: 30,
        max_facts: 20,
        iteration_config: IterationLimitConfig::Unlimited,
        task_token_budget: None,
        spawn_family: Default::default(),
        llm_call_timeout_secs: None,
        mcp_registry: None,
        goal_token_registry: Some(goal_token_registry),
        hub: None,
        record_decision_points: true,
        context_window_config: crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        policy_config: crate::testing::test_policy_config(),
        path_aliases: crate::config::PathAliasConfig::default(),
        inbox_dir: None,
        inherited_project_scope: None,
        specialists: Arc::new(crate::agent::specialists::SpecialistRegistry::load(None)),
        interactive_slot: None,
        vision_config: crate::config::VisionConfig::from_files(
            &crate::config::FilesConfig::default(),
        ),
        audio_config: crate::config::AudioConfig::from_files(&crate::config::FilesConfig::default()),
        stt_config: crate::config::SttConfig::from_files(&crate::config::FilesConfig::default()),
        harness_eval_config: (&crate::config::DiagnosticsHarnessEvalConfig::default()).into(),
    });

    let channel = Arc::new(TestChannel::new());

    Ok(TestHarness {
        agent,
        state,
        provider,
        channel,
        _db_file: db_file,
        _skills_dir: skills_dir,
    })
}

/// Build a test agent in orchestrator mode with task leads enabled.
///
/// Currently identical to `setup_test_agent_orchestrator()` (task leads are always-on),
/// but kept as a convenience wrapper for integration tests.
#[allow(dead_code)]
pub async fn setup_test_agent_orchestrator_task_leads(
    provider: MockProvider,
) -> anyhow::Result<TestHarness> {
    setup_test_agent_orchestrator(provider).await
}

// ---------------------------------------------------------------------------
// MockTool — configurable fake tool for testing
// ---------------------------------------------------------------------------

/// A configurable mock tool for simulating any tool in tests.
#[allow(dead_code)]
pub struct MockTool {
    tool_name: String,
    tool_description: String,
    return_value: String,
    role: ToolRole,
    available: bool,
    return_metadata: crate::traits::ToolCallMetadata,
    return_finish: Option<aidaemon_core::Finish>,
    dispatch_facts: crate::traits::ToolDispatchFacts,
    output_trust: aidaemon_core::SourceTrust,
}

#[allow(dead_code)]
impl MockTool {
    pub fn new(name: &str, description: &str, return_value: &str) -> Self {
        Self {
            tool_name: name.to_string(),
            tool_description: description.to_string(),
            return_value: return_value.to_string(),
            role: ToolRole::Action,
            available: true,
            return_metadata: Default::default(),
            return_finish: None,
            dispatch_facts: Default::default(),
            output_trust: aidaemon_core::SourceTrust::Trusted,
        }
    }

    /// Its output is written by someone other than the owner, as a real web
    /// or channel tool declares.
    pub fn untrusted(mut self) -> Self {
        self.output_trust = aidaemon_core::SourceTrust::Untrusted;
        self
    }

    /// State these dispatch facts, as the real tool of this name does.
    pub fn with_dispatch_facts(mut self, facts: crate::traits::ToolDispatchFacts) -> Self {
        self.dispatch_facts = facts;
        self
    }

    /// End the run on this tool's call, as a tool's own typed finish.
    pub fn with_finish(mut self, finish: aidaemon_core::Finish) -> Self {
        self.return_finish = Some(finish);
        self
    }

    pub fn with_role(mut self, role: ToolRole) -> Self {
        self.role = role;
        self
    }

    pub fn with_availability(mut self, available: bool) -> Self {
        self.available = available;
        self
    }

    pub fn with_metadata(mut self, metadata: crate::traits::ToolCallMetadata) -> Self {
        self.return_metadata = metadata;
        self
    }
}

#[async_trait]
impl Tool for MockTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn schema(&self) -> Value {
        json!({
            "name": self.tool_name,
            "description": self.tool_description,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        Ok(crate::traits::ToolCallOutcome {
            output: self.return_value.clone(),
            metadata: self.return_metadata.clone(),
            finish: self.return_finish.clone(),
        })
    }
}

impl crate::traits::ToolSemantics for MockTool {
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        self.output_trust
    }

    fn tool_role(&self) -> ToolRole {
        self.role
    }

    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        self.dispatch_facts
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ---------------------------------------------------------------------------
// FullStackTestHarness — agent + TerminalTool + ChannelHub wiring
// ---------------------------------------------------------------------------

/// Full-stack test harness with TerminalTool and ChannelHub approval wiring.
///
/// Unlike `TestHarness`, this includes a real `TerminalTool` in Yolo mode
/// (auto-approves all commands) and a `ChannelHub` with an approval listener,
/// enabling tests that exercise real shell commands through the agent loop.
#[allow(dead_code)]
pub struct FullStackTestHarness {
    pub agent: Agent,
    pub state: Arc<SqliteStateStore>,
    pub provider: Arc<MockProvider>,
    pub channel: Arc<TestChannel>,
    pub hub: Arc<ChannelHub>,
    pub session_map: SessionMap,
    _db_file: tempfile::NamedTempFile,
    _skills_dir: tempfile::TempDir,
    _approval_task: tokio::task::JoinHandle<()>,
}

/// Build a full-stack agent with TerminalTool + ChannelHub approval wiring.
///
/// The TerminalTool runs in Yolo mode (auto-approves everything), so tests
/// can exercise real shell commands without user interaction.
#[allow(dead_code)]
pub async fn setup_full_stack_test_agent(
    provider: MockProvider,
) -> anyhow::Result<FullStackTestHarness> {
    setup_full_stack_test_agent_with_extra_tools(provider, vec![]).await
}

/// Build a full-stack agent with TerminalTool + ChannelHub + extra tools.
///
/// Use this when tests need additional mock tools (e.g., a mock `cli_agent`).
#[allow(dead_code)]
pub async fn setup_full_stack_test_agent_with_extra_tools(
    provider: MockProvider,
    extra_tools: Vec<Arc<dyn Tool>>,
) -> anyhow::Result<FullStackTestHarness> {
    // Temp file for SQLite
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();

    // Temp dir for skills
    let skills_dir = tempfile::TempDir::new()?;

    // Embedding service
    let embedding_service = Arc::new(EmbeddingService::deterministic());

    // State store
    let state =
        Arc::new(SqliteStateStore::for_tests(&db_path, 100, None, embedding_service).await?);

    // Reuse the state store pool for all DB-backed components so tests match production.
    let pool = state.pool();
    let event_store = Arc::new(EventStore::new(pool.clone()).await?);

    // Approval channel
    let (approval_tx, approval_rx) = mpsc::channel(16);
    let approval_tx = crate::tools::ApprovalBroker::new(approval_tx);

    // TerminalTool in Yolo mode — auto-approves everything
    let terminal_tool = Arc::new(
        TerminalTool::new(
            crate::execution::active_execution_backend(),
            vec!["*".to_string()],
            approval_tx,
            30,
            8000,
            PermissionMode::Yolo,
            pool.clone(),
        )
        .await,
    );

    // Tools: SystemInfoTool + RememberFactTool + TerminalTool + extras
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool::new(
            crate::execution::active_execution_backend(),
        )),
        Arc::new(RememberFactTool::new(state.memory_stores())),
        terminal_tool,
    ];
    tools.extend(extra_tools);

    // Provider
    let provider = Arc::new(provider);

    // Models config
    let models_config = ModelsConfig {
        default_model: "mock-model".to_string(),
        fallback_models: Vec::new(),
        primary: "mock-model".to_string(),
        fast: "mock-model".to_string(),
        smart: "mock-model".to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as crate::llm_runtime::ModelBackend,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );
    let goal_token_registry = crate::goal_tokens::GoalTokenRegistry::new();

    let mut agent = crate::agent::new_agent(crate::agent::AgentParts {
        dependencies: crate::agent::AgentDependencies {
            memory_store: state.memory_stores(),
            side_calls: crate::llm_runtime::gateway::SideCallGateway::new(
                llm_runtime.clone(),
                state.clone(),
                event_store.clone(),
            ),
            llm_runtime,
            state: state.clone() as Arc<dyn crate::traits::StateStore>,
            event_store,
            tools,
        },
        model: "mock-model".to_string(),
        system_prompt: "You are a helpful test assistant.".to_string(),
        config_path: PathBuf::from("config.toml"),
        daemon_paths: Default::default(),
        execution_backend: crate::execution::active_execution_backend(),
        skills_dir: skills_dir.path().to_path_buf(),
        max_depth: 3,
        max_iterations: 50,
        max_iterations_cap: 100,
        max_response_chars: 8000,
        timeout_secs: 30,
        max_facts: 20,
        iteration_config: IterationLimitConfig::Unlimited,
        task_token_budget: None,
        spawn_family: Default::default(),
        llm_call_timeout_secs: None,
        mcp_registry: None,
        goal_token_registry: Some(goal_token_registry),
        hub: None,
        record_decision_points: true,
        context_window_config: crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        policy_config: crate::testing::test_policy_config(),
        path_aliases: crate::config::PathAliasConfig::default(),
        inbox_dir: None,
        inherited_project_scope: None,
        specialists: Arc::new(crate::agent::specialists::SpecialistRegistry::load(None)),
        interactive_slot: None,
        vision_config: crate::config::VisionConfig::from_files(
            &crate::config::FilesConfig::default(),
        ),
        audio_config: crate::config::AudioConfig::from_files(&crate::config::FilesConfig::default()),
        stt_config: crate::config::SttConfig::from_files(&crate::config::FilesConfig::default()),
        harness_eval_config: (&crate::config::DiagnosticsHarnessEvalConfig::default()).into(),
    });

    // Set executor mode so tests exercise the execution loop directly
    agent.set_test_executor_mode();

    // Channel
    let channel = Arc::new(TestChannel::new());

    // SessionMap pre-populated with a test session
    let mut map = HashMap::new();
    map.insert("telegram_test".to_string(), "test".to_string());
    let session_map: SessionMap = Arc::new(tokio::sync::RwLock::new(map));

    // ChannelHub
    let hub = Arc::new(ChannelHub::new(
        vec![channel.clone() as Arc<dyn HubChannel>],
        session_map.clone(),
    ));

    // Spawn approval listener
    let hub_for_approvals = hub.clone();
    let approval_task = tokio::spawn(async move {
        hub_for_approvals.approval_listener(approval_rx).await;
    });

    Ok(FullStackTestHarness {
        agent,
        state,
        provider,
        channel,
        hub,
        session_map,
        _db_file: db_file,
        _skills_dir: skills_dir,
        _approval_task: approval_task,
    })
}

/// Policy defaults for the in-process harness. The production default keeps
/// the Guided-tier in-loop supervision calls (pre-execution plan, critique,
/// re-planner) off; the harness turns them on so the tests that script those
/// auxiliary calls keep exercising them. Tests that need the production
/// default construct `PolicyConfig::default()` explicitly.
/// A root agent on a REAL provider and model, for ignored live checks of the
/// agent loop. Tools: system info, remember_fact, read/write file and a
/// yolo terminal (auto-approved). Nothing here is mocked except the channel.
pub struct LiveHarness {
    pub agent: Agent,
    /// For inspecting a live run while debugging; no check reads it yet.
    #[allow(dead_code)]
    pub state: Arc<SqliteStateStore>,
    pub event_store: Arc<EventStore>,
    _db_file: tempfile::NamedTempFile,
    _skills_dir: tempfile::TempDir,
}

pub async fn setup_live_agent(
    provider: crate::llm_runtime::ModelBackend,
    provider_kind: ProviderKind,
    model: &str,
) -> anyhow::Result<LiveHarness> {
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state =
        Arc::new(SqliteStateStore::for_tests(&db_path, 100, None, embedding_service).await?);
    let pool = state.pool();
    let event_store = Arc::new(EventStore::new(pool.clone()).await?);

    let (approval_tx, mut approval_rx) =
        mpsc::channel::<crate::tools::terminal::ApprovalRequest>(16);
    tokio::spawn(async move {
        while let Some(request) = approval_rx.recv().await {
            let _ = request.response_tx.send(ApprovalResponse::AllowOnce);
        }
    });
    let terminal_tool = Arc::new(
        TerminalTool::new(
            crate::execution::active_execution_backend(),
            vec!["*".to_string()],
            crate::tools::ApprovalBroker::new(approval_tx),
            60,
            8000,
            PermissionMode::Yolo,
            pool.clone(),
        )
        .await,
    );
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool::new(
            crate::execution::active_execution_backend(),
        )),
        Arc::new(RememberFactTool::new(state.memory_stores())),
        Arc::new(crate::tools::ReadFileTool::new(
            crate::execution::active_execution_backend(),
        )),
        Arc::new(crate::tools::WriteFileTool::new(
            crate::execution::active_execution_backend(),
        )),
        terminal_tool,
    ];

    let models_config = ModelsConfig {
        default_model: model.to_string(),
        fallback_models: Vec::new(),
        primary: model.to_string(),
        fast: model.to_string(),
        smart: model.to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider,
        router_from_models(models_config.clone()),
        provider_kind,
        models_config.primary.clone(),
    );
    let agent = crate::agent::new_agent(crate::agent::AgentParts {
        dependencies: crate::agent::AgentDependencies {
            memory_store: state.memory_stores(),
            side_calls: crate::llm_runtime::gateway::SideCallGateway::new(
                llm_runtime.clone(),
                state.clone(),
                event_store.clone(),
            ),
            llm_runtime,
            state: state.clone() as Arc<dyn crate::traits::StateStore>,
            event_store: event_store.clone(),
            tools,
        },
        model: model.to_string(),
        system_prompt: "You are aidaemon, a helpful personal assistant with tools.".to_string(),
        config_path: PathBuf::from("config.toml"),
        daemon_paths: Default::default(),
        execution_backend: crate::execution::active_execution_backend(),
        skills_dir: skills_dir.path().to_path_buf(),
        max_depth: 3,
        max_iterations: 50,
        max_iterations_cap: 100,
        max_response_chars: 8000,
        timeout_secs: 120,
        max_facts: 20,
        iteration_config: IterationLimitConfig::Unlimited,
        task_token_budget: None,
        spawn_family: Default::default(),
        llm_call_timeout_secs: None,
        mcp_registry: None,
        goal_token_registry: Some(crate::goal_tokens::GoalTokenRegistry::new()),
        hub: None,
        record_decision_points: true,
        context_window_config: crate::config::ContextWindowConfig::default(),
        policy_config: crate::config::PolicyConfig::default(),
        path_aliases: crate::config::PathAliasConfig::default(),
        inbox_dir: None,
        inherited_project_scope: None,
        specialists: Arc::new(crate::agent::specialists::SpecialistRegistry::load(None)),
        interactive_slot: None,
        vision_config: crate::config::VisionConfig::from_files(
            &crate::config::FilesConfig::default(),
        ),
        audio_config: crate::config::AudioConfig::from_files(&crate::config::FilesConfig::default()),
        stt_config: crate::config::SttConfig::from_files(&crate::config::FilesConfig::default()),
        harness_eval_config: (&crate::config::DiagnosticsHarnessEvalConfig::default()).into(),
    });
    Ok(LiveHarness {
        agent,
        state,
        event_store,
        _db_file: db_file,
        _skills_dir: skills_dir,
    })
}

pub fn test_policy_config() -> crate::config::PolicyConfig {
    crate::config::PolicyConfig {
        pre_execution_supervision: true,
        ..crate::config::PolicyConfig::default()
    }
}

/// Redirects secret storage away from the OS keychain for one test.
///
/// `AIDAEMON_NO_KEYCHAIN` and the runtime env-file path are process-global,
/// and libtest runs every module's tests concurrently in one process. A
/// module-local mutex only orders the tests inside that module; a test in
/// another module that restores (unsets) the variable mid-flight sends the
/// lock holder through the real keychain, where an unsigned test binary
/// either blocks the whole suite behind a macOS Keychain prompt or writes
/// into the developer's keychain. Every test that needs the redirect must go
/// through this one guard so the ordering is crate-wide.
///
/// The guard restores each variable it touched when dropped, so a panicking
/// test cannot leak its redirect into the next one, and a poisoned lock is
/// recovered instead of cascading one failure into every later holder.
pub struct KeychainIsolation {
    // Declared first so the variables are restored before the lock is let go.
    vars: EnvVars,
    _lock: std::sync::MutexGuard<'static, ()>,
    env_file: tempfile::NamedTempFile,
}

impl KeychainIsolation {
    /// Serialize on the crate-wide lock, disable the keychain, and point the
    /// runtime env-file path at a fresh empty temp file.
    pub fn acquire() -> Self {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        let lock = LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let env_file = tempfile::NamedTempFile::new().expect("create isolated env file");
        let mut isolation = Self {
            vars: EnvVars::default(),
            _lock: lock,
            env_file,
        };
        isolation.set("AIDAEMON_NO_KEYCHAIN", "1");
        let env_path = isolation.env_file.path().to_string_lossy().to_string();
        isolation.set(crate::RUNTIME_ENV_FILE_ENV_KEY, &env_path);
        isolation
    }

    /// The temp env file that now backs secret storage.
    pub fn env_file_path(&self) -> &std::path::Path {
        self.env_file.path()
    }

    /// Override another process-global variable for the guard's lifetime.
    pub fn set(&mut self, name: &str, value: &str) {
        self.vars.set(name, value);
    }

    /// Clear another process-global variable for the guard's lifetime.
    pub fn unset(&mut self, name: &str) {
        self.vars.unset(name);
    }
}

/// Process environment overrides for one test, undone when dropped.
///
/// This is the one place the crate's tests write the process environment.
/// It takes no lock of its own: tests that share a variable serialize on
/// `#[serial_test::serial(env_vars)]` or a module lock, and secret-store
/// tests go through [`KeychainIsolation`], which holds one of these.
#[derive(Default)]
pub struct EnvVars {
    previous: Vec<(String, Option<String>)>,
}

impl EnvVars {
    /// Set `name` for the guard's lifetime.
    pub fn set(&mut self, name: &str, value: &str) {
        self.remember(name);
        // SAFETY: test-only. std serializes its own environment reads and
        // writes, and the tests sharing this variable are serialized as
        // described on the type; what remains is foreign code calling
        // getenv concurrently, a risk the test binary accepts.
        unsafe { std::env::set_var(name, value) };
    }

    /// Clear `name` for the guard's lifetime.
    pub fn unset(&mut self, name: &str) {
        self.remember(name);
        // SAFETY: as in `set`.
        unsafe { std::env::remove_var(name) };
    }

    fn remember(&mut self, name: &str) {
        self.previous
            .push((name.to_string(), std::env::var(name).ok()));
    }
}

impl Drop for EnvVars {
    fn drop(&mut self) {
        // Reverse order so a variable set twice lands on its original value.
        for (name, previous) in self.previous.drain(..).rev() {
            // SAFETY: as in `EnvVars::set`.
            unsafe {
                match previous {
                    Some(value) => std::env::set_var(&name, value),
                    None => std::env::remove_var(&name),
                }
            }
        }
    }
}

/// The tools a model request offers: those whose schema it carries, and
/// the deferred ones `load_tools` lists by name (callable, schema sent once
/// loaded).
pub fn offered_tool_names(defs: &[serde_json::Value]) -> Vec<String> {
    let mut names = Vec::new();
    for def in defs {
        let function = def.get("function").unwrap_or(def);
        let Some(name) = function.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        names.push(name.to_string());
        if name == aidaemon_core::LOAD_TOOLS {
            let description = function
                .get("description")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            names.extend(description.lines().filter_map(|line| {
                let entry = line.strip_prefix("- ")?;
                Some(entry.split(':').next()?.trim().to_string())
            }));
        }
    }
    names
}

#[cfg(test)]
mod keychain_isolation_tests {
    use super::{EnvVars, KeychainIsolation};

    #[test]
    fn acquire_redirects_and_drop_restores_original_values() {
        // Only the private marker is asserted after the drop: the shared
        // variables belong to whichever test acquires the lock next.
        let marker = "AIDAEMON_TEST_ISOLATION_MARKER";
        let mut outer = EnvVars::default();
        outer.set(marker, "before");

        {
            let mut isolation = KeychainIsolation::acquire();
            assert_eq!(std::env::var("AIDAEMON_NO_KEYCHAIN").as_deref(), Ok("1"));
            assert_eq!(
                std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY)
                    .ok()
                    .as_deref(),
                Some(isolation.env_file_path().to_string_lossy().as_ref())
            );
            isolation.set(marker, "during");
            isolation.unset(marker);
            isolation.set(marker, "again");
            assert_eq!(std::env::var(marker).as_deref(), Ok("again"));
        }

        assert_eq!(std::env::var(marker).as_deref(), Ok("before"));
        drop(outer);
    }
}
