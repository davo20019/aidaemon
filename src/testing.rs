//! Test infrastructure: MockProvider, TestChannel, and TestHarness.
//!
//! Provides a fully wired Agent with a mock LLM and in-memory state,
//! suitable for integration tests that exercise the real agent loop.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};

use crate::agent::Agent;
use crate::channels::{ChannelHub, SessionMap};
use crate::config::{IterationLimitConfig, ModelsConfig, ProviderKind};
use crate::events::EventStore;
use crate::llm_runtime::{router_from_models, SharedLlmRuntime};
use crate::memory::embeddings::EmbeddingService;
use crate::state::SqliteStateStore;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::memory::RememberFactTool;
use crate::tools::{SystemInfoTool, TerminalTool};
use crate::traits::{
    Channel, ChannelCapabilities, ModelProvider, ProviderResponse, TokenUsage, Tool, ToolCall,
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
}

/// Mock LLM provider that returns scripted responses.
pub struct MockProvider {
    responses: Mutex<Vec<ProviderResponse>>,
    pub call_log: Mutex<Vec<MockChatCall>>,
}

impl MockProvider {
    /// Create a provider that always returns "Mock response".
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(Vec::new()),
            call_log: Mutex::new(Vec::new()),
        }
    }

    /// Create a provider with a FIFO queue of scripted responses.
    pub fn with_responses(responses: Vec<ProviderResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
            call_log: Mutex::new(Vec::new()),
        }
    }

    /// Helper: build a text-only ProviderResponse.
    pub fn text_response(text: &str) -> ProviderResponse {
        ProviderResponse {
            content: Some(text.to_string()),
            tool_calls: vec![],
            usage: Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                model: "mock".to_string(),
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
                model: "mock".to_string(),
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

#[async_trait]
impl ModelProvider for MockProvider {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse> {
        // Record the call
        self.call_log.lock().await.push(MockChatCall {
            model: model.to_string(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
        });

        // Return next scripted response, or a default
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            Ok(MockProvider::text_response("Mock response"))
        } else {
            Ok(responses.remove(0))
        }
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        Ok(vec!["mock-model".to_string()])
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
impl Channel for TestChannel {
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
    ) -> anyhow::Result<ApprovalResponse> {
        Ok(self.default_approval.lock().await.clone())
    }
}

// ---------------------------------------------------------------------------
// TestHarness
// ---------------------------------------------------------------------------

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

/// Build a fully-wired agent with mock provider and temp-file SQLite DB.
///
/// Each call creates an isolated database, so tests can run in parallel.
pub async fn setup_test_agent(provider: MockProvider) -> anyhow::Result<TestHarness> {
    // Temp file for SQLite (pool needs a real file, not :memory:)
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();

    // Temp dir for skills
    let skills_dir = tempfile::TempDir::new()?;

    // Embedding service (downloads ~25MB model on first run, cached afterwards)
    let embedding_service = Arc::new(EmbeddingService::new()?);

    // State store
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    // Event store (reuse the same pool/options as the state store)
    let event_store = Arc::new(EventStore::new(state.pool()).await?);

    // Provider
    let provider = Arc::new(provider);

    // Tools — SystemInfoTool + RememberFactTool (no side effects, no approval)
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
    ];

    // Models config (all tiers point to "mock-model")
    let models_config = ModelsConfig {
        primary: "mock-model".to_string(),
        fast: "mock-model".to_string(),
        smart: "mock-model".to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as Arc<dyn ModelProvider>,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );

    let mut agent = Agent::new(
        llm_runtime,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        tools,
        "mock-model".to_string(),                        // model
        "You are a helpful test assistant.".to_string(), // system_prompt
        PathBuf::from("config.toml"),                    // config_path
        skills_dir.path().to_path_buf(),                 // skills_dir
        3,                                               // max_depth
        50,                                              // max_iterations
        100,                                             // max_iterations_cap
        8000,                                            // max_response_chars
        30,                                              // timeout_secs
        20,                                              // max_facts
        None,                                            // daily_token_budget
        IterationLimitConfig::Unlimited,
        None, // task_timeout_secs
        None, // task_token_budget
        None, // llm_call_timeout_secs
        None, // mcp_registry
        None, // goal_token_registry
        None, // hub
        true, // record_decision_points
        crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        crate::config::PolicyConfig::default(),
    );

    // Set executor mode so integration tests exercise the execution loop directly,
    // bypassing orchestrator routing (consultant pass, intent classification).
    agent.set_test_executor_mode();

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

/// Build a test agent with non-uniform model tiers (enables the smart router
/// and consultant pass).
///
/// The `primary_model` name is used as the agent's default model and for the
/// Primary router tier.  `smart_model` is used for the Smart tier (and Fast
/// tier).  Because `MockProvider` ignores model names (pops from its response
/// queue), the different names only affect routing logic and the consultant
/// pass activation check (`first_turn_model != execution_model`).
#[allow(dead_code)]
pub async fn setup_test_agent_with_models(
    provider: MockProvider,
    primary_model: &str,
    smart_model: &str,
) -> anyhow::Result<TestHarness> {
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;
    let embedding_service = Arc::new(EmbeddingService::new()?);
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    let event_store = Arc::new(EventStore::new(state.pool()).await?);

    let provider = Arc::new(provider);

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
    ];

    // Non-uniform: smart != primary → router is enabled, consultant pass activates
    let models_config = ModelsConfig {
        primary: primary_model.to_string(),
        fast: smart_model.to_string(),
        smart: smart_model.to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as Arc<dyn ModelProvider>,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );

    let agent = Agent::new(
        llm_runtime,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        tools,
        primary_model.to_string(),
        "You are a helpful test assistant.".to_string(),
        PathBuf::from("config.toml"),
        skills_dir.path().to_path_buf(),
        3,    // max_depth
        50,   // max_iterations
        100,  // max_iterations_cap
        8000, // max_response_chars
        30,   // timeout_secs
        20,   // max_facts
        None, // daily_token_budget
        IterationLimitConfig::Unlimited,
        None, // task_timeout_secs
        None, // task_token_budget
        None, // llm_call_timeout_secs
        None, // mcp_registry
        None, // goal_token_registry
        None, // hub
        true, // record_decision_points
        crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        crate::config::PolicyConfig::default(),
    );
    // Note: keeps orchestrator mode (depth=0) — used by consultant pass tests

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
/// This enables the smart router + consultant pass so integration tests can
/// exercise orchestration routing (intent gate + confirmation gate + full loop).
#[allow(dead_code)]
pub async fn setup_test_agent_orchestrator(provider: MockProvider) -> anyhow::Result<TestHarness> {
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;
    let embedding_service = Arc::new(EmbeddingService::new()?);
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    let event_store = Arc::new(EventStore::new(state.pool()).await?);

    let provider = Arc::new(provider);

    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
    ];

    // Non-uniform: smart != primary → router is enabled, consultant pass activates
    let models_config = ModelsConfig {
        primary: "primary-model".to_string(),
        fast: "fast-model".to_string(),
        smart: "smart-model".to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as Arc<dyn ModelProvider>,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );

    let agent = Agent::new(
        llm_runtime,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        tools,
        "primary-model".to_string(),
        "You are a helpful test assistant.".to_string(),
        PathBuf::from("config.toml"),
        skills_dir.path().to_path_buf(),
        3,    // max_depth
        50,   // max_iterations
        100,  // max_iterations_cap
        8000, // max_response_chars
        30,   // timeout_secs
        20,   // max_facts
        None, // daily_token_budget
        IterationLimitConfig::Unlimited,
        None, // task_timeout_secs
        None, // task_token_budget
        None, // llm_call_timeout_secs
        None, // mcp_registry
        None, // goal_token_registry
        None, // hub
        true, // record_decision_points
        crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        crate::config::PolicyConfig::default(),
    );

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
}

#[allow(dead_code)]
impl MockTool {
    pub fn new(name: &str, description: &str, return_value: &str) -> Self {
        Self {
            tool_name: name.to_string(),
            tool_description: description.to_string(),
            return_value: return_value.to_string(),
        }
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

    async fn call(&self, _args: &str) -> anyhow::Result<String> {
        Ok(self.return_value.clone())
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
    let embedding_service = Arc::new(EmbeddingService::new()?);

    // State store
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    // Reuse the state store pool for all DB-backed components so tests match production.
    let pool = state.pool();
    let event_store = Arc::new(EventStore::new(pool.clone()).await?);

    // Approval channel
    let (approval_tx, approval_rx) = mpsc::channel(16);

    // TerminalTool in Yolo mode — auto-approves everything
    let terminal_tool = Arc::new(
        TerminalTool::new(
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
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
        terminal_tool,
    ];
    tools.extend(extra_tools);

    // Provider
    let provider = Arc::new(provider);

    // Models config
    let models_config = ModelsConfig {
        primary: "mock-model".to_string(),
        fast: "mock-model".to_string(),
        smart: "mock-model".to_string(),
    };
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone() as Arc<dyn ModelProvider>,
        router_from_models(models_config.clone()),
        ProviderKind::OpenaiCompatible,
        models_config.primary.clone(),
    );

    let mut agent = Agent::new(
        llm_runtime,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        tools,
        "mock-model".to_string(),
        "You are a helpful test assistant.".to_string(),
        PathBuf::from("config.toml"),
        skills_dir.path().to_path_buf(),
        3,    // max_depth
        50,   // max_iterations
        100,  // max_iterations_cap
        8000, // max_response_chars
        30,   // timeout_secs
        20,   // max_facts
        None, // daily_token_budget
        IterationLimitConfig::Unlimited,
        None, // task_timeout_secs
        None, // task_token_budget
        None, // llm_call_timeout_secs
        None, // mcp_registry
        None, // goal_token_registry
        None, // hub
        true, // record_decision_points
        crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        crate::config::PolicyConfig::default(),
    );

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
        vec![channel.clone() as Arc<dyn Channel>],
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
