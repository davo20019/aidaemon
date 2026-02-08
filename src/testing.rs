//! Test infrastructure: MockProvider, TestChannel, and TestHarness.
//!
//! Provides a fully wired Agent with a mock LLM and in-memory state,
//! suitable for integration tests that exercise the real agent loop.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::config::{IterationLimitConfig, ModelsConfig};
use crate::events::EventStore;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::{PlanStore, StepTracker};
use crate::skills::SharedSkillRegistry;
use crate::state::SqliteStateStore;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::memory::RememberFactTool;
use crate::tools::SystemInfoTool;
use crate::traits::{
    Channel, ChannelCapabilities, ModelProvider, ProviderResponse, TokenUsage, Tool, ToolCall,
};
use crate::types::{ApprovalResponse, MediaMessage};

use crate::agent::Agent;

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
}

/// Build a fully-wired agent with mock provider and temp-file SQLite DB.
///
/// Each call creates an isolated database, so tests can run in parallel.
pub async fn setup_test_agent(provider: MockProvider) -> anyhow::Result<TestHarness> {
    // Temp file for SQLite (pool needs a real file, not :memory:)
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();

    // Embedding service (downloads ~25MB model on first run, cached afterwards)
    let embedding_service = Arc::new(EmbeddingService::new()?);

    // State store
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    // Event store & plan store share the same SQLite pool
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db_path)).await?;
    let event_store = Arc::new(EventStore::new(pool.clone()).await?);
    let plan_store = Arc::new(PlanStore::new(pool).await?);
    let step_tracker = Arc::new(StepTracker::new(plan_store.clone()));

    // Provider
    let provider = Arc::new(provider);

    // Tools — SystemInfoTool + RememberFactTool (no side effects, no approval)
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
    ];

    // Empty skill registry
    let skills = SharedSkillRegistry::new(vec![]);

    // Models config (all tiers point to "mock-model")
    let models_config = ModelsConfig {
        primary: "mock-model".to_string(),
        fast: "mock-model".to_string(),
        smart: "mock-model".to_string(),
    };

    let agent = Agent::new(
        provider.clone() as Arc<dyn ModelProvider>,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        plan_store,
        step_tracker,
        tools,
        "mock-model".to_string(),                        // model
        "You are a helpful test assistant.".to_string(), // system_prompt
        PathBuf::from("config.toml"),                    // config_path
        skills,
        3,    // max_depth
        50,   // max_iterations
        100,  // max_iterations_cap
        8000, // max_response_chars
        30,   // timeout_secs
        models_config,
        20,   // max_facts
        None, // daily_token_budget
        IterationLimitConfig::Unlimited,
        None, // task_timeout_secs
        None, // task_token_budget
        None, // llm_call_timeout_secs
        None, // mcp_registry
    );

    // Channel (not wired to hub — tests call agent.handle_message directly)
    let channel = Arc::new(TestChannel::new());

    Ok(TestHarness {
        agent,
        state,
        provider,
        channel,
        _db_file: db_file,
    })
}
