//! Live end-to-end smoke tests using a real LLM provider.
//!
//! These tests are **opt-in** — they only run when the `AIDAEMON_LIVE_TEST=1`
//! environment variable is set. They require a valid `config.toml` in the
//! current directory with real provider credentials.
//!
//! ```bash
//! # Opt-in: run live tests
//! AIDAEMON_LIVE_TEST=1 cargo test test_live -- --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::channels::{ChannelHub, SessionMap};
use crate::config::{AppConfig, IterationLimitConfig, ProviderKind};
use crate::events::EventStore;
use crate::llm_runtime::{router_from_models, SharedLlmRuntime};
use crate::memory::embeddings::EmbeddingService;
use crate::state::SqliteStateStore;
use crate::testing::TestChannel;
use crate::tools::command_risk::PermissionMode;
use crate::tools::memory::RememberFactTool;
use crate::tools::{SystemInfoTool, TerminalTool};
use crate::traits::{Channel, ModelProvider};
use crate::types::{ChannelContext, UserRole};

/// Check if live tests should run.
fn should_run_live_tests() -> bool {
    std::env::var("AIDAEMON_LIVE_TEST").is_ok_and(|v| v == "1")
}

/// Set up a live agent with a real LLM provider from config.toml.
async fn setup_live_agent() -> anyhow::Result<LiveTestHarness> {
    let config_path = PathBuf::from("config.toml");
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "config.toml not found — live tests require a valid config"
        ));
    }

    let mut config = AppConfig::load(&config_path)?;
    config.provider.models.apply_defaults(&config.provider.kind);

    // Create real provider
    let provider: Arc<dyn ModelProvider> = match config.provider.kind {
        ProviderKind::OpenaiCompatible => Arc::new(
            crate::providers::OpenAiCompatibleProvider::new(
                &config.provider.base_url,
                &config.provider.api_key,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?,
        ),
        ProviderKind::GoogleGenai => Arc::new(crate::providers::GoogleGenAiProvider::new(
            &config.provider.api_key,
        )),
        ProviderKind::Anthropic => Arc::new(crate::providers::AnthropicNativeProvider::new(
            &config.provider.api_key,
        )),
    };

    // Temp DB for isolation
    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;

    let embedding_service = Arc::new(EmbeddingService::new()?);
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db_path)).await?;
    let event_store = Arc::new(EventStore::new(pool.clone()).await?);
    // Approval channel + TerminalTool
    let (approval_tx, approval_rx) = mpsc::channel(16);
    let terminal_tool = Arc::new(
        TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            30,
            8000,
            PermissionMode::Yolo,
            pool,
        )
        .await,
    );

    let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
        terminal_tool,
    ];

    let model = config.provider.models.primary.clone();
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone(),
        router_from_models(config.provider.models.clone()),
        config.provider.kind.clone(),
        model.clone(),
    );

    let agent = Agent::new(
        llm_runtime,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        tools,
        model,
        "You are a helpful assistant. Use terminal commands to gather information when asked."
            .to_string(),
        config_path,
        skills_dir.path().to_path_buf(),
        3,
        50,
        100,
        8000,
        30,
        20,
        None,
        IterationLimitConfig::Unlimited,
        None,
        None,
        None,
        None,
        None, // goal_token_registry
        None, // hub
        true, // record_decision_points
        crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        crate::config::PolicyConfig::default(),
        crate::config::PathAliasConfig::default(),
    );

    let channel = Arc::new(TestChannel::new());
    let mut map = HashMap::new();
    map.insert("live_test".to_string(), "test".to_string());
    let session_map: SessionMap = Arc::new(tokio::sync::RwLock::new(map));
    let hub = Arc::new(ChannelHub::new(
        vec![channel.clone() as Arc<dyn Channel>],
        session_map,
    ));

    let hub_for_approvals = hub.clone();
    let approval_task = tokio::spawn(async move {
        hub_for_approvals.approval_listener(approval_rx).await;
    });

    Ok(LiveTestHarness {
        agent,
        state,
        _channel: channel,
        _hub: hub,
        _db_file: db_file,
        _skills_dir: skills_dir,
        _approval_task: approval_task,
    })
}

#[allow(dead_code)]
struct LiveTestHarness {
    agent: Agent,
    state: Arc<SqliteStateStore>,
    _channel: Arc<TestChannel>,
    _hub: Arc<ChannelHub>,
    _db_file: tempfile::NamedTempFile,
    _skills_dir: tempfile::TempDir,
    _approval_task: tokio::task::JoinHandle<()>,
}

/// Live test: complex prompt that triggers multiple terminal calls.
///
/// Sends a system exploration prompt that should trigger 5-10+ terminal calls.
/// Verifies the agent completes normally without stall detection firing.
#[tokio::test]
async fn test_live_complex_prompt_no_stall() {
    if !should_run_live_tests() {
        eprintln!("Skipping live test (set AIDAEMON_LIVE_TEST=1 to run)");
        return;
    }

    let harness = setup_live_agent()
        .await
        .expect("Failed to set up live agent");

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        harness.agent.handle_message(
            "live_test",
            "Check this system thoroughly — OS, disk, running processes, and installed runtimes \
             (node, python, rust). Give me a complete report.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        ),
    )
    .await
    .expect("Live test timed out after 120s")
    .expect("Live test agent returned error");

    assert!(
        !response.contains("stuck in a loop"),
        "Live agent should not trigger stall detection. Got: {}",
        &response[..response.len().min(500)]
    );
    assert!(
        response.len() > 100,
        "Live agent should produce a substantial response. Got {} chars: {}",
        response.len(),
        &response[..response.len().min(200)]
    );
}

/// Set up a live agent with a custom system prompt.
async fn setup_live_agent_with_prompt(system_prompt: &str) -> anyhow::Result<LiveTestHarness> {
    let config_path = PathBuf::from("config.toml");
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "config.toml not found — live tests require a valid config"
        ));
    }

    let mut config = AppConfig::load(&config_path)?;
    config.provider.models.apply_defaults(&config.provider.kind);

    let provider: Arc<dyn ModelProvider> = match config.provider.kind {
        ProviderKind::OpenaiCompatible => Arc::new(
            crate::providers::OpenAiCompatibleProvider::new(
                &config.provider.base_url,
                &config.provider.api_key,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?,
        ),
        ProviderKind::GoogleGenai => Arc::new(crate::providers::GoogleGenAiProvider::new(
            &config.provider.api_key,
        )),
        ProviderKind::Anthropic => Arc::new(crate::providers::AnthropicNativeProvider::new(
            &config.provider.api_key,
        )),
    };

    let db_file = tempfile::NamedTempFile::new()?;
    let db_path = db_file.path().to_str().unwrap().to_string();
    let skills_dir = tempfile::TempDir::new()?;

    let embedding_service = Arc::new(EmbeddingService::new()?);
    let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);

    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", db_path)).await?;
    let event_store = Arc::new(EventStore::new(pool.clone()).await?);
    let (approval_tx, approval_rx) = mpsc::channel(16);
    let terminal_tool = Arc::new(
        TerminalTool::new(
            vec!["*".to_string()],
            approval_tx,
            30,
            8000,
            PermissionMode::Yolo,
            pool,
        )
        .await,
    );

    let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(RememberFactTool::new(
            state.clone() as Arc<dyn crate::traits::StateStore>
        )),
        terminal_tool,
    ];

    let model = config.provider.models.primary.clone();
    let llm_runtime = SharedLlmRuntime::new(
        provider.clone(),
        router_from_models(config.provider.models.clone()),
        config.provider.kind.clone(),
        model.clone(),
    );

    let agent = Agent::new(
        llm_runtime,
        state.clone() as Arc<dyn crate::traits::StateStore>,
        event_store,
        tools,
        model,
        system_prompt.to_string(),
        config_path,
        skills_dir.path().to_path_buf(),
        3,
        50,
        100,
        8000,
        30,
        20,
        None,
        IterationLimitConfig::Unlimited,
        None,
        None,
        None,
        None,
        None,
        None,
        true,
        crate::config::ContextWindowConfig {
            progressive_facts: false,
            ..Default::default()
        },
        crate::config::PolicyConfig::default(),
        crate::config::PathAliasConfig::default(),
    );

    let channel = Arc::new(TestChannel::new());
    let mut map = HashMap::new();
    map.insert("live_test".to_string(), "test".to_string());
    let session_map: SessionMap = Arc::new(tokio::sync::RwLock::new(map));
    let hub = Arc::new(ChannelHub::new(
        vec![channel.clone() as Arc<dyn Channel>],
        session_map,
    ));

    let hub_for_approvals = hub.clone();
    let approval_task = tokio::spawn(async move {
        hub_for_approvals.approval_listener(approval_rx).await;
    });

    Ok(LiveTestHarness {
        agent,
        state,
        _channel: channel,
        _hub: hub,
        _db_file: db_file,
        _skills_dir: skills_dir,
        _approval_task: approval_task,
    })
}

/// Live test: simple question that shouldn't trigger any tools.
#[tokio::test]
async fn test_live_simple_response() {
    if !should_run_live_tests() {
        eprintln!("Skipping live test (set AIDAEMON_LIVE_TEST=1 to run)");
        return;
    }

    let harness = setup_live_agent()
        .await
        .expect("Failed to set up live agent");

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        harness.agent.handle_message(
            "live_test",
            "What is 2+2? Answer with just the number.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        ),
    )
    .await
    .expect("Live test timed out after 30s")
    .expect("Live test agent returned error");

    assert!(
        response.contains('4'),
        "Live agent should answer 2+2=4. Got: {}",
        response
    );
}

// ---------------------------------------------------------------------------
// Behavioral regression tests
// ---------------------------------------------------------------------------

/// The decision-framework rules from the real system prompt that we're testing.
const BEHAVIORAL_SYSTEM_PROMPT: &str = "\
You are a helpful AI assistant running on the user's local machine. \
You have full access to the filesystem via the `terminal` tool.\n\n\
## CRITICAL RULES (MUST FOLLOW)\n\n\
1. **You CAN access any file or folder** on this machine using `terminal`. \
   NEVER say \"I can't access\", \"I don't have access\", or \"I can't browse\" — that is FALSE.\n\n\
2. **When the user gives a path or location hint** (e.g. \"it's in /tmp/foo\", \"check the projects folder\"), \
   you MUST immediately run `ls` or `cat` on that path using `terminal`. Do NOT ask for clarification — ACT.\n\n\
3. **When the user says \"check it yourself\"**, you MUST use `terminal` to explore. \
   Asking another question after \"check it yourself\" is WRONG.\n\n\
4. **Fuzzy-match names**: If the user says \"site-cars\" but the directory is \"cars-site\", \
   recognize the match. List the directory first, then find the closest match.\n\n\
5. **Local-first**: When the user says something is on the local filesystem, \
   ALWAYS use `terminal` (ls, cat, find). NEVER use web_search for local files.\n\n\
6. After using `terminal`, include the results in your response.";

/// Behavioral test: when given a location hint, the agent should explore with
/// terminal instead of asking for clarification.
///
/// Simulates:
///   User: "What's in the site-cars project?"
///   Agent: (should ask for clarification — ambiguous)
///   User: "It's in the projects folder, check it yourself"
///   Agent: (should use terminal to `ls` the projects folder)
#[tokio::test]
async fn test_live_behavior_explores_on_location_hint() {
    if !should_run_live_tests() {
        eprintln!("Skipping live test (set AIDAEMON_LIVE_TEST=1 to run)");
        return;
    }

    // Create a temp directory structure to simulate the scenario
    let workspace = tempfile::TempDir::new().expect("Failed to create temp dir");
    let projects_dir = workspace.path().join("projects");
    std::fs::create_dir(&projects_dir).unwrap();
    std::fs::create_dir(projects_dir.join("cars-site")).unwrap();
    std::fs::create_dir(projects_dir.join("blog")).unwrap();
    std::fs::create_dir(projects_dir.join("portfolio")).unwrap();
    // Add a marker file so we can verify the agent found it
    std::fs::write(
        projects_dir.join("cars-site").join("README.md"),
        "# Cars Site\nDeployed at: https://cars-site.example.com\n",
    )
    .unwrap();

    let harness = setup_live_agent_with_prompt(BEHAVIORAL_SYSTEM_PROMPT)
        .await
        .expect("Failed to set up live agent");

    let session = "behavior_location_hint";
    let workspace_path = workspace.path().to_str().unwrap();

    // Turn 1: ambiguous request (agent may ask for clarification — that's OK)
    let _response1 = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        harness.agent.handle_message(
            session,
            "What's in the site-cars project?",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        ),
    )
    .await
    .expect("Turn 1 timed out")
    .expect("Turn 1 error");

    // Turn 2: user gives a location hint and says "check it yourself"
    let response2 = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        harness.agent.handle_message(
            session,
            &format!(
                "It's in the projects folder at {}. Check it yourself.",
                workspace_path
            ),
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        ),
    )
    .await
    .expect("Turn 2 timed out")
    .expect("Turn 2 error");

    let resp_lower = response2.to_lowercase();

    // PRIMARY CHECK: response should show directory contents (cars-site, blog, portfolio).
    // These names only exist in the temp directory, so finding them proves terminal was used.
    let found_directory_content = resp_lower.contains("cars-site")
        || resp_lower.contains("cars site")
        || resp_lower.contains("blog")
        || resp_lower.contains("portfolio");

    assert!(
        found_directory_content,
        "Agent should have explored the directory and found its contents \
         (cars-site, blog, portfolio). Got: {}",
        &response2[..response2.len().min(500)]
    );

    // The agent must NOT ask for clarification when given an explicit path
    let asks_clarification = resp_lower.contains("clarify")
        || resp_lower.contains("could you")
        || resp_lower.contains("which site")
        || resp_lower.contains("which project");
    assert!(
        !asks_clarification,
        "Agent should NOT ask for clarification when given an explicit path. Got: {}",
        &response2[..response2.len().min(500)]
    );

    // The agent must NOT claim it can't access files
    let bad_phrases = [
        "don't have access",
        "can't access",
        "cannot access",
        "can't browse",
        "cannot browse",
        "unable to browse",
        "unable to access",
        "not able to access",
    ];
    for phrase in &bad_phrases {
        assert!(
            !resp_lower.contains(phrase),
            "Agent falsely claimed it can't access files. Found '{}' in: {}",
            phrase,
            &response2[..response2.len().min(500)]
        );
    }
}

/// Behavioral test: agent should never claim it can't access local files.
///
/// Sends a direct instruction to check a specific local path. The agent must
/// use terminal, not give up or claim inability.
#[tokio::test]
async fn test_live_behavior_never_claims_no_access() {
    if !should_run_live_tests() {
        eprintln!("Skipping live test (set AIDAEMON_LIVE_TEST=1 to run)");
        return;
    }

    let workspace = tempfile::TempDir::new().expect("Failed to create temp dir");
    std::fs::write(
        workspace.path().join("hello.txt"),
        "Hello from behavioral test!",
    )
    .unwrap();

    let harness = setup_live_agent_with_prompt(BEHAVIORAL_SYSTEM_PROMPT)
        .await
        .expect("Failed to set up live agent");

    let session = "behavior_no_access_claim";
    let file_path = workspace.path().join("hello.txt");

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        harness.agent.handle_message(
            session,
            &format!(
                "Read the file at {} and tell me what it says.",
                file_path.display()
            ),
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        ),
    )
    .await
    .expect("Test timed out")
    .expect("Test error");

    // PRIMARY CHECK: response must contain the exact file contents.
    // "Hello from behavioral test!" can only come from reading the temp file,
    // so this proves the agent used terminal.
    assert!(
        response.contains("Hello from behavioral test"),
        "Agent should have read and reported the file contents. Got: {}",
        &response[..response.len().min(500)]
    );

    // Must NOT claim inability
    let bad_phrases = [
        "don't have access",
        "can't access",
        "cannot access",
        "can't browse",
        "cannot browse",
        "unable to access",
    ];
    let resp_lower = response.to_lowercase();
    for phrase in &bad_phrases {
        assert!(
            !resp_lower.contains(phrase),
            "Agent falsely claimed inability. Found '{}' in response",
            phrase
        );
    }
}

/// Behavioral test: agent should NOT do web search when the user says something
/// is on the local filesystem.
#[tokio::test]
async fn test_live_behavior_no_web_search_for_local_files() {
    if !should_run_live_tests() {
        eprintln!("Skipping live test (set AIDAEMON_LIVE_TEST=1 to run)");
        return;
    }

    let workspace = tempfile::TempDir::new().expect("Failed to create temp dir");
    let projects_dir = workspace.path().join("projects");
    std::fs::create_dir(&projects_dir).unwrap();
    std::fs::create_dir(projects_dir.join("my-app")).unwrap();
    std::fs::write(
        projects_dir.join("my-app").join("package.json"),
        r#"{"name": "my-app", "version": "1.0.0"}"#,
    )
    .unwrap();

    let harness = setup_live_agent_with_prompt(BEHAVIORAL_SYSTEM_PROMPT)
        .await
        .expect("Failed to set up live agent");

    let session = "behavior_no_web_search";

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        harness.agent.handle_message(
            session,
            &format!(
                "List the projects in {}. What's in there?",
                workspace.path().display()
            ),
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        ),
    )
    .await
    .expect("Test timed out")
    .expect("Test error");

    let resp_lower = response.to_lowercase();

    // PRIMARY CHECK: response should contain directory contents.
    // "my-app" only exists in the temp directory, so finding it proves terminal was used.
    assert!(
        resp_lower.contains("my-app"),
        "Agent should have listed the directory and found 'my-app'. Got: {}",
        &response[..response.len().min(500)]
    );

    // Should NOT ask for clarification when given an explicit path
    let asks_clarification = resp_lower.contains("clarify")
        || resp_lower.contains("could you specify")
        || resp_lower.contains("which project");
    assert!(
        !asks_clarification,
        "Agent should NOT ask for clarification when given an explicit path. Got: {}",
        &response[..response.len().min(500)]
    );
}
