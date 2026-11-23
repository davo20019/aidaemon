//! Tests of the cli_agent tool. Shared helpers live here; the tests are
//! grouped by area in the child modules.

use super::tool_impl::cli_agent_outcome;
use super::*;
use crate::tools::traits::ToolSemantics;
use aidaemon_tools::testing::ToolCalls as _;

use crate::stores::prelude::*;
use crate::tools::traits::Tool;
use crate::tools::traits::{Goal, Task};
use std::collections::HashMap;
use std::sync::Arc;

mod agents;
mod background;
mod basics;
mod claims;
mod launch;
mod output;
mod prompt;

/// The work stores on a fresh database, an empty chat history and an
/// in-memory invocation log: what the tool reads and writes.
struct TestStore {
    work: aidaemon_work::sqlite::SqliteWorkStore,
    invocations: std::sync::Mutex<Vec<aidaemon_tools::services::CliAgentInvocation>>,
    /// Kept events: (invocation, seq, kind, data).
    events: std::sync::Mutex<Vec<(i64, i64, String, String)>>,
    sessions: std::sync::Mutex<Vec<(i64, String)>>,
}

impl TestStore {
    async fn open(path: &str) -> Self {
        Self {
            work: aidaemon_work::sqlite::open_for_tests(path).await,
            invocations: Default::default(),
            events: Default::default(),
            sessions: Default::default(),
        }
    }
}

/// No mode stored: each agent runs in its kind's default.
#[async_trait::async_trait]
impl aidaemon_tools::services::SettingsStore for TestStore {}

impl aidaemon_work::sqlite::WorkDatabase for TestStore {
    fn work_pool(&self) -> &sqlx::SqlitePool {
        self.work.work_pool()
    }

    fn redact_secrets(&self, text: &str) -> String {
        self.work.redact_secrets(text)
    }
}

#[async_trait]
impl aidaemon_events::stores::MessageStore for TestStore {
    async fn append_message(&self, _msg: &Message) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_history(&self, _session_id: &str, _limit: usize) -> anyhow::Result<Vec<Message>> {
        Ok(Vec::new())
    }

    async fn clear_session(&self, _session_id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

#[async_trait]
impl crate::tools::traits::DynamicCliAgentStore for TestStore {
    async fn set_cli_agent_session(&self, id: i64, agent_session: &str) -> anyhow::Result<()> {
        self.sessions
            .lock()
            .unwrap()
            .push((id, agent_session.to_string()));
        Ok(())
    }

    async fn record_cli_agent_event(
        &self,
        id: i64,
        seq: i64,
        kind: &str,
        data: &str,
    ) -> anyhow::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push((id, seq, kind.to_string(), data.to_string()));
        Ok(())
    }

    async fn log_cli_agent_start(
        &self,
        task_id: Option<&str>,
        session_id: &str,
        agent_name: &str,
        prompt_summary: &str,
        working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        let mut log = self.invocations.lock().unwrap();
        let id = log.len() as i64 + 1;
        log.push(aidaemon_tools::services::CliAgentInvocation {
            id,
            task_id: task_id.map(str::to_string),
            session_id: session_id.to_string(),
            agent_name: agent_name.to_string(),
            prompt_summary: prompt_summary.to_string(),
            working_dir: working_dir.map(str::to_string),
            started_at: String::new(),
            completed_at: None,
            exit_code: None,
            output_summary: None,
            success: None,
            duration_secs: None,
        });
        Ok(id)
    }

    async fn log_cli_agent_complete(
        &self,
        id: i64,
        exit_code: Option<i32>,
        output_summary: &str,
        success: bool,
        duration_secs: f64,
    ) -> anyhow::Result<()> {
        let mut log = self.invocations.lock().unwrap();
        if let Some(entry) = log.iter_mut().find(|entry| entry.id == id) {
            entry.completed_at = Some(String::new());
            entry.exit_code = exit_code;
            entry.output_summary = Some(output_summary.to_string());
            entry.success = Some(success);
            entry.duration_secs = Some(duration_secs);
        }
        Ok(())
    }

    async fn get_cli_agent_invocations(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<aidaemon_tools::services::CliAgentInvocation>> {
        let log = self.invocations.lock().unwrap();
        Ok(log.iter().rev().take(limit).cloned().collect())
    }
}

fn extract_task_id_from_background_message(msg: &str) -> String {
    let marker = "task_id=";
    let start = msg
        .find(marker)
        .expect("background response should include task_id")
        + marker.len();
    msg[start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

/// Create a CliAgentTool with `echo` registered as a test tool.
/// Uses a real temp-file SQLite DB for state persistence.
async fn setup_echo_tool() -> (CliAgentTool, tempfile::NamedTempFile) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);

    let mut tools_map = HashMap::new();
    tools_map.insert(
        "echo".to_string(),
        CliToolEntry {
            command: "echo".to_string(),
            args: vec![],
            description: "Echo agent for testing".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );

    let tool = CliAgentTool {
        context_facts: None,
        project_notes: None,
        backend: aidaemon_tools::testing::active_execution_backend(),
        tools: Arc::new(std::sync::RwLock::new(tools_map)),
        tool_names: Arc::new(std::sync::RwLock::new(vec!["echo".to_string()])),
        running: Arc::new(Mutex::new(HashMap::new())),
        completed: Arc::new(Mutex::new(HashMap::new())),
        working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        default_max_output: 10000,
        max_concurrent: 3,
        concurrency_limiter: Arc::new(Semaphore::new(3)),
        hub: OnceLock::new(),
        runs: OnceLock::new(),
        plan_store: OnceLock::new(),
        agent: OnceLock::new(),
        reengagements: Arc::new(Mutex::new(HashMap::new())),
        coding_agents: aidaemon_coding_agents::CodingAgents::builtin(),
        permissions: Arc::new(OnceLock::new()),
    };

    (tool, db_file)
}

/// Create a CliAgentTool with `bash` registered, for testing scripts.
async fn setup_bash_tool() -> (CliAgentTool, tempfile::NamedTempFile) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);

    let mut tools_map = HashMap::new();
    tools_map.insert(
        "bash-agent".to_string(),
        CliToolEntry {
            command: "bash".to_string(),
            args: vec!["-c".to_string()],
            description: "Bash agent for testing".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );

    let tool = CliAgentTool {
        context_facts: None,
        project_notes: None,
        backend: aidaemon_tools::testing::active_execution_backend(),
        tools: Arc::new(std::sync::RwLock::new(tools_map)),
        tool_names: Arc::new(std::sync::RwLock::new(vec!["bash-agent".to_string()])),
        running: Arc::new(Mutex::new(HashMap::new())),
        completed: Arc::new(Mutex::new(HashMap::new())),
        working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        default_max_output: 10000,
        max_concurrent: 3,
        concurrency_limiter: Arc::new(Semaphore::new(3)),
        hub: OnceLock::new(),
        runs: OnceLock::new(),
        plan_store: OnceLock::new(),
        agent: OnceLock::new(),
        reengagements: Arc::new(Mutex::new(HashMap::new())),
        coding_agents: aidaemon_coding_agents::CodingAgents::builtin(),
        permissions: Arc::new(OnceLock::new()),
    };

    (tool, db_file)
}

// -----------------------------------------------------------------------
// Basic run tests
// -----------------------------------------------------------------------
