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
mod peers;
mod prompt;
mod queue;

/// The work stores on a fresh database, an empty chat history and an
/// in-memory invocation log: what the tool reads and writes.
struct TestStore {
    work: aidaemon_work::sqlite::SqliteWorkStore,
    invocations: std::sync::Mutex<Vec<aidaemon_tools::services::CliAgentInvocation>>,
    /// Kept events: (invocation, seq, kind, data).
    events: std::sync::Mutex<Vec<(i64, i64, String, String)>>,
    sessions: std::sync::Mutex<Vec<(i64, String)>>,
    /// Run metadata keyed by invocation id: (run_id, mode, keeps_session, thread).
    run_meta: std::sync::Mutex<HashMap<i64, RunMeta>>,
    /// Settings a test stored (a conversation set to a coding agent).
    settings: std::sync::Mutex<HashMap<String, String>>,
}

/// A kept run's metadata: (run_id, mode, keeps_session, thread).
type RunMeta = (Option<String>, Option<String>, bool, Option<String>);

impl TestStore {
    async fn open(path: &str) -> Self {
        Self {
            work: aidaemon_work::sqlite::open_for_tests(path).await,
            invocations: Default::default(),
            events: Default::default(),
            sessions: Default::default(),
            run_meta: Default::default(),
            settings: Default::default(),
        }
    }
}

/// Only what a test stored: with no agent mode stored, each agent runs in
/// its kind's default.
#[async_trait::async_trait]
impl aidaemon_tools::services::SettingsStore for TestStore {
    async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
        Ok(self.settings.lock().unwrap().get(key).cloned())
    }

    async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
        self.settings
            .lock()
            .unwrap()
            .insert(key.to_string(), value.to_string());
        Ok(())
    }
}

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

    async fn note_cli_agent_run(
        &self,
        id: i64,
        run_id: &str,
        mode: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut meta = self.run_meta.lock().unwrap();
        let entry = meta.entry(id).or_insert((None, None, false, None));
        entry.0 = Some(run_id.to_string());
        entry.1 = mode.map(str::to_string);
        Ok(())
    }

    async fn set_cli_agent_keeps_session(&self, id: i64, kept: bool) -> anyhow::Result<()> {
        let mut meta = self.run_meta.lock().unwrap();
        let entry = meta.entry(id).or_insert((None, None, false, None));
        entry.2 = kept;
        Ok(())
    }

    async fn set_cli_agent_thread(&self, id: i64, thread: &str) -> anyhow::Result<()> {
        let mut meta = self.run_meta.lock().unwrap();
        let entry = meta.entry(id).or_insert((None, None, false, None));
        entry.3 = Some(thread.to_string());
        Ok(())
    }

    async fn latest_kept_cli_agent_run(
        &self,
        session_id: &str,
        agent_name: &str,
    ) -> anyhow::Result<Option<String>> {
        let log = self.invocations.lock().unwrap();
        let meta = self.run_meta.lock().unwrap();
        let sessions = self.sessions.lock().unwrap();
        for inv in log.iter().rev() {
            if inv.session_id != session_id || inv.agent_name != agent_name {
                continue;
            }
            let Some((Some(run_id), _, true, _)) = meta.get(&inv.id) else {
                continue;
            };
            if sessions.iter().any(|(sid, _)| *sid == inv.id) {
                return Ok(Some(run_id.clone()));
            }
        }
        Ok(None)
    }

    async fn cli_agent_run(
        &self,
        run_id: &str,
    ) -> anyhow::Result<Option<aidaemon_tools::services::CliAgentRun>> {
        let log = self.invocations.lock().unwrap();
        let meta = self.run_meta.lock().unwrap();
        let sessions = self.sessions.lock().unwrap();
        for inv in log.iter().rev() {
            let Some((Some(rid), _, keeps_session, thread)) = meta.get(&inv.id) else {
                continue;
            };
            if rid != run_id {
                continue;
            }
            let agent_session = sessions
                .iter()
                .rev()
                .find(|(sid, _)| *sid == inv.id)
                .map(|(_, s)| s.clone());
            return Ok(Some(aidaemon_tools::services::CliAgentRun {
                invocation: inv.id,
                session_id: inv.session_id.clone(),
                agent_name: inv.agent_name.clone(),
                prompt_summary: inv.prompt_summary.clone(),
                task: None,
                working_dir: inv.working_dir.clone(),
                output_summary: inv.output_summary.clone(),
                success: inv.success,
                agent_session,
                keeps_session: *keeps_session,
                thread: thread.clone(),
            }));
        }
        Ok(None)
    }

    async fn latest_cli_agent_run_in_thread(
        &self,
        session_id: &str,
        thread: &str,
    ) -> anyhow::Result<Option<String>> {
        let log = self.invocations.lock().unwrap();
        let meta = self.run_meta.lock().unwrap();
        for inv in log.iter().rev() {
            if inv.session_id != session_id {
                continue;
            }
            let Some((Some(rid), _, _, inv_thread)) = meta.get(&inv.id) else {
                continue;
            };
            if rid == thread || inv_thread.as_deref() == Some(thread) {
                return Ok(Some(rid.clone()));
            }
        }
        Ok(None)
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
        claims_changed: Arc::new(tokio::sync::Notify::new()),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        background_timeout: Duration::from_secs(DEFAULT_BACKGROUND_TIMEOUT_SECS),
        idle_timeout: Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS),
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
        peer_coord: OnceLock::new(),
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
        claims_changed: Arc::new(tokio::sync::Notify::new()),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        background_timeout: Duration::from_secs(DEFAULT_BACKGROUND_TIMEOUT_SECS),
        idle_timeout: Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS),
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
        peer_coord: OnceLock::new(),
    };

    (tool, db_file)
}

// -----------------------------------------------------------------------
// Basic run tests
// -----------------------------------------------------------------------

/// The scope and new folders a call declares become write roots of its run
/// (`helper_write_paths`), so the scope check must see them as write targets.
#[cfg(unix)]
#[tokio::test]
async fn declared_scope_and_new_folders_are_write_targets() {
    let (tool, _db) = setup_echo_tool().await;
    let manifest = tool.call_access_manifest(
        &serde_json::json!({
            "tool": "echo-agent",
            "prompt": "synthetic task",
            "working_dir": "/work/project",
            "write_paths": ["src"],
            "scope": ["/work"],
            "new_folders": ["/work/new-folder"],
        })
        .to_string(),
    );
    let writes: Vec<&str> = manifest
        .write_targets
        .iter()
        .map(|target| target.value.as_str())
        .collect();
    for expected in ["/work/project/src", "/work", "/work/new-folder"] {
        assert!(
            writes.contains(&expected),
            "{expected} missing from {writes:?}"
        );
    }
}

/// A stored agent with a built-in agent's name does not replace it at
/// startup (it would inherit the built-in's permission mode).
#[tokio::test]
async fn a_stored_agent_never_replaces_a_built_in_of_its_name() {
    let (tool, _db) = setup_echo_tool().await;
    tool.state
        .save_dynamic_cli_agent(&DynamicCliAgent {
            id: 0,
            name: "echo".to_string(),
            command: "sh".to_string(),
            args_json: "[]".to_string(),
            description: "shadow".to_string(),
            timeout_secs: None,
            max_output_chars: None,
            enabled: true,
            created_at: String::new(),
        })
        .await
        .unwrap();
    tool.load_dynamic_agents().await;
    let tools = tool.tools.read().unwrap();
    let entry = tools.get("echo").unwrap();
    assert!(!entry.is_dynamic);
    assert_eq!(entry.command, "echo");
}
