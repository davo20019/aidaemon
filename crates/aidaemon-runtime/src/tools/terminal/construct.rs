//! Building and wiring a `TerminalTool`.

use super::*;

impl TerminalTool {
    /// The owner's approval cards come from the agent kernel
    /// (`AppApprovalRules`), never from the tool.
    pub async fn new(
        backend: SharedExecutionBackend,
        allowed_prefixes: Vec<String>,
        initial_timeout_secs: u64,
        max_output_chars: usize,
        permission_mode: PermissionMode,
        pool: SqlitePool,
    ) -> Self {
        // Log permission mode on startup
        match permission_mode {
            PermissionMode::Yolo => {
                warn!(
                    "⚠️  YOLO mode enabled: all command approvals persist forever, including critical commands"
                );
            }
            PermissionMode::Cautious => {
                info!("Cautious mode: all command approvals are session-only");
            }
            PermissionMode::Default => {
                info!("Default permission mode: critical commands require per-session approval");
            }
        }

        // Configured prefixes, then the ones the owner allowed for good. A
        // configured `*` (and YOLO mode) is Full access, never a prefix; the
        // daemon converts both into its level at start
        // (`startup::permissions`), so this is for a terminal built alone.
        let mut trust_all = permission_mode == PermissionMode::Yolo;
        let mut merged: Vec<String> = Vec::new();
        for prefix in allowed_prefixes {
            if prefix == "*" {
                trust_all = true;
            } else if !merged.contains(&prefix) {
                merged.push(prefix);
            }
        }
        let mut persisted_capabilities = HashSet::new();
        let loaded = async {
            crate::tools::terminal_grants::ensure_schema(&pool).await?;
            crate::tools::terminal_grants::load(&pool, backend.approval_scope()).await
        }
        .await;
        match loaded {
            Ok(grants) => {
                let mut loaded = 0usize;
                for prefix in grants.prefixes {
                    if !merged.contains(&prefix) {
                        merged.push(prefix);
                        loaded += 1;
                    }
                }
                persisted_capabilities = grants.capabilities;
                // Counts, not the entries: some are whole command lines.
                info!(
                    loaded,
                    capabilities = persisted_capabilities.len(),
                    "Loaded the terminal's standing approvals"
                );
            }
            Err(error) => warn!(%error, "Could not load the terminal's standing approvals"),
        }

        Self {
            backend,
            allowed_prefixes: Arc::new(RwLock::new(merged)),
            session_approved: Arc::new(RwLock::new(HashMap::new())),
            session_capability_grants: Arc::new(RwLock::new(HashSet::new())),
            capability_grants: Arc::new(RwLock::new(persisted_capabilities)),
            permissions: crate::permissions::Permissions::new(if trust_all {
                crate::permissions::OwnerLevel::FullAccess
            } else {
                crate::permissions::OwnerLevel::Full
            }),
            daemon_permissions: OnceLock::new(),
            permission_mode,
            confinement: aidaemon_tools::TerminalConfinement::default(),
            running: Arc::new(Mutex::new(HashMap::new())),
            running_by_dedupe_key: Arc::new(Mutex::new(HashMap::new())),
            task_processes: Arc::new(Mutex::new(HashMap::new())),
            completed: Arc::new(Mutex::new(HashMap::new())),
            launch_trust: Arc::new(std::sync::Mutex::new(HashMap::new())),
            initial_timeout: Duration::from_secs(initial_timeout_secs),
            max_output_chars,
            pool: Some(pool),
            event_store: None,
            goals: None,
            notifications: None,
            hub: OnceLock::new(),
            agent: OnceLock::new(),
            reengagements: Arc::new(Mutex::new(HashMap::new())),
            recent_background_deliveries: Arc::new(Mutex::new(HashMap::new())),
            delivered_deliverables: Arc::new(Mutex::new(HashMap::new())),
            plan_store: OnceLock::new(),
            runs: OnceLock::new(),
            inbox_dir: std::env::temp_dir(),
            outbox_dirs: Vec::new(),
            side_calls: OnceLock::new(),
        }
    }

    /// Wire the inbox + outbox directories used by harness-side background
    /// deliverable delivery (`prepare_delivery`). Mirrors the dirs `send_file`
    /// is constructed with. Without this, delivery defaults to the system temp
    /// dir as inbox with no extra outbox dirs.
    pub fn with_delivery_dirs(mut self, inbox_dir: PathBuf, outbox_dirs: Vec<PathBuf>) -> Self {
        self.inbox_dir = inbox_dir;
        self.outbox_dirs = outbox_dirs;
        self
    }

    /// Choose host or sandbox execution (see `[terminal] confinement`).
    pub fn with_confinement(mut self, confinement: aidaemon_tools::TerminalConfinement) -> Self {
        match confinement {
            aidaemon_tools::TerminalConfinement::Host => {
                info!(
                    "Terminal runs commands on the host (set [terminal] confinement = \"sandbox\" to confine them)"
                )
            }
            aidaemon_tools::TerminalConfinement::Sandbox => {
                info!("Terminal runs commands inside the OS sandbox with task-scoped policy")
            }
        }
        self.confinement = confinement;
        self
    }

    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    /// The host's goals and notification queue.
    pub fn with_stores(
        mut self,
        goals: Arc<dyn GoalStore>,
        notifications: Arc<dyn NotificationStore>,
    ) -> Self {
        self.goals = Some(goals);
        self.notifications = Some(notifications);
        self
    }

    /// Enable semantic risk assessment for novel terminal commands, through
    /// the side-call gateway. Wired after provider startup; the gateway reads
    /// the shared runtime, so model hot-swaps stay visible.
    pub fn set_side_calls(&self, side_calls: crate::llm::gateway::SideCallGateway) {
        assert!(
            self.side_calls.set(side_calls).is_ok(),
            "TerminalTool::set_side_calls called more than once"
        );
    }

    /// Set channel hub reference for immediate background progress/completion delivery.
    pub fn set_hub(&self, hub: Weak<dyn OutboundRouter>) {
        self.hub
            .set(hub)
            .expect("TerminalTool::set_hub called more than once");
    }

    pub(super) fn get_hub(&self) -> Option<Arc<dyn OutboundRouter>> {
        self.hub.get().and_then(|w| w.upgrade())
    }

    /// Set agent reference so background command completions can re-engage
    /// the agent loop to process the output and continue the original task.
    pub fn set_agent(&self, agent: Weak<dyn ConversationRuntime>) {
        self.agent
            .set(agent)
            .expect("TerminalTool::set_agent called more than once");
    }

    /// Wire the durable plan store so the background completion notifier can read
    /// the session's incomplete checklist (for deliverable attribution + the
    /// conservative delivery-step write-back).
    pub fn set_plan_store(&self, plan_store: Arc<aidaemon_work::plans::PlanStore>) {
        let _ = self.plan_store.set(plan_store);
    }

    /// Wire the runtime's run records (see `runs`).
    pub fn set_runs(&self, runs: Arc<crate::run_state::RunStates>) {
        let _ = self.runs.set(runs);
    }

    /// Where a background run of this terminal reports.
    pub(super) fn run_home(&self) -> crate::background_run::RunHome {
        crate::background_run::RunHome {
            runs: self.runs.get_or_init(Default::default).clone(),
            hub: self.hub.get().cloned(),
            plans: self.plan_store.get().cloned(),
        }
    }

    pub fn wiring_ready(&self) -> bool {
        self.hub.get().and_then(Weak::upgrade).is_some()
            && self.agent.get().and_then(Weak::upgrade).is_some()
            && self.plan_store.get().is_some()
    }
}
