//! Multi-CLI peer coordination for [`CliAgentTool`], built on
//! [`aidaemon_coding_agents::peer`] and [`aidaemon_mcp::LocalMcpServer`].
//!
//! Works identically whether AIdaemon's LLM is in the loop or bypassed in
//! direct-agent mode (`SessionModelSetting::agent`): the daemon hosts a single
//! loopback [`LocalMcpServer`] and injects a scoped MCP endpoint into each CLI
//! run so CLIs can delegate subtasks (in git worktrees, read-only, or narrowed
//! subpaths), exchange messages, resume finished peers, and coordinate through
//! a shared tree board with zero AIdaemon LLM calls.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;

use aidaemon_coding_agents::McpEndpoint;
use aidaemon_coding_agents::peer::{
    PeerAgentInfo, PeerLaunchRequest, PeerLaunchResult, PeerLauncher, PeerLimits, PeerMcpContext,
    PeerMcpHandler, PeerRunStatus, PeerState, PeerTree, PeerWorkspaceMode,
    coordination_prompt_hint,
};
use aidaemon_mcp::{LocalMcpServer, McpToolHandler, ScopedEndpoint, ServerMeta};
use aidaemon_tools::backend::BackendKind;
use aidaemon_tools::{ExecutionBackend, ExecutionRequest};
use async_trait::async_trait;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tracing::info;

use super::{CliAgentTool, CliWorkspaceMode, StatusUpdate, cli_agent_task_id, cli_run_id};

/// Shared permit holder for a running CLI so a blocking peer wait can park its
/// concurrency slot and avoid pool deadlocks.
#[derive(Clone)]
pub(super) struct PermitSlot {
    permit: Arc<StdMutex<Option<OwnedSemaphorePermit>>>,
    limiter: Arc<Semaphore>,
    finished: Arc<AtomicBool>,
    in_flight_waits: Arc<AtomicUsize>,
}

impl PermitSlot {
    pub(super) fn new(
        permit: OwnedSemaphorePermit,
        limiter: Arc<Semaphore>,
        finished: Arc<AtomicBool>,
        in_flight_waits: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            permit: Arc::new(StdMutex::new(Some(permit))),
            limiter,
            finished,
            in_flight_waits,
        }
    }

    /// Release the held permit while the owning CLI blocks inside a peer MCP
    /// call, and pause its idle/wall-clock timer until the returned guard drops.
    pub(super) fn park(&self) -> ParkedPermitGuard {
        self.in_flight_waits.fetch_add(1, Ordering::AcqRel);
        let dropped = self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(dropped);
        ParkedPermitGuard { slot: self.clone() }
    }

    /// Release any held permit when the run exits.
    pub(super) fn release(&self) {
        let dropped = self.permit.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(dropped);
    }
}

/// RAII guard restoring a parked permit and unpausing idle timeouts on drop.
pub(super) struct ParkedPermitGuard {
    slot: PermitSlot,
}

impl Drop for ParkedPermitGuard {
    fn drop(&mut self) {
        self.slot.in_flight_waits.fetch_sub(1, Ordering::AcqRel);
        if self.slot.finished.load(Ordering::Acquire) {
            return;
        }
        if let Ok(permit) = self.slot.limiter.clone().try_acquire_owned() {
            let mut guard = self.slot.permit.lock().unwrap_or_else(|e| e.into_inner());
            if !self.slot.finished.load(Ordering::Acquire) {
                *guard = Some(permit);
            }
            return;
        }
        let slot = self.slot.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Ok(permit) = slot.limiter.clone().acquire_owned().await {
                    let mut guard = slot.permit.lock().unwrap_or_else(|e| e.into_inner());
                    if !slot.finished.load(Ordering::Acquire) {
                        *guard = Some(permit);
                    }
                }
            });
        }
    }
}

/// Daemon-wide peer coordination service attached to a [`CliAgentTool`].
pub(super) struct PeerCoordinator {
    tool: Weak<CliAgentTool>,
    mcp: PeerMcpContext,
    limits: PeerLimits,
    /// The daemon's inbox, where a file outside a run's folders (a
    /// screenshot in a temp folder) is staged for delivery; unset, no peer
    /// can send files.
    pub(super) file_inbox: std::sync::OnceLock<PathBuf>,
}

impl PeerCoordinator {
    /// Bind a local loopback MCP server and construct a coordinator using
    /// `bridge_exe` + `bridge_args` as the stdio MCP bridge command.
    pub(super) async fn start(
        tool: Weak<CliAgentTool>,
        bridge_exe: String,
        bridge_args: Vec<String>,
        limits: PeerLimits,
    ) -> std::io::Result<Arc<Self>> {
        let server = LocalMcpServer::bind(
            ServerMeta::new("aidaemon-peers", env!("CARGO_PKG_VERSION")).with_instructions(
                "Coordinate subtasks, shared board notes, and peer messages across CLI coding agents.",
            ),
        )
        .await?;
        let mcp = PeerMcpContext::new(server, bridge_exe, bridge_args);
        Ok(Arc::new(Self {
            tool,
            mcp,
            limits,
            file_inbox: std::sync::OnceLock::new(),
        }))
    }

    /// Create a fresh [`PeerTree`] and scoped MCP endpoint for a root CLI run.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_root_tree(
        &self,
        root_agent: &str,
        root_title: &str,
        root_ctx: PeerRootContext,
        available_agents: &[String],
    ) -> RootPeerSession {
        let tree = PeerTree::new(
            root_agent,
            root_title,
            root_ctx.working_dir.clone(),
            self.limits,
        );
        let permit_slots: Arc<StdMutex<HashMap<String, PermitSlot>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let launcher = Arc::new(RuntimePeerLauncher {
            tool: self.tool.clone(),
            tree: tree.clone(),
            root: Arc::new(root_ctx),
            permit_slots: permit_slots.clone(),
            worktrees: StdMutex::new(Vec::new()),
            files_sent: AtomicUsize::new(0),
            sent_files: StdMutex::new(Vec::new()),
        });
        let handler: Arc<dyn McpToolHandler> = PeerMcpHandler::new(
            tree.clone(),
            "peer-root",
            launcher.clone(),
            self.mcp.clone(),
        );
        let (endpoint, scoped) = self.mcp.endpoint_for(handler);
        let prompt_hint = coordination_prompt_hint("peer-root", None, available_agents);
        RootPeerSession {
            tree,
            endpoint,
            scoped,
            prompt_hint,
            permit_slots,
            launcher,
        }
    }
}

/// Authority and routing context captured from a root CLI run and inherited
/// (strictly narrowed, never widened) by every child peer it spawns.
#[derive(Clone)]
pub(super) struct PeerRootContext {
    /// The lead agent's mode, the most any peer in its tree may run in
    /// (`None`: it may not run delegated work, so neither may its peers).
    pub(super) lead_mode: Option<aidaemon_coding_agents::PermissionMode>,
    pub(super) session_id: String,
    pub(super) grant_session: String,
    pub(super) working_dir: Option<String>,
    pub(super) read_paths: Vec<String>,
    pub(super) write_paths: Vec<String>,
    pub(super) enforce_native_access: bool,
    pub(super) project: Option<aidaemon_core::ProjectRef>,
    pub(super) detached_judge: Option<Arc<crate::approvals::DetachedNestedJudge>>,
    pub(super) spawn_family: Option<crate::authority::SpawnFamily>,
    pub(super) status_tx: Option<mpsc::Sender<StatusUpdate>>,
    /// The run works for the owner in their own private conversation, the
    /// only place its peers may send files (`send_file`).
    pub(super) may_send_files: bool,
}

/// Active peer-coordination session attached to a root CLI run.
pub(super) struct RootPeerSession {
    pub(super) tree: Arc<PeerTree>,
    pub(super) endpoint: McpEndpoint,
    pub(super) scoped: ScopedEndpoint,
    pub(super) prompt_hint: String,
    pub(super) permit_slots: Arc<StdMutex<HashMap<String, PermitSlot>>>,
    pub(super) launcher: Arc<RuntimePeerLauncher>,
}

/// Active peer metadata carried by a running CLI (root or child) so its
/// concurrency permit can be parked during blocking peer waits and its subtree
/// cancelled when it ends or is cancelled.
#[derive(Clone)]
pub(super) struct ActivePeerBinding {
    pub(super) tree: Arc<PeerTree>,
    pub(super) peer_id: String,
    pub(super) permit_slots: Arc<StdMutex<HashMap<String, PermitSlot>>>,
    pub(super) is_root: bool,
    pub(super) launcher: Option<Arc<RuntimePeerLauncher>>,
    pub(super) branch: Option<String>,
}

/// How much of each board post a lead run's result keeps.
const PEER_RECORD_BODY_CHARS: usize = 2_000;
/// How much of the board, all posts together, a lead run's result keeps.
const PEER_RECORD_BOARD_CHARS: usize = 16_000;

impl ActivePeerBinding {
    /// The tree's record for a lead run's result (every peer and board
    /// post): its own report only summarizes them, and this is what its
    /// caller can quote. `None` for a peer, or a tree with nothing in it.
    pub(super) fn lead_record(&self) -> Option<String> {
        if !self.is_root {
            return None;
        }
        let mut record = self
            .tree
            .record(PEER_RECORD_BODY_CHARS, PEER_RECORD_BOARD_CHARS);
        let sent = self
            .launcher
            .as_ref()
            .map(|launcher| {
                launcher
                    .sent_files
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
            })
            .unwrap_or_default();
        if !sent.is_empty() {
            let text = record.get_or_insert_with(String::new);
            text.push_str("Files sent to the conversation:\n");
            for (peer, file, caption) in sent {
                text.push_str(&format!("- {file} by {peer}"));
                if !caption.is_empty() {
                    text.push_str(&format!(" (caption: {caption})"));
                }
                text.push('\n');
            }
        }
        record
    }

    /// Whether a lead run acted through its tree: launched a peer, posted
    /// to the board or sent a file. Such a run changed things outside its
    /// folders even when it could not edit them.
    pub(super) fn acted(&self) -> bool {
        self.is_root
            && (self.tree.acted()
                || self.launcher.as_ref().is_some_and(|launcher| {
                    !launcher
                        .sent_files
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .is_empty()
                }))
    }

    pub(super) fn register_permit(&self, slot: PermitSlot) {
        self.permit_slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(self.peer_id.clone(), slot);
    }

    pub(super) fn unregister_permit(&self) {
        if let Some(slot) = self
            .permit_slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.peer_id)
        {
            slot.release();
        }
    }

    /// Cancel any still-running descendant peers when a root CLI ends or is
    /// cancelled.
    pub(super) async fn cleanup_children(&self) {
        if !self.is_root {
            return;
        }
        let runs = self.tree.cancel_subtree(&self.peer_id);
        let Some(launcher) = self.launcher.as_ref() else {
            return;
        };
        for (peer_id, run_id) in runs {
            if peer_id == self.peer_id {
                continue;
            }
            let _ = launcher.cancel(&run_id).await;
        }
        launcher.release_worktrees().await;
    }
}

/// Host [`PeerLauncher`] implementation bridging [`PeerMcpHandler`] directly
/// onto [`CliAgentTool`].
pub(super) struct RuntimePeerLauncher {
    tool: Weak<CliAgentTool>,
    tree: Arc<PeerTree>,
    root: Arc<PeerRootContext>,
    permit_slots: Arc<StdMutex<HashMap<String, PermitSlot>>>,
    /// Worktrees provisioned in this tree, released when its root ends.
    worktrees: StdMutex<Vec<PeerWorktree>>,
    /// Files this tree's peers sent to the conversation.
    files_sent: AtomicUsize,
    /// Each file sent: the peer, the file's name and its caption.
    sent_files: StdMutex<Vec<(String, String, String)>>,
}

impl RuntimePeerLauncher {
    /// Release every worktree this tree provisioned (its root has ended).
    pub(super) async fn release_worktrees(&self) {
        let worktrees =
            std::mem::take(&mut *self.worktrees.lock().unwrap_or_else(|e| e.into_inner()));
        let Ok(tool) = self.upgrade_tool() else {
            return;
        };
        for worktree in worktrees {
            if let Err(error) = release_peer_worktree(tool.backend.as_ref(), &worktree).await {
                tracing::warn!(%error, worktree = %worktree.path, "Could not release a peer worktree");
            }
        }
    }

    fn upgrade_tool(&self) -> Result<Arc<CliAgentTool>, String> {
        self.tool
            .upgrade()
            .ok_or_else(|| "CLI agent runtime is no longer available".to_string())
    }
}

#[async_trait]
impl PeerLauncher for RuntimePeerLauncher {
    async fn available_agents(&self) -> Vec<PeerAgentInfo> {
        let Ok(tool) = self.upgrade_tool() else {
            return Vec::new();
        };
        let entries: Vec<(String, String)> = {
            let map = tool.tools.read().unwrap_or_else(|e| e.into_inner());
            let mut list: Vec<(String, String)> = map
                .iter()
                .map(|(name, entry)| (name.clone(), entry.description.clone()))
                .collect();
            list.sort_by(|a, b| a.0.cmp(&b.0));
            list
        };
        entries
            .into_iter()
            .map(|(name, description)| PeerAgentInfo { name, description })
            .collect()
    }

    async fn launch(&self, request: PeerLaunchRequest) -> Result<PeerLaunchResult, String> {
        if request.cancel.is_cancelled() {
            return Err("peer launch cancelled before start".to_string());
        }
        let tool = self.upgrade_tool()?;
        if tool.backend.kind() != BackendKind::Local {
            return Err("peer CLI coordination requires a local execution backend".to_string());
        }
        if tool.configured_tool(&request.agent).is_none() {
            let names: Vec<String> = tool
                .tool_names
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            return Err(format!(
                "unknown CLI agent `{}`; configured agents: {}",
                request.agent,
                names.join(", ")
            ));
        }

        // Resolve workspace strictly within the root run's authority.
        let (workspace_mode, working_dir, read_paths, write_paths, branch) = match request.workspace
        {
            PeerWorkspaceMode::ReadOnly => (
                CliWorkspaceMode::ReadOnly,
                self.root.working_dir.clone(),
                self.root.read_paths.clone(),
                Vec::new(),
                None,
            ),
            PeerWorkspaceMode::Worktree => {
                // A worktree peer writes: a lead with no write authority
                // (a read-only run) cannot hand one out.
                if self.root.write_paths.is_empty() && self.root.enforce_native_access {
                    return Err(
                        "lead CLI has no write authority; use workspace=\"read_only\"".to_string(),
                    );
                }
                if request.continue_run.is_some() {
                    // Resuming an existing worktree peer keeps its existing working dir.
                    let existing = self.tree.peer(&request.peer_id);
                    let dir = existing
                        .as_ref()
                        .and_then(|p| p.working_dir.clone())
                        .or_else(|| self.root.working_dir.clone());
                    let branch = existing.and_then(|p| p.branch);
                    let writes = dir.iter().cloned().collect();
                    (
                        CliWorkspaceMode::ReadWrite,
                        dir.clone(),
                        dir.into_iter().collect(),
                        writes,
                        branch,
                    )
                } else {
                    let base = self.root.working_dir.as_deref().ok_or_else(|| {
                        "workspace=\"worktree\" requires the lead CLI to have a git working directory"
                            .to_string()
                    })?;
                    let worktree = provision_peer_worktree(
                        tool.backend.as_ref(),
                        base,
                        &request.tree_id,
                        &request.peer_id,
                    )
                    .await?;
                    let (wt_path, branch_name) = (worktree.path.clone(), worktree.branch.clone());
                    self.worktrees
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(worktree);
                    (
                        CliWorkspaceMode::ReadWrite,
                        Some(wt_path.clone()),
                        vec![wt_path.clone()],
                        vec![wt_path],
                        Some(branch_name),
                    )
                }
            }
            PeerWorkspaceMode::Subpath => {
                if self.root.write_paths.is_empty() && self.root.enforce_native_access {
                    return Err(
                        "lead CLI has no write authority; use workspace=\"read_only\"".to_string(),
                    );
                }
                let validated = validate_narrowed_write_paths(
                    &request.write_paths,
                    self.root.working_dir.as_deref(),
                    &self.root.write_paths,
                    self.root.enforce_native_access,
                )?;
                (
                    CliWorkspaceMode::ReadWrite,
                    self.root.working_dir.clone(),
                    self.root.read_paths.clone(),
                    validated,
                    None,
                )
            }
        };

        // Optional session continuation when `message_peer(resume=true)` resumes a peer.
        let continues = match request.continue_run.as_deref() {
            Some(earlier_run_id) => match tool
                .earlier_run(earlier_run_id, &self.root.session_id)
                .await
            {
                Ok(earlier) => Some(earlier.continues),
                Err(why) => return Err(format!("could not resume peer session: {why}")),
            },
            None => None,
        };

        let has_known_agent = tool
            .tools
            .read()
            .ok()
            .and_then(|tools| {
                tools
                    .get(&request.agent)
                    .map(|entry| tool.entry_agent(entry))
            })
            .is_some_and(|agent| agent.launch_arguments().is_some());
        let full_prompt = if has_known_agent && request.mcp_endpoint.is_some() {
            let hint =
                coordination_prompt_hint(&request.peer_id, Some(&request.parent_peer_id), &[]);
            format!("{}\n\n{hint}", request.prompt.trim_end())
        } else {
            request.prompt.clone()
        };

        let spec = crate::helper_run::HelperSpec {
            title: Some(if request.title.trim().is_empty() {
                format!("{} peer task", request.agent)
            } else {
                request.title.clone()
            }),
            mission: request.prompt.clone(),
            new_folders: Vec::new(),
            scope: write_paths.clone(),
            continues,
            owner_follow_up: false,
        };

        let binding = ActivePeerBinding {
            tree: self.tree.clone(),
            peer_id: request.peer_id.clone(),
            permit_slots: self.permit_slots.clone(),
            is_root: false,
            launcher: None,
            branch: branch.clone(),
        };

        // A subpath child writes inside its ancestors' folder by their own
        // delegation (its paths were narrowed from theirs above): their claims
        // on that folder don't block it. Anyone else's still do.
        let nests_in = match request.workspace {
            PeerWorkspaceMode::Subpath => ancestor_task_ids(&self.tree, &request.parent_peer_id),
            _ => Vec::new(),
        };

        let launched = tool
            .launch_peer_run(PeerLaunchSpec {
                tool_name: request.agent,
                prompt: full_prompt,
                task_prompt: request.prompt,
                working_dir: working_dir.clone(),
                session_id: self.root.session_id.clone(),
                grant_session: self.root.grant_session.clone(),
                workspace_mode,
                read_paths,
                write_paths,
                enforce_native_access: self.root.enforce_native_access,
                wait: request.wait,
                status_tx: self.root.status_tx.clone(),
                spec,
                project: self.root.project.clone(),
                detached_judge: self.root.detached_judge.clone(),
                spawn_family: self.root.spawn_family.clone(),
                mcp_endpoints: request.mcp_endpoint.into_iter().collect(),
                mcp_keep_alive: request.mcp_keep_alive,
                peer_binding: binding,
                cancel: request.cancel,
                nests_in,
                mode_cap: Some(
                    self.root
                        .lead_mode
                        .ok_or("the lead agent may not run delegated work, so its peers may not")?,
                ),
            })
            .await?;

        Ok(PeerLaunchResult {
            run_id: launched.run_id,
            working_dir,
            branch,
            finished: launched.finished,
            success: launched.success,
            output: launched.output,
            resumable: launched.resumable,
        })
    }

    async fn check(&self, run_id: &str, wait: bool) -> Result<PeerRunStatus, String> {
        let tool = self.upgrade_tool()?;
        let task_id = cli_agent_task_id(run_id).unwrap_or(run_id);
        if wait {
            tool.wait_for_finish(task_id, Duration::from_secs(300))
                .await;
        }
        let running = tool.running.lock().await;
        if let Some(agent) = running.get(task_id) {
            let finished = agent.finished.load(Ordering::Acquire);
            if !finished {
                let queued = agent.process.is_queued();
                let partial = agent.display_buf.lock().await.clone();
                let preview = if partial.is_empty() {
                    None
                } else {
                    Some(crate::tools::terminal::keep_head_and_tail(&partial, 4000))
                };
                return Ok(PeerRunStatus {
                    run_id: cli_run_id(task_id),
                    state: if queued {
                        PeerState::Queued
                    } else {
                        PeerState::Running
                    },
                    output: preview,
                    working_dir: agent.claim_dir.clone(),
                    branch: None,
                    resumable: false,
                });
            }
            let verdict = agent.verdict.get().cloned();
            let output = match verdict.as_ref() {
                Some(v) => Some(v.text.clone()),
                None => Some(CliAgentTool::build_finished_result(&tool.backend, agent).await),
            };
            let state = match verdict {
                Some(v) if !v.success => PeerState::Failed,
                _ => PeerState::Completed,
            };
            let dir = agent.claim_dir.clone();
            drop(running);
            let resumable = tool
                .state
                .cli_agent_run(&cli_run_id(task_id))
                .await
                .ok()
                .flatten()
                .is_some_and(|r| r.keeps_session && r.agent_session.is_some());
            return Ok(PeerRunStatus {
                run_id: cli_run_id(task_id),
                state,
                output,
                working_dir: dir,
                branch: None,
                resumable,
            });
        }
        drop(running);

        if let Some(done) = tool.completed.lock().await.get(task_id).cloned() {
            let state = if done.result.trim_start().starts_with("ERROR: CLI agent") {
                PeerState::Failed
            } else {
                PeerState::Completed
            };
            let resumable = tool
                .state
                .cli_agent_run(&cli_run_id(task_id))
                .await
                .ok()
                .flatten()
                .is_some_and(|r| r.keeps_session && r.agent_session.is_some());
            return Ok(PeerRunStatus {
                run_id: cli_run_id(task_id),
                state,
                output: Some(done.result),
                working_dir: None,
                branch: None,
                resumable,
            });
        }

        if let Ok(Some(found)) = tool.state.cli_agent_run(&cli_run_id(task_id)).await {
            let state = match found.success {
                Some(true) => PeerState::Completed,
                Some(false) => PeerState::Failed,
                None => PeerState::Running,
            };
            return Ok(PeerRunStatus {
                run_id: cli_run_id(task_id),
                state,
                output: found.output_summary,
                working_dir: found.working_dir,
                branch: None,
                resumable: found.keeps_session && found.agent_session.is_some(),
            });
        }

        Err(format!("no CLI peer run matched `{run_id}`"))
    }

    async fn cancel(&self, run_id: &str) -> Result<String, String> {
        let tool = self.upgrade_tool()?;
        let task_id = cli_agent_task_id(run_id).unwrap_or(run_id);
        tool.handle_cancel(task_id)
            .await
            .map_err(|error| error.to_string())
    }

    async fn send_file(
        &self,
        caller_peer_id: &str,
        path: &str,
        caption: Option<&str>,
    ) -> Result<String, String> {
        if !self.root.may_send_files {
            return Err(
                "files go only to the owner's own private conversation, and this run's \
                 conversation is not one: name the path in your reply instead"
                    .to_string(),
            );
        }
        if path.is_empty() {
            return Err("give the path of the file to send".to_string());
        }
        let tool = self.upgrade_tool()?;
        let inbox = tool
            .peer_coord
            .get()
            .and_then(|coord| coord.file_inbox.get().cloned())
            .ok_or("this daemon has no file delivery set up")?;
        if self.files_sent.load(Ordering::Acquire) >= MAX_FILES_PER_TREE {
            return Err(format!(
                "this run already sent {MAX_FILES_PER_TREE} files, the most one run may send"
            ));
        }
        // A relative path is the caller's own working folder's.
        let caller_dir = self
            .tree
            .peer(caller_peer_id)
            .and_then(|peer| peer.working_dir)
            .or_else(|| self.root.working_dir.clone());
        let requested = match (Path::new(path).is_absolute(), caller_dir.as_deref()) {
            (false, Some(dir)) => Path::new(dir).join(path).to_string_lossy().into_owned(),
            _ => path.to_string(),
        };
        // What the run may read: its folders and its peers' worktrees. A file
        // elsewhere in a temp folder is staged through the inbox; anything
        // else, and secrets anywhere, is refused (`prepare_delivery`).
        let mut roots: Vec<PathBuf> = self
            .root
            .working_dir
            .iter()
            .chain(self.root.read_paths.iter())
            .chain(self.root.write_paths.iter())
            .map(PathBuf::from)
            .collect();
        roots.extend(
            self.worktrees
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|worktree| PathBuf::from(&worktree.path)),
        );
        let cwd = caller_dir.map(PathBuf::from).unwrap_or_default();
        use aidaemon_tools::file_delivery::{DeliveryError, prepare_delivery};
        let ready = prepare_delivery(&requested, &cwd, &inbox, &roots).map_err(|error| match error {
            DeliveryError::FileNotFound(_) => format!("no file at `{requested}`"),
            DeliveryError::NotRegularFile(_) => format!("`{requested}` is not a regular file"),
            DeliveryError::Blocked(_) => {
                format!("`{requested}` may hold secrets and is never sent")
            }
            DeliveryError::OutsideAllowedDirs(_) => format!(
                "`{requested}` is outside this run's folders; copy it into the working folder first"
            ),
            DeliveryError::RecoveryFailed { error, .. } => {
                format!("could not stage `{requested}` for delivery: {error}")
            }
        })?;
        if ready.size_bytes > MAX_FILE_BYTES {
            return Err(format!(
                "`{}` is {} bytes, more than the {MAX_FILE_BYTES} a run may send",
                ready.filename, ready.size_bytes
            ));
        }
        let hub = tool
            .get_hub()
            .ok_or("the conversation's channel is not connected")?;
        let media = aidaemon_channels::hub::MediaMessage {
            session_id: self.root.session_id.clone(),
            caption: caption.unwrap_or_default().to_string(),
            kind: aidaemon_channels::hub::MediaKind::Document {
                file_path: ready.canonical_path.to_string_lossy().into_owned(),
                filename: ready.filename.clone(),
            },
            result_tx: None,
        };
        hub.send_media_strict(&self.root.session_id, &media)
            .await
            .map_err(|error| format!("the conversation did not take the file: {error}"))?;
        self.files_sent.fetch_add(1, Ordering::AcqRel);
        self.sent_files
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((
                caller_peer_id.to_string(),
                ready.filename.clone(),
                caption.unwrap_or_default().to_string(),
            ));
        info!(
            peer_id = %caller_peer_id,
            file = %ready.filename,
            bytes = ready.size_bytes,
            "A peer CLI sent a file to its conversation"
        );
        Ok(format!(
            "sent {} ({} bytes) to the owner's conversation",
            ready.filename, ready.size_bytes
        ))
    }

    async fn before_blocking_wait(
        &self,
        caller_peer_id: &str,
    ) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        let slot = self
            .permit_slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(caller_peer_id)
            .cloned()?;
        Some(Box::new(slot.park()))
    }
}

/// The most files one tree's peers may send to its conversation.
const MAX_FILES_PER_TREE: usize = 20;
/// The largest file a peer may send.
const MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;

/// Internal spec passed from [`RuntimePeerLauncher`] to [`CliAgentTool::launch_peer_run`].
pub(super) struct PeerLaunchSpec {
    pub(super) tool_name: String,
    pub(super) prompt: String,
    pub(super) task_prompt: String,
    pub(super) working_dir: Option<String>,
    pub(super) session_id: String,
    pub(super) grant_session: String,
    pub(super) workspace_mode: CliWorkspaceMode,
    pub(super) read_paths: Vec<String>,
    pub(super) write_paths: Vec<String>,
    pub(super) enforce_native_access: bool,
    pub(super) wait: bool,
    pub(super) status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub(super) spec: crate::helper_run::HelperSpec,
    pub(super) project: Option<aidaemon_core::ProjectRef>,
    pub(super) detached_judge: Option<Arc<crate::approvals::DetachedNestedJudge>>,
    pub(super) spawn_family: Option<crate::authority::SpawnFamily>,
    pub(super) mcp_endpoints: Vec<McpEndpoint>,
    pub(super) mcp_keep_alive: Option<Box<dyn std::any::Any + Send + Sync>>,
    pub(super) peer_binding: ActivePeerBinding,
    pub(super) cancel: tokio_util::sync::CancellationToken,
    /// Runs (task ids) whose folder claims this run may nest inside.
    pub(super) nests_in: Vec<String>,
    /// The lead's mode: the peer runs in no more than this.
    pub(super) mode_cap: Option<aidaemon_coding_agents::PermissionMode>,
}

/// The task ids of `peer_id` and its ancestors' runs in `tree`.
pub(super) fn ancestor_task_ids(tree: &PeerTree, peer_id: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut next = Some(peer_id.to_string());
    // Bounded: a tree is shallow, and a malformed parent chain must not spin.
    for _ in 0..64 {
        let Some(peer) = next.as_deref().and_then(|id| tree.peer(id)) else {
            break;
        };
        if let Some(run_id) = peer.run_id.as_deref() {
            ids.push(cli_agent_task_id(run_id).unwrap_or(run_id).to_string());
        }
        next = peer.parent_peer_id;
    }
    ids
}

/// Result returned by [`CliAgentTool::launch_peer_run`].
pub(super) struct PeerLaunched {
    pub(super) run_id: String,
    pub(super) finished: bool,
    pub(super) success: Option<bool>,
    pub(super) output: Option<String>,
    pub(super) resumable: bool,
}

/// Ensure every requested subpath write target stays strictly inside the root
/// run's allowed write paths (authority narrowing, never widening).
pub(super) fn validate_narrowed_write_paths(
    requested: &[String],
    working_dir: Option<&str>,
    parent_write_paths: &[String],
    enforce: bool,
) -> Result<Vec<String>, String> {
    if requested.is_empty() {
        return Err("workspace=\"subpath\" requires at least one write path".to_string());
    }
    // Compared and handed out as real paths: a symlink inside the lead's
    // folder that points elsewhere would pass a lexical prefix check and
    // then give the peer's sandbox the folder it points to.
    let parent_roots: Vec<PathBuf> = parent_write_paths
        .iter()
        .map(|p| real_path(&normalize_lexically(Path::new(p))))
        .collect();
    let mut out = Vec::with_capacity(requested.len());
    for raw in requested {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let path = Path::new(trimmed);
        let resolved = if path.is_absolute() {
            normalize_lexically(path)
        } else if let Some(cwd) = working_dir {
            normalize_lexically(&Path::new(cwd).join(path))
        } else {
            return Err(format!(
                "relative write path `{trimmed}` requires a working directory"
            ));
        };
        let resolved = real_path(&resolved);
        if enforce && !parent_roots.iter().any(|root| resolved.starts_with(root)) {
            return Err(format!(
                "write path `{}` is outside the lead CLI's allowed write scope",
                resolved.display()
            ));
        }
        out.push(resolved.to_string_lossy().into_owned());
    }
    if out.is_empty() {
        return Err("workspace=\"subpath\" requires at least one non-empty write path".to_string());
    }
    Ok(out)
}

/// `path` with every symlink in its existing part resolved; the part that
/// does not exist yet (a folder the peer will make) is appended as given.
fn real_path(path: &Path) -> PathBuf {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    loop {
        if let Ok(real) = std::fs::canonicalize(&existing) {
            let mut out = real;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (
            existing.file_name().map(|n| n.to_os_string()),
            existing.parent(),
        ) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => return path.to_path_buf(),
        }
    }
}

fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

async fn git_cmd(
    backend: &dyn ExecutionBackend,
    cwd: &str,
    args: Vec<String>,
) -> Result<String, String> {
    // The repository is one agents write to: its config and hooks are not
    // the daemon's to run. No hooks, no fsmonitor command, no signing
    // program the repository names.
    let mut full: Vec<String> = [
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "tag.gpgSign=false",
        "-C",
    ]
    .iter()
    .map(|arg| arg.to_string())
    .collect();
    full.push(cwd.to_string());
    full.extend(args);
    let output = backend
        .execute(ExecutionRequest::argv("git", full), Duration::from_secs(30))
        .await
        .map_err(|e| format!("failed to run git: {e}"))?;
    if output.timed_out || output.exit_code != 0 {
        return Err(format!(
            "git command failed: {}",
            output.stderr_lossy().trim()
        ));
    }
    Ok(output.stdout_lossy().trim().to_string())
}

/// Provision an isolated git worktree on a new branch `aidaemon/peer-<tree>-<peer>`
/// off the repository's current `HEAD` so parallel peer CLIs can edit the same
/// repository without triggering working-directory claim conflicts.
pub(super) async fn provision_peer_worktree(
    backend: &dyn ExecutionBackend,
    base_dir: &str,
    tree_id: &str,
    peer_id: &str,
) -> Result<PeerWorktree, String> {
    let git_root = git_cmd(
        backend,
        base_dir,
        vec!["rev-parse".into(), "--show-toplevel".into()],
    )
    .await
    .map_err(|_| {
        format!(
            "cannot provision worktree: `{base_dir}` is not inside a git repository; use workspace=\"read_only\" or \"subpath\""
        )
    })?;
    let head = git_cmd(backend, &git_root, vec!["rev-parse".into(), "HEAD".into()]).await?;

    let safe_tree: String = tree_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .take(12)
        .collect();
    let safe_peer: String = peer_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .take(16)
        .collect();
    let unique = uuid::Uuid::new_v4().simple().to_string();
    let short_uid = &unique[..6];
    let branch_name = format!("aidaemon/peer-{safe_tree}-{safe_peer}-{short_uid}");
    let root = peer_worktrees_root();
    aidaemon_coding_agents::private_files::private_dir(&root)
        .map_err(|e| format!("failed to create the private worktree dir: {e}"))?;
    let worktree_parent = root.join(&safe_tree);
    std::fs::create_dir_all(&worktree_parent)
        .map_err(|e| format!("failed to create worktree parent dir: {e}"))?;
    let worktree_dir = worktree_parent.join(format!("{safe_peer}-{short_uid}"));
    let worktree_path = worktree_dir.to_string_lossy().into_owned();

    git_cmd(
        backend,
        &git_root,
        vec![
            "worktree".into(),
            "add".into(),
            "-b".into(),
            branch_name.clone(),
            worktree_path.clone(),
            head.clone(),
        ],
    )
    .await?;

    info!(
        tree_id = %safe_tree,
        peer_id = %safe_peer,
        branch = %branch_name,
        worktree = %worktree_path,
        "Provisioned isolated git worktree for peer CLI run"
    );
    Ok(PeerWorktree {
        git_root,
        path: worktree_path,
        branch: branch_name,
        base: Some(head),
        peer_id: peer_id.to_string(),
    })
}

/// The root of every peer worktree this daemon provisions.
///
/// The user's own cache directory, owner-only: a peer's checkout holds the
/// repository's source, and a fixed name in the shared temp directory could
/// be read, or made first, by another local user.
fn peer_worktrees_root() -> PathBuf {
    dirs::cache_dir()
        .map(|cache| cache.join("aidaemon").join("peer-worktrees"))
        .unwrap_or_else(|| std::env::temp_dir().join("aidaemon-peer-worktrees"))
}

/// A git worktree provisioned for one peer, released when its tree ends.
#[derive(Debug, Clone)]
pub(super) struct PeerWorktree {
    pub(super) git_root: String,
    pub(super) path: String,
    pub(super) branch: String,
    /// The commit it was branched from; `None` when unknown (a worktree left
    /// by an earlier daemon), so its branch is always kept.
    pub(super) base: Option<String>,
    pub(super) peer_id: String,
}

/// What releasing a peer worktree kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ReleasedWorktree {
    /// The branch still holding the peer's work; `None` when it had none and
    /// was deleted.
    pub(super) kept_branch: Option<String>,
    /// Whether uncommitted changes were committed to the branch first.
    pub(super) committed: bool,
}

/// Release a peer worktree: commit what the peer left uncommitted to its own
/// branch (its work is never discarded), remove the worktree, and delete the
/// branch when it holds nothing beyond the commit it started from.
pub(super) async fn release_peer_worktree(
    backend: &dyn ExecutionBackend,
    worktree: &PeerWorktree,
) -> Result<ReleasedWorktree, String> {
    let mut committed = false;
    if Path::new(&worktree.path).is_dir() {
        let dirty = git_cmd(
            backend,
            &worktree.path,
            vec!["status".into(), "--porcelain".into()],
        )
        .await?;
        if !dirty.is_empty() {
            git_cmd(backend, &worktree.path, vec!["add".into(), "-A".into()]).await?;
            let message = format!(
                "aidaemon peer {}: work left uncommitted when its run ended",
                worktree.peer_id
            );
            let commit = |identity: bool| {
                let mut args: Vec<String> = Vec::new();
                if identity {
                    // Only when the repository has no identity of its own.
                    args.extend([
                        "-c".into(),
                        "user.name=aidaemon".into(),
                        "-c".into(),
                        "user.email=aidaemon@localhost".into(),
                    ]);
                }
                args.extend([
                    "commit".into(),
                    "--no-verify".into(),
                    "-q".into(),
                    "-m".into(),
                    message.clone(),
                ]);
                args
            };
            if git_cmd(backend, &worktree.path, commit(false))
                .await
                .is_err()
            {
                git_cmd(backend, &worktree.path, commit(true)).await?;
            }
            committed = true;
        }
        git_cmd(
            backend,
            &worktree.git_root,
            vec![
                "worktree".into(),
                "remove".into(),
                "--force".into(),
                worktree.path.clone(),
            ],
        )
        .await?;
    } else {
        // Its folder is gone already: drop git's record of it.
        let _ = git_cmd(
            backend,
            &worktree.git_root,
            vec!["worktree".into(), "prune".into()],
        )
        .await;
    }
    if let Some(parent) = Path::new(&worktree.path).parent() {
        // Removes the tree's folder only once its last worktree is gone.
        let _ = std::fs::remove_dir(parent);
    }

    let empty = match worktree.base.as_deref() {
        Some(base) => git_cmd(
            backend,
            &worktree.git_root,
            vec![
                "rev-list".into(),
                "--count".into(),
                format!("{base}..{}", worktree.branch),
            ],
        )
        .await
        .is_ok_and(|count| count == "0"),
        None => false,
    };
    if empty {
        git_cmd(
            backend,
            &worktree.git_root,
            vec!["branch".into(), "-D".into(), worktree.branch.clone()],
        )
        .await?;
    }
    info!(
        peer_id = %worktree.peer_id,
        branch = %worktree.branch,
        committed,
        kept = !empty,
        "Released peer worktree"
    );
    Ok(ReleasedWorktree {
        kept_branch: (!empty).then(|| worktree.branch.clone()),
        committed,
    })
}

/// Release worktrees an earlier daemon left behind (it ended before their
/// trees did). Their work is committed to their branches, which are kept.
pub(super) async fn sweep_orphaned_peer_worktrees(backend: &dyn ExecutionBackend) {
    let Ok(trees) = std::fs::read_dir(peer_worktrees_root()) else {
        return;
    };
    for tree in trees.flatten() {
        let Ok(peers) = std::fs::read_dir(tree.path()) else {
            continue;
        };
        for peer in peers.flatten() {
            let path = peer.path().to_string_lossy().into_owned();
            let branch = git_cmd(
                backend,
                &path,
                vec!["rev-parse".into(), "--abbrev-ref".into(), "HEAD".into()],
            )
            .await;
            let common = git_cmd(
                backend,
                &path,
                vec![
                    "rev-parse".into(),
                    "--path-format=absolute".into(),
                    "--git-common-dir".into(),
                ],
            )
            .await;
            let (Ok(branch), Ok(common)) = (branch, common) else {
                // Not a git worktree any more: nothing of a peer's to keep.
                let _ = std::fs::remove_dir_all(peer.path());
                continue;
            };
            let Some(git_root) = Path::new(&common)
                .parent()
                .map(|root| root.to_string_lossy().into_owned())
            else {
                continue;
            };
            let worktree = PeerWorktree {
                git_root,
                path,
                branch,
                base: None,
                peer_id: peer.file_name().to_string_lossy().into_owned(),
            };
            if let Err(error) = release_peer_worktree(backend, &worktree).await {
                tracing::warn!(%error, worktree = %worktree.path, "Could not release an orphaned peer worktree");
            }
        }
        let _ = std::fs::remove_dir(tree.path());
    }
}
