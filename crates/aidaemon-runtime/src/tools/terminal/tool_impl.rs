//! The `Tool` and `ToolSemantics` impls.

use super::*;

#[async_trait]
impl Tool for TerminalTool {
    fn name(&self) -> &str {
        "terminal"
    }

    /// The terminal's cards go out in its configured mode (which picks the
    /// buttons offered).
    fn approval_card_mode(&self) -> Option<PermissionMode> {
        Some(self.permission_mode)
    }

    async fn record_owner_answer(
        &self,
        call: &crate::tools::traits::ApprovalCall<'_>,
        facts: &ToolCallContext,
        response: Option<ApprovalResponse>,
    ) {
        let trust = TerminalTrust::for_approval(facts);
        TerminalTool::record_owner_answer(self, call.arguments, &trust, response).await;
    }

    fn description(&self) -> &str {
        "Run shell commands in local, Docker, or SSH workspaces. Dangerous or uncertain commands require owner approval. Check or kill long-running commands later; use write_file instead of shell redirection. If a chain (&&, ||, ;, |) has a dangerous segment, refuse the whole chain and ask which operation the user wants."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "terminal",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "One shell command for action=run. Use this when exact native shell exit semantics are required. Mutually exclusive with script."
                    },
                    "script": {
                        "type": "string",
                        "description": "A multi-step POSIX shell workflow for action=run, passed through stdin to /bin/sh -eu -s so a failed step stops later steps without nested command-string quoting. Mutually exclusive with command."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Explicit execution directory for action=run; relative command paths resolve from this directory"
                    },
                    "read_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Additional exact readable files/directories for a confined run. Paths outside the task scope (e.g. a deploy tool's config dir under $HOME) are allowed but prompt the user for approval; credential stores such as ~/.ssh or ~/.aws never are."
                    },
                    "description": {
                        "type": "string",
                        "description": "A few plain words naming what this command does, shown to the owner while it runs (e.g. 'running the site installer', 'counting script'). Never the command itself."
                    },
                    "network": {
                        "type": "boolean",
                        "description": "Allow outbound network for this run (deploys, package installs, API calls). Confined runs have no network by default; set true when the command needs it. Prompts the user for approval unless already granted for this command."
                    },
                    "write_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Exact files the confined run may create or change (e.g. /tmp/out/result.txt). Never list a file under write_roots: roots are prepared as directories before launch, so a file declared there would be created as a directory and the write would fail. Existing directories you need to write into may also be listed here; use write_roots for directories that must be created."
                    },
                    "read_roots": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Typed directory roots whose descendants the confined run may read"
                    },
                    "write_roots": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Directories (never files) whose descendants the confined run may create or change. A root that does not exist yet is created as a directory before launch, so list only directories here; put exact output files in write_paths."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["run", "check", "kill"],
                        "description": "run, check, or kill"
                    },
                    "detach": {
                        "type": "boolean",
                        "description": "Launch-and-return for a long-lived process that never ends by itself (server, tunnel, watcher): after a ~1.5s startup check the command is handed to background tracking (background_started=true, detached=true, pid returned) and survives task end until explicitly killed. A command that exits within the startup check is reported synchronously with its exit code. For finite work use background instead."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run finite work (a build, a count, a long script) in the background: after a ~1.5s startup check this turn ends and the result is sent to the user when the command finishes. Do not wait for it, poll it or start another command to watch it. A command that exits within the startup check is reported synchronously."
                    },
                    "pid": {
                        "type": "integer",
                        "description": "Process ID for check/kill"
                    },
                    "checklist_items": {
                        "type": "array",
                        "items": { "type": "integer", "minimum": 0 },
                        "uniqueItems": true,
                        "description": "The indices of your checklist items this command carries out, so a change it makes settles them."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        let status_tx = aidaemon_tools::status::status_sender(facts);
        let exec_ctx = facts.clone();
        // The checklist items the command said it carries out: its typed
        // link to the declared work, on its receipt.
        let checklist_items: Vec<usize> = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| value.get("checklist_items").cloned())
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default();

        // A run that says it carries out declared items and is not provably
        // read-only changed something, even when no declared write path says
        // what: work inside `docker exec` or `ssh` names paths this computer
        // does not have. Its receipt says so (`UNSPECIFIED`, which proves
        // only the items the call claims), or the ledger reads it as an
        // observation and keeps asking for the change (live 2026-11-20:
        // every container task got two passes that re-ran finished work).
        // Whether it changes something: inside a remote exec (`docker
        // exec`, `ssh`) the command it runs decides, so a look inside a
        // container is a look; otherwise the authority view does.
        let run_text = serde_json::from_str::<TerminalArgs>(arguments)
            .ok()
            .and_then(|args| args.command.or(args.script));
        let changes = match run_text
            .as_deref()
            .and_then(aidaemon_tools::command_effects::remote_inner_only_reads)
        {
            Some(only_reads) => !only_reads,
            None => {
                !self
                    .call_annotations_in(arguments, &crate::tools::traits::TrustRoots::new(&[]))
                    .read_only
            }
        };
        let opaque_change = !checklist_items.is_empty() && changes;
        self.execute_terminal(arguments, status_tx, &exec_ctx)
            .await
            .map(|mut outcome| {
                if outcome.metadata.checklist_items.is_empty() {
                    outcome.metadata.checklist_items = checklist_items;
                }
                if !outcome.metadata.semantics.mutates_state() {
                    if opaque_change {
                        outcome.metadata.semantics =
                            outcome.metadata.semantics.with_untyped_change();
                    } else if run_text
                        .as_deref()
                        .is_some_and(aidaemon_tools::command_effects::recognizably_changes)
                    {
                        // A recognized change is work done even before any
                        // checklist exists: its receipt is a change
                        // (`UNSPECIFIED`, which proves no item it does not
                        // name), so the owner's request reads as attempted
                        // (live 2026-11-20: a script written with a
                        // here-document inside `docker exec` before the
                        // checklist read as "nothing attempted", and the run
                        // was sent back). The one item in progress may be
                        // credited with it. An unknown command stays a look:
                        // it can miss a credit, never invent one.
                        outcome.metadata.semantics =
                            outcome.metadata.semantics.with_untyped_change();
                        outcome.metadata.changes_untyped = true;
                    }
                }
                outcome
            })
    }
}

#[async_trait::async_trait]
impl crate::tools::traits::ToolSemantics for TerminalTool {
    fn canonicalize_arguments(
        &self,
        arguments: &str,
    ) -> Result<String, ToolArgumentContractViolation> {
        canonicalize_terminal_arguments(arguments)
    }

    fn validate_arguments(&self, arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        validate_terminal_argument_contract(arguments)
    }

    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            runs_as_judged: true,
            keeps_own_grants: true,
            shell_command_line: true,
            offering: aidaemon_tools::semantics::Offering::Core,
            same_tool_streak: crate::tools::traits::SameToolStreak::Versatile,
            records_launch_trust: true,
            goal_context: crate::tools::traits::GoalContext::BackgroundDelivery,
            recap: aidaemon_tools::semantics::RecapKind::Command,
            activity: aidaemon_tools::semantics::ToolActivity::Fixed(
                aidaemon_core::card::ActivityKind::Run,
            ),
            project_dir_argument: aidaemon_tools::semantics::ProjectDirArgument::ShellWorkingDir,
            opens_project_instructions: true,
            status_label: Some("running a command"),
            result_fit: aidaemon_tools::semantics::ResultFit {
                strip_control_markers: true,
                pages_spilled_results: true,
                ..Default::default()
            },
            evidence: aidaemon_tools::semantics::EvidenceSource::Shell,
            semantic_scope: Some(aidaemon_tools::semantics::ToolSemanticScope::LocalWorkspace),
            ..Default::default()
        }
    }

    /// The T8 ladder on the call as the dispatcher will run it, with the
    /// turn's facts. A call whose arguments cannot be read asks, Critical,
    /// with the arguments as the model sent them; a call the dispatcher
    /// refuses needs nobody.
    async fn owner_approval_request(
        &self,
        call: &crate::tools::traits::ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> crate::tools::traits::OwnerApprovalNeed {
        use crate::tools::traits::OwnerApprovalNeed;
        if let Some(reason) = Self::unreadable_reason(call.arguments) {
            return OwnerApprovalNeed::Ask(self.unreadable_request(call.as_sent, &reason));
        }
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        let trust = TerminalTrust::for_approval(facts);
        match TerminalTool::owner_approval_request(self, call.arguments, &trust).await {
            Ok(Some(request)) => OwnerApprovalNeed::Ask(request),
            Ok(None) => OwnerApprovalNeed::NotNeeded,
            Err(reason) => OwnerApprovalNeed::Ask(self.unreadable_request(call.as_sent, &reason)),
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::EXTERNAL_HIGH_IMPACT
    }

    fn receipt_kind(&self, arguments: &str) -> crate::tools::traits::ToolReceiptKind {
        terminal_receipt_kind(arguments)
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        terminal_call_semantics(arguments)
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        terminal_call_effects(&*self.backend, arguments)
    }

    /// The declared files, plus every path operand the command names
    /// (`cp <capture> <site>/homepage.png` reads the capture though no
    /// `read_paths` says so), resolved against the run's directory.
    fn named_files(&self, arguments: &str) -> Vec<String> {
        let mut named = aidaemon_tools::semantics::declared_named_files(self, arguments);
        for operand in terminal_command_operands(arguments) {
            if !named.contains(&operand) {
                named.push(operand);
            }
        }
        named
    }

    fn call_annotations_in(
        &self,
        arguments: &str,
        roots: &crate::tools::traits::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        let Ok(args) = serde_json::from_str::<TerminalArgs>(arguments) else {
            return aidaemon_core::ToolAnnotations::WRITE;
        };
        match TerminalAction::parse(&args.action) {
            TerminalAction::Check => aidaemon_core::ToolAnnotations::READ_ONLY,
            TerminalAction::Kill => aidaemon_core::ToolAnnotations::WRITE,
            // Retired: the call only says so.
            TerminalAction::TrustAll => aidaemon_core::ToolAnnotations::READ_ONLY,
            // A non-canonical spelling executes as a run.
            TerminalAction::Run | TerminalAction::Unrecognized => {
                terminal_run_annotations(&args, self.confinement, roots)
            }
        }
    }

    fn output_trust(
        &self,
        arguments: &str,
        roots: &crate::tools::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        let Ok(args) = serde_json::from_str::<TerminalArgs>(arguments) else {
            return aidaemon_core::SourceTrust::Untrusted;
        };
        // `check` and `kill` return a background process's output: it
        // carries the trust judged when that process was launched.
        if matches!(
            TerminalAction::parse(&args.action),
            TerminalAction::Check | TerminalAction::Kill
        ) {
            return args
                .pid
                .and_then(|pid| {
                    self.launch_trust
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get(&pid)
                        .copied()
                })
                .unwrap_or(aidaemon_core::SourceTrust::Untrusted);
        }
        let text = args
            .command
            .as_deref()
            .or(args.script.as_deref())
            .unwrap_or("");
        if args.network || aidaemon_tools::command_effects::fetches_network(text) {
            return aidaemon_core::SourceTrust::Untrusted;
        }
        // The run's own absolute directory, else the turn's cwd (where a
        // run without one executes).
        let working_dir = args
            .working_dir
            .as_deref()
            .map(str::trim)
            .filter(|dir| std::path::Path::new(dir).is_absolute())
            .or(roots.cwd());
        let roots = roots.anchored_at(working_dir);
        let trusted = if run_is_confined(self.confinement) {
            // The sandbox enforces the declared read set, so it decides.
            args.read_paths
                .iter()
                .chain(args.read_roots.iter())
                .all(|path| roots.contains(path))
        } else {
            // On the host nothing enforces declared reads: trust comes from
            // the command's structure. An observation or ordinary local work,
            // run inside the roots, on operands that all resolve inside them.
            let effects = aidaemon_tools::command_effects::command_effects(text);
            (effects.observation_only
                || effects.local_work_only
                || aidaemon_tools::command_effects::is_observation_reading_input(text))
                && working_dir.is_some_and(|dir| roots.contains(dir))
                && aidaemon_tools::command_effects::path_operands(text)
                    .is_some_and(|operands| operands.iter().all(|path| roots.contains(path)))
        };
        if trusted {
            aidaemon_core::SourceTrust::Trusted
        } else {
            aidaemon_core::SourceTrust::Untrusted
        }
    }

    fn call_title(&self, arguments: &str) -> Option<String> {
        let args = serde_json::from_str::<TerminalArgs>(arguments).ok()?;
        matches!(
            TerminalAction::parse(&args.action),
            TerminalAction::Run | TerminalAction::Unrecognized
        )
        .then(|| call_title(&args))
        .flatten()
    }

    fn call_access_manifest(
        &self,
        arguments: &str,
    ) -> crate::tools::traits::ToolCallAccessManifest {
        terminal_access_manifest(arguments)
    }

    fn adapter_owned_access_manifest(
        &self,
        _arguments: &str,
    ) -> crate::tools::traits::ToolCallAccessManifest {
        // A confined process must load its shell and immutable system
        // executables before it can exercise the task manifest.  These roots
        // are adapter-owned runtime capabilities, not task data grants.  The
        // native sandbox still installs the narrower executable/runtime paths
        // it resolves for the actual command; this lane merely prevents a
        // model from having to repeat host-loader paths in `read_paths`.
        let adapter_read_targets = [
            "/bin",
            "/usr/bin",
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/usr/sbin",
            "/sbin",
        ]
        .into_iter()
        .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path.to_string()))
        .collect();
        crate::tools::traits::ToolCallAccessManifest {
            adapter_read_targets,
            ..Default::default()
        }
    }

    async fn on_task_end(&self, task_id: &str, _session_id: &str) -> anyhow::Result<()> {
        let cleaned = self.cleanup_task_processes(task_id).await?;
        if cleaned > 0 {
            info!(
                task_id,
                cleaned, "Cleaned up task-owned terminal background process(es)"
            );
        }
        Ok(())
    }

    async fn background_jobs(
        &self,
        session_id: &str,
    ) -> Vec<aidaemon_tools::semantics::BackgroundJob> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return Vec::new();
        }
        let now = chrono::Utc::now();
        let mut jobs: Vec<aidaemon_tools::semantics::BackgroundJob> = self
            .running
            .lock()
            .await
            .iter()
            .filter(|(_, proc)| {
                proc.notify_session_id == session_id && !proc.reader_handle.is_finished()
            })
            .map(|(pid, proc)| aidaemon_tools::semantics::BackgroundJob {
                key: pid.to_string(),
                title: background_job_title(&proc.command, proc.title.as_deref()),
                started_at: now
                    - chrono::Duration::from_std(proc.started_at.elapsed()).unwrap_or_default(),
                task_id: proc.origin_task_id.clone(),
                run: proc.run_id.clone(),
            })
            .collect();
        jobs.sort_by_key(|job| job.started_at);
        jobs
    }

    async fn stop_background_job(&self, key: &str) -> bool {
        let Ok(pid) = key.parse::<u32>() else {
            return false;
        };
        let taken = self.running.lock().await.remove(&pid);
        let Some(proc) = taken else {
            // Not tracked (any more): it is gone.
            return true;
        };
        self.remove_indexes_for_process(pid, &proc).await;
        self.completed.lock().await.remove(&pid);
        match self
            .terminate_running_process_verified(pid, proc, "stopped by the owner")
            .await
        {
            Ok((_, _, exited)) => {
                info!(pid, exited, "Stopped a background process for the owner");
                exited
            }
            Err(error) => {
                warn!(pid, %error, "Could not stop a background process for the owner");
                false
            }
        }
    }

    async fn jobs_ended_by_task_end(&self, task_id: &str) -> usize {
        let pids = self
            .task_processes
            .lock()
            .await
            .get(task_id)
            .cloned()
            .unwrap_or_default();
        if pids.is_empty() {
            return 0;
        }
        let running = self.running.lock().await;
        pids.iter()
            .filter_map(|pid| running.get(pid))
            .filter(|proc| {
                !proc.notifier_active && !proc.detached && !proc.reader_handle.is_finished()
            })
            .count()
    }

    async fn session_context_note(&self, session_id: &str) -> Option<String> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return None;
        }
        let mut lines: Vec<(u32, String)> = Vec::new();
        for (pid, proc) in self.running.lock().await.iter() {
            if proc.notify_session_id != session_id || proc.reader_handle.is_finished() {
                continue;
            }
            let mode = if proc.detached {
                "detached, runs until killed"
            } else {
                "background, user is notified when it finishes"
            };
            lines.push((
                *pid,
                format!(
                    "- pid={} ({}), running for {}: `{}`",
                    pid,
                    mode,
                    humanize_elapsed(proc.started_at.elapsed().as_secs()),
                    truncate_str(&proc.command, 160)
                ),
            ));
        }
        if lines.is_empty() {
            return None;
        }
        lines.sort_by_key(|(pid, _)| *pid);
        let mut note = String::from(
            "Background processes started earlier in this conversation are still running:\n",
        );
        for (_, line) in lines {
            note.push_str(&line);
            note.push('\n');
        }
        note.push_str(
            "Read one's output with terminal action=\"check\" and its pid, and stop it with action=\"kill\". Do not launch a running one again.",
        );
        Some(note)
    }
}
