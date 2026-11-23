//! What a call declares and proves: semantics, access manifests, call
//! effects and background completion receipts.

use super::*;

/// A background command that ended, as its terminal receipt records it.
pub(super) struct BackgroundCompletion<'a> {
    pub(super) session_id: &'a str,
    pub(super) task_id: &'a str,
    /// The launch's call: the receipt ends that same invocation.
    pub(super) tool_call_id: &'a str,
    pub(super) exit_code: Option<i32>,
    pub(super) output: &'a str,
    pub(super) truncation: Option<crate::tools::traits::TruncationInfo>,
    pub(super) stderr: &'a str,
    pub(super) semantics: ToolCallSemantics,
    pub(super) elapsed_secs: u64,
    /// The process ran inside the confining sandbox.
    pub(super) confined: bool,
}

/// Persist the terminal receipt of a background command that ended, under
/// the launch's own call: the launch receipt is provisional, and until this
/// one lands the run's ledger reads the command as still running, so a run
/// that waited for it and read its result still could not count as done
/// (live 2026-11-13: helpers that finished were reported as "Didn't
/// finish"). Returns the receipt's result id.
pub(super) async fn persist_background_completion_receipt(
    event_store: &Arc<EventStore>,
    ended: &BackgroundCompletion<'_>,
) -> Option<String> {
    let outcome_status = ended.exit_code.map_or(
        ToolOutcomeStatus::FailedPermanent,
        ToolOutcomeStatus::from_process_exit_code,
    );
    let mut metadata = foreground_terminal_metadata(ended.exit_code);
    metadata.receipt_kind = crate::tools::traits::ToolReceiptKind::Process;
    metadata.effective_tool_name = Some("terminal".to_string());
    metadata.truncation = ended.truncation.clone();
    metadata.semantics = ended.semantics.clone();
    metadata.access_denial = sandbox_access_denial(ended.stderr, ended.exit_code, ended.confined);
    let provenance = crate::tools::traits::ToolResultProvenance::from_authoritative_result(
        ended.output,
        &metadata,
        crate::tools::traits::ToolResultContentSource::ToolOutput,
    );
    let result_id = provenance.result_id.clone();
    metadata.result_provenance = Some(provenance);
    let mut receipt = aidaemon_events::ToolReceiptV1::from_metadata(
        &metadata,
        outcome_status,
        aidaemon_events::ToolOutcomeEvidenceSource::ToolReported,
        None,
    );
    receipt.completion_obligation_ids = event_store
        .tool_completion_obligation_ids(ended.session_id, ended.task_id, ended.tool_call_id)
        .await
        .unwrap_or_default();
    receipt.continuation_obligation_ids = receipt.completion_obligation_ids.clone();
    let emitter =
        aidaemon_events::EventEmitter::new(event_store.clone(), ended.session_id.to_string())
            .with_task_id(ended.task_id.to_string());
    if let Err(error) = emitter
        .emit(
            aidaemon_events::EventType::ToolResult,
            aidaemon_events::ToolResultData {
                message_id: None,
                tool_call_id: ended.tool_call_id.to_string(),
                name: "terminal".to_string(),
                result: ended.output.to_string(),
                success: outcome_status == ToolOutcomeStatus::Succeeded,
                duration_ms: ended.elapsed_secs.saturating_mul(1000),
                error: (outcome_status == ToolOutcomeStatus::FailedPermanent)
                    .then(|| "background process ended without an exit status".to_string()),
                task_id: Some(ended.task_id.to_string()),
                annotations: Vec::new(),
                turn_id: None,
                attachments: Vec::new(),
                receipt: Some(receipt),
            },
        )
        .await
    {
        warn!(
            %error,
            task_id = %ended.task_id,
            tool_call_id = %ended.tool_call_id,
            "Failed to persist terminal background completion receipt"
        );
    }
    result_id
}

pub(super) fn foreground_terminal_metadata(exit_code: Option<i32>) -> ToolCallMetadata {
    let enforcement = confined_process_access_enforcement();
    ToolCallMetadata {
        outcome_status: exit_code.map(ToolOutcomeStatus::from_process_exit_code),
        exit_code,
        invocation_stage: crate::tools::traits::ToolInvocationStage::Dispatched,
        access_enforcement: enforcement,
        timed_out: false,
        background_started: false,
        detached: false,
        completion_notifications_enabled: false,
        transport_error: None,
        http_status: None,
        semantics: ToolCallSemantics::default(),
        read_file: None,
        ..Default::default()
    }
}

pub(crate) fn confined_process_access_enforcement() -> crate::tools::traits::ToolAccessEnforcement {
    #[cfg(target_os = "macos")]
    {
        crate::tools::traits::ToolAccessEnforcement::KernelEnforced
    }
    #[cfg(not(target_os = "macos"))]
    {
        crate::tools::traits::ToolAccessEnforcement::AdapterEnforced
    }
}

/// Receipt for a background process that has FINISHED (collected by
/// `action="check"` or the reaper). This is a terminal completion receipt:
/// `background_started` is reserved for launch receipts, because the loop's
/// detach boundary treats it as "this call moved work to the background" and
/// ends the task with a handoff. A finished process's receipt must instead
/// let its result be delivered and credited like any process completion.
pub(super) fn background_completion_metadata(exit_code: Option<i32>) -> ToolCallMetadata {
    ToolCallMetadata {
        outcome_status: exit_code.map(ToolOutcomeStatus::from_process_exit_code),
        exit_code,
        invocation_stage: crate::tools::traits::ToolInvocationStage::Dispatched,
        access_enforcement: confined_process_access_enforcement(),
        ..Default::default()
    }
}

/// Receipt for observing a STILL-RUNNING background process
/// (`action="check"`) or terminating one (`action="kill"`). Observations and
/// kills never claim `background_started`; only the launch receipt does.
pub(super) fn background_observation_metadata(
    outcome_status: ToolOutcomeStatus,
    detached: bool,
    completion_notifications_enabled: bool,
) -> ToolCallMetadata {
    ToolCallMetadata {
        outcome_status: Some(outcome_status),
        invocation_stage: crate::tools::traits::ToolInvocationStage::Dispatched,
        access_enforcement: confined_process_access_enforcement(),
        detached,
        completion_notifications_enabled,
        ..Default::default()
    }
}

pub(super) fn terminal_receipt_kind(arguments: &str) -> crate::tools::traits::ToolReceiptKind {
    let action = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("action")
                .and_then(Value::as_str)
                .map(TerminalAction::parse)
        })
        .unwrap_or(TerminalAction::Run);
    if action.runs_command() {
        crate::tools::traits::ToolReceiptKind::Process
    } else {
        crate::tools::traits::ToolReceiptKind::Generic
    }
}

/// Absolute-path candidates whose existence transition the run may need to
/// prove: every declared write target plus path operands of structurally
/// parsed deletion invocations (`rm`, `rmdir`, `unlink`, `shred`,
/// `find … -delete`). Command syntax only NOMINATES paths to observe — shell
/// source is program input, not a trusted declaration of effects — so the
/// receipt's DESTRUCTIVE effect is typed from the observed present→absent
/// transition, never from a program name alone.
pub(super) fn deletion_observation_candidates(
    command: &str,
    working_dir: Option<&str>,
    write_paths: &[String],
    write_roots: &[String],
) -> Vec<String> {
    const MAX_CANDIDATES: usize = 16;
    let mut candidates: Vec<String> = Vec::new();
    let mut push = |raw: &str| {
        let trimmed = raw.trim().trim_end_matches('/');
        if trimmed.is_empty() || trimmed.contains(['*', '?', '[']) {
            return;
        }
        let absolute = if aidaemon_tools::BackendPath::is_absolute(trimmed) {
            trimmed.to_string()
        } else if let Some(cwd) =
            working_dir.filter(|cwd| aidaemon_tools::BackendPath::is_absolute(cwd))
        {
            format!("{}/{}", cwd.trim_end_matches('/'), trimmed)
        } else {
            return;
        };
        if !candidates.contains(&absolute) {
            candidates.push(absolute);
        }
    };
    for path in write_paths.iter().chain(write_roots.iter()) {
        push(path);
    }
    for (program, args) in parsed_command_invocations(command) {
        let is_deletion = matches!(program.as_str(), "rm" | "rmdir" | "unlink" | "shred")
            || (program == "find" && args.iter().any(|arg| arg == "-delete"));
        if !is_deletion {
            continue;
        }
        for operand in args.iter().filter(|arg| !arg.starts_with('-')) {
            push(operand);
        }
    }
    candidates.truncate(MAX_CANDIDATES);
    candidates
}

/// Derive lifecycle semantics exclusively from the capability manifest that
/// the process sandbox enforces. Shell source is program input, not a trusted
/// declaration of effects: inspecting words in it cannot prove what an
/// executable, script, trap, alias, or quoted expression will do.
pub(super) fn terminal_run_semantics_from_access(
    confinement_active: bool,
    has_write_targets: bool,
) -> ToolCallSemantics {
    if has_write_targets {
        // A process receipt always observes its own typed outcome (exit status
        // and authoritative stdout/stderr), even when that same process was
        // permitted to mutate a bounded output tree. Mutation effects and
        // process-result evidence are independent facets; collapsing them into
        // one enum made successful build/test scripts impossible to verify.
        return ToolCallSemantics::observation_and_mutation_with(
            ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
        )
        .with_verification_mode(ToolVerificationMode::ResultContent);
    }
    if confinement_active {
        return ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
    }
    ToolCallSemantics::default()
}

/// Authority view of a `run`: typed from the command's structure, not from
/// the declared write targets alone (the ledger view,
/// `terminal_call_semantics`, deliberately stays separate).
pub(super) fn terminal_run_annotations(
    args: &TerminalArgs,
    confinement: aidaemon_tools::TerminalConfinement,
    roots: &crate::tools::traits::TrustRoots<'_>,
) -> aidaemon_core::ToolAnnotations {
    let text = match (args.command.as_deref(), args.script.as_deref()) {
        (Some(command), None) => command,
        (None, Some(script)) => script,
        // Neither or both is an invalid run; never read-only.
        _ => "",
    };
    let effects = aidaemon_tools::command_effects::command_effects(text);
    let has_write_targets = args
        .write_paths
        .iter()
        .chain(args.write_roots.iter())
        .any(|path| !path.trim().is_empty());
    // A read of this computer alone (observations, a local project's own
    // query, a check of a dev site the work set up) is read-only; a run
    // given the network capability never is, since nothing in a sandbox
    // with network keeps it to this computer.
    let read_only = !has_write_targets
        && !args.network
        && (effects.observation_only
            || aidaemon_tools::command_effects::only_reads_this_computer(text));
    // Writing into git configuration changes what later git commands run:
    // a declared target that is git configuration, a declared root holding a
    // repository, or a command word naming git configuration.
    let plants_git_config = args
        .write_paths
        .iter()
        .chain(args.write_roots.iter())
        .any(|path| aidaemon_tools::command_effects::is_git_config_path(path))
        || args.write_roots.iter().any(|root| {
            aidaemon_tools::command_effects::root_holds_repository(root, args.working_dir.as_deref())
        })
        // An observation-only command cannot write by construction.
        || (!effects.observation_only
            && aidaemon_tools::command_effects::command_names_git_config_path(text));
    // A delete is egress unless every target is inside the trusted roots or
    // a folder the request's work made for itself (everything there is the
    // work's own: live 2026-11-13, removing a reset script the Drupal
    // install had just written in its own new folder asked the owner).
    // Relative targets resolve against the call's own absolute working
    // directory; without one (the dispatcher may still pick it) they are
    // unknown.
    let base = args
        .working_dir
        .as_deref()
        .map(str::trim)
        .filter(|dir| std::path::Path::new(dir).is_absolute());
    let deletes_outside_roots = effects.deletes
        && effects.delete_targets.as_ref().is_none_or(|targets| {
            targets.is_empty()
                || targets.iter().any(|target| {
                    let resolved =
                        if std::path::Path::new(target).is_absolute() || target.starts_with('~') {
                            Some(target.clone())
                        } else {
                            base.map(|base| format!("{}/{target}", base.trim_end_matches('/')))
                        };
                    resolved.is_none_or(|path| !roots.holds_effect(&path))
                })
        });
    // On the host every program can reach the network; only observations
    // and ordinary local work (builds, tests, local git) are not egress.
    let host_beyond_local_work =
        !run_is_confined(confinement) && !effects.observation_only && !effects.local_work_only;
    let reaches_network = args.network
        || aidaemon_tools::command_effects::fetches_network(text)
        || host_beyond_local_work;
    // A request to a site on this computer (a dev site the work set up) or
    // a local project's own query reaches nothing outside the machine, even
    // when the sandbox needs the network capability to reach a local port
    // or the container engine. Judged only when the call would otherwise be
    // network (it resolves names).
    //
    // A project's own Drush or WP-CLI command run in a folder the request's
    // work created acts on that work's own site in its own containers
    // (live 2026-11-13: `ddev drush watchdog:show` in the Drupal site a
    // helper had just installed asked the owner after the run read a page).
    let runs_in_owned_folder = args
        .working_dir
        .as_deref()
        .map(str::trim)
        .filter(|dir| std::path::Path::new(dir).is_absolute())
        .or(roots.cwd())
        .is_some_and(|dir| roots.owns(dir));
    let stays_on_this_computer = reaches_network
        && (aidaemon_tools::command_effects::reaches_only_this_computer(text)
            || (runs_in_owned_folder
                && aidaemon_tools::command_effects::acts_on_its_ddev_project(text)));
    aidaemon_core::ToolAnnotations {
        read_only,
        destructive: effects.deletes,
        idempotent: read_only,
        egress: plants_git_config
            || deletes_outside_roots
            || (reaches_network && !stays_on_this_computer),
    }
}

pub(super) fn terminal_call_semantics(arguments: &str) -> ToolCallSemantics {
    let args = serde_json::from_str::<Value>(arguments).ok();
    let action = args
        .as_ref()
        .and_then(|value| value.get("action"))
        .and_then(Value::as_str)
        .map_or(TerminalAction::Run, TerminalAction::parse);
    match action {
        TerminalAction::Check => ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent),
        TerminalAction::Kill => ToolCallSemantics::mutation(),
        TerminalAction::TrustAll => ToolCallSemantics::observation(),
        TerminalAction::Run | TerminalAction::Unrecognized => {
            let has_write_targets = args.as_ref().is_some_and(|value| {
                ["write_paths", "write_roots"].into_iter().any(|field| {
                    value
                        .get(field)
                        .and_then(Value::as_array)
                        .is_some_and(|paths| {
                            paths.iter().any(|path| {
                                path.as_str().is_some_and(|path| !path.trim().is_empty())
                            })
                        })
                })
            });
            let command = args.as_ref().and_then(|value| {
                value
                    .get("command")
                    .or_else(|| value.get("script"))
                    .and_then(Value::as_str)
            });
            // A command that asks a remote service to change something is a
            // remote mutation, however it reaches the service.
            if command.is_some_and(aidaemon_tools::command_effects::sends_remote_write) {
                let effects = if has_write_targets {
                    ToolMutationEffects::REMOTE_MUTATION
                        .union(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
                } else {
                    ToolMutationEffects::REMOTE_MUTATION
                };
                return ToolCallSemantics::observation_and_mutation_with(effects)
                    .with_verification_mode(ToolVerificationMode::ResultContent);
            }
            // What a run does comes from what it is given and what its
            // command is, never from the paths the model declared it reads.
            // Without write targets or the network, the sandbox keeps it from
            // changing anything. With the network it reaches what the sandbox
            // does not contain (the container engine, services): unless its
            // command structurally only reads, it changes something there
            // (`ddev exec drush site:install` installs a site, whatever its
            // `read_paths`).
            let has_network = args
                .as_ref()
                .and_then(|value| value.get("network"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !has_write_targets
                && has_network
                && command.is_some_and(|command| {
                    !aidaemon_tools::command_effects::command_effects(command).observation_only
                        && !aidaemon_tools::command_effects::only_reads(command)
                })
            {
                return ToolCallSemantics::observation_and_mutation_with(
                    ToolMutationEffects::UNSPECIFIED,
                )
                .with_verification_mode(ToolVerificationMode::ResultContent);
            }
            // Every process call observes its own typed receipt, and every run
            // crosses the native sandbox even when its task-data manifest is
            // empty. An empty manifest means no task-data authority; it never
            // means ambient host access.
            terminal_run_semantics_from_access(true, has_write_targets)
        }
    }
}

/// The literal paths a `run`'s command names, read structurally
/// (`command_effects::literal_path_operands`), each resolved against the
/// run's absolute `working_dir` when relative. Empty for a call that is not
/// a run.
pub(super) fn terminal_command_operands(arguments: &str) -> Vec<String> {
    let Ok(args) = serde_json::from_str::<TerminalArgs>(arguments) else {
        return Vec::new();
    };
    if !TerminalAction::parse(&args.action).runs_command() {
        return Vec::new();
    }
    let Some(text) = args.command.as_deref().or(args.script.as_deref()) else {
        return Vec::new();
    };
    let cwd = args
        .working_dir
        .as_deref()
        .map(str::trim)
        .filter(|dir| aidaemon_tools::BackendPath::is_absolute(dir));
    aidaemon_tools::command_effects::literal_path_operands(text)
        .into_iter()
        .map(|operand| {
            if aidaemon_tools::BackendPath::is_absolute(&operand)
                || operand == "~"
                || operand.starts_with("~/")
            {
                operand
            } else if let Some(cwd) = cwd {
                aidaemon_tools::BackendPath::new(cwd)
                    .join(&operand)
                    .to_string()
            } else {
                operand
            }
        })
        .collect()
}

pub(super) fn terminal_access_manifest(
    arguments: &str,
) -> crate::tools::traits::ToolCallAccessManifest {
    let parsed = serde_json::from_str::<Value>(arguments).ok();
    let execution_cwd = parsed
        .as_ref()
        .and_then(|value| value.get("working_dir").or_else(|| value.get("cwd")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let resolve = |path: &str| {
        execution_cwd
            .as_deref()
            .filter(|_| {
                !aidaemon_tools::BackendPath::is_absolute(path)
                    && path != "~"
                    && !path.starts_with("~/")
            })
            .map(|cwd| aidaemon_tools::BackendPath::new(cwd).join(path).to_string())
            .unwrap_or_else(|| path.to_string())
    };
    // Execution location and data authority are independent. Merely selecting
    // a cwd must not expose every file below it; callers declare readable data
    // explicitly through `read_paths`.
    let mut read_targets = Vec::new();
    for path in parsed
        .as_ref()
        .and_then(|value| value.get("read_paths"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(resolve)
        .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
    {
        if !read_targets.contains(&path) {
            read_targets.push(path);
        }
    }
    for path in parsed
        .as_ref()
        .and_then(|value| value.get("read_roots"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(resolve)
        .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path))
    {
        if !read_targets.contains(&path) {
            read_targets.push(path);
        }
    }
    let mut write_targets: Vec<ToolTargetHint> = parsed
        .as_ref()
        .and_then(|value| value.get("write_paths"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(resolve)
        .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
        .collect();
    write_targets.extend(
        parsed
            .as_ref()
            .and_then(|value| value.get("write_roots"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(resolve)
            .filter_map(|path| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, path)),
    );
    crate::tools::traits::ToolCallAccessManifest {
        execution_cwd,
        read_targets,
        write_targets,
        adapter_read_targets: Vec::new(),
    }
}

/// Shell here-documents are parsed by the shell before the command body runs.
/// Routing command-form here-documents through the same stdin boundary keeps
/// the sandbox transient parser allowance and scratch preparation consistent.
/// The marker is shell syntax, not natural-language classification; a false
/// positive only changes the equivalent transport form.
pub(super) fn command_requires_stdin_shell(command: &str) -> bool {
    command.contains("<<")
}

pub(super) fn validate_terminal_argument_contract(
    arguments: &str,
) -> Result<(), ToolArgumentContractViolation> {
    let Ok(parsed) = serde_json::from_str::<Value>(arguments) else {
        return Ok(());
    };
    let action = parsed
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("run");
    if action != "run" {
        return Ok(());
    }
    let command = parsed
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let script = parsed
        .get("script")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (command, script) {
        (Some(_), Some(_)) => {
            return Err(ToolArgumentContractViolation::new(
                "terminal action=run accepts exactly one of command or script",
            ));
        }
        (Some(_), None) | (None, Some(_)) => {}
        (None, None) => {
            return Err(ToolArgumentContractViolation::new(
                "terminal action=run requires command or script",
            ));
        }
    }
    Ok(())
}

/// A terminal call's action, read once for execution, effects, semantics
/// and annotations so they agree on every input. Only the exact schema
/// spellings are recognized; execution runs anything else as `run` (its
/// historical default), and effects declare such a call as unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalAction {
    Run,
    Check,
    Kill,
    TrustAll,
    /// Not an exact schema spelling (`"Check"`, `" run "`).
    Unrecognized,
}

impl TerminalAction {
    pub(crate) fn parse(action: &str) -> Self {
        match action {
            "run" => Self::Run,
            "check" => Self::Check,
            "kill" => Self::Kill,
            "trust_all" => Self::TrustAll,
            _ => Self::Unrecognized,
        }
    }

    /// Execution runs the command or script.
    pub(crate) fn runs_command(self) -> bool {
        matches!(self, Self::Run | Self::Unrecognized)
    }
}

/// The typed effects of a terminal call: `check` reads a tracked process,
/// `run` declares what its command does (see
/// `command_effects::shell_call_effects`), anything else is undeclared.
pub(super) fn terminal_call_effects(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    arguments: &str,
) -> aidaemon_core::CallEffects {
    terminal_call_effects_on(
        arguments,
        aidaemon_tools::command_effects::execution_backend_is_local(backend),
    )
}

/// `terminal_call_effects` for a backend that is (`local`) or is not the
/// daemon's own machine. On a remote (SSH, Docker) workspace the paths a
/// command touches live there, so they are not resolved on the local disk:
/// path effects are left undeclared and the call is incomplete.
pub(super) fn terminal_call_effects_on(arguments: &str, local: bool) -> aidaemon_core::CallEffects {
    use aidaemon_tools::command_effects::{path_effect, shell_call_effects, without_local_paths};
    let Ok(args) = serde_json::from_str::<TerminalArgs>(arguments) else {
        return aidaemon_core::CallEffects::unknown();
    };
    match TerminalAction::parse(&args.action) {
        TerminalAction::Check => aidaemon_core::CallEffects::none(),
        TerminalAction::Run => {
            let Some(text) = args
                .command
                .as_deref()
                .or(args.script.as_deref())
                .filter(|text| !text.trim().is_empty())
            else {
                return aidaemon_core::CallEffects::unknown();
            };
            // Only an explicit absolute directory is known here: a run
            // without one executes wherever the dispatcher anchors it.
            let cwd = args
                .working_dir
                .as_deref()
                .map(str::trim)
                .filter(|dir| aidaemon_tools::BackendPath::is_absolute(dir));
            // Never complete (see `shell_call_effects`): the keys are best
            // effort and bind the owner's declines.
            let mut effects = shell_call_effects(text, cwd);
            // Declared write targets are what a confined run may change.
            for target in args.write_paths.iter().chain(args.write_roots.iter()) {
                match path_effect(aidaemon_core::EffectClass::Write, target, cwd) {
                    Some(key) => {
                        if !effects.keys.contains(&key) {
                            effects.keys.push(key);
                        }
                    }
                    None => effects.understood = false,
                }
            }
            effects.complete = false;
            if local {
                // A site on this computer is not outbound: the owner's
                // declines and re-asks are about what leaves the machine.
                effects.keys.retain(|key| {
                    key.class != aidaemon_core::EffectClass::Egress
                        || aidaemon_tools::web::local_site_key(&format!("https://{}/", key.target))
                            .is_none()
                });
                effects
            } else {
                without_local_paths(effects)
            }
        }
        // The retired `trust_all` changes nothing.
        TerminalAction::TrustAll => aidaemon_core::CallEffects::none(),
        // `kill` signals a process; a non-canonical spelling is executed as
        // a run nobody declared.
        TerminalAction::Kill | TerminalAction::Unrecognized => {
            aidaemon_core::CallEffects::unknown()
        }
    }
}

pub(super) fn canonicalize_terminal_arguments(
    arguments: &str,
) -> Result<String, ToolArgumentContractViolation> {
    let mut parsed = serde_json::from_str::<Value>(arguments).map_err(|error| {
        ToolArgumentContractViolation::new(format!("terminal arguments must be JSON: {error}"))
    })?;
    if let Some(object) = parsed.as_object_mut() {
        // Provider schema projections may materialize an inactive optional
        // string as "". Normalize the tagged union before any policy or
        // semantic derivation; empty input is absence, not a second mode.
        for field in ["command", "script"] {
            let remove = object
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|value| value.trim().is_empty());
            if remove {
                object.remove(field);
            }
        }
    }
    serde_json::to_string(&parsed).map_err(|error| {
        ToolArgumentContractViolation::new(format!(
            "terminal arguments could not be canonicalized: {error}"
        ))
    })
}
