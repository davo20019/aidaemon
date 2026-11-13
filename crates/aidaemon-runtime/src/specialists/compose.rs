//! How a child's prompt and tools follow from its specialist: the kind a
//! spawn resolves to, the registry-rendered task-lead and executor prompts,
//! and the executor's handoff contract.

use std::sync::Arc;

use aidaemon_agents::specialist::SpecialistKind;
use aidaemon_events::traits::AgentRole;
use aidaemon_tools::OutcomeTool;
use serde_json::{json, Value};
use tracing::warn;
use uuid::Uuid;

use super::{SpecialistRegistry, SpecialistRenderContext};
use crate::prompt::rules::format_goal_context;
use crate::spawn::task_lead_execution_mode;
use crate::turn::validation_state::ExecutorHandoff;

/// Every helper's contract: its report is about the work, and the daemon
/// adds everything about time (and what reached the owner) from its own
/// record. Put as the principle it is, so a helper neither states a time
/// nor comments on one, whatever its task asks (live 2026-11-13: told not to
/// state its run's time, a helper wrote "I don't have a measured elapsed
/// time" right above the measured "Took 3m 16s", and wrote "I sent a
/// screenshot" for one that never arrived).
pub const OWN_DURATION_RULE: &str = "Your report is about the work: what you did, what you \
     found, and what resulted. The daemon adds everything about time under your report, from \
     its own measurement, so time is not yours to state or comment on in any form, not how long \
     anything took, not when it happened, and not whether a time was or could be measured, even \
     when your task asks for timings: the measured times your task wants appear beside your \
     report. Likewise, what reached the owner is listed under your report from the daemon's \
     record of its deliveries: say what you captured or produced, not whether it was sent.";

pub fn bounded_completed_task_results_context(
    ctx_json: &str,
    max_results: usize,
) -> Option<String> {
    let value: Value = serde_json::from_str(ctx_json).ok()?;
    let results = value.get("task_results")?.as_array()?;
    if results.is_empty() {
        return None;
    }
    let start = results.len().saturating_sub(max_results.max(1));
    let bounded = json!({ "task_results": results[start..].to_vec() });
    let formatted = format_goal_context(&bounded.to_string());
    (!formatted.trim().is_empty()).then_some(formatted)
}

pub fn select_specialist_kind(role: AgentRole, _mission: &str, _task: &str) -> SpecialistKind {
    match role {
        AgentRole::TaskLead => SpecialistKind::TaskLead,
        AgentRole::Orchestrator => SpecialistKind::Generic,
        // An omitted specialist is an explicit choice of the broad executor
        // profile. Mission prose is not a stable authority or capability
        // signal and must not silently narrow the child's tools or budget.
        AgentRole::Executor => SpecialistKind::Executor,
    }
}

pub fn build_specialist_session_id(kind: SpecialistKind, id: Uuid) -> String {
    format!("specialist:{}:{}", kind.as_str(), id)
}

/// Resolve the effective `SpecialistKind` for a spawn given:
/// - explicit role (role-typed spawns normally ignore the arg),
/// - optional caller-supplied `arg_specialist` (from `spawn_agent`'s schema),
/// - broad executor fallback when no typed specialist is supplied.
///
/// "task_lead" is role-typed and rejected when passed as an arg (the
/// `AgentRole::TaskLead` spawn path produces it). Invalid arg values fall
/// through to the broad executor and produce a warn log.
pub fn resolve_specialist_kind(
    role: Option<AgentRole>,
    arg_specialist: Option<&str>,
    _mission: &str,
    _task: &str,
) -> SpecialistKind {
    if matches!(role, Some(AgentRole::TaskLead)) {
        return SpecialistKind::TaskLead;
    }
    if matches!(role, Some(AgentRole::Orchestrator)) {
        return SpecialistKind::Generic;
    }
    if let Some(s) = arg_specialist {
        match SpecialistKind::from_str(s) {
            Some(kind) if kind != SpecialistKind::TaskLead => return kind,
            Some(_) => {
                // Caller passed "task_lead" but that's role-typed only.
                warn!(arg = %s, "ignoring 'task_lead' specialist arg; role-typed only");
            }
            None => {
                warn!(
                    arg = %s,
                    "ignoring invalid specialist arg; using broad executor"
                );
            }
        }
    }
    SpecialistKind::Executor
}

/// Apply a specialist's declared tool allowlist to the in-flight tool set.
///
/// This function is intentionally a simple `declared ∩ available` intersection
/// (with the existing `intersect_tools` unknown-tool warning). The role
/// boundary (Executor vs. TaskLead vs. Orchestrator) is enforced upstream
/// by the caller's pre-filtering of `tools` — by the time we get here,
/// `tools` already reflects the role scope, so feeding `tools` as both the
/// working set and the `role_scope` parameter to `intersect_tools` is the
/// honest expression of that contract (no tautological double-check).
///
/// Tools declared by the specialist but not present in `tools` are dropped
/// with a warn. An empty `declared` allowlist also produces a warn so
/// operators notice that the specialist will have no tools.
pub fn apply_specialist_tool_allowlist(
    kind: SpecialistKind,
    declared: &[String],
    tools: &mut Vec<Arc<dyn OutcomeTool>>,
) {
    if declared.is_empty() {
        warn!(
            kind = %kind.as_str(),
            "specialist declared an empty tool allowlist — child will have no tools"
        );
    }
    let known_owned: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
    let known: Vec<&str> = known_owned.iter().map(|s| s.as_str()).collect();
    // Role boundary is enforced upstream by the caller's pre-filtering of
    // `tools`. Here `role_scope == known`: real enforcement is upstream;
    // this call only retains declared tools that are still in-flight.
    let permitted = super::validation::intersect_tools(kind, declared, &known, &known);
    tools.retain(|t| permitted.iter().any(|p| p == t.name()));
}

/// Render the task-lead system prompt using the `SpecialistRegistry` as the
/// source of truth for the base template, then append the same dynamic
/// sections (Prior Knowledge, CLI Agent Delegation) the legacy builder
/// produced. This keeps byte-equivalence with `build_task_lead_prompt`
/// when `goal_context=None` and `has_cli_agent=false`.
#[allow(clippy::too_many_arguments)]
pub fn compose_task_lead_prompt_from_registry(
    registry: &SpecialistRegistry,
    goal_id: &str,
    goal_description: &str,
    goal_context: Option<&str>,
    depth: usize,
    max_depth: usize,
    has_cli_agent: bool,
    is_scheduled: bool,
) -> String {
    let execution_mode = task_lead_execution_mode(is_scheduled).to_string();
    let ctx = SpecialistRenderContext {
        mission: goal_description.to_string(),
        task: String::new(),
        depth,
        max_depth,
        max_iterations: 0,
        goal_id: goal_id.to_string(),
        working_dir: String::new(),
        is_scheduled,
        parent_session_id: String::new(),
        execution_mode,
        can_spawn: depth < max_depth,
    };
    let mut prompt = registry.render(SpecialistKind::TaskLead, &ctx);
    // Markdown templates conventionally end in a newline while the legacy
    // builder's base string does not. Normalize before appending dynamic
    // sections so `\n\n## ...` produces exactly one blank line in either
    // path; a single final newline is restored below.
    while prompt.ends_with('\n') {
        prompt.pop();
    }

    if let Some(ctx_text) = goal_context {
        prompt.push_str(&format!(
            "\n\n## Prior Knowledge\n\
             The following knowledge was gathered from previous tasks and may be relevant:\n{}",
            format_goal_context(ctx_text)
        ));
    }

    if has_cli_agent {
        prompt.push_str(&format!(
            "{}\n\
                 If the work should stay tied to a claimed task with executor results or \
                 blocker handling, claim the task and use `spawn_agent`.\n\
                 When calling `cli_agent`, use `action=\"run\"` with a non-empty `prompt` \
                 describing the work.",
            crate::helper_run::DELEGATED_CLI_SANDBOX.delegation_section()
        ));
    }

    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    prompt
}

/// Render the executor system prompt using the `SpecialistRegistry` as the
/// source of truth for the base template, then splice in the same dynamic
/// sections (working directory, task contract, cli-agent suffix) the
/// legacy builder produced. Keeps byte-equivalence with
/// `build_executor_prompt` when `specialist_kind == Executor` and all
/// dynamic inputs are empty. Other kinds render their own .md (which
/// includes the shared `{{executor_base}}` partial plus a kind-specific
/// tagline), so the child sees a role-appropriate prompt.
#[allow(clippy::too_many_arguments)]
pub fn compose_executor_prompt_from_registry(
    registry: &SpecialistRegistry,
    specialist_kind: SpecialistKind,
    task_description: &str,
    parent_mission: &str,
    depth: usize,
    max_depth: usize,
    has_cli_agent: bool,
    task_id: Option<&str>,
    project_scope: Option<&str>,
    can_spawn: bool,
) -> String {
    let mut all_dirs = extract_directory_paths(parent_mission);
    for dir in extract_directory_paths(task_description) {
        if !all_dirs.contains(&dir) {
            all_dirs.push(dir);
        }
    }

    // Render the base template (header + body) verbatim from the registry.
    let ctx = SpecialistRenderContext {
        mission: parent_mission.to_string(),
        task: task_description.to_string(),
        depth,
        max_depth,
        max_iterations: 0,
        goal_id: String::new(),
        working_dir: String::new(),
        is_scheduled: false,
        parent_session_id: String::new(),
        execution_mode: String::new(),
        can_spawn,
    };
    // Markdown source files conventionally end with a newline, while the
    // dynamic suffix below owns its leading separator. Normalize that
    // boundary so prompt output does not depend on editor EOF settings.
    let base = registry
        .render(specialist_kind, &ctx)
        .trim_end_matches('\n')
        .to_string();

    // Build the dynamic mid-section (working directory + task contract)
    // that the legacy builder inserts between the sub-agent header and
    // "## Original User Request".
    let mut middle = String::new();
    if !all_dirs.is_empty() {
        middle.push_str("## WORKING DIRECTORY (CRITICAL)\n");
        middle.push_str("All files for this task are in: ");
        middle.push_str(&all_dirs.join(", "));
        middle.push_str("\n\nYou MUST use absolute paths when calling read_file, edit_file, write_file, search_files.\n");
        middle.push_str("Examples:\n");
        for dir in &all_dirs {
            middle.push_str(&format!(
                "- read_file: path=\"{dir}/filename.py\"\n\
                 - edit_file: path=\"{dir}/filename.py\"\n\
                 - search_files: path=\"{dir}\"\n"
            ));
        }
        middle.push_str(
            "Do NOT use relative paths. Do NOT search in the default project directory.\n\n",
        );
    }

    if let Some(task_id) = task_id {
        let handoff = build_executor_handoff(
            task_id,
            parent_mission,
            task_description,
            &[],
            project_scope,
        );
        middle.push_str(&handoff.render_prompt_section());
        middle.push_str("\n\n");
    }

    // Splice `middle` immediately before the "## Original User Request"
    // section so the result matches the legacy builder layout exactly.
    let marker = "## Original User Request";
    let mut prompt = if middle.is_empty() {
        base.clone()
    } else if let Some(idx) = base.find(marker) {
        let (head, tail) = base.split_at(idx);
        let mut out = String::with_capacity(base.len() + middle.len());
        out.push_str(head);
        out.push_str(&middle);
        out.push_str(tail);
        out
    } else {
        // Defensive: marker should always be present in the bundled
        // executor template; if it isn't, fall back to base + middle.
        warn!(
            "executor template missing '## Original User Request' marker; appending dynamic content"
        );
        let mut out = base.clone();
        out.push_str(&middle);
        out
    };

    if has_cli_agent {
        prompt.push_str(
            "\n- Delegation mode is active: `terminal`, `browser`, and `run_command` are not available here.\n\
             Use direct file tools (`read_file`, `edit_file`, `write_file`, `search_files`) for narrow file work.\n\
             Use `cli_agent` for shell/test flows or multi-step coding and research work.\n\
             For public URL reachability or returned text, use an available HTTP read tool or ask `cli_agent` to run curl; do not require browser access unless the task is visual or interactive.\n\
             When you use `cli_agent`, always provide `action=\"run\"`, a concrete `prompt`, and `working_dir` when you know the repo path.",
        );
    }

    prompt
}

pub fn collect_executor_expected_targets(
    mission: &str,
    task_description: &str,
    project_scope: Option<&str>,
) -> Vec<aidaemon_tools::semantics::ToolTargetHint> {
    let mut targets = Vec::new();

    if let Some(scope) = project_scope
        && let Some(target) = aidaemon_tools::semantics::ToolTargetHint::new(
            aidaemon_tools::semantics::ToolTargetHintKind::ProjectScope,
            scope,
        ) {
            targets.push(target);
        }

    let mut add_dir = |dir: String| {
        if let Some(target) = aidaemon_tools::semantics::ToolTargetHint::new(
            aidaemon_tools::semantics::ToolTargetHintKind::Path,
            dir,
        )
            && !targets.iter().any(|existing| existing == &target) {
                targets.push(target);
            }
    };

    for dir in extract_directory_paths(mission) {
        add_dir(dir);
    }
    for dir in extract_directory_paths(task_description) {
        add_dir(dir);
    }

    targets
}

pub fn build_executor_handoff(
    task_id: &str,
    mission: &str,
    task_description: &str,
    tools: &[Arc<dyn OutcomeTool>],
    project_scope: Option<&str>,
) -> ExecutorHandoff {
    let expected_targets =
        collect_executor_expected_targets(mission, task_description, project_scope);
    let allowed_targets = if let Some(scope) = project_scope {
        aidaemon_tools::semantics::ToolTargetHint::new(
            aidaemon_tools::semantics::ToolTargetHintKind::ProjectScope,
            scope,
        )
        .into_iter()
        .collect()
    } else {
        expected_targets.clone()
    };

    ExecutorHandoff {
        task_id: task_id.to_string(),
        mission: mission.to_string(),
        task_description: task_description.to_string(),
        target_scope: crate::turn::execution_state::TargetScope {
            allowed_targets,
            hard_fail_outside_scope: project_scope.is_some(),
        },
        expected_targets,
        allowed_tools: Some(
            tools
                .iter()
                .map(|tool| tool.name().to_string())
                .collect::<Vec<_>>(),
        ),
    }
}

/// Extract absolute directory paths from text (e.g. /tmp/debugme3/, /home/user/project/).
/// Returns deduplicated list of directory paths found.
pub fn extract_directory_paths(text: &str) -> Vec<String> {
    let mut dirs = Vec::new();
    // Match absolute paths: /word/word... optionally ending with /
    for word in text.split_whitespace() {
        // Strip trailing punctuation
        let clean = word.trim_end_matches(|c: char| {
            c == '.' || c == ',' || c == ':' || c == ';' || c == ')' || c == '\''
        });
        if clean.starts_with('/')
            && clean.len() > 2
            && !clean.starts_with("//")
            // Must have at least 2 path components
            && clean.matches('/').count() >= 2
            // Skip common non-directory paths
            && !clean.ends_with(".rs")
            && !clean.ends_with(".toml")
        {
            // Normalize to directory (remove trailing filename if it has an extension)
            let path = std::path::Path::new(clean);
            let dir = if path.extension().is_some() {
                // Looks like a file path — take parent directory
                path.parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| clean.to_string())
            } else {
                clean.trim_end_matches('/').to_string()
            };
            if !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    dirs
}

/// The legacy prompt builders, kept as the oracle the registry-rendered
/// prompts are checked against (`equivalence_tests`).
#[cfg(test)]
pub mod legacy {
    use super::*;

    /// Build system prompt for a Task Lead agent.
    ///
    /// Retained as the oracle for `specialists::equivalence_tests` and as the
    /// reference legacy implementation. Production callers now go through
    /// `compose_task_lead_prompt_from_registry`, which renders the same base
    /// text from `task_lead.md` and appends the same dynamic sections.
    pub fn build_task_lead_prompt(
        goal_id: &str,
        goal_description: &str,
        goal_context: Option<&str>,
        depth: usize,
        max_depth: usize,
        has_cli_agent: bool,
        is_scheduled: bool,
    ) -> String {
        let execution_mode = task_lead_execution_mode(is_scheduled);

        let mut prompt = format!(
            "You are a Task Lead managing goal: {goal_id}\n\
             Goal: {goal_description}\n\n\
             You are a sub-agent (depth {depth}/{max_depth}).\n\
             {execution_mode}\n\n\
             ## Workflow\n\
             1. Analyze the goal and break it into concrete tasks using manage_goal_tasks(create_task)\n\
                - Start with 1-5 tasks for the NEXT PHASE (not the entire project)\n\
                - Keep one cohesive target in one task even when it has sequential build, deploy, and verification stages; split only independent workstreams or ownership boundaries\n\
                - After those tasks complete, reassess and create more tasks if the goal isn't done\n\
                - Set `depends_on` (array of task IDs) for tasks that require prior tasks to complete\n\
                - Set `parallel_group` for tasks that belong to the same logical phase\n\
                - Set `idempotent: true` for tasks safe to retry on failure\n\
                - Set `task_order` for display ordering\n\
                - Set `worker_profile` to the best named profile: profile-code, profile-research, profile-review, profile-browser-verifier, profile-artifact-writer, profile-comms-draft, or profile-executor\n\
                - Use `workspace_policy: isolated` for a new project, `worktree` for parallel or collision-prone edits in an existing Git project, and `shared` only for one explicit existing project\n\
                - If the objective concerns an existing repository or site that is NOT present in the attempt workspace (empty directory, no Git repository, no build/deploy workflow), do not block: locate the project under the configured project roots with `project_inspect` (a site hostname usually matches a sibling directory name), then bind it durably with `scheduled_goal_runs(action=\"bind_workspace\", goal_id=..., workspace_path=...)` and create the work tasks against that directory. Report a blocker only when no such directory exists anywhere.\n\
                - For a scheduled objective whose target is a website or repository, do this FIRST, before creating any task: call `scheduled_goal_runs(action=\"overview\")` and check `workspace_binding`; when it is null, resolve and bind the project directory as above. Executors inherit the bound directory; tasks created before binding run in the wrong workspace and their reads are rejected by the scope lock.\n\
             2. Before spawning an executor, claim the task: manage_goal_tasks(claim_task, task_id=...)\n\
                - This verifies dependencies are met and atomically reserves the task\n\
                - If claiming fails due to unmet dependencies, work on other available tasks first\n\
             3. Spawn an executor: spawn_agent(mission=..., task=..., task_id=<the task ID>)\n\
                - Always pass the task_id so executor activity is tracked\n\
                - Your tools are for planning and tracking; executors carry the working tools you do not have. A tool missing or refused in your own session never shows a task cannot be done: spawn its executor, and fail or block a task only on what an executor reports\n\
             4. After each executor returns, update: manage_goal_tasks(update_task, task_id, status, result)\n\
             5. If a task fails and is idempotent: manage_goal_tasks(retry_task, task_id) then re-spawn\n\
                - If not idempotent or max retries exceeded: create an alternative task or fail the goal\n\
                - If an alternative task successfully replaces failed work, update the original task to \
                  status `superseded`; its result MUST name the replacement task ID and explain why the \
                  replacement satisfies the original requirement\n\
                - Never leave a replaced failure in `failed`: that incorrectly poisons the run result\n\
             6. When every required task is completed/skipped and every obsolete task is explicitly \
                superseded: manage_goal_tasks(complete_goal, summary)\n\n\
             ## Rules\n\
             - Keep each planning step small: 1-5 tasks at a time, then iterate\n\
             - Execute sequentially unless independent tasks share an explicit `parallel_group`; bounded parallel groups may run up to four executors\n\
             - Each executor gets a single, focused task\n\
             - Executors do not automatically see this Task Lead's prompt. If a task depends on \
               Prior Knowledge, Completed Task Results, or another context section, copy the \
               necessary evidence into the task text; never tell an executor to inspect context \
               it was not given\n\
             - Always check list_tasks before spawning the next executor\n\
             - If an executor reports a blocker, inspect the recorded task status/result and resolve it or adjust the plan\n\
             - Do not escalate a local file merely because it existed before the run. When build, test, validation, or deployment evidence identifies a file inside the authorized project as a concrete blocker, treat the smallest reversible mechanical repair as a causal dependency of the task. Preserve its substantive content; only escalate broader behavior/content changes, destructive actions, genuinely unrelated files, or missing external authority\n\
             - Executors persist a structured handoff/result contract onto the claimed task record; do not treat vague prose alone as proof of completion\n\
             - When finishing the goal, your final reply MUST include concrete executor results (outputs, paths, data), not just \"goal completed\"\n\
             - Make the final reply easy to scan on a phone: lead with the outcome, use short paragraphs or bullets, and label important links, paths, verification status, and IDs. Do not repeat the goal instructions.\n\n\
             ## Pre-flight and Verification\n\
             - Keep readiness checks, the mutation, and immediate verification in the same task when \
             they concern one target and one worker can perform them safely. Put the concrete checks in \
             that task's acceptance criteria and structured handoff\n\
             - Create a separate prerequisite or verification task only for a real ownership boundary, \
             an independent parallel review, an external wait/monitoring period, or a prerequisite that \
             must be handed to another worker\n\
             - For public endpoint reachability and rendered text, prefer an HTTP read first. Require a \
             browser only for visual layout or interactive behavior. If one verification surface is \
             unavailable, use another surface for every claim it can prove instead of asking the user \
             to repair the tool session\n\
             - Never mark the goal as complete until you have a completion signal — but the completion \
             signal is the mutating call's OWN success response (e.g. HTTP 2xx with a created/updated \
             resource ID), not necessarily a separate read-back\n\
             - A failed verification task means \"I could not confirm,\" not \"the change didn't happen\" \
             — before creating a remediation task, check whether the original mutating executor already \
             reported a success response (2xx, created ID, etc.). If it did, do NOT remediate by repeating \
             the mutating action (re-posting, re-sending, re-publishing): that risks duplicate real-world \
             side effects (duplicate posts, duplicate sends, duplicate charges). Instead, mark the task \
             complete, note the verification limitation in the result, and stop\n\
             - Only create a remediation task that repeats the mutating action when the ORIGINAL mutating \
             call itself failed or errored — never solely because a downstream verification/read step \
             failed or is unavailable (e.g. read-restricted API tier, eventual consistency delay, transient \
             tool failure)"
        );

        if let Some(ctx) = goal_context {
            prompt.push_str(&format!(
                "\n\n## Prior Knowledge\n\
                 The following knowledge was gathered from previous tasks and may be relevant:\n{}",
                format_goal_context(ctx)
            ));
        }

        if has_cli_agent {
            prompt.push_str(&format!(
                "{}\n\
                 If the work should stay tied to a claimed task with executor results or \
                 blocker handling, claim the task and use `spawn_agent`.\n\
                 When calling `cli_agent`, use `action=\"run\"` with a non-empty `prompt` \
                 describing the work.",
                crate::helper_run::DELEGATED_CLI_SANDBOX.delegation_section()
            ));
        }

        if !prompt.ends_with('\n') {
            prompt.push('\n');
        }
        prompt
    }

    /// Build system prompt for an Executor agent.
    ///
    /// Retained as the oracle for `specialists::equivalence_tests` and as the
    /// reference legacy implementation. Production callers now go through
    /// `compose_executor_prompt_from_registry`, which renders the same base
    /// text from `executor.md` and splices in the same dynamic sections.
    pub fn build_executor_prompt(
        task_description: &str,
        parent_mission: &str,
        depth: usize,
        max_depth: usize,
        has_cli_agent: bool,
        task_id: Option<&str>,
        project_scope: Option<&str>,
    ) -> String {
        // Extract directory paths from both parent mission and task description
        let mut all_dirs = extract_directory_paths(parent_mission);
        for dir in extract_directory_paths(task_description) {
            if !all_dirs.contains(&dir) {
                all_dirs.push(dir);
            }
        }

        let mut prompt = format!(
            "You are an Executor. Complete this single task and return your results.\n\n\
             You are a sub-agent (depth {depth}/{max_depth}).\n\n"
        );

        // Inject extracted directory paths at the very top — before anything else
        if !all_dirs.is_empty() {
            prompt.push_str("## WORKING DIRECTORY (CRITICAL)\n");
            prompt.push_str("All files for this task are in: ");
            prompt.push_str(&all_dirs.join(", "));
            prompt.push_str("\n\nYou MUST use absolute paths when calling read_file, edit_file, write_file, search_files.\n");
            prompt.push_str("Examples:\n");
            for dir in &all_dirs {
                prompt.push_str(&format!(
                    "- read_file: path=\"{dir}/filename.py\"\n\
                     - edit_file: path=\"{dir}/filename.py\"\n\
                     - search_files: path=\"{dir}\"\n"
                ));
            }
            prompt.push_str(
                "Do NOT use relative paths. Do NOT search in the default project directory.\n\n",
            );
        }

        if let Some(task_id) = task_id {
            let handoff = build_executor_handoff(
                task_id,
                parent_mission,
                task_description,
                &[],
                project_scope,
            );
            prompt.push_str(&handoff.render_prompt_section());
            prompt.push_str("\n\n");
        }

        prompt.push_str(&format!(
            "## Original User Request\n\
             {parent_mission}\n\n\
             ## Your Specific Task\n\
             {task_description}\n\n\
             Rules:\n\
             - Focus ONLY on your specific task. Do not expand scope.\n\
             - EXECUTE the task immediately. Do NOT ask for permission or confirmation.\n\
             - Do NOT ask \"Shall I proceed?\" or \"Would you like me to...?\". Just do the work.\n\
             - There is no human in this loop — you are an autonomous executor.\n\
             - For modifying code: use `edit_file` (preferred) or `write_file`. NEVER use `python3 -c` to rewrite files — it is blocked.\n\
             - For reading code: use `read_file` with ABSOLUTE paths. For searching: use `search_files` with ABSOLUTE directory path.\n\
             - For running commands, use the execution surface actually available in your tool set.\n\
             - If `terminal` is available, keep commands simple and single-line.\n\
             - If `terminal` is available, scope commands to explicit directories and avoid scanning `target`, `node_modules`, and `.git` trees.\n\
             - Treat operational failures as recovery work inside this task. After a failed command, tool, validation, or stale-state contradiction, inspect current state and choose a safe in-scope RETRY, REPAIR, SUBSTITUTE, or RECONCILE action. Then rerun the original verification. Do not ask the owner to perform recovery that your current tools and authority permit.\n\
             - Judge scope by causal necessity, not by whether a file predates this run. If the documented build, test, validation, or deployment workflow identifies a file inside the authorized project as the concrete blocker, that file is a task dependency rather than an unrelated file. Inspect it and make the smallest reversible local repair that restores an existing mechanical invariant (for example required frontmatter, formatting, generated metadata, or a manifest entry), preserving body content and unrelated behavior. Dirty or untracked status alone does not make a causal dependency off-limits. A file is genuinely unrelated only when no observed failure or required workflow connects it to this task; broader content/behavior changes, destructive changes, secrets, and external authority remain out of scope.\n\
             - Keep recovery bounded. Before declaring recovery_exhausted, make at least two concrete recovery attempts and retain the action, outcome, and evidence for each. If newer evidence contradicts an earlier failure, treat the earlier result as stale and retry the original operation against current state.\n\
             - Never retry a mutation whose external effect is ambiguous. Reconcile read-only when authorized; otherwise report an ambiguous_external_effect blocker for owner reconciliation.\n\
             - For public URL reachability or text, use an HTTP-capable tool or curl when a browser is unavailable; require browser access only for visual or interactive claims.\n\
             - Use report_blocker only for owner_input, missing_authority, external_dependency, ambiguous_external_effect, safety_boundary, or genuinely recovery_exhausted conditions.\n\
             - When using report_blocker, include blocker_class, external_effect_state, recovery_attempts, outcome, reason, partial_work when applicable, exact_need, next_step, and target. For missing_authority, set dependency_repair=true only for a minimal reversible local repair of a path named by failed required-workflow evidence; otherwise set it false.\n\
             - When setup needs a new password or secret, generate a strong random one and save it alone in a project-folder file readable only by the owner (permissions 600). Keep it out of every command's text, a `$(cat file)` substitution included (process lists, logs and approval cards show it): use a way that reads the file, standard input or an environment variable (wp-cli: `--prompt=admin_password < file`); when the tool only takes an argument, let it make a throwaway one (output to /dev/null) and set yours afterwards with code that reads the file (Drupal: `drush php:eval` setting the admin user's password from `trim(file_get_contents(...))`). Report the file's path, never the secret.\n\
             - Return the FULL content you produced — not a meta-description of what you did.\n\
             - NEVER return just \"I researched X\" or \"Generated a report about Y\". Return the actual content.\n\
             - Include specific outputs (file paths, data retrieved, commands run).\n\
             - {OWN_DURATION_RULE}\n\
             - When your task is to make something new (a fresh install, a new project or folder), something already in its place is not yours to reuse: leave it untouched, never report it as new, and finish not done (outcome blocked), saying what was there.\n\
             - If you create or write a file, include its FULL ABSOLUTE PATH in your result text.\n\
             - Format the result for a small chat screen: lead with the outcome, use short paragraphs or bullets, and label links, paths, verification results, versions, and IDs. Do not repeat the original request or task instructions, and do not dump the execution chronology.\n\
             - Do NOT claim the overall goal is complete. You may only finish this single task.\n\
             - Do NOT spawn sub-agents."
        ));

        if has_cli_agent {
            prompt.push_str(
                "\n- Delegation mode is active: `terminal`, `browser`, and `run_command` are not available here.\n\
                 Use direct file tools (`read_file`, `edit_file`, `write_file`, `search_files`) for narrow file work.\n\
                 Use `cli_agent` for shell/test flows or multi-step coding and research work.\n\
                 For public URL reachability or returned text, use an available HTTP read tool or ask `cli_agent` to run curl; do not require browser access unless the task is visual or interactive.\n\
                 When you use `cli_agent`, always provide `action=\"run\"`, a concrete `prompt`, and `working_dir` when you know the repo path.",
            );
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spawn::task_references_parent_context;

    #[test]
    fn executor_parent_context_reference_gets_bounded_recent_results() {
        assert!(task_references_parent_context(
            "Review Completed Task Results and select one event"
        ));
        let ctx = serde_json::json!({
            "task_results": (0..12).map(|i| format!("result-{i}")).collect::<Vec<_>>()
        });
        let formatted = bounded_completed_task_results_context(&ctx.to_string(), 3).unwrap();
        assert!(formatted.contains("### Completed Task Results"));
        assert!(!formatted.contains("result-8"));
        assert!(formatted.contains("result-9"));
        assert!(formatted.contains("result-11"));
    }

    #[test]
    fn specialist_arg_wins_over_heuristic() {
        let kind = resolve_specialist_kind(
            None,
            Some("research"),
            "Implement the sorting algorithm in src/sort.rs",
            "Add a test for the edge case",
        );
        assert_eq!(kind, SpecialistKind::Research);
    }

    #[test]
    fn invalid_specialist_arg_falls_back_to_broad_executor() {
        let kind = resolve_specialist_kind(
            None,
            Some("not_a_real_kind"),
            "Implement the sorting algorithm in src/sort.rs",
            "Add a unit test",
        );
        assert_eq!(kind, SpecialistKind::Executor);
    }

    #[test]
    fn role_typed_spawn_ignores_specialist_arg() {
        let kind = resolve_specialist_kind(
            Some(AgentRole::TaskLead),
            Some("code"),
            "any mission",
            "any task",
        );
        assert_eq!(kind, SpecialistKind::TaskLead);
    }

    #[test]
    fn executor_role_accepts_generic_recovery_override() {
        let kind = resolve_specialist_kind(
            Some(AgentRole::Executor),
            Some("executor"),
            "Recover a website deployment",
            "Verify https://example.com in a browser and finish every unmet requirement",
        );
        assert_eq!(kind, SpecialistKind::Executor);
    }

    #[test]
    fn executor_role_honors_explicit_typed_specialist() {
        let kind = resolve_specialist_kind(
            Some(AgentRole::Executor),
            Some("research"),
            "Mission wording is deliberately irrelevant",
            "Task wording is deliberately irrelevant",
        );
        assert_eq!(kind, SpecialistKind::Research);
    }

    #[test]
    fn task_lead_arg_is_rejected_falling_back_to_broad_executor() {
        // "task_lead" is role-typed only — when passed as an arg (without
        // explicit role) it must be ignored and the broad executor should run.
        let kind = resolve_specialist_kind(
            None,
            Some("task_lead"),
            "Implement the sorting algorithm in src/sort.rs",
            "Add a unit test",
        );
        assert_eq!(kind, SpecialistKind::Executor);
    }

    #[test]
    fn omitted_specialist_never_depends_on_mission_wording() {
        for (mission, task) in [
            (
                "Compile and format a report",
                "Create a markdown report in /tmp/report.md",
            ),
            (
                "Smoke-check the landing page",
                "Open the homepage in a browser and run the smoke test",
            ),
            (
                "Fix the broken assertion",
                "Run cargo test until src/math.rs passes",
            ),
        ] {
            assert_eq!(
                select_specialist_kind(AgentRole::Executor, mission, task),
                SpecialistKind::Executor
            );
        }
    }

    #[test]
    fn specialist_session_id_uses_kind_prefix() {
        let id = Uuid::parse_str("344ee9c6-a93f-48ef-84bf-ae3f4d68fc5b").unwrap();
        let session_id = build_specialist_session_id(SpecialistKind::Research, id);
        assert_eq!(
            session_id,
            "specialist:research:344ee9c6-a93f-48ef-84bf-ae3f4d68fc5b"
        );
    }

    #[test]
    fn specialist_session_id_format_holds_for_every_kind() {
        let id = Uuid::parse_str("344ee9c6-a93f-48ef-84bf-ae3f4d68fc5b").unwrap();
        let kinds = [
            SpecialistKind::TaskLead,
            SpecialistKind::Executor,
            SpecialistKind::Research,
            SpecialistKind::ArtifactWriter,
            SpecialistKind::Code,
            SpecialistKind::BrowserVerifier,
            SpecialistKind::Review,
            SpecialistKind::CommsDraft,
            SpecialistKind::Generic,
        ];
        for kind in kinds {
            let session_id = build_specialist_session_id(kind, id);
            assert!(
                session_id.starts_with("specialist:"),
                "{:?}: missing prefix in {}",
                kind,
                session_id
            );
            let expected_segment = format!(":{}:", kind.as_str());
            assert!(
                session_id.contains(&expected_segment),
                "{:?}: missing kind segment in {}",
                kind,
                session_id
            );
            assert!(
                session_id.ends_with(&id.to_string()),
                "{:?}: missing uuid suffix in {}",
                kind,
                session_id
            );
        }
    }

    #[test]
    fn specialist_session_ids_are_unique_per_invocation() {
        let a = build_specialist_session_id(SpecialistKind::Code, Uuid::new_v4());
        let b = build_specialist_session_id(SpecialistKind::Code, Uuid::new_v4());
        assert_ne!(a, b, "fresh uuids must produce unique session ids");
    }

    #[test]
    fn executor_prompt_includes_search_files_preference() {
        let prompt = legacy::build_executor_prompt(
            "find async fns",
            "user request",
            2,
            4,
            false,
            None,
            None,
        );
        assert!(prompt.contains("search_files"));
        assert!(prompt.contains("edit_file"));
        assert!(prompt.contains("avoid scanning `target`, `node_modules`, and `.git`"));
    }

    #[test]
    fn executor_prompt_extracts_directory_paths_from_mission() {
        let prompt = legacy::build_executor_prompt(
            "Fix the bug in task_scheduler.py",
            "There are 5 bugs in /tmp/debugme3/. Fix them all.",
            2,
            4,
            false,
            None,
            Some("/tmp/debugme3"),
        );
        assert!(
            prompt.contains("WORKING DIRECTORY"),
            "Should have WORKING DIRECTORY section"
        );
        assert!(
            prompt.contains("/tmp/debugme3"),
            "Should extract /tmp/debugme3 path"
        );
        assert!(
            prompt.contains("read_file: path=\"/tmp/debugme3/filename.py\""),
            "Should show read_file example"
        );
    }

    #[test]
    fn extract_directory_paths_basic() {
        let dirs = extract_directory_paths("Fix bugs in /tmp/debugme3/ and run tests");
        assert_eq!(dirs, vec!["/tmp/debugme3"]);

        let dirs = extract_directory_paths("Edit /home/user/project/foo.py");
        assert_eq!(dirs, vec!["/home/user/project"]);

        let dirs = extract_directory_paths("No paths here");
        assert!(dirs.is_empty());
    }

    #[test]
    fn task_lead_prompt_requires_concrete_final_results() {
        let prompt =
            legacy::build_task_lead_prompt("goal_1", "audit disk usage", None, 1, 3, false, false);
        assert!(prompt.contains("final reply MUST include concrete executor results"));
        assert!(prompt.contains("not just \"goal completed\""));
    }

    #[test]
    fn task_lead_keeps_cohesive_delivery_and_verification_with_one_worker() {
        let prompt =
            legacy::build_task_lead_prompt("goal_1", "deploy the site", None, 1, 3, false, false);
        assert!(prompt.contains(
            "Keep readiness checks, the mutation, and immediate verification in the same task"
        ));
        assert!(prompt.contains("real ownership boundary"));
        assert!(!prompt.contains("ALWAYS create a verification task"));
    }

    #[test]
    fn executor_prompt_mentions_cli_delegate_mode_when_cli_present() {
        let prompt =
            legacy::build_executor_prompt("refactor auth", "user request", 2, 4, true, None, None);
        assert!(prompt.contains("Delegation mode is active"));
        assert!(prompt.contains("`terminal`, `browser`, and `run_command` are not available"));
        assert!(!prompt.contains("prefer `terminal` directly"));
        assert!(prompt.contains("action=\"run\""));
        assert!(prompt.contains("working_dir"));
        assert!(prompt.contains("public URL reachability"));
        assert!(prompt.contains("do not require browser access"));
    }

    #[test]
    fn executor_prompt_includes_task_contract_when_task_id_present() {
        let prompt = legacy::build_executor_prompt(
            "patch /tmp/demo/src/main.rs",
            "fix the scoped regression in /tmp/demo",
            2,
            4,
            false,
            Some("task-123"),
            Some("/tmp/demo"),
        );
        assert!(prompt.contains("## Task Contract"));
        assert!(prompt.contains("task_id: task-123"));
        assert!(prompt.contains("allowed targets (hard boundary): /tmp/demo"));
        assert!(prompt.contains("report_blocker"));
    }

    #[test]
    fn task_lead_prompt_mentions_cli_agent_when_available() {
        let prompt =
            legacy::build_task_lead_prompt("goal_2", "build release", None, 1, 3, true, false);
        assert!(prompt.contains("## CLI Agent Delegation"));
        assert!(prompt.contains("Its sandbox cannot reach the network"));
        assert!(prompt.contains("goes to spawn_agent"));
        assert!(prompt.contains("claim the task and use `spawn_agent`"));
        assert!(prompt.contains("action=\"run\""));
        assert!(prompt.contains("working_dir"));
        assert!(prompt.contains("do not retry it unchanged"));
    }

    #[test]
    fn scheduled_task_lead_prompt_allows_direct_execution() {
        let prompt =
            legacy::build_task_lead_prompt("goal_3", "deploy blog", None, 1, 3, false, true);
        assert!(
            prompt.contains("full tool access including `terminal`"),
            "Scheduled task lead should mention terminal access"
        );
        assert!(
            !prompt.contains("MUST NOT execute tasks yourself"),
            "Scheduled task lead should NOT prohibit direct execution"
        );
    }

    #[test]
    fn non_scheduled_task_lead_prompt_allows_fallback_direct_execution() {
        let prompt =
            legacy::build_task_lead_prompt("goal_4", "deploy blog", None, 1, 3, false, false);
        assert!(
            prompt.contains("plan and delegate work"),
            "Non-scheduled task lead should prefer delegation"
        );
        assert!(
            prompt.contains("switch to direct execution"),
            "Non-scheduled task lead should allow fallback to direct execution"
        );
        assert!(
            !prompt.contains("full tool access including `terminal`"),
            "Non-scheduled task lead should NOT mention full tool access"
        );
    }
}
