//! The enriched prompt a run is given, and the git diff it reports.

use super::*;

impl CliAgentTool {
    /// For a direct coding-agent conversation whose agent did not keep a
    /// resumable session, prepend a compact transcript of prior turns so
    /// follow-up messages keep conversation context.
    pub(super) async fn prepend_direct_conversation_context(
        &self,
        session_id: &str,
        prompt: &str,
    ) -> String {
        let Ok(history) = self.state.get_history(session_id, 10).await else {
            return prompt.to_string();
        };
        let mut lines: Vec<String> = Vec::new();
        let trimmed_prompt = prompt.trim();
        for msg in &history {
            if !matches!(msg.role.as_str(), "user" | "assistant") {
                continue;
            }
            let Some(content) = msg.content.as_deref().map(str::trim) else {
                continue;
            };
            if content.is_empty() {
                continue;
            }
            // Skip background-launch handoff notices; they add noise, not context.
            if msg.role == "assistant" && crate::background_run::is_launch_line(content) {
                continue;
            }
            let snippet: String = content.chars().take(400).collect();
            lines.push(format!("{}: {snippet}", msg.role));
        }
        // Drop the trailing entry if it is the current message itself.
        if lines.last().is_some_and(|last| {
            last.strip_prefix("user: ")
                .is_some_and(|u| u == trimmed_prompt)
        }) {
            lines.pop();
        }
        if lines.is_empty() {
            return prompt.to_string();
        }
        let prior = lines
            .into_iter()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        format!("## Prior conversation\n{prior}\n\n## Current request\n{prompt}")
    }

    /// Build an enriched prompt with context from memory and conversation history.
    pub(super) async fn build_enriched_prompt(
        &self,
        session_id: &str,
        system_instruction: &str,
        task_prompt: &str,
        working_dir: Option<&str>,
        project: Option<&aidaemon_core::ProjectRef>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        // System instruction (never truncated)
        if !system_instruction.trim().is_empty() {
            parts.push(system_instruction.to_string());
        }

        // Task prompt (never truncated)
        parts.push(format!("## Task\n{}", task_prompt));

        let budget = MAX_PROMPT_SIZE.saturating_sub(
            parts.iter().map(|p| p.len()).sum::<usize>() + 200, // headroom for section headers
        );

        // Conversation context (truncated first if over budget)
        let mut context_text = String::new();
        if let Ok(history) = self.state.get_history(session_id, 10).await
            && !history.is_empty()
        {
            let mut lines = Vec::new();
            for msg in history.iter().rev().take(10) {
                let role = &msg.role;
                let content: String = msg
                    .content
                    .as_deref()
                    .unwrap_or("")
                    .chars()
                    .take(400)
                    .collect();
                lines.push(format!("{}: {}", role, content));
            }
            context_text = lines.join("\n");
        }

        // Relevant facts (query both task prompt and recent user message)
        let mut facts_text = String::new();
        if let Some(store) = &self.context_facts {
            let mut seen = HashSet::new();
            let mut facts_accum = Vec::new();

            if let Ok(facts) = store.get_relevant_facts(task_prompt, 15).await {
                for fact in facts {
                    if seen.insert((fact.category.clone(), fact.key.clone())) {
                        facts_accum.push(fact);
                    }
                }
            }

            if let Ok(history) = self.state.get_history(session_id, 10).await
                && let Some(last_user_msg) = history.iter().rev().find(|m| m.role == "user")
                && let Some(content) = last_user_msg.content.as_deref()
            {
                let user_text: String = content.chars().take(500).collect();
                if user_text != task_prompt
                    && let Ok(facts) = store.get_relevant_facts(&user_text, 10).await
                {
                    for fact in facts {
                        if seen.insert((fact.category.clone(), fact.key.clone())) {
                            facts_accum.push(fact);
                        }
                    }
                }
            }

            facts_accum.truncate(20);
            let recalled: Vec<i64> = facts_accum.iter().map(|fact| fact.id).collect();
            let _ = store.record_memory_recall(&recalled, &[]).await;
            if !facts_accum.is_empty() {
                let fact_lines: Vec<String> = facts_accum
                    .iter()
                    .map(|f| format!("- {}: {}", f.key, f.value))
                    .collect();
                facts_text = fact_lines.join("\n");
            }
        }

        // Active goal context
        let mut active_goal_text = String::new();
        if let Ok(goals) = self.state.get_goals_for_session(session_id).await
            && let Some(active_goal) = goals
                .iter()
                .find(|g| g.status == "active" || g.status == "in_progress")
        {
            active_goal_text = active_goal.description.chars().take(500).collect();
        }

        // Scoped repository instructions. Native coding CLIs also perform
        // their own discovery from cwd; this shared snapshot keeps enriched
        // specialist prompts and non-native/custom CLI agents consistent with
        // AIDaemon's direct-agent behavior.
        let mut project_instructions_text = String::new();
        if let Some(dir) = working_dir {
            match crate::turn::project_instructions::load_project_instructions(
                self.backend.clone(),
                dir,
            )
            .await
            {
                Ok(Some(instructions)) => {
                    project_instructions_text = instructions.render_for_prompt();
                }
                Ok(None) => {}
                Err(error) => {
                    warn!(working_dir = dir, %error, "Could not load CLI project instructions");
                }
            }
        }

        // Native file listing from working directory (up to 3 levels)
        let mut files_text = String::new();
        if let Some(dir) = working_dir {
            let skip: HashSet<&str> = [
                ".git",
                "node_modules",
                "target",
                "__pycache__",
                ".venv",
                "dist",
                "build",
            ]
            .into_iter()
            .collect();
            let mut file_paths =
                list_backend_files(self.backend.clone(), BackendPath::new(dir), &skip, 3, 200)
                    .await;

            if !file_paths.is_empty() {
                file_paths.sort();
                files_text = file_paths.join("\n");
            }
        }

        // Spatial awareness: list ~/projects/ contents so sub-agents can discover
        // files/projects outside the current working directory.
        let mut projects_listing_text = String::new();
        if let Ok(home) = self.backend.home_dir().await {
            let projects_dir = home.join("projects");
            if let Ok(entries) = self.backend.read_dir(&projects_dir).await {
                let mut dirs_list: Vec<String> = Vec::new();
                let mut files_list: Vec<String> = Vec::new();
                for entry in entries {
                    let name = entry.path.file_name().unwrap_or_default().to_string();
                    if name.starts_with('.') {
                        continue;
                    }
                    if entry.metadata.file_type == BackendFileType::Directory {
                        dirs_list.push(format!("  {}/", name));
                    } else if entry.metadata.file_type == BackendFileType::File {
                        files_list.push(format!("  {}", name));
                    }
                    if dirs_list.len() + files_list.len() >= 80 {
                        break;
                    }
                }
                dirs_list.sort();
                files_list.sort();
                if !dirs_list.is_empty() || !files_list.is_empty() {
                    let mut listing = format!("{}:\n", projects_dir);
                    let mut all_entries = dirs_list;
                    all_entries.append(&mut files_list);
                    listing.push_str(&all_entries.join("\n"));
                    projects_listing_text = listing;
                }
            }
        }

        // Fit within budget by truncating lower-priority sections first.
        let total = active_goal_text.len()
            + project_instructions_text.len()
            + facts_text.len()
            + files_text.len()
            + projects_listing_text.len()
            + context_text.len();
        if total > budget {
            let goal_budget = budget / 10;
            let docs_budget = budget * 3 / 10;
            let facts_budget = budget * 2 / 10;
            let files_budget = budget * 2 / 10;
            let projects_budget = budget / 10;
            let context_budget = budget.saturating_sub(
                goal_budget + docs_budget + facts_budget + files_budget + projects_budget,
            );

            if active_goal_text.len() > goal_budget {
                active_goal_text = active_goal_text.chars().take(goal_budget).collect();
                active_goal_text.push_str("...[truncated]");
            }
            if project_instructions_text.len() > docs_budget {
                project_instructions_text = project_instructions_text
                    .chars()
                    .take(docs_budget)
                    .collect();
                project_instructions_text.push_str("...[truncated]");
            }
            if facts_text.len() > facts_budget {
                facts_text = facts_text.chars().take(facts_budget).collect();
                facts_text.push_str("...[truncated]");
            }
            if files_text.len() > files_budget {
                files_text = files_text.chars().take(files_budget).collect();
                files_text.push_str("...[truncated]");
            }
            if projects_listing_text.len() > projects_budget {
                projects_listing_text = projects_listing_text
                    .chars()
                    .take(projects_budget)
                    .collect();
                projects_listing_text.push_str("...[truncated]");
            }
            if context_text.len() > context_budget {
                context_text = context_text.chars().take(context_budget).collect();
                context_text.push_str("...[truncated]");
            }
        }

        if !active_goal_text.is_empty() {
            parts.push(format!("## Active Goal\n{}", active_goal_text));
        }
        if !project_instructions_text.is_empty() {
            parts.push(format!(
                "## Project Instructions\n{}",
                project_instructions_text
            ));
        }
        // The project's memory index, after its instructions.
        if let (Some(notes), Some(project)) = (self.project_notes.clone(), project.cloned())
            && let Ok(Some(section)) =
                tokio::task::spawn_blocking(move || notes.section(&project)).await
        {
            parts.push(section);
        }
        if !facts_text.is_empty() {
            parts.push(format!("## Known Facts\n{}", facts_text));
        }
        if !files_text.is_empty() {
            parts.push(format!("## Project Files\n{}", files_text));
        }
        if !projects_listing_text.is_empty() {
            parts.push(format!(
                "## Available Project Directories\n{}",
                projects_listing_text
            ));
        }
        if !context_text.is_empty() {
            parts.push(format!("## Conversation Context\n{}", context_text));
        }

        parts.push(
            "## Instructions\n\
             - Focus exclusively on the task above\n\
             - Do NOT attempt to directly inspect or modify aidaemon's state database (SQLite/SQLCipher). Do not run sqlite3/sqlcipher, do not install sqlcipher, and do not look for encryption keys. Use aidaemon tools/APIs instead.\n\
             - Do NOT install system packages (brew/apt/dnf/pacman/pip) unless the user explicitly asked\n\
             - Report what you did and what changed when done"
                .to_string(),
        );

        parts.join("\n\n")
    }

    /// The commit `working_dir`'s checkout is at, when it is a repository
    /// with one.
    pub(super) async fn git_head(
        backend: &SharedExecutionBackend,
        working_dir: &str,
    ) -> Option<String> {
        let head = aidaemon_tools::fs_utils::run_cmd_backend(
            backend.as_ref(),
            "git rev-parse HEAD",
            Some(&BackendPath::new(working_dir)),
            5,
        )
        .await
        .ok()?;
        let sha = head.stdout.trim();
        // A commit id is hexadecimal (machine format): anything else is not one.
        (head.exit_code == 0 && !sha.is_empty() && sha.chars().all(|c| c.is_ascii_hexdigit()))
            .then(|| sha.to_string())
    }

    /// Capture git diff after CLI agent completes (for any exit code): what
    /// changed since `base` was taken, uncommitted or committed by the run.
    /// A commit made before the run started is never its own.
    pub(super) async fn capture_git_diff(
        backend: &SharedExecutionBackend,
        base: &DiffBase,
    ) -> Option<String> {
        let working_dir = base.dir.as_str();
        let directory = BackendPath::new(working_dir);
        let git_check = aidaemon_tools::fs_utils::run_cmd_backend(
            backend.as_ref(),
            "git rev-parse --git-dir",
            Some(&directory),
            5,
        )
        .await
        .ok()?;
        if git_check.exit_code != 0 {
            return None;
        }

        // Check for uncommitted changes first
        let diff_stat = Self::guarded_git(
            backend,
            &directory,
            &["diff".into(), "--no-ext-diff".into(), "--stat".into()],
            10,
        )
        .await?;
        let stat_output = diff_stat.stdout;

        if !stat_output.trim().is_empty() {
            // There are uncommitted changes — capture them
            let diff = Self::guarded_git(
                backend,
                &directory,
                &["diff".into(), "--no-ext-diff".into()],
                10,
            )
            .await?;
            let diff_text = diff.stdout;
            if !diff_text.trim().is_empty() {
                return Some(truncate_with_note(&diff_text, MAX_DIFF_SIZE));
            }
        }

        // No uncommitted changes: what the run committed, when HEAD moved
        // since it started.
        let start = base.head.as_deref()?;
        let now = Self::git_head(backend, working_dir).await?;
        if now == start {
            return None;
        }
        let log = Self::guarded_git(
            backend,
            &directory,
            &["log".into(), "--format=%s".into(), format!("{start}..HEAD")],
            10,
        )
        .await?;
        let log_output = log.stdout;

        if !log_output.trim().is_empty() {
            let committed_diff = Self::guarded_git(
                backend,
                &directory,
                &[
                    "diff".into(),
                    "--no-ext-diff".into(),
                    format!("{start}..HEAD"),
                ],
                10,
            )
            .await?;
            let committed_text = committed_diff.stdout;
            if !committed_text.trim().is_empty() {
                return Some(format!(
                    "Committed: {}\n{}",
                    log_output.lines().next().unwrap_or(""),
                    truncate_with_note(&committed_text, MAX_DIFF_SIZE)
                ));
            }
        }

        None
    }

    /// Run git in `directory` (argv, no shell) with the overrides that keep
    /// the repository's own hooks and drivers from running
    /// (`aidaemon_tools::git_guard`): the agent wrote to this repository.
    /// `None` when git could not run, or a driver cannot be turned off.
    async fn guarded_git(
        backend: &SharedExecutionBackend,
        directory: &BackendPath,
        args: &[String],
        timeout_secs: u64,
    ) -> Option<aidaemon_tools::fs_utils::CommandOutput> {
        let run = |argv: Vec<String>| {
            let mut request = ExecutionRequest::argv("git", argv);
            request.cwd = Some(directory.clone());
            backend.execute(request, Duration::from_secs(timeout_secs))
        };
        let listing = run(aidaemon_tools::git_guard::DRIVER_LISTING_ARGS
            .iter()
            .map(|arg| arg.to_string())
            .collect())
        .await
        .map(|output| output.stdout)
        .unwrap_or_default();
        let mut argv = aidaemon_tools::git_guard::untrusted_repo_config_args(&listing).ok()?;
        argv.extend(args.iter().cloned());
        let output = run(argv).await.ok()?;
        if output.timed_out {
            return None;
        }
        Some(aidaemon_tools::fs_utils::CommandOutput {
            exit_code: output.exit_code,
            stdout: output.stdout_lossy(),
            stderr: output.stderr_lossy(),
            duration_ms: output.duration_ms,
        })
    }

    /// Return the number of tracked or untracked entries in an existing Git
    /// worktree. A non-repository (or an unavailable backend) is not classified
    /// as dirty here; normal CLI-agent approval and execution policy still apply.
    pub(super) async fn dirty_worktree_entry_count(
        backend: &SharedExecutionBackend,
        working_dir: &str,
    ) -> Option<usize> {
        let directory = BackendPath::new(working_dir);
        let status = Self::guarded_git(
            backend,
            &directory,
            &[
                "status".into(),
                "--porcelain=v1".into(),
                "--untracked-files=all".into(),
            ],
            10,
        )
        .await?;
        if status.exit_code != 0 {
            return None;
        }
        Some(
            status
                .stdout
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count(),
        )
    }
}

/// Where a run's diff is taken and from what: its folder (clean when it
/// started) and the commit it was at then.
#[derive(Debug, Clone)]
pub(super) struct DiffBase {
    pub(super) dir: String,
    pub(super) head: Option<String>,
}
