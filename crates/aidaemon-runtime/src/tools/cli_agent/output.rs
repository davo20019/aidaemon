//! Reading a coding agent's output: the bounded buffer, loop detection, and
//! progress, commands and the final result parsed from its JSON stream.

use super::*;

/// Max bytes for output buffer (1 MB) to prevent unbounded memory growth.
pub(super) const BUFFER_CAP: usize = 1_048_576;

pub(super) const BUFFER_TRUNCATION_MARKER: &str = "\n[... middle of CLI output omitted ...]\n";

/// Append one output line while preserving both startup diagnostics and the
/// most recent events. Final-result records are normally at the end of JSONL
/// streams, so a prefix-only cap can erase the authoritative outcome.
pub(super) fn append_bounded_line(buffer: &mut String, prefix: &str, line: &str, cap: usize) {
    if cap == 0 {
        return;
    }

    buffer.push_str(prefix);
    buffer.push_str(line);
    buffer.push('\n');
    if buffer.len() <= cap {
        return;
    }

    if cap <= BUFFER_TRUNCATION_MARKER.len() {
        let tail_start = buffer.ceil_char_boundary(buffer.len().saturating_sub(cap));
        *buffer = buffer[tail_start..].to_string();
        return;
    }

    let head_end = floor_char_boundary(buffer, cap / 4);
    let tail_budget = cap - head_end - BUFFER_TRUNCATION_MARKER.len();
    let tail_start = buffer.ceil_char_boundary(buffer.len().saturating_sub(tail_budget));

    let mut retained = String::with_capacity(cap);
    retained.push_str(&buffer[..head_end]);
    retained.push_str(BUFFER_TRUNCATION_MARKER);
    retained.push_str(&buffer[tail_start..]);
    *buffer = retained;
}

/// Tracks recent output lines to detect infinite loops
pub(super) struct LoopDetector {
    pub(super) recent_lines: Vec<u64>, // Store hashes to save memory
    pub(super) line_counts: HashMap<u64, usize>,
}

impl LoopDetector {
    pub(super) fn new() -> Self {
        Self {
            recent_lines: Vec::with_capacity(LOOP_DETECTION_WINDOW),
            line_counts: HashMap::new(),
        }
    }

    /// Add a line and return true if an infinite loop is detected
    pub(super) fn add_line(&mut self, line: &str) -> bool {
        // Hash the line (normalized - trim whitespace)
        let normalized = line.trim();
        if normalized.is_empty() {
            return false; // Don't count empty lines
        }

        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        let hash = hasher.finish();

        // Add to window
        self.recent_lines.push(hash);
        *self.line_counts.entry(hash).or_insert(0) += 1;

        // Remove old lines if window is full
        if self.recent_lines.len() > LOOP_DETECTION_WINDOW {
            let old_hash = self.recent_lines.remove(0);
            if let Some(count) = self.line_counts.get_mut(&old_hash) {
                *count -= 1;
                if *count == 0 {
                    self.line_counts.remove(&old_hash);
                }
            }
        }

        // Check if any line appears too frequently
        self.line_counts
            .values()
            .any(|&count| count >= LOOP_DETECTION_THRESHOLD)
    }

    /// Get the most repeated line pattern for error reporting
    pub(super) fn get_loop_pattern(&self) -> Option<usize> {
        self.line_counts.values().max().copied()
    }
}

/// Check if a string looks like JSON (starts with { or [).
pub(super) fn looks_like_json(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.starts_with('{') || trimmed.starts_with('[')
}

/// Try to extract human-readable progress from a JSON line.
/// Returns None if the line isn't JSON or doesn't contain useful progress info.
pub(super) fn extract_progress_from_json(line: &str) -> Option<String> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;

    // Claude Code: tool_use events
    if let Some(tool_name) = v.get("name").and_then(|n| n.as_str()) {
        // Tool being used
        if let Some(input) = v.get("input") {
            // Extract key info based on tool type
            if tool_name == "Read" || tool_name == "read" {
                if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
                    let short_path: String = path
                        .chars()
                        .rev()
                        .take(50)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    return Some(format!("📖 Reading: ...{}", short_path));
                }
            } else if tool_name == "Write"
                || tool_name == "write"
                || tool_name == "Edit"
                || tool_name == "edit"
            {
                if let Some(path) = input.get("file_path").and_then(|p| p.as_str()) {
                    let short_path: String = path
                        .chars()
                        .rev()
                        .take(50)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect();
                    return Some(format!("✏️ Writing: ...{}", short_path));
                }
            } else if tool_name == "Bash" || tool_name == "bash" || tool_name == "terminal" {
                if let Some(cmd) = input.get("command").and_then(|c| c.as_str()) {
                    let short_cmd: String = cmd.chars().take(60).collect();
                    return Some(format!("⚡ Running: {}", short_cmd));
                }
            } else if tool_name == "Glob" || tool_name == "glob" {
                if let Some(pattern) = input.get("pattern").and_then(|p| p.as_str()) {
                    return Some(format!("🔍 Searching: {}", pattern));
                }
            } else if tool_name == "Grep" || tool_name == "grep" {
                if let Some(pattern) = input.get("pattern").and_then(|p| p.as_str()) {
                    let short: String = pattern.chars().take(40).collect();
                    return Some(format!("🔍 Grep: {}", short));
                }
            } else {
                return Some(format!("🔧 Using: {}", tool_name));
            }
        }
        return Some(format!("🔧 Using: {}", tool_name));
    }

    // Claude Code: type field events
    if let Some(event_type) = v.get("type").and_then(|t| t.as_str()) {
        match event_type {
            "assistant" => {
                // Assistant is thinking/responding - extract tool use details
                if let Some(content) = v.get("message").and_then(|m| m.get("content"))
                    && let Some(arr) = content.as_array()
                {
                    for item in arr {
                        if item.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                            let name = item
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown");
                            let input = item.get("input");

                            // Extract details based on tool type
                            let detail = match name {
                                "Bash" | "bash" | "terminal" => input
                                    .and_then(|i| i.get("command"))
                                    .and_then(|c| c.as_str())
                                    .map(|cmd| {
                                        let short: String = cmd.chars().take(50).collect();
                                        format!("⚡ {}", short)
                                    }),
                                "Read" | "read" => input
                                    .and_then(|i| i.get("file_path"))
                                    .and_then(|p| p.as_str())
                                    .map(|path| {
                                        let short: String = path
                                            .chars()
                                            .rev()
                                            .take(40)
                                            .collect::<String>()
                                            .chars()
                                            .rev()
                                            .collect();
                                        format!("📖 ...{}", short)
                                    }),
                                "Write" | "write" | "Edit" | "edit" => input
                                    .and_then(|i| i.get("file_path"))
                                    .and_then(|p| p.as_str())
                                    .map(|path| {
                                        let short: String = path
                                            .chars()
                                            .rev()
                                            .take(40)
                                            .collect::<String>()
                                            .chars()
                                            .rev()
                                            .collect();
                                        format!("✏️ ...{}", short)
                                    }),
                                "Glob" | "glob" => input
                                    .and_then(|i| i.get("pattern"))
                                    .and_then(|p| p.as_str())
                                    .map(|pat| format!("🔍 {}", pat)),
                                "Grep" | "grep" => input
                                    .and_then(|i| i.get("pattern"))
                                    .and_then(|p| p.as_str())
                                    .map(|pat| {
                                        let short: String = pat.chars().take(30).collect();
                                        format!("🔍 grep: {}", short)
                                    }),
                                "Task" => input
                                    .and_then(|i| i.get("description"))
                                    .and_then(|d| d.as_str())
                                    .map(|desc| format!("🚀 {}", desc)),
                                _ => None,
                            };

                            return Some(detail.unwrap_or_else(|| format!("🔧 {}", name)));
                        }
                    }
                }
                return None; // Don't show generic "assistant" events
            }
            "tool_use" => {
                if let Some(name) = v.get("tool").and_then(|n| n.as_str()) {
                    return Some(format!("🔧 Using: {}", name));
                }
            }
            "thinking" => {
                return Some("💭 Thinking...".to_string());
            }
            _ => {}
        }
    }

    None
}

/// The daemon's state database's file names: the database and SQLite's
/// side files beside it.
const DAEMON_DB_FILES: &[&str] = &[
    "aidaemon.db",
    "aidaemon.db-wal",
    "aidaemon.db-shm",
    "aidaemon.db-journal",
];

/// Programs that run code given in an argument (`python3 -c`, `bash -c`,
/// `sqlite3 file 'SQL'`): their code is text, so a database path inside it
/// is not its own token.
const INLINE_CODE_PROGRAMS: &[&str] = &[
    "python",
    "python3",
    "node",
    "perl",
    "ruby",
    "php",
    "bash",
    "sh",
    "zsh",
    "sqlite3",
    "sqlcipher",
    "osascript",
];

/// Why a coding agent's shell command is stopped, when it is: it changes or
/// opens the daemon's own state database, or reaches for its encryption key.
///
/// E2E incidents: external coding agents tried to "fix scheduling" by
/// hacking the SQLCipher DB directly (sqlite/sqlcipher + key hunting). This
/// tool runs in an auto-approve mode and outside any path sandbox, so no
/// typed filesystem guard covers the database here: the command is read as a
/// shell command line (machine format), never as prose.
///
/// The daemon's own harness works on its own code, so only what would touch
/// the live state is stopped (live 2026-11-27: a memory-improvement run was
/// killed for `ls -la aidaemon.db*`). A command that only reads
/// ([`aidaemon_tools::command_effects::only_reads`]: `ls`, `stat`, `du`,
/// `rg`, `cat`) runs, whatever it names; so does a change whose path operands
/// are not the database (editing source that mentions it, `cargo test`).
/// A project's own `sqlite3 app.db` is not the daemon's database and runs.
pub(super) fn prohibited_cli_agent_command_reason(command: &str) -> Option<&'static str> {
    if aidaemon_tools::command_effects::only_reads(command) {
        return None;
    }
    for (segment, _) in aidaemon_tools::command_risk::split_by_operators(command) {
        if let Some(reason) = segment_reason(&segment) {
            return Some(reason);
        }
    }
    None
}

fn segment_reason(segment: &str) -> Option<&'static str> {
    let segment = segment.trim();
    if segment.is_empty() || aidaemon_tools::command_effects::only_reads(segment) {
        return None;
    }
    // Unparsable quoting: read the whole text as one token, so a path in it
    // still counts.
    let tokens = shell_words::split(segment).unwrap_or_else(|_| vec![segment.to_string()]);
    let program_at = tokens.iter().position(|token| !is_env_assignment(token))?;
    let program = token_basename(&tokens[program_at]).to_ascii_lowercase();
    let args = &tokens[program_at + 1..];
    let inline_code = INLINE_CODE_PROGRAMS.contains(&program.as_str());

    // SQLCipher's key pragmas (machine format: SQL), in code it runs.
    if inline_code
        && args.iter().any(|arg| {
            let lower = arg.to_ascii_lowercase();
            lower.contains("pragma key")
                || lower.contains("pragma rekey")
                || lower.contains("pragma cipher")
        })
    {
        return Some(
            "Reading or setting aidaemon's database key is prohibited. Use aidaemon's built-in tools \
             (manage_memories and the like) instead of terminal SQL.",
        );
    }
    // Installing the SQLCipher shell: the first step of opening the
    // database by hand (a common rabbit hole).
    if is_installer(&program)
        && args
            .iter()
            .any(|arg| arg.to_ascii_lowercase().contains("sqlcipher"))
    {
        return Some(
            "Installing sqlcipher is prohibited in coding-agent runs: aidaemon's database is \
             changed through its own tools.",
        );
    }
    let names_db = tokens
        .iter()
        .any(|token| names_daemon_db_path(token) || (inline_code && mentions_daemon_db(token)));
    names_db.then_some(
        "Changing or opening aidaemon's live state database is prohibited (reading its file \
         listing is fine). Use manage_memories / built-in tools, or `db_probe` read-only, \
         instead of terminal SQL or file operations on it.",
    )
}

fn is_env_assignment(token: &str) -> bool {
    token.split_once('=').is_some_and(|(name, _)| {
        !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

fn token_basename(token: &str) -> &str {
    token.rsplit(['/', '\\']).next().unwrap_or(token)
}

fn is_installer(program: &str) -> bool {
    matches!(
        program,
        "brew"
            | "apt"
            | "apt-get"
            | "yum"
            | "dnf"
            | "port"
            | "pip"
            | "pip3"
            | "npm"
            | "cargo"
            | "nix-env"
            | "apk"
    )
}

/// A token that is a path to the database: its last component is one of
/// its files, a glob over them (`aidaemon.db*`), or a redirection into one
/// (`>aidaemon.db`, `2>>aidaemon.db-wal`).
fn names_daemon_db_path(token: &str) -> bool {
    let token =
        token.trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '>' | '<' | '&'));
    let name = token_basename(token).to_ascii_lowercase();
    let name = name.trim_end_matches(['*', '?']);
    DAEMON_DB_FILES.contains(&name)
}

/// Code text that names the database file (`sqlite3.connect('aidaemon.db')`,
/// `ATTACH 'aidaemon.db'`): the name bounded by anything a file name does
/// not continue with.
fn mentions_daemon_db(code: &str) -> bool {
    let lower = code.to_ascii_lowercase();
    lower.match_indices("aidaemon.db").any(|(at, _)| {
        let before = lower[..at].chars().next_back();
        !before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    })
}

pub(super) fn extract_terminal_commands_from_json(line: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
        return vec![];
    };

    let mut out = Vec::new();

    // Claude Code stream-json tool_use: { "name": "Bash", "input": {"command": "..."} }
    if let Some(tool_name) = v.get("name").and_then(|n| n.as_str())
        && matches!(tool_name, "Bash" | "bash" | "terminal")
        && let Some(cmd) = v
            .get("input")
            .and_then(|i| i.get("command"))
            .and_then(|c| c.as_str())
    {
        out.push(cmd.to_string());
    }

    // Claude Code assistant wrapper:
    // { "type": "assistant", "message": {"content":[{"type":"tool_use","name":"Bash","input":{"command":"..."}}]}}
    if v.get("type").and_then(|t| t.as_str()) == Some("assistant")
        && let Some(items) = v
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array())
    {
        for item in items {
            if item.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
                continue;
            }
            let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
            if !matches!(name, "Bash" | "bash" | "terminal") {
                continue;
            }
            if let Some(cmd) = item
                .get("input")
                .and_then(|i| i.get("command"))
                .and_then(|c| c.as_str())
            {
                out.push(cmd.to_string());
            }
        }
    }

    // Generic tool_use format: { "type": "tool_use", "tool": "bash", "command": "..." }
    if v.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
        let tool = v
            .get("tool")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if (tool == "bash" || tool == "terminal")
            && let Some(cmd) = v.get("command").and_then(|c| c.as_str())
        {
            out.push(cmd.to_string());
        }
    }

    // Codex JSONL / app-server translated item events:
    // { "type": "item.started" | "item.completed", "item": { "type": "command_execution", "command": "..." } }
    if matches!(
        v.get("type").and_then(Value::as_str),
        Some("item.started" | "item.updated" | "item.completed")
    ) && v.pointer("/item/type").and_then(Value::as_str) == Some("command_execution")
        && let Some(cmd) = v.pointer("/item/command").and_then(Value::as_str)
    {
        out.push(cmd.to_string());
    }

    // ACP host translated tool_call events:
    // { "type": "tool_call", "kind": "execute", "command": "..." }
    if v.get("type").and_then(Value::as_str) == Some("tool_call")
        && v.get("kind").and_then(Value::as_str) == Some("execute")
        && let Some(cmd) = v.get("command").and_then(Value::as_str)
    {
        out.push(cmd.to_string());
    }

    out
}

/// Try to extract meaningful content from CLI output.
/// A schema-held agent's report reads as its text
/// ([`aidaemon_coding_agents::Report::to_text`]), never as its JSON.
pub(super) fn extract_meaningful_output(raw: &str, max_chars: usize) -> String {
    // Try JSON extraction first
    if let Some(content) = extract_json_content(raw) {
        return truncate_with_note(&report_or_text(content), max_chars);
    }
    if let Some(content) = extract_jsonl_content(raw) {
        return truncate_with_note(&report_or_text(content), max_chars);
    }
    truncate_with_note(raw, max_chars)
}

/// A final message that is a report's JSON (what a schema-held agent
/// writes) as the report's text; any other text as it is.
fn report_or_text(text: String) -> String {
    match aidaemon_coding_agents::Report::from_text(&text) {
        Some(report) => report.to_text(),
        None => text,
    }
}

/// The report in a final event's `structured_output` (Claude Code with
/// `--json-schema` writes it there, its `result` text may be empty), as text.
fn structured_report(v: &Value) -> Option<String> {
    v.get("structured_output")
        .and_then(aidaemon_coding_agents::Report::from_value)
        .map(|report| report.to_text())
}

/// Try to extract content from JSON output.
pub(super) fn extract_json_content(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;

    if let Some(report) = structured_report(&v) {
        return Some(report);
    }

    // Claude Code JSON: "result" field
    if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
        return Some(result.to_string());
    }

    // Gemini CLI JSON: "output" field
    if let Some(output) = v.get("output").and_then(|o| o.as_str()) {
        return Some(output.to_string());
    }

    // Generic: "content" or "message" fields
    if let Some(content) = v.get("content").and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }
    if let Some(message) = v.get("message").and_then(|m| m.as_str()) {
        return Some(message.to_string());
    }

    None
}

/// Try to extract content from JSONL output.
pub(super) fn extract_jsonl_content(raw: &str) -> Option<String> {
    let mut last_content: Option<String> = None;
    for line in raw.lines().rev() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if let Some(report) = structured_report(&v) {
                return Some(report);
            }
            // Codex CLI JSONL emits the final response as:
            // {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
            //
            // Command events can contain very large `aggregated_output` values.
            // If this final message is not recognized, `extract_meaningful_output`
            // falls back to truncating the raw stream from the front and loses the
            // authoritative outcome. That can make a completed side effect look
            // unfinished and cause the orchestrator to repeat it.
            if v.get("type").and_then(Value::as_str) == Some("item.completed")
                && v.pointer("/item/type").and_then(Value::as_str) == Some("agent_message")
                && let Some(text) = v.pointer("/item/text").and_then(Value::as_str)
            {
                return Some(text.to_string());
            }

            if let Some(content) = v
                .pointer("/item/content")
                .or_else(|| v.pointer("/content"))
                .or_else(|| v.pointer("/result"))
            {
                if let Some(text) = content.as_str() {
                    last_content = Some(text.to_string());
                    break;
                }
                if let Some(arr) = content.as_array() {
                    let texts: Vec<&str> = arr
                        .iter()
                        .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                        .collect();
                    if !texts.is_empty() {
                        last_content = Some(texts.join("\n"));
                        break;
                    }
                }
            }
        }
    }
    last_content
}
