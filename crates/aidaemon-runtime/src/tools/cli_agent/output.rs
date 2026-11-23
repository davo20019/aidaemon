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
        let mut tail_start = buffer.len().saturating_sub(cap);
        while tail_start < buffer.len() && !buffer.is_char_boundary(tail_start) {
            tail_start += 1;
        }
        *buffer = buffer[tail_start..].to_string();
        return;
    }

    let head_end = floor_char_boundary(buffer, cap / 4);
    let tail_budget = cap - head_end - BUFFER_TRUNCATION_MARKER.len();
    let mut tail_start = buffer.len().saturating_sub(tail_budget);
    while tail_start < buffer.len() && !buffer.is_char_boundary(tail_start) {
        tail_start += 1;
    }

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

pub(super) fn prohibited_cli_agent_command_reason(command: &str) -> Option<&'static str> {
    let lower = command.trim().to_ascii_lowercase();

    // E2E incidents: external coding agents tried to "fix scheduling" by hacking the
    // SQLCipher DB directly (sqlite/sqlcipher + key hunting). This tool runs in an
    // auto-approve mode and outside any path sandbox, so no typed filesystem
    // guard covers the daemon's state database here; the command is checked
    // for machine formats only: the database's file name (`aidaemon.db`, and
    // its `-wal`/`-shm` siblings), the SQLCipher tool, and SQLCipher's key
    // pragmas. A project's own `sqlite3 app.db` is not the daemon's database
    // and runs.
    if lower.contains("aidaemon.db")
        || lower.contains("sqlcipher")
        || lower.contains("pragma key")
        || lower.contains("pragma cipher")
    {
        return Some(
            "Direct manipulation of aidaemon's state database is prohibited. Use manage_memories / built-in tools instead of terminal SQL.",
        );
    }

    // Narrow install block for sqlcipher specifically (common rabbit-hole).
    if lower.contains("install") && lower.contains("sqlcipher") {
        return Some(
            "Installing sqlcipher is prohibited in cli_agent runs. Use aidaemon tools instead.",
        );
    }

    None
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

    out
}

/// Try to extract meaningful content from CLI output.
pub(super) fn extract_meaningful_output(raw: &str, max_chars: usize) -> String {
    // Try JSON extraction first
    if let Some(content) = extract_json_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    if let Some(content) = extract_jsonl_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    truncate_with_note(raw, max_chars)
}

/// Try to extract content from JSON output.
pub(super) fn extract_json_content(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;

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
