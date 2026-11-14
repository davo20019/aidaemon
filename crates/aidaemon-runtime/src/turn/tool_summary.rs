//! One-line summaries of a tool call's arguments, for status updates and
//! post-task learning.

use serde_json::Value;

/// Extract a brief human-readable summary from tool arguments JSON.
/// Helper to truncate a string and append "..." if it exceeds `max` chars.
fn truncate_summary(s: &str, max: usize) -> String {
    let truncated: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

/// Helper to extract the last path component (file/dir name) for compact display.
fn short_path(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

pub fn summarize_tool_args(name: &str, arguments: &str) -> String {
    let val: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };

    // Helper closure to get a string field from the JSON args.
    let get_str = |key: &str| val.get(key).and_then(|v| v.as_str());

    match name {
        // --- Command execution ---
        "terminal" | "run_command" => get_str("command")
            .map(|cmd| {
                // Order matters: shorten the home dir first (the "File path"
                // secret pattern would otherwise swallow `/Users/...` whole),
                // redact before truncating (a secret cut at the boundary no
                // longer matches its pattern and its prefix would leak), then
                // collapse whitespace so multi-line commands render one-line.
                let cmd = aidaemon_tools::sanitize::shorten_home_dir(cmd);
                let cmd = aidaemon_tools::sanitize::redact_secrets(&cmd);
                let one_line = cmd.split_whitespace().collect::<Vec<_>>().join(" ");
                // 75 + "..." + 2 backticks = 80 — fits the downstream
                // STATUS_SUMMARY_MAX_CHARS cap exactly, so the closing
                // backtick survives the second truncation in sanitize.rs.
                format!("`{}`", truncate_summary(&one_line, 75))
            })
            .unwrap_or_default(),

        // --- File operations ---
        "read_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "write_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "edit_file" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),
        "search_files" => {
            let pattern = get_str("pattern").or_else(|| get_str("glob")).unwrap_or("");
            if pattern.is_empty() {
                String::new()
            } else {
                truncate_summary(pattern, 40)
            }
        }
        "list_files" => get_str("path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),

        // --- Web & network ---
        "web_search" => get_str("query")
            .map(|q| truncate_summary(q, 50))
            .unwrap_or_default(),
        "web_fetch" => get_str("url")
            .map(|u| truncate_summary(u, 60))
            .unwrap_or_default(),
        "http_request" => {
            let method = get_str("method").unwrap_or("GET");
            let url = get_str("url").unwrap_or("");
            if url.is_empty() {
                method.to_string()
            } else {
                format!("{} {}", method, truncate_summary(url, 50))
            }
        }

        // --- Browser ---
        "browser" => {
            let action = get_str("action").unwrap_or("");
            let url = get_str("url").unwrap_or("");
            if !url.is_empty() {
                format!("{} {}", action, truncate_summary(url, 50))
            } else {
                action.to_string()
            }
        }

        // --- Desktop GUI ---
        "computer_use" => {
            let action = get_str("action").unwrap_or("");
            let app = get_str("app").unwrap_or("");
            let index = val
                .get("element_index")
                .and_then(|v| v.as_u64())
                .map(|i| i.to_string());
            match (action, app, index) {
                ("click", app, Some(index)) if !app.is_empty() => {
                    format!("click {app} #{index}")
                }
                (_, app, _) if !app.is_empty() => format!("{action} {app}"),
                _ => action.to_string(),
            }
        }

        // --- Memory ---
        "remember_fact" => {
            let fact = get_str("fact").or_else(|| get_str("value")).unwrap_or("");
            if fact.is_empty() {
                "saving to memory".to_string()
            } else {
                truncate_summary(fact, 40)
            }
        }
        "manage_memories" => get_str("action").unwrap_or("").to_string(),
        "search_history" => get_str("action").unwrap_or("").to_string(),

        // --- Skills ---
        "use_skill" => get_str("skill_name").unwrap_or("").to_string(),
        "manage_skills" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- People ---
        "manage_people" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- Agents ---
        "spawn_agent" => get_str("mission")
            .map(|m| truncate_summary(m, 50))
            .unwrap_or_default(),
        "cli_agent" => {
            let action = get_str("action").unwrap_or("run");
            if action != "run" {
                return format!("action={}", action);
            }
            let tool = get_str("tool").unwrap_or("auto");
            let prompt = get_str("prompt")
                .or_else(|| get_str("task"))
                .or_else(|| get_str("mission"))
                .or_else(|| get_str("description"))
                .or_else(|| get_str("command"))
                .unwrap_or("");
            let task_desc = truncate_summary(prompt, 50);
            if task_desc.is_empty() {
                format!("→ {}", tool)
            } else {
                format!("→ {}: {}", tool, task_desc)
            }
        }
        "manage_cli_agents" => get_str("action").unwrap_or("").to_string(),

        // --- Config / diagnostic ---
        "manage_config" => get_str("action").unwrap_or("").to_string(),
        "manage_mcp" => {
            let action = get_str("action").unwrap_or("");
            let name_val = get_str("name").unwrap_or("");
            if name_val.is_empty() {
                action.to_string()
            } else {
                format!("{} {}", action, name_val)
            }
        }

        // --- Channel operations ---
        "read_channel_history" => {
            let channel = get_str("channel_id").unwrap_or("");
            if channel.is_empty() {
                String::new()
            } else {
                truncate_summary(channel, 30)
            }
        }
        "send_file" => get_str("file_path")
            .map(|p| short_path(p).to_string())
            .unwrap_or_default(),

        // --- MCP tools: extract a human-readable name from the prefix ---
        _ if name.starts_with("mcp__") => {
            // mcp__chrome-devtools__take_screenshot → chrome-devtools: take_screenshot
            let without_prefix = &name[5..]; // skip "mcp__"
            if let Some(idx) = without_prefix.find("__") {
                let server = &without_prefix[..idx];
                let tool = &without_prefix[idx + 2..];
                // For common tools, add key arg info
                let arg_info = match tool {
                    "navigate_page" => get_str("url")
                        .map(|u| format!(" {}", truncate_summary(u, 40)))
                        .unwrap_or_default(),
                    "click" | "hover" | "fill" => get_str("uid")
                        .map(|u| format!(" #{}", u))
                        .unwrap_or_default(),
                    "evaluate_script" => get_str("function")
                        .map(|f| format!(" {}", truncate_summary(f, 30)))
                        .unwrap_or_default(),
                    _ => String::new(),
                };
                format!("{}: {}{}", server, tool.replace('_', " "), arg_info)
            } else {
                without_prefix.replace('_', " ")
            }
        }

        _ => String::new(),
    }
}
