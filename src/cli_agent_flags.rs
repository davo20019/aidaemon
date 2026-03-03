use std::collections::{HashMap, HashSet};
use std::process::Stdio;
use std::time::Duration;

use once_cell::sync::Lazy;
use regex::Regex;
use tokio::process::Command;

use crate::traits::StateStore;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub(crate) const AGENT_FLAGS_CACHE_TTL_SECS: i64 = 24 * 60 * 60;
pub(crate) const MAX_DISCOVERED_AGENT_FLAGS: usize = 512;
pub(crate) const MAX_DISCOVERED_AGENT_FLAG_CHARS: usize = 96;
pub(crate) const MAX_DISCOVERED_AGENT_FLAG_DESC_CHARS: usize = 240;
pub(crate) const AGENT_FLAGS_PAGE_SIZE: usize = 12;
pub(crate) const SUPPORTED_TERMINAL_AGENTS: &[&str] = &["codex", "claude", "gemini", "opencode"];
pub(crate) const MAX_TERMINAL_AGENT_ARGS: usize = 24;
pub(crate) const MAX_TERMINAL_AGENT_ARG_CHARS: usize = 256;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct AgentFlagDoc {
    pub(crate) flag: String,
    #[serde(default)]
    pub(crate) description: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct AgentFlagsCacheEntry {
    pub(crate) updated_at_unix: i64,
    #[serde(default)]
    pub(crate) flags: Vec<String>,
    #[serde(default)]
    pub(crate) docs: Vec<AgentFlagDoc>,
}

// ---------------------------------------------------------------------------
// Normalization helpers
// ---------------------------------------------------------------------------

/// Normalize a terminal agent name, returning the canonical lowercase form.
///
/// Reuses the `normalize_terminal_agent_name` from `lib.rs` (which returns
/// `Option<&'static str>`) and converts to `Option<String>`.
pub(crate) fn normalize_terminal_agent_name(value: &str) -> Option<String> {
    crate::normalize_terminal_agent_name(value).map(String::from)
}

pub(crate) fn normalize_terminal_agent_args(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && !v.contains('\0'))
        .map(|v| {
            v.chars()
                .take(MAX_TERMINAL_AGENT_ARG_CHARS)
                .collect::<String>()
        })
        .take(MAX_TERMINAL_AGENT_ARGS)
        .collect()
}

pub(crate) fn normalize_discovered_agent_flags(values: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for value in values {
        let cleaned = value.trim().to_string();
        if cleaned.is_empty() || !cleaned.starts_with("--") || cleaned.contains('\0') {
            continue;
        }
        let clipped = cleaned
            .chars()
            .take(MAX_DISCOVERED_AGENT_FLAG_CHARS)
            .collect::<String>();
        if seen.insert(clipped.clone()) {
            out.push(clipped);
        }
        if out.len() >= MAX_DISCOVERED_AGENT_FLAGS {
            break;
        }
    }
    out
}

pub(crate) fn normalize_agent_flag_docs(values: Vec<AgentFlagDoc>) -> Vec<AgentFlagDoc> {
    let mut out: Vec<AgentFlagDoc> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();

    for value in values {
        let normalized_flags = normalize_discovered_agent_flags(vec![value.flag]);
        let Some(flag) = normalized_flags.first().cloned() else {
            continue;
        };

        let description = value
            .description
            .map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "))
            .map(|d| d.trim().to_string())
            .filter(|d| !d.is_empty())
            .map(|d| {
                d.chars()
                    .take(MAX_DISCOVERED_AGENT_FLAG_DESC_CHARS)
                    .collect::<String>()
            });

        if let Some(idx) = seen.get(&flag).copied() {
            let existing = out.get_mut(idx).expect("index from seen must exist");
            let replace = match (&existing.description, &description) {
                (None, Some(_)) => true,
                (Some(old), Some(new)) => new.len() > old.len(),
                _ => false,
            };
            if replace {
                existing.description = description;
            }
        } else {
            seen.insert(flag.clone(), out.len());
            out.push(AgentFlagDoc { flag, description });
            if out.len() >= MAX_DISCOVERED_AGENT_FLAGS {
                break;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

pub(crate) fn format_agent_flag_docs(
    agent: &str,
    docs: &[AgentFlagDoc],
    cached: bool,
) -> Vec<String> {
    if docs.is_empty() {
        return vec![format!(
            "### No flags found for `{}`{}\nTry `/agent flags {} refresh`.",
            agent,
            if cached { " (cached)" } else { "" },
            agent
        )];
    }

    let page_size = AGENT_FLAGS_PAGE_SIZE.max(1);
    let total = docs.len();
    let total_pages = total.div_ceil(page_size);
    let mut pages = Vec::new();

    for (idx, chunk) in docs.chunks(page_size).enumerate() {
        let mut lines = Vec::new();
        lines.push(format!(
            "### Flags for `{}`{}",
            agent,
            if cached { " (cached)" } else { "" }
        ));
        lines.push(format!("**{} total**", total));
        if total_pages > 1 {
            let start = idx * page_size + 1;
            let end = start + chunk.len() - 1;
            lines.push(format!(
                "**Showing {}-{} of {} (page {}/{})**",
                start,
                end,
                total,
                idx + 1,
                total_pages
            ));
        }
        lines.push(String::new());

        for doc in chunk {
            lines.push(format!("- `{}`", doc.flag));
            if let Some(desc) = doc.description.as_deref() {
                lines.push(format!("Description: {}", desc));
            }
            lines.push(String::new());
        }

        if idx + 1 == total_pages {
            lines.push("Set defaults with `/agent defaults set <agent> [flags...]`.".to_string());
            lines.push("Bypass once with `--no-default-flags`.".to_string());
            lines.push("Refresh with `/agent flags <agent> refresh`.".to_string());
        }

        while lines.last().map(|line| line.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        pages.push(lines.join("\n"));
    }

    pages
}

pub(crate) fn strip_no_default_flag(agent_args: &mut Vec<String>) -> bool {
    let before = agent_args.len();
    agent_args.retain(|arg| arg != "--no-default-flags");
    before != agent_args.len()
}

// ---------------------------------------------------------------------------
// Help text parsing
// ---------------------------------------------------------------------------

pub(crate) fn extract_long_flags_from_help(help_text: &str) -> Vec<String> {
    static LONG_FLAG_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"--[a-zA-Z0-9][a-zA-Z0-9\-]*").expect("valid long flag regex"));
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for cap in LONG_FLAG_RE.find_iter(help_text) {
        let flag = cap.as_str().to_string();
        if seen.insert(flag.clone()) {
            out.push(flag);
        }
    }
    out
}

pub(crate) fn extract_flag_docs_from_help(help_text: &str) -> Vec<AgentFlagDoc> {
    static LONG_FLAG_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"--[a-zA-Z0-9][a-zA-Z0-9\-]*").expect("valid long flag regex"));
    let mut docs = Vec::new();
    for raw_line in help_text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let matches = LONG_FLAG_RE.find_iter(line).collect::<Vec<_>>();
        if matches.is_empty() {
            continue;
        }
        let first_end = matches[0].end();
        let desc_raw = line
            .get(first_end..)
            .unwrap_or("")
            .trim()
            .trim_start_matches([':', ';', ',', '|', '-', ' ']);
        let description = if desc_raw.is_empty() {
            None
        } else {
            Some(desc_raw.to_string())
        };
        for m in matches {
            docs.push(AgentFlagDoc {
                flag: m.as_str().to_string(),
                description: description.clone(),
            });
        }
    }
    normalize_agent_flag_docs(docs)
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

pub(crate) async fn discover_agent_flags(agent: &str) -> anyhow::Result<Vec<AgentFlagDoc>> {
    let run_help_cmd = |help_arg: &str| {
        let mut cmd = Command::new(agent);
        cmd.arg(help_arg);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);
        cmd.env_remove("CLAUDECODE");
        cmd.env_remove("CLAUDE_CODE");
        cmd
    };

    let output = match tokio::time::timeout(
        Duration::from_secs(10),
        run_help_cmd("--help").output(),
    )
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                anyhow::bail!(
                    "`{}` is not installed or not in PATH on this machine.",
                    agent
                );
            }
            match tokio::time::timeout(Duration::from_secs(10), run_help_cmd("-h").output()).await {
                Ok(Ok(v)) => v,
                Ok(Err(second_err)) => {
                    anyhow::bail!("Failed to run `{}` help: {}", agent, second_err)
                }
                Err(_) => {
                    anyhow::bail!("`{} -h` timed out while fetching help output.", agent)
                }
            }
        }
        Err(_) => anyhow::bail!("`{} --help` timed out while fetching help output.", agent),
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = if stdout.is_empty() {
        stderr.clone()
    } else if stderr.is_empty() {
        stdout.clone()
    } else {
        format!("{}\n{}", stdout, stderr)
    };

    let flags = extract_long_flags_from_help(&combined);
    if flags.is_empty() {
        anyhow::bail!(
            "No long-form flags were detected in `{}` help output. Try running `{}` manually.",
            agent,
            agent
        );
    }
    let docs = extract_flag_docs_from_help(&combined);
    if docs.is_empty() {
        let fallback = normalize_discovered_agent_flags(flags)
            .into_iter()
            .map(|flag| AgentFlagDoc {
                flag,
                description: None,
            })
            .collect::<Vec<_>>();
        return Ok(fallback);
    }
    Ok(docs)
}

// ---------------------------------------------------------------------------
// Cache load/save
// ---------------------------------------------------------------------------

fn agent_flags_cache_key(namespace: &str, user_id: u64, agent: &str) -> String {
    format!(
        "telegram:agent_flags_cache:{}:{}:{}",
        namespace,
        user_id,
        agent.to_ascii_lowercase()
    )
}

fn terminal_agent_defaults_key(namespace: &str, user_id: u64, chat_id: i64) -> String {
    format!(
        "telegram:agent_defaults:{}:{}:{}",
        namespace, user_id, chat_id
    )
}

pub(crate) async fn load_agent_flags_cache(
    state: &dyn StateStore,
    namespace: &str,
    user_id: u64,
    agent: &str,
) -> Option<AgentFlagsCacheEntry> {
    let key = agent_flags_cache_key(namespace, user_id, agent);
    let raw = state.get_setting(&key).await.ok().flatten()?;
    let mut parsed: AgentFlagsCacheEntry = serde_json::from_str(&raw).ok()?;
    let mut docs = normalize_agent_flag_docs(parsed.docs.clone());
    if docs.is_empty() && !parsed.flags.is_empty() {
        docs = normalize_discovered_agent_flags(parsed.flags.clone())
            .into_iter()
            .map(|flag| AgentFlagDoc {
                flag,
                description: None,
            })
            .collect();
    }
    if docs.is_empty() {
        return None;
    }
    parsed.docs = docs.clone();
    parsed.flags = docs.iter().map(|d| d.flag.clone()).collect();
    Some(parsed)
}

pub(crate) async fn save_agent_flags_cache(
    state: &dyn StateStore,
    namespace: &str,
    user_id: u64,
    agent: &str,
    docs: &[AgentFlagDoc],
) -> anyhow::Result<()> {
    let key = agent_flags_cache_key(namespace, user_id, agent);
    let normalized_docs = normalize_agent_flag_docs(docs.to_vec());
    let payload = AgentFlagsCacheEntry {
        updated_at_unix: chrono::Utc::now().timestamp(),
        flags: normalized_docs.iter().map(|d| d.flag.clone()).collect(),
        docs: normalized_docs,
    };
    let serialized = serde_json::to_string(&payload)?;
    state.set_setting(&key, &serialized).await
}

pub(crate) async fn load_terminal_agent_defaults(
    state: &dyn StateStore,
    namespace: &str,
    chat_id: i64,
    user_id: u64,
) -> HashMap<String, Vec<String>> {
    let key = terminal_agent_defaults_key(namespace, user_id, chat_id);
    let raw = match state.get_setting(&key).await {
        Ok(Some(v)) => v,
        _ => return HashMap::new(),
    };
    let parsed: HashMap<String, Vec<String>> =
        serde_json::from_str(&raw).unwrap_or_else(|_| HashMap::new());
    let mut sanitized = HashMap::new();
    for (agent, args) in parsed {
        let Some(agent_name) = normalize_terminal_agent_name(&agent) else {
            continue;
        };
        let cleaned = normalize_terminal_agent_args(args);
        let (cleaned, _) =
            crate::normalize_terminal_agent_permission_aliases(Some(agent_name.as_str()), cleaned);
        if cleaned.is_empty() {
            continue;
        }
        sanitized.insert(agent_name, cleaned);
    }
    sanitized
}

pub(crate) async fn save_terminal_agent_defaults(
    state: &dyn StateStore,
    namespace: &str,
    chat_id: i64,
    user_id: u64,
    defaults: &HashMap<String, Vec<String>>,
) -> anyhow::Result<()> {
    let key = terminal_agent_defaults_key(namespace, user_id, chat_id);
    let serialized = serde_json::to_string(defaults)?;
    state.set_setting(&key, &serialized).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_agent_flag_docs_paginates_and_keeps_footer_actions() {
        let docs = (1..=13)
            .map(|n| AgentFlagDoc {
                flag: format!("--flag-{}", n),
                description: Some(format!("Description {}", n)),
            })
            .collect::<Vec<_>>();

        let pages = format_agent_flag_docs("claude", &docs, false);
        assert_eq!(pages.len(), 2);
        assert!(pages[0].contains("Showing 1-12 of 13 (page 1/2)"));
        assert!(pages[1].contains("Showing 13-13 of 13 (page 2/2)"));
        assert!(pages[1].contains("Set defaults with `/agent defaults set <agent> [flags...]`."));
        assert!(pages[1].contains("Bypass once with `--no-default-flags`."));
        assert!(pages[1].contains("Refresh with `/agent flags <agent> refresh`."));
    }

    #[test]
    fn format_agent_flag_docs_includes_cached_badge() {
        let docs = vec![AgentFlagDoc {
            flag: "--print".to_string(),
            description: Some("Output format.".to_string()),
        }];
        let pages = format_agent_flag_docs("claude", &docs, true);
        assert_eq!(pages.len(), 1);
        assert!(pages[0].contains("Flags for `claude` (cached)"));
    }
}
