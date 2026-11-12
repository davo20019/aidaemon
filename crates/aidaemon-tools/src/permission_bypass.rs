//! Launch arguments that switch a CLI coding agent (claude, codex, gemini,
//! copilot, aider) into unattended auto-approve mode, and their removal. The
//! single list for every place that launches an agent.

/// Boolean flags that switch a CLI agent into unattended auto-approve mode,
/// across every agent we launch. The single list for the cli_agent tool and
/// the terminal-bridge review profiles.
pub const PERMISSION_BYPASS_FLAGS: &[&str] = &[
    // claude
    "--dangerously-skip-permissions",
    "--allow-dangerously-skip-permissions",
    "--dangerously-bypass-hook-trust",
    // codex
    "--dangerously-bypass-approvals-and-sandbox",
    "--full-auto",
    // gemini
    "--yolo",
    // copilot
    "--allow-all-tools",
    "--allow-all-paths",
    // aider
    "--yes",
    "--yes-always",
];

/// Options whose given value turns them into a bypass: (flag, value).
const PERMISSION_BYPASS_OPTION_VALUES: &[(&str, &str)] = &[
    ("--sandbox", "false"),
    ("--sandbox", "danger-full-access"),
    ("--approval-mode", "yolo"),
    ("--permission-mode", "bypassPermissions"),
];

fn is_permission_bypass_flag(arg: &str, agent: &str) -> bool {
    if arg == "-y" {
        // `-y` means something else for most CLIs; it is gemini's --yolo.
        return agent.eq_ignore_ascii_case("gemini");
    }
    PERMISSION_BYPASS_FLAGS.iter().any(|flag| {
        arg == *flag
            || arg
                .strip_prefix(flag)
                .is_some_and(|rest| rest.starts_with('='))
    }) || PERMISSION_BYPASS_OPTION_VALUES.iter().any(|(flag, value)| {
        arg.strip_prefix(flag).and_then(|r| r.strip_prefix('=')) == Some(value)
    })
}

pub fn without_permission_bypass(args: &[String], agent: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if is_permission_bypass_flag(arg, agent) {
            index += 1;
            continue;
        }
        if let Some(value) = args.get(index + 1) {
            if PERMISSION_BYPASS_OPTION_VALUES
                .iter()
                .any(|(flag, v)| arg == *flag && value == v)
            {
                index += 2;
                continue;
            }
        }
        out.push(args[index].clone());
        index += 1;
    }
    out
}

/// Remove auto-approve flags from a launch argument list. `agent` is the
/// agent's name or executable name (it only matters for gemini's `-y`).
/// Every launch goes through this function; it refuses if a flag survives.
pub fn strip_permission_bypass_flags(args: &mut Vec<String>, agent: &str) -> Result<(), String> {
    *args = without_permission_bypass(args, agent);
    if without_permission_bypass(args, agent).len() != args.len() {
        return Err("its permission-bypass flags could not be removed".to_string());
    }
    Ok(())
}

/// Rewrites claude's `--allow-dangerously-skip-permissions` alias to
/// `--dangerously-skip-permissions` (once). Returns the arguments and whether
/// anything was rewritten; other agents' arguments are returned unchanged.
pub fn normalize_terminal_agent_permission_aliases(
    agent: Option<&str>,
    args: Vec<String>,
) -> (Vec<String>, bool) {
    if !agent
        .map(|value| value.trim().eq_ignore_ascii_case("claude"))
        .unwrap_or(false)
    {
        return (args, false);
    }

    let mut out = Vec::with_capacity(args.len());
    let mut has_dangerous_skip = false;
    let mut rewrote = false;

    for arg in args {
        if arg == "--dangerously-skip-permissions" {
            has_dangerous_skip = true;
            out.push(arg);
        } else if arg == "--allow-dangerously-skip-permissions" {
            rewrote = true;
            if !has_dangerous_skip {
                out.push("--dangerously-skip-permissions".to_string());
                has_dangerous_skip = true;
            }
        } else {
            out.push(arg);
        }
    }

    (out, rewrote)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_terminal_agent_permission_aliases_rewrites_claude_allow_flag() {
        let args = vec![
            "--allow-dangerously-skip-permissions".to_string(),
            "--model".to_string(),
            "sonnet".to_string(),
        ];
        let (normalized, rewrote) =
            normalize_terminal_agent_permission_aliases(Some("claude"), args);
        assert!(rewrote);
        assert_eq!(
            normalized,
            vec![
                "--dangerously-skip-permissions".to_string(),
                "--model".to_string(),
                "sonnet".to_string()
            ]
        );
    }

    #[test]
    fn normalize_terminal_agent_permission_aliases_leaves_other_agents_unchanged() {
        let args = vec!["--allow-dangerously-skip-permissions".to_string()];
        let (normalized, rewrote) =
            normalize_terminal_agent_permission_aliases(Some("codex"), args.clone());
        assert!(!rewrote);
        assert_eq!(normalized, args);
    }
}
