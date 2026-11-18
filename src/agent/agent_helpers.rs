//! Free-function helpers extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Groups status/heartbeat helpers, tool-arg
//! summarization and resume-checkpoint rendering.

#[cfg(test)]
pub use aidaemon_runtime::turn::helpers::touch_heartbeat;

#[cfg(test)]
pub(in crate::agent) use aidaemon_runtime::turn::tool_summary::summarize_tool_args;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_summary_wraps_command_in_backticks() {
        let args = serde_json::json!({"command": "cargo build --release"}).to_string();
        assert_eq!(
            summarize_tool_args("terminal", &args),
            "`cargo build --release`"
        );
    }

    #[test]
    fn terminal_summary_collapses_multiline_commands() {
        let args = serde_json::json!({"command": "echo one &&\n  echo   two"}).to_string();
        assert_eq!(
            summarize_tool_args("terminal", &args),
            "`echo one && echo two`"
        );
    }

    #[test]
    fn terminal_summary_redacts_secret_before_truncation() {
        // The key starts just before the 80-char truncation boundary. If
        // truncation ran first, the pattern would no longer match and a
        // prefix of the key would leak.
        let cmd = format!("echo {} sk-{}", "x".repeat(60), "a".repeat(30));
        let args = serde_json::json!({ "command": cmd }).to_string();
        let summary = summarize_tool_args("terminal", &args);
        assert!(
            !summary.contains("sk-a"),
            "no fragment of the key may survive: {summary}"
        );
    }

    #[test]
    fn terminal_summary_shortens_home_dir() {
        let home = dirs::home_dir().unwrap();
        let cmd = format!(
            "ls {}/projects",
            home.to_string_lossy().trim_end_matches('/')
        );
        let args = serde_json::json!({ "command": cmd }).to_string();
        assert_eq!(summarize_tool_args("terminal", &args), "`ls ~/projects`");
    }

    #[test]
    fn terminal_summary_truncation_preserves_closing_backtick() {
        // Long commands must stay within the downstream 80-char status cap
        // (sanitize::STATUS_SUMMARY_MAX_CHARS) so the second truncation never
        // strips the closing backtick.
        let cmd = "x".repeat(200);
        let args = serde_json::json!({ "command": cmd }).to_string();
        let summary = summarize_tool_args("terminal", &args);
        assert!(
            summary.chars().count() <= 80,
            "got {}",
            summary.chars().count()
        );
        assert!(
            summary.starts_with('`') && summary.ends_with('`'),
            "got: {summary}"
        );
        assert!(
            summary.contains("..."),
            "long command should show truncation"
        );
    }
}
