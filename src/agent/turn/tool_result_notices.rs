#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) enum ToolResultNotice {
    ScopeLockBlockedResult { tool_name: String, reason: String },
    ProjectInstructionsDiscovered { tool_name: String, sources: String },
    PathAutoInjectedFromProjectContext { injected_dir: String },
}

impl ToolResultNotice {
    pub(in crate::agent) fn render(&self) -> String {
        match self {
            Self::ScopeLockBlockedResult { tool_name, reason } => format!(
                "[SYSTEM] Scope lock blocked `{}`: {}. Continue with tools that stay inside the active request scope.",
                tool_name, reason
            ),
            Self::ProjectInstructionsDiscovered { tool_name, sources } => format!(
                "[SYSTEM] `{}` was deliberately NOT executed because newly applicable project \
                 instructions were discovered at {}. Review the new system-level Project \
                 Instructions, adjust the action if needed, then retry it. This deferral made \
                 no filesystem or external change.",
                tool_name, sources
            ),
            Self::PathAutoInjectedFromProjectContext { injected_dir } => format!(
                "[SYSTEM] Path was auto-injected from known project context: {}",
                injected_dir
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ToolResultNotice;

    #[test]
    fn project_instruction_discovery_notice_is_an_explicit_noop() {
        let rendered = ToolResultNotice::ProjectInstructionsDiscovered {
            tool_name: "write_file".to_string(),
            sources: "crates/widget/AGENTS.md".to_string(),
        }
        .render();

        assert!(rendered.contains("deliberately NOT executed"));
        assert!(rendered.contains("crates/widget/AGENTS.md"));
        assert!(rendered.contains("no filesystem or external change"));
    }
}
