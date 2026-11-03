//! Approval as a `before_tool` hook: pause (Interrupt) or refuse (Deny)
//! based on annotations, name rules and tokenized command rules.

use std::collections::HashSet;

use aidaemon_core::{Hook, HookContext, Safeguard, ToolCall, ToolDecision, ToolInfo};
use async_trait::async_trait;

use crate::shell::{catastrophic_reason, command_segments, segment_matches_prefix};

/// When to ask a human before a call runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalMode {
    /// Never ask (rules and hard blocks still apply).
    Never,
    /// Ask before tools annotated `destructive`.
    #[default]
    DestructiveOnly,
    /// Ask before anything not annotated `read_only`.
    AllWrites,
    /// Ask before every call.
    Always,
}

/// Approval policy.
///
/// Evaluation order, first match wins:
/// 1. **Hard blocks** (`Deny`): catastrophic commands found by tokenizing
///    `access.commands` (see [`crate::catastrophic_reason`]). Cannot be
///    allowed by any rule.
/// 2. **Deny rules** (`Deny`): tool name in the deny list, or any command
///    segment starting with a denied prefix.
/// 3. **Allow rules** (`Allow`): tool name in the allow list, or every
///    command segment of every declared command starts with an allowed
///    prefix.
/// 4. **Mode** (`Interrupt` or `Allow`).
///
/// Command prefixes are matched by tokens, not characters: `git push`
/// matches `git push origin` but not `git pushy`, and a prefix never
/// matches text inside quotes or in a later `&&` segment by accident.
#[derive(Debug, Clone, Default)]
pub struct ApprovalPolicy {
    mode: ApprovalMode,
    allow_tools: HashSet<String>,
    deny_tools: HashSet<String>,
    allow_commands: Vec<Vec<String>>,
    deny_commands: Vec<Vec<String>>,
}

impl ApprovalPolicy {
    pub fn new(mode: ApprovalMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    pub fn allow_tool(mut self, name: impl Into<String>) -> Self {
        self.allow_tools.insert(name.into());
        self
    }

    pub fn deny_tool(mut self, name: impl Into<String>) -> Self {
        self.deny_tools.insert(name.into());
        self
    }

    /// Allow commands starting with these tokens (e.g. `"git status"`).
    pub fn allow_command(mut self, prefix: &str) -> Self {
        if let Some(tokens) = command_segments(prefix).into_iter().next() {
            self.allow_commands.push(tokens);
        }
        self
    }

    /// Deny commands starting with these tokens (e.g. `"git push"`).
    pub fn deny_command(mut self, prefix: &str) -> Self {
        if let Some(tokens) = command_segments(prefix).into_iter().next() {
            self.deny_commands.push(tokens);
        }
        self
    }

    pub fn mode(&self) -> ApprovalMode {
        self.mode
    }

    /// The decision for one call, without a hook context (handy in tests
    /// and for previewing a policy).
    pub fn decide(&self, tool_name: &str, info: &ToolInfo) -> ToolDecision {
        for command in &info.access.commands {
            if let Some(reason) = catastrophic_reason(command) {
                return ToolDecision::Deny {
                    reason: format!("blocked by policy: {reason} (command: `{command}`)"),
                };
            }
        }

        if self.deny_tools.contains(tool_name) {
            return ToolDecision::Deny {
                reason: format!("blocked by policy: the tool `{tool_name}` is denied"),
            };
        }
        let segments: Vec<Vec<String>> = info
            .access
            .commands
            .iter()
            .flat_map(|c| command_segments(c))
            .collect();
        for segment in &segments {
            if let Some(prefix) = self
                .deny_commands
                .iter()
                .find(|p| segment_matches_prefix(segment, p))
            {
                return ToolDecision::Deny {
                    reason: format!(
                        "blocked by policy: commands starting with `{}` are denied",
                        prefix.join(" ")
                    ),
                };
            }
        }

        if self.allow_tools.contains(tool_name) {
            return ToolDecision::Allow;
        }
        if !segments.is_empty()
            && segments.iter().all(|segment| {
                self.allow_commands
                    .iter()
                    .any(|p| segment_matches_prefix(segment, p))
            })
        {
            return ToolDecision::Allow;
        }

        let ann = info.annotations;
        let why = match self.mode {
            ApprovalMode::Never => None,
            ApprovalMode::DestructiveOnly => ann
                .destructive
                .then(|| format!("`{tool_name}` is marked destructive")),
            ApprovalMode::AllWrites => {
                (!ann.read_only).then(|| format!("`{tool_name}` is not read-only"))
            }
            ApprovalMode::Always => Some(format!("every call needs approval (`{tool_name}`)")),
        };
        match why {
            Some(why) => ToolDecision::Interrupt {
                reason: format!("approval required: {why}"),
            },
            None => ToolDecision::Allow,
        }
    }
}

#[async_trait]
impl Hook for ApprovalPolicy {
    fn safeguards(&self) -> &'static [Safeguard] {
        &[Safeguard::Authority]
    }

    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        self.decide(&call.name, tool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::{AccessManifest, ToolAnnotations};

    fn info(annotations: ToolAnnotations, commands: &[&str]) -> ToolInfo {
        ToolInfo {
            annotations,
            access: AccessManifest {
                commands: commands.iter().map(|c| c.to_string()).collect(),
                ..AccessManifest::default()
            },
        }
    }

    fn interrupts(d: &ToolDecision) -> bool {
        matches!(d, ToolDecision::Interrupt { .. })
    }

    #[test]
    fn modes_follow_annotations() {
        let ro = info(ToolAnnotations::READ_ONLY, &[]);
        let w = info(ToolAnnotations::WRITE, &[]);
        let d = info(ToolAnnotations::DESTRUCTIVE, &[]);

        let never = ApprovalPolicy::new(ApprovalMode::Never);
        assert_eq!(never.decide("t", &d), ToolDecision::Allow);

        let destructive = ApprovalPolicy::new(ApprovalMode::DestructiveOnly);
        assert_eq!(destructive.decide("t", &w), ToolDecision::Allow);
        assert!(interrupts(&destructive.decide("t", &d)));

        let writes = ApprovalPolicy::new(ApprovalMode::AllWrites);
        assert_eq!(writes.decide("t", &ro), ToolDecision::Allow);
        assert!(interrupts(&writes.decide("t", &w)));

        let always = ApprovalPolicy::new(ApprovalMode::Always);
        assert!(interrupts(&always.decide("t", &ro)));
    }

    #[test]
    fn deny_wins_over_allow() {
        let policy = ApprovalPolicy::new(ApprovalMode::Always)
            .allow_tool("shell")
            .deny_tool("shell");
        assert!(matches!(
            policy.decide("shell", &info(ToolAnnotations::READ_ONLY, &[])),
            ToolDecision::Deny { .. }
        ));

        let policy = ApprovalPolicy::new(ApprovalMode::Always)
            .allow_command("git")
            .deny_command("git push");
        let shell = |cmd: &str| info(ToolAnnotations::DESTRUCTIVE, &[cmd]);
        assert!(matches!(
            policy.decide("shell", &shell("git push origin main")),
            ToolDecision::Deny { .. }
        ));
        assert_eq!(
            policy.decide("shell", &shell("git status")),
            ToolDecision::Allow
        );
    }

    #[test]
    fn command_allow_needs_every_segment() {
        let policy = ApprovalPolicy::new(ApprovalMode::DestructiveOnly).allow_command("git status");
        let shell = |cmd: &str| info(ToolAnnotations::DESTRUCTIVE, &[cmd]);
        assert_eq!(
            policy.decide("shell", &shell("git status --short")),
            ToolDecision::Allow
        );
        assert!(interrupts(&policy.decide(
            "shell",
            &shell("git status && curl https://example.invalid")
        )));
        assert!(interrupts(&policy.decide("shell", &shell("git statusx"))));
    }

    #[test]
    fn hard_blocks_cannot_be_allowed() {
        let policy = ApprovalPolicy::new(ApprovalMode::Never)
            .allow_tool("shell")
            .allow_command("rm");
        let decision = policy.decide("shell", &info(ToolAnnotations::WRITE, &["rm -rf ~"]));
        match decision {
            ToolDecision::Deny { reason } => assert!(reason.contains("never allowed"), "{reason}"),
            other => panic!("expected deny, got {other:?}"),
        }
        // The same words quoted as data are not a command.
        assert_eq!(
            policy.decide("shell", &info(ToolAnnotations::WRITE, &["echo 'rm -rf ~'"])),
            ToolDecision::Allow
        );
    }
}
