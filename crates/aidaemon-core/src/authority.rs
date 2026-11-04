//! Explicit authority: how much a run may do, and the hook that enforces it.
//!
//! An [`Authority`] is a value, not a mood. A child's authority is derived
//! with [`Authority::narrow`], which can only shrink it.

use std::collections::BTreeSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::hook::{Hook, HookContext, ToolDecision};
use crate::message::ToolCall;
use crate::safeguard::Safeguard;
use crate::tool::{ToolAnnotations, ToolInfo};

/// How much a run may change. `ReadOnly < Write`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AccessLevel {
    ReadOnly,
    Write,
}

/// What a run is allowed to do: a level plus an optional tool allow-list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Authority {
    pub level: AccessLevel,
    /// `None` means every tool.
    pub tools: Option<BTreeSet<String>>,
}

impl Authority {
    /// Every tool, writes allowed.
    pub fn full() -> Self {
        Self {
            level: AccessLevel::Write,
            tools: None,
        }
    }

    /// Every tool, but only calls that do not change anything.
    pub fn read_only() -> Self {
        Self {
            level: AccessLevel::ReadOnly,
            tools: None,
        }
    }

    /// No tool at all.
    pub fn no_tools() -> Self {
        Self {
            level: AccessLevel::ReadOnly,
            tools: Some(BTreeSet::new()),
        }
    }

    /// The most that both `self` and `requested` allow. Never wider than
    /// either: lowest level, tool intersection (`None` is every tool).
    pub fn narrow(&self, requested: &Authority) -> Authority {
        let tools = match (&self.tools, &requested.tools) {
            (None, None) => None,
            (Some(t), None) | (None, Some(t)) => Some(t.clone()),
            (Some(a), Some(b)) => Some(a.intersection(b).cloned().collect()),
        };
        Authority {
            level: self.level.min(requested.level),
            tools,
        }
    }

    /// Whether a call to `tool` with these (per-call) annotations is allowed.
    /// The error is the reason, as short text.
    pub fn permits(&self, tool: &str, annotations: ToolAnnotations) -> Result<(), String> {
        if let Some(tools) = &self.tools {
            if !tools.contains(tool) {
                return Err("tool not available".into());
            }
        }
        if self.level == AccessLevel::ReadOnly && !annotations.read_only {
            return Err("read-only".into());
        }
        Ok(())
    }
}

/// Enforces the granted [`Authority`] (level and tool set) on every call, and
/// declares [`Safeguard::Authority`] so the agent builder accepts it. It
/// never prompts: a composition root that wants approvals for some calls adds
/// its own hook.
#[derive(Debug, Clone)]
pub struct AuthorityHook {
    authority: Authority,
}

impl AuthorityHook {
    pub fn new(authority: Authority) -> Self {
        Self { authority }
    }
}

#[async_trait]
impl Hook for AuthorityHook {
    fn safeguards(&self) -> &'static [Safeguard] {
        &[Safeguard::Authority]
    }

    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        match self.authority.permits(&call.name, tool.annotations) {
            Ok(()) => ToolDecision::Allow,
            Err(reason) => ToolDecision::Deny {
                reason: format!(
                    "[AUTHORITY BLOCKED] `{}` is outside what this session allows ({reason}). \
                     Use a permitted tool or report the limitation.",
                    call.name
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{call, spec, text, tool_calls, ScriptedProvider};
    use crate::tool::{FnTool, Tool, ToolContext, ToolOutput, ToolResult};
    use crate::{Agent, ReceiptStatus};
    use serde_json::{json, Value};

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn samples() -> Vec<Authority> {
        vec![
            Authority::full(),
            Authority::read_only(),
            Authority::no_tools(),
            Authority {
                level: AccessLevel::Write,
                tools: Some(set(&["read_file", "write_file"])),
            },
        ]
    }

    fn within(tools: &Option<BTreeSet<String>>, outer: &Option<BTreeSet<String>>) -> bool {
        match (tools, outer) {
            (_, None) => true,
            (None, Some(_)) => false,
            (Some(t), Some(o)) => t.is_subset(o),
        }
    }

    #[test]
    fn narrow_never_widens() {
        for a in samples() {
            for b in samples() {
                let n = a.narrow(&b);
                assert!(n.level <= a.level.min(b.level));
                assert!(within(&n.tools, &a.tools), "{a:?} {b:?}");
                assert!(within(&n.tools, &b.tools), "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn narrow_intersects_tool_sets() {
        let ab = Authority {
            level: AccessLevel::Write,
            tools: Some(set(&["a", "b"])),
        };
        let bc = Authority {
            level: AccessLevel::Write,
            tools: Some(set(&["b", "c"])),
        };
        assert_eq!(ab.narrow(&bc).tools, Some(set(&["b"])));
        let b = Authority {
            level: AccessLevel::Write,
            tools: Some(set(&["b"])),
        };
        assert_eq!(Authority::full().narrow(&b).tools, Some(set(&["b"])));
        assert_eq!(b.narrow(&Authority::full()).tools, Some(set(&["b"])));
        assert_eq!(Authority::full().narrow(&Authority::full()).tools, None);
    }

    #[test]
    fn read_only_denies_non_read_only_calls() {
        let a = Authority::read_only();
        assert!(a.permits("write_file", ToolAnnotations::WRITE).is_err());
        assert!(a.permits("read_file", ToolAnnotations::READ_ONLY).is_ok());
    }

    #[test]
    fn tool_set_denies_unlisted_tools() {
        let a = Authority {
            level: AccessLevel::Write,
            tools: Some(set(&["read_file"])),
        };
        assert!(a.permits("terminal", ToolAnnotations::READ_ONLY).is_err());
        assert!(a.permits("read_file", ToolAnnotations::READ_ONLY).is_ok());
    }

    #[tokio::test]
    async fn hook_declares_authority_and_denies() {
        let writer = FnTool::new(
            spec("write_file", "writes", &[]),
            ToolAnnotations::WRITE,
            |_: Value| async { Ok(ToolOutput::text("wrote")) },
        );
        let agent = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([call("c1", "write_file", json!({}))]),
                text("done"),
            ]))
            .tool(writer)
            .hook(AuthorityHook::new(Authority::read_only()))
            .attest(Safeguard::Scope)
            .attest(Safeguard::UntrustedOutput)
            .build()
            .unwrap();
        let out = agent.run("s", "go").await.unwrap();
        assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
    }

    struct Mixed;
    #[async_trait]
    impl Tool for Mixed {
        fn spec(&self) -> crate::ToolSpec {
            spec("mixed", "read or write", &[])
        }
        fn annotations(&self) -> ToolAnnotations {
            ToolAnnotations::DESTRUCTIVE
        }
        fn annotations_for(&self, arguments: &Value) -> ToolAnnotations {
            if arguments["op"] == "read" {
                ToolAnnotations::READ_ONLY
            } else {
                self.annotations()
            }
        }
        async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
            Ok(ToolOutput::text("ok"))
        }
    }

    #[tokio::test]
    async fn annotations_for_overrides_per_call() {
        let agent = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([
                    call("c1", "mixed", json!({"op": "read"})),
                    call("c2", "mixed", json!({"op": "write"})),
                ]),
                text("done"),
            ]))
            .tool(Mixed)
            .hook(AuthorityHook::new(Authority::read_only()))
            .attest(Safeguard::Scope)
            .attest(Safeguard::UntrustedOutput)
            .build()
            .unwrap();
        let out = agent.run("s", "go").await.unwrap();
        let r = out.receipts();
        assert_ne!(r[0].status, ReceiptStatus::Denied);
        assert_eq!(r[1].status, ReceiptStatus::Denied);
    }
}
