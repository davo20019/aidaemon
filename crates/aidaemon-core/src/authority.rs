//! Explicit authority: how much a run may do, and the hook that enforces it.
//!
//! An [`Authority`] is a value, not a mood. A child's authority is derived
//! with [`Authority::narrow`], which can only shrink it.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::approval::AUTHORITY_BLOCKED;
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
        if let Some(tools) = &self.tools
            && !tools.contains(tool) {
                return Err("tool not available".into());
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
#[derive(Clone)]
pub struct AuthorityHook {
    authority: Authority,
    ceiling: Option<Arc<dyn AuthorityCeiling>>,
    delegated: bool,
    /// How the person in charge can lift this hook's own limit, added to
    /// its refusals (see [`Self::with_limit_note`]).
    limit_note: Option<String>,
}

impl AuthorityHook {
    pub fn new(authority: Authority) -> Self {
        Self {
            authority,
            ceiling: None,
            delegated: false,
            limit_note: None,
        }
    }

    /// Add `note` to every refusal this hook's own authority makes: what
    /// set the limit and how it can be lifted, so the run tells the person
    /// that instead of asking them to do the work by hand.
    pub fn with_limit_note(mut self, note: impl Into<String>) -> Self {
        self.limit_note = Some(note.into());
        self
    }

    /// This run's authority was set by the agent that started it (a
    /// sub-agent's): a refusal by that authority says so, so the run reports
    /// the limit to its parent instead of to the session's owner. A refusal
    /// by the ceiling still names the session.
    pub fn delegated(mut self) -> Self {
        self.delegated = true;
        self
    }

    /// Read `ceiling` on every call and judge with the narrower of this
    /// hook's authority and the ceiling's. A ceiling can narrow a running
    /// turn; it can never widen it.
    pub fn with_ceiling(mut self, ceiling: Arc<dyn AuthorityCeiling>) -> Self {
        self.ceiling = Some(ceiling);
        self
    }
}

impl std::fmt::Debug for AuthorityHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthorityHook")
            .field("authority", &self.authority)
            .field("ceiling", &self.ceiling.is_some())
            .field("delegated", &self.delegated)
            .finish()
    }
}

/// A limit on what a run may do that can tighten while the run is in flight,
/// read before every call. `None` means no extra limit right now.
#[async_trait]
pub trait AuthorityCeiling: Send + Sync {
    async fn ceiling(&self) -> Option<Authority>;

    /// What set the ceiling and how it can be lifted, added to a refusal
    /// the ceiling (and not the run's own authority) made.
    fn limit_note(&self) -> Option<&str> {
        None
    }
}

#[async_trait]
impl Hook for AuthorityHook {
    fn safeguards(&self) -> &'static [Safeguard] {
        &[Safeguard::Authority]
    }

    fn authority(&self) -> Option<Authority> {
        Some(self.authority.clone())
    }

    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        if self.delegated
            && let Err(reason) = self.authority.permits(&call.name, tool.annotations) {
                return ToolDecision::Deny {
                    reason: format!(
                        "{AUTHORITY_BLOCKED} `{}` is outside the access this sub-agent was \
                         given ({reason}). Use a permitted tool or report the limitation: the \
                         agent that started this one chose that access, and only it can start \
                         a sub-agent with more (never more than its own).",
                        call.name
                    ),
                };
            }
        let narrowed = match &self.ceiling {
            Some(ceiling) => ceiling.ceiling().await.map(|c| self.authority.narrow(&c)),
            None => None,
        };
        let judged = narrowed.as_ref().unwrap_or(&self.authority);
        match judged.permits(&call.name, tool.annotations) {
            Ok(()) => ToolDecision::Allow,
            Err(reason) => {
                // Whose limit refused it: the run's own authority, or only
                // the ceiling narrowing it now.
                let note = if self.authority.permits(&call.name, tool.annotations).is_ok() {
                    self.ceiling.as_ref().and_then(|c| c.limit_note())
                } else {
                    self.limit_note.as_deref()
                };
                let mut reason = format!(
                    "{AUTHORITY_BLOCKED} `{}` is outside what this session allows ({reason}). \
                     Use a permitted tool or report the limitation.",
                    call.name
                );
                if let Some(note) = note {
                    reason.push(' ');
                    reason.push_str(note);
                }
                ToolDecision::Deny { reason }
            }
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
                assert_eq!(n.level, a.level.min(b.level));
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

    struct FlagCeiling(std::sync::Arc<std::sync::atomic::AtomicBool>);
    #[async_trait]
    impl AuthorityCeiling for FlagCeiling {
        async fn ceiling(&self) -> Option<Authority> {
            self.0
                .load(std::sync::atomic::Ordering::SeqCst)
                .then(Authority::read_only)
        }
    }

    fn writer_tool() -> impl Tool {
        FnTool::new(
            spec("write_file", "writes", &[]),
            ToolAnnotations::WRITE,
            |_: Value| async { Ok(ToolOutput::text("wrote")) },
        )
    }

    #[tokio::test]
    async fn ceiling_narrows_mid_run() {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flipper = {
            let flag = flag.clone();
            FnTool::new(
                spec("flip", "flips the ceiling", &[]),
                ToolAnnotations::READ_ONLY,
                move |_: Value| {
                    let flag = flag.clone();
                    async move {
                        flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        Ok(ToolOutput::text("flipped"))
                    }
                },
            )
        };
        let agent = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([call("c1", "write_file", json!({}))]),
                tool_calls([call("c2", "flip", json!({}))]),
                tool_calls([call("c3", "write_file", json!({}))]),
                text("done"),
            ]))
            .tool(writer_tool())
            .tool(flipper)
            .hook(
                AuthorityHook::new(Authority::full())
                    .with_ceiling(std::sync::Arc::new(FlagCeiling(flag))),
            )
            .attest(Safeguard::Scope)
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .attest(Safeguard::UntrustedOutput)
            .build()
            .unwrap();
        let out = agent.run("s", "go").await.unwrap();
        let r = out.receipts();
        assert_eq!(r[0].status, ReceiptStatus::Succeeded);
        assert_eq!(r[2].status, ReceiptStatus::Denied);
    }

    struct FullCeiling;
    #[async_trait]
    impl AuthorityCeiling for FullCeiling {
        async fn ceiling(&self) -> Option<Authority> {
            Some(Authority::full())
        }
    }

    #[tokio::test]
    async fn ceiling_never_widens() {
        let agent = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([call("c1", "write_file", json!({}))]),
                text("done"),
            ]))
            .tool(writer_tool())
            .hook(
                AuthorityHook::new(Authority::read_only())
                    .with_ceiling(std::sync::Arc::new(FullCeiling)),
            )
            .attest(Safeguard::Scope)
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .attest(Safeguard::UntrustedOutput)
            .build()
            .unwrap();
        let out = agent.run("s", "go").await.unwrap();
        assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
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
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .attest(Safeguard::UntrustedOutput)
            .build()
            .unwrap();
        let out = agent.run("s", "go").await.unwrap();
        assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
        let reason = out.receipts()[0].reason.as_deref().unwrap_or_default();
        assert!(
            reason.starts_with("[AUTHORITY BLOCKED] `write_file`"),
            "{reason}"
        );
    }

    /// A sub-agent refused by the access its parent gave it is told so (the
    /// parent can start it again with more); a refusal by the session's
    /// ceiling still names the session.
    #[tokio::test]
    async fn a_delegated_refusal_names_the_parent_not_the_session() {
        let run = |hook: AuthorityHook| async move {
            let agent = Agent::builder("a")
                .model("m")
                .provider(ScriptedProvider::new([
                    tool_calls([call("c1", "write_file", json!({}))]),
                    text("done"),
                ]))
                .tool(writer_tool())
                .hook(hook)
                .attest(Safeguard::Scope)
                .attest(Safeguard::Approval)
                .attest(Safeguard::Egress)
                .attest(Safeguard::UntrustedOutput)
                .build()
                .unwrap();
            let out = agent.run("s", "go").await.unwrap();
            assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
            out.receipts()[0].reason.clone().unwrap_or_default()
        };
        let given = run(AuthorityHook::new(Authority::read_only()).delegated()).await;
        assert!(
            given.contains("this sub-agent was given (read-only)"),
            "{given}"
        );
        assert!(!given.contains("session"), "{given}");

        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let session = run(AuthorityHook::new(Authority::full())
            .with_ceiling(std::sync::Arc::new(FlagCeiling(flag)))
            .delegated())
        .await;
        assert!(
            session.contains("this session allows (read-only)"),
            "{session}"
        );
    }

    struct NotedCeiling;
    #[async_trait]
    impl AuthorityCeiling for NotedCeiling {
        async fn ceiling(&self) -> Option<Authority> {
            Some(Authority::read_only())
        }
        fn limit_note(&self) -> Option<&str> {
            Some("ceiling-note")
        }
    }

    /// A refusal carries the note of whichever limit made it: the run's own
    /// authority, or the ceiling narrowing a run that could otherwise write.
    #[tokio::test]
    async fn a_refusal_says_how_its_own_limit_is_lifted() {
        let run = |hook: AuthorityHook| async move {
            let agent = Agent::builder("a")
                .model("m")
                .provider(ScriptedProvider::new([
                    tool_calls([call("c1", "write_file", json!({}))]),
                    text("done"),
                ]))
                .tool(writer_tool())
                .hook(hook)
                .attest(Safeguard::Scope)
                .attest(Safeguard::Approval)
                .attest(Safeguard::Egress)
                .attest(Safeguard::UntrustedOutput)
                .build()
                .unwrap();
            let out = agent.run("s", "go").await.unwrap();
            assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
            out.receipts()[0].reason.clone().unwrap_or_default()
        };
        let own = run(AuthorityHook::new(Authority::read_only())
            .with_limit_note("own-note")
            .with_ceiling(std::sync::Arc::new(NotedCeiling)))
        .await;
        assert!(own.ends_with(" own-note"), "{own}");
        let ceiling = run(AuthorityHook::new(Authority::full())
            .with_limit_note("own-note")
            .with_ceiling(std::sync::Arc::new(NotedCeiling)))
        .await;
        assert!(ceiling.ends_with(" ceiling-note"), "{ceiling}");
        let plain = run(AuthorityHook::new(Authority::read_only())).await;
        assert!(plain.ends_with("report the limitation."), "{plain}");
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
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .attest(Safeguard::UntrustedOutput)
            .build()
            .unwrap();
        let out = agent.run("s", "go").await.unwrap();
        let r = out.receipts();
        assert_eq!(r[0].status, ReceiptStatus::Succeeded);
        assert_eq!(r[1].status, ReceiptStatus::Denied);
    }

    #[test]
    fn an_empty_tool_set_permits_nothing() {
        assert_eq!(
            Authority::no_tools().permits("read_file", ToolAnnotations::READ_ONLY),
            Err("tool not available".to_string())
        );
    }
}
