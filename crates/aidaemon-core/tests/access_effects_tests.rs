//! A tool that declares what it writes (`access`) but not its effects
//! (`effects`) still has effects the kernel can bind a decline to: they are
//! derived from the declared access, so a write the owner refused is not
//! made by the same tool again or by another one.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aidaemon_core::testing::{ScriptedProvider, call, spec, text, tool_calls};
use aidaemon_core::{
    AccessManifest, Agent, ApprovalRisk, CallEffects, EffectClass, EffectKey, Hook, HookContext,
    InterruptKind, OWNER_DECLINED, OwnerApprovalRequest, Resolution, RunStatus, Tool,
    ToolAnnotations, ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde_json::{Value, json};

const TARGET: &str = "/srv/notes/plan.txt";

/// Writes the path it is given; declares it, nothing more.
struct DeclaredWrite {
    name: &'static str,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for DeclaredWrite {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec(self.name, "writes a file", &["path"])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    fn access(&self, arguments: &Value) -> AccessManifest {
        AccessManifest {
            writes: vec![arguments["path"].as_str().unwrap_or("").to_string()],
            ..AccessManifest::default()
        }
    }
    async fn call(&self, _arguments: Value, _ctx: &ToolContext) -> ToolResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::text("written"))
    }
}

/// Asks the owner before the first tool's calls, naming its effects.
struct AskFirst;

#[async_trait]
impl Hook for AskFirst {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        if call.name != "write_note" {
            return ToolDecision::Allow;
        }
        ToolDecision::Interrupt {
            reason: "the owner must confirm this write".to_string(),
            kind: InterruptKind::OwnerApproval(OwnerApprovalRequest {
                card: None,
                effects: tool.effects.keys.clone(),
                summary: format!("Write {}", call.arguments["path"]),
                subject: format!("{} {}", call.name, call.arguments),
                risk: ApprovalRisk::High,
                warnings: Vec::new(),
            }),
        }
    }
}

#[test]
fn declared_access_implies_effects() {
    let writes = AccessManifest {
        writes: vec![TARGET.into(), "relative.txt".into()],
        hosts: vec![
            "https://api.example.invalid/v1".into(),
            "h.example.invalid:8443".into(),
        ],
        commands: vec!["/usr/bin/git push".into()],
        ..AccessManifest::default()
    };
    let effects = CallEffects::from_access(&writes, ToolAnnotations::WRITE);
    assert_eq!(
        effects.keys,
        vec![
            EffectKey::path(EffectClass::Write, TARGET).unwrap(),
            EffectKey::named(EffectClass::Process, "git"),
            EffectKey::egress("https", "api.example.invalid", None),
            EffectKey::egress("https", "h.example.invalid", Some(8443)),
        ]
    );
    assert!(!effects.complete, "derived, never vouched for");
    let destructive = CallEffects::from_access(&writes, ToolAnnotations::DESTRUCTIVE);
    assert!(
        destructive
            .keys
            .contains(&EffectKey::path(EffectClass::Delete, TARGET).unwrap())
    );
    // A read-only tool changes nothing; only where it reaches counts.
    let read = CallEffects::from_access(&writes, ToolAnnotations::READ_ONLY);
    assert!(read.keys.iter().all(|key| key.class == EffectClass::Egress));
    assert_eq!(
        CallEffects::from_access(&AccessManifest::default(), ToolAnnotations::WRITE),
        CallEffects::unknown()
    );
}

#[tokio::test]
async fn a_declined_declared_write_binds_the_rest_of_the_run() {
    let first = Arc::new(AtomicUsize::new(0));
    let second = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "write_note", json!({"path": TARGET}))]),
        tool_calls([call("c2", "save_file", json!({"path": TARGET}))]),
        text("not done"),
    ]));
    let agent = Agent::builder("declines")
        .model("m")
        .provider(provider)
        .tool(DeclaredWrite {
            name: "write_note",
            calls: first.clone(),
        })
        .tool(DeclaredWrite {
            name: "save_file",
            calls: second.clone(),
        })
        .hook(AskFirst)
        .allow_unguarded()
        .build()
        .unwrap();
    let out = agent.run("s", "write the plan").await.unwrap();
    let RunStatus::Interrupted { interruption } = &out.status else {
        panic!("{:?}", out.status);
    };
    let InterruptKind::OwnerApproval(request) = &interruption.kind else {
        panic!("{:?}", interruption.kind);
    };
    assert_eq!(
        request.effects,
        vec![EffectKey::path(EffectClass::Write, TARGET).unwrap()]
    );
    let out = agent
        .resume(
            out.state,
            Resolution::Deny {
                reason: "not that file".into(),
            },
        )
        .await
        .unwrap();
    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(
        second.load(Ordering::SeqCst),
        0,
        "the other tool did not write the declined path"
    );
    assert!(
        out.state
            .declined
            .contains(&EffectKey::path(EffectClass::Write, TARGET).unwrap()),
        "{:?}",
        out.state.declined
    );
    let refused = out
        .receipts()
        .iter()
        .find(|receipt| receipt.tool == "save_file")
        .expect("a receipt for the second call");
    assert!(
        refused
            .reason
            .as_deref()
            .is_some_and(|reason| reason.starts_with(OWNER_DECLINED)),
        "{refused:?}"
    );
}
