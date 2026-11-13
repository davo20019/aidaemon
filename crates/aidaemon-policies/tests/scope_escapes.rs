//! Path-scope escapes found by an outside security review, each turned into
//! a check that now fails safely. The tool under test trusts the guard: it
//! declares the raw path it was given and writes it relative to the root.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aidaemon_core::testing::{ScriptedProvider, call, text, tool_calls};
use aidaemon_core::{
    AccessManifest, Agent, HookContext, ReceiptStatus, Resolution, RunStatus, Tool,
    ToolAnnotations, ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult,
    ToolSpec,
};
use aidaemon_policies::{
    ApprovalMode, ApprovalPolicy, ApprovalRules, RuleDecision, ScopeGuard, resolve_path,
};
use async_trait::async_trait;
use serde_json::{Value, json};

struct Env {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    outside: PathBuf,
}

fn env() -> Env {
    let tmp = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(tmp.path()).unwrap();
    let root = base.join("notes");
    let outside = base.join("outside");
    std::fs::create_dir_all(root.join("drafts")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("existing.txt"), "ORIGINAL").unwrap();
    Env {
        _tmp: tmp,
        root,
        outside,
    }
}

fn write_info(path: &str) -> ToolInfo {
    ToolInfo {
        annotations: ToolAnnotations::WRITE,
        access: AccessManifest {
            writes: vec![path.to_string()],
            ..AccessManifest::default()
        },
        effects: aidaemon_core::CallEffects::unknown(),
        outbound: Default::default(),
    }
}

fn denied(guard: &ScopeGuard, path: &str) -> bool {
    matches!(
        guard.decide("naive_write", &write_info(path)),
        ToolDecision::Deny { .. }
    )
}

/// Writes `path` relative to the root, trusting the guard entirely.
struct NaiveWrite {
    root: PathBuf,
}

#[async_trait]
impl Tool for NaiveWrite {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "naive_write".into(),
            description: "Write a file".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]
            }),
        }
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    fn access(&self, args: &Value) -> AccessManifest {
        AccessManifest {
            writes: vec![args["path"].as_str().unwrap_or("").to_string()],
            ..AccessManifest::default()
        }
    }
    async fn call(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        let path = Path::new(args["path"].as_str().unwrap_or(""));
        let target = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        std::fs::write(&target, args["content"].as_str().unwrap_or(""))?;
        Ok(ToolOutput::text("wrote"))
    }
}

/// Run one naive write under the guard, approving every question.
async fn naive(env: &Env, path: &str) -> aidaemon_core::RunOutcome {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call(
            "c1",
            "naive_write",
            json!({"path": path, "content": "PWNED"}),
        )]),
        text("done"),
    ]));
    let agent = Agent::builder("scope")
        .model("m")
        .provider(provider)
        .tool(NaiveWrite {
            root: env.root.clone(),
        })
        .hook(ScopeGuard::new([&env.root]))
        .hook(ApprovalPolicy::new(ApprovalMode::AllWrites))
        .allow_unguarded()
        .build()
        .unwrap();
    let mut out = agent.run("s", "go").await.unwrap();
    while matches!(out.status, RunStatus::Interrupted { .. }) {
        out = agent.resume(out.state, Resolution::Approve).await.unwrap();
    }
    out
}

#[tokio::test]
async fn a_dangling_symlink_cannot_create_a_file_outside() {
    let e = env();
    let target = e.outside.join("new.txt");
    std::os::unix::fs::symlink(&target, e.root.join("drafts/dangling")).unwrap();
    assert!(denied(&ScopeGuard::new([&e.root]), "drafts/dangling"));
    let out = naive(&e, "drafts/dangling").await;
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
    assert!(!target.exists(), "dangling symlink escaped");
}

#[tokio::test]
async fn dot_dot_after_a_symlink_follows_the_link_first() {
    let e = env();
    std::fs::create_dir_all(e.outside.join("sub")).unwrap();
    std::os::unix::fs::symlink(e.outside.join("sub"), e.root.join("drafts/link")).unwrap();
    assert!(denied(
        &ScopeGuard::new([&e.root]),
        "drafts/link/../esc.txt"
    ));
    let out = naive(&e, "drafts/link/../esc.txt").await;
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
    assert!(
        !e.outside.join("esc.txt").exists(),
        "symlink then .. escaped"
    );
}

#[tokio::test]
async fn the_default_guard_follows_a_symlinked_directory() {
    let e = env();
    std::os::unix::fs::symlink(&e.outside, e.root.join("drafts/link")).unwrap();
    assert!(denied(&ScopeGuard::new([&e.root]), "drafts/link/x.txt"));
    let out = naive(&e, "drafts/link/x.txt").await;
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
    assert!(!e.outside.join("x.txt").exists(), "symlinked dir escaped");
}

#[tokio::test]
async fn a_symlinked_file_cannot_overwrite_its_outside_target() {
    let e = env();
    std::os::unix::fs::symlink(e.outside.join("existing.txt"), e.root.join("drafts/f")).unwrap();
    let out = naive(&e, "drafts/f").await;
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
    assert_eq!(
        std::fs::read_to_string(e.outside.join("existing.txt")).unwrap(),
        "ORIGINAL"
    );
}

#[tokio::test]
async fn lexical_escapes_and_prefix_siblings_stay_out() {
    let e = env();
    std::fs::create_dir_all(e.root.parent().unwrap().join("notes-evil")).unwrap();
    let guard = ScopeGuard::new([&e.root]);
    let absolute = e.outside.join("x.txt").display().to_string();
    for path in [
        "../outside/x.txt",
        "drafts/../../outside/x.txt",
        "drafts/../../outside/./x.txt",
        "../notes-evil/x.txt",
        absolute.as_str(),
    ] {
        assert!(denied(&guard, path), "{path} allowed");
    }
    // A leading space is a different (relative) name, never trimmed into
    // the absolute path.
    assert!(!denied(&guard, &format!(" {absolute}")));
    assert!(!denied(&guard, "drafts/a.txt"));
}

#[tokio::test]
async fn a_root_given_through_a_symlink_still_admits_its_own_files() {
    let e = env();
    let link_root = e.root.parent().unwrap().join("notes-link");
    std::os::unix::fs::symlink(&e.root, &link_root).unwrap();
    let guard = ScopeGuard::new([&link_root]);
    assert!(!denied(&guard, "drafts/ok.txt"));
    assert!(!denied(
        &guard,
        &e.root.join("drafts/ok.txt").display().to_string()
    ));
    assert!(denied(&guard, "../outside/x.txt"));
}

#[test]
fn the_resolver_walks_links_one_component_at_a_time() {
    let e = env();
    std::fs::create_dir_all(e.outside.join("sub")).unwrap();
    std::os::unix::fs::symlink(e.outside.join("sub"), e.root.join("drafts/link")).unwrap();
    std::os::unix::fs::symlink(e.outside.join("gone/new.txt"), e.root.join("dangling")).unwrap();
    std::os::unix::fs::symlink("loop", e.root.join("loop")).unwrap();
    assert_eq!(
        resolve_path(&e.root.join("drafts/link/../esc.txt")).unwrap(),
        e.outside.join("esc.txt")
    );
    // A broken link is followed to its target, as `open(O_CREAT)` would.
    assert_eq!(
        resolve_path(&e.root.join("dangling")).unwrap(),
        e.outside.join("gone/new.txt")
    );
    assert_eq!(
        resolve_path(&e.root.join("drafts/new/deeper.txt")).unwrap(),
        e.root.join("drafts/new/deeper.txt")
    );
    assert!(resolve_path(&e.root.join("loop/x")).is_err());
    assert!(resolve_path(Path::new("relative/path")).is_err());
}

/// Writes under `drafts/` run freely; anything else asks.
struct DraftsOnly {
    drafts: PathBuf,
}

#[async_trait]
impl ApprovalRules for DraftsOnly {
    async fn judge(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        info: &ToolInfo,
    ) -> RuleDecision {
        if info
            .access
            .writes
            .iter()
            .all(|w| Path::new(w).starts_with(&self.drafts))
        {
            RuleDecision::Allow
        } else {
            RuleDecision::Ask(aidaemon_core::OwnerApprovalRequest {
                effects: vec![],
                summary: format!("{} wants to write", call.name),
                subject: format!("{} {}", call.name, call.arguments),
                risk: aidaemon_core::ApprovalRisk::High,
                warnings: vec![],
                card: None,
            })
        }
    }
}

#[tokio::test]
async fn approval_rules_judge_the_normalized_manifest() {
    let e = env();
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([
            call(
                "c1",
                "naive_write",
                json!({"path": "drafts/a.txt", "content": "x"}),
            ),
            call(
                "c2",
                "naive_write",
                json!({"path": "drafts/../final.txt", "content": "x"}),
            ),
        ]),
        text("done"),
    ]));
    let agent = Agent::builder("rules")
        .model("m")
        .provider(provider)
        .tool(NaiveWrite {
            root: e.root.clone(),
        })
        .hook(ScopeGuard::new([&e.root]))
        .hook(
            ApprovalPolicy::new(ApprovalMode::AllWrites)
                .base_dir(&e.root)
                .with_rules(Arc::new(DraftsOnly {
                    drafts: e.root.join("drafts"),
                })),
        )
        .allow_unguarded()
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    // The rule sees `<root>/drafts/a.txt`, so the draft runs unasked.
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);
    assert!(
        matches!(&out.status, RunStatus::Interrupted { interruption } if interruption.call.id == "c2"),
        "drafts/../final.txt is not under drafts/: {:?}",
        out.status
    );
    assert!(!e.root.join("final.txt").exists());
}
