//! The shipped scenarios against a live model: the plain kernel first, then
//! each policy from `aidaemon-policies` on its own, compared with
//! [`aidaemon_eval::compare`] (Decision 4: a policy stays only if it shows a
//! gain).
//!
//! ```text
//! ANTHROPIC_API_KEY=... AIDAEMON_MODEL=claude-haiku-5-5 EVAL_N=5 \
//!   cargo run -p aidaemon-eval --features examples --example live_policies -- <out_dir> [config ...]
//! ```
//! Configs: `baseline` (default), `completion_check`, `streak_guard`,
//! `untrusted_output`, `redactor`, `context_window`, `context_window_8k`,
//! `egress_guard`, `scope_guard`, `approval`, `all`. Each writes
//! `<out_dir>/<config>.json` and prints its report, plus a comparison against
//! `<out_dir>/baseline.json` when present. Every agent also has the
//! scenarios' own `DenyDestructive` fixture hook (unless a scenario is
//! tagged `allow_destructive`). `EVAL_ONLY=name,name` runs only those
//! scenarios.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use aidaemon_core::{Hook, Provider};
use aidaemon_eval::examples::{ExampleExpect, ExampleFactory, ExampleSubject};
use aidaemon_eval::{compare, NoExtension, Report, Suite};
use aidaemon_policies::{
    ApprovalMode, ApprovalPolicy, ContextWindow, EgressGuard, ReceiptCompletionCheck, Redactor,
    Requirement, SameToolStreakGuard, ScopeGuard, StreakLimits, UntrustedOutput,
};
use aidaemon_providers::Anthropic;

const CONFIGS: &[&str] = &[
    "baseline",
    "completion_check",
    "streak_guard",
    "untrusted_output",
    "redactor",
    "context_window",
    "context_window_8k",
    "egress_guard",
    "scope_guard",
    "approval",
    "all",
];

fn policy(name: &str) -> Vec<Arc<dyn Hook>> {
    match name {
        "baseline" => vec![],
        "completion_check" => vec![Arc::new(ReceiptCompletionCheck::new([
            Requirement::RequireNoUnresolvedFailures,
        ]))],
        "streak_guard" => vec![Arc::new(SameToolStreakGuard {
            read_only_tools: HashSet::new(),
            allowances: HashMap::new(),
            limits: StreakLimits::default(),
        })],
        "untrusted_output" => vec![Arc::new(UntrustedOutput::default())],
        "redactor" => vec![Arc::new(Redactor::default())],
        "context_window" => vec![Arc::new(ContextWindow::new(120_000))],
        // Below the 10k window `context_trail` gives the model.
        "context_window_8k" => vec![Arc::new(ContextWindow::new(8_000))],
        "egress_guard" => vec![Arc::new(EgressGuard::new())],
        // `scope_outside` declares /workspace/app as its workspace.
        "scope_guard" => vec![Arc::new(ScopeGuard::new([
            std::env::temp_dir(),
            PathBuf::from("/workspace/app"),
        ]))],
        "approval" => vec![Arc::new(ApprovalPolicy::new(ApprovalMode::DestructiveOnly))],
        // Registration order as `aidaemon-policies` documents it.
        "all" => ["scope_guard", "egress_guard", "streak_guard", "redactor"]
            .into_iter()
            .chain(["untrusted_output", "context_window", "completion_check"])
            .flat_map(policy)
            .collect(),
        other => panic!("unknown config `{other}`; one of {CONFIGS:?}"),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: live_policies <out_dir> [config ...]")?,
    );
    let mut configs: Vec<String> = args.collect();
    if configs.is_empty() {
        configs.push("baseline".into());
    }
    std::fs::create_dir_all(&out)?;
    let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| "set ANTHROPIC_API_KEY")?;
    let model = std::env::var("AIDAEMON_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".into());
    let n: usize = std::env::var("EVAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let provider: Arc<dyn Provider> = Arc::new(Anthropic::builder().api_key(key).build()?);
    let mut suite: Suite<NoExtension, ExampleExpect> =
        Suite::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/scenarios"))?
            .live()
            .parallelism(4);
    if let Ok(only) = std::env::var("EVAL_ONLY") {
        let names: Vec<&str> = only.split(',').map(str::trim).collect();
        suite.scenarios.retain(|s| names.contains(&s.name.as_str()));
        if suite.scenarios.is_empty() {
            return Err(format!("EVAL_ONLY matched no scenario: {only}").into());
        }
    }

    for config in configs {
        let name = config.clone();
        let _ = policy(&name); // reject unknown names before spending anything
        let factory = ExampleFactory::with_live_provider(provider.clone())
            .model(model.clone())
            .with_hooks(move |_| policy(&name));
        let report = suite.run_subject(&ExampleSubject::new(&factory), n).await;
        println!("## {config} ({model}, N={n})\n\n{}", report.to_markdown());
        std::fs::write(out.join(format!("{config}.json")), report.to_json())?;
        let baseline = out.join("baseline.json");
        if config != "baseline" && baseline.exists() {
            let base = Report::from_json(&std::fs::read_to_string(baseline)?)?;
            println!("{}", compare(&base, &report).to_markdown());
        }
    }
    Ok(())
}
