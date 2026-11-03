//! A small coding agent built only from SDK crates.
//!
//! It works inside one directory, asks on the terminal before destructive
//! actions, keeps runs in SQLite, and will not accept "done" while a tool
//! failure is unresolved.
//!
//! ```text
//! ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example coding_agent -- <dir> "<task>"
//! OPENAI_API_KEY=...    cargo run -p aidaemon-sdk --example coding_agent -- <dir> "<task>"
//! ```
//! Optional: AIDAEMON_MODEL (model id), OPENAI_BASE_URL (any OpenAI-compatible server).

use std::io::{BufRead, Write};
use std::sync::Arc;

use aidaemon_sdk::policies::{
    ApprovalMode, ApprovalPolicy, ContextWindow, ProjectInstructions, ReceiptCompletionCheck,
    Redactor, Requirement, ScopeGuard, UntrustedOutput,
};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::providers::{Anthropic, OpenAiCompatible};
use aidaemon_sdk::store::SqliteStore;
use aidaemon_sdk::tools::{default_tools, Workspace};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(task)) = (args.next(), args.next()) else {
        eprintln!("usage: coding_agent <dir> \"<task>\"");
        std::process::exit(2);
    };

    let (provider, default_model): (Box<dyn FnOnce(AgentBuilder) -> AgentBuilder>, &str) =
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            let p = Anthropic::builder().api_key(key).build()?;
            (Box::new(move |b| b.provider(p)), "claude-sonnet-5-5")
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            let mut builder = OpenAiCompatible::builder().api_key(key);
            if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
                builder = builder.base_url(url);
            }
            let p = builder.build()?;
            (Box::new(move |b| b.provider(p)), "gpt-5")
        } else {
            eprintln!("set ANTHROPIC_API_KEY or OPENAI_API_KEY");
            std::process::exit(2);
        };
    let model = std::env::var("AIDAEMON_MODEL").unwrap_or_else(|_| default_model.to_string());

    let workspace = Workspace::new(&dir)?;
    let store = Arc::new(SqliteStore::open(workspace.root().join(".aidaemon-runs.db")).await?);

    let mut builder = provider(Agent::builder("coding-agent"))
        .model(model)
        .instructions(
            "You are a careful software engineer working in the user's repository. \
             Read before editing, make the smallest change that solves the task, \
             and verify with the project's own tests before saying you are done.",
        )
        .hook(ProjectInstructions::discover(workspace.root())?)
        .hook(ContextWindow::new(120_000))
        .hook(ApprovalPolicy::new(ApprovalMode::DestructiveOnly))
        .hook(ScopeGuard::new([workspace.root()]))
        .hook(Redactor::default())
        .hook(UntrustedOutput::default())
        .hook(ReceiptCompletionCheck::new([
            Requirement::RequireNoUnresolvedFailures,
        ]))
        .store(store)
        .event_sink(|_run, event: &Event| match event {
            Event::TextDelta { text, .. } => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            Event::ToolFinished { receipt } => {
                println!("\n  [{}] {:?}", receipt.tool, receipt.status)
            }
            _ => {}
        });
    for tool in default_tools(workspace) {
        builder = builder.tool_arc(tool);
    }
    let agent = builder.build()?;

    let mut outcome = agent.run("cli:local", &task).await?;
    while let RunStatus::Interrupted { interruption } = &outcome.status {
        print!(
            "\nAllow `{}` {}? [y/N] ",
            interruption.call.name, interruption.call.arguments
        );
        std::io::stdout().flush()?;
        let mut answer = String::new();
        std::io::stdin().lock().read_line(&mut answer)?;
        let resolution = if answer.trim().eq_ignore_ascii_case("y") {
            Resolution::Approve
        } else {
            Resolution::Deny {
                reason: "the user declined".into(),
            }
        };
        outcome = agent.resume(outcome.state, resolution).await?;
    }

    let usage = outcome.usage();
    println!(
        "\n\n{:?} after {} turns, {} billable tokens, {} tool calls",
        outcome.status,
        outcome.turns(),
        usage.billable_tokens(),
        outcome.receipts().len()
    );
    Ok(())
}
