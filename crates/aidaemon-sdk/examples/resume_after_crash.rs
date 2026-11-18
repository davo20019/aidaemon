//! Resume after a crash. Runs are journaled in SQLite as they go; this
//! example starts itself as a child process that dies in the middle of a
//! tool call, then finds the unfinished run in the database and finishes
//! it. A call declared idempotent is re-run; any other is reported to the
//! model as indeterminate, never silently repeated.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example resume_after_crash --features testing
use aidaemon_sdk::CancellationToken;
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::store::SqliteStore;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use serde_json::json;

const CRASH: &str = "AIDAEMON_EXAMPLE_CRASH";

/// Build the weekly sales report (synthetic data). Safe to run twice.
#[tool(write, idempotent)]
async fn build_report(week: u32) -> String {
    if std::env::var_os(CRASH).is_some() {
        eprintln!("child: crashing in the middle of build_report");
        std::process::exit(1);
    }
    format!("week {week}: 412 orders, $18,240 revenue (synthetic)")
}

async fn agent(
    db: &std::path::Path,
    script: ScriptedProvider,
) -> Result<Agent, aidaemon_sdk::PresetError> {
    aidaemon_sdk::preset("reporter")
        .provider_from_env_or(script)
        .instructions("Build the report you are asked for, then summarize it in one sentence.")
        .tool(BuildReport)
        // The report touches no files: run it without asking the owner.
        .path_free_tool("build_report")
        .sqlite(db)
        .build()
        .await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("--child") {
        let db = std::path::PathBuf::from(args.next().expect("db path"));
        let script = ScriptedProvider::new([tool_calls([call(
            "c1",
            "build_report",
            json!({"week": 45}),
        )])]);
        agent(&db, script)
            .await?
            .run("cli:synthetic-user-1", "Report for week 45")
            .await?;
        unreachable!("the tool exits the process");
    }

    let dir = tempfile::tempdir()?;
    let db = dir.path().join("runs.db");
    let status = std::process::Command::new(std::env::current_exe()?)
        .arg("--child")
        .arg(&db)
        .env(CRASH, "1")
        .status()?;
    println!("child exited with {status}");

    let store = SqliteStore::open(&db).await?;
    let unfinished = store.in_flight_runs().await?;
    store.close().await;
    println!("unfinished runs in the database: {}", unfinished.len());

    let script = ScriptedProvider::new([text("Week 45: 412 orders and $18,240 in revenue.")]);
    let agent = agent(&db, script).await?;
    for state in unfinished {
        let outcome = agent.run_from(state, CancellationToken::new()).await?;
        println!(
            "{:?}: {}",
            outcome.status,
            outcome.final_text.unwrap_or_default()
        );
    }
    Ok(())
}
