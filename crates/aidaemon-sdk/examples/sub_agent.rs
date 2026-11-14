//! A sub-agent: the parent delegates a bounded research task to a child
//! agent that it calls like any other tool. The child runs in its own
//! session with its own budget; the parent gets its condensed answer, and
//! the child's token use counts against the parent's budget.
//!
//! A delegation declares that it changes no file itself (each of the
//! child's calls is judged by the child's own hooks), so under the preset
//! it runs without asking the owner. `.read_only()` lets several
//! delegations in one model turn run at the same time; without it they run
//! one after another. A child that fails becomes a failed call for the
//! parent's model, not a failed parent run: read how each ended from
//! `outcome.tool_results()`. To fan out without a model choosing, use
//! `agents::run_all`, shown at the end.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example sub_agent --features testing
use aidaemon_sdk::agents::{AgentTool, ChildBudget, run_all};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use serde_json::json;

/// Search the synthetic product catalog.
#[tool(read_only)]
async fn search_catalog(query: String) -> String {
    format!("3 results for {query:?}: Acme Kettle $45, Acme Blender $89, Acme Toaster $39")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let researcher = aidaemon_sdk::preset("researcher")
        .provider_from_env_or(ScriptedProvider::new([
            tool_calls([call(
                "r1",
                "search_catalog",
                json!({"query": "kitchen under $50"}),
            )]),
            text("Under $50: Acme Kettle ($45) and Acme Toaster ($39)."),
        ]))
        .instructions("You research the product catalog and answer in one sentence.")
        .tool(SearchCatalog)
        .build()
        .await?;
    let delegate = AgentTool::new(researcher)
        .budget(ChildBudget {
            max_turns: 8,
            max_tokens: Some(50_000),
            ..Default::default()
        })
        .description("Ask the researcher a question about the product catalog.")
        .read_only();

    let parent = aidaemon_sdk::preset("concierge")
        .provider_from_env_or(ScriptedProvider::new([
            tool_calls([call(
                "c1",
                "researcher",
                json!({"task": "Kitchen items under $50?"}),
            )]),
            text("For under $50 I'd pick the Acme Kettle ($45) or the Acme Toaster ($39)."),
        ]))
        .instructions("You help shoppers. Delegate catalog questions to the researcher.")
        .tool(delegate)
        .build()
        .await?;

    let outcome = parent
        .run(
            "shop:synthetic-user-1",
            "A gift for the kitchen, under $50?",
        )
        .await?;
    // Each delegation: how it ended and the answer the parent's model saw.
    for result in outcome.tool_results() {
        println!(
            "[{}] {:?}: {}",
            result.receipt.tool,
            result.receipt.status,
            result.text.unwrap_or_default()
        );
    }
    println!(
        "billable tokens, child included: {}",
        outcome.usage().billable_tokens()
    );
    println!("{}", outcome.final_text.unwrap_or_default());

    // Fan out without a model: two tasks at once, results in order.
    let helper = aidaemon_sdk::preset("summarizer")
        .provider_from_env_or(
            ScriptedProvider::new([]).then_repeat(text("A short synthetic summary.")),
        )
        .build()
        .await?;
    let tasks = [
        "Summarize the kettle reviews",
        "Summarize the toaster reviews",
    ];
    let outcomes = run_all(
        "shop:synthetic-user-1",
        tasks.iter().map(|task| (&helper, task.to_string())),
    )
    .await;
    for (task, outcome) in tasks.iter().zip(outcomes) {
        let answer = outcome?.final_text.unwrap_or_default();
        println!("{task}: {answer}");
    }
    Ok(())
}
