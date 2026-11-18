//! Custom tools with `#[tool]`: the schema comes from the parameters (or a
//! typed args struct), the description from the doc comments, and the
//! annotations are written out. Synthetic order data.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example custom_tool --features testing
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// Read an order: customer, item, total and status.
#[tool(read_only)]
async fn lookup_order(
    /// The order id, like ORD-1001.
    order_id: String,
) -> Result<String, String> {
    match order_id.as_str() {
        "ORD-1001" => Ok("Alice Rivera, Acme Blender, $89.00, delivered".into()),
        _ => Err(format!("no order {order_id}")),
    }
}

/// A hand-off to a person.
#[derive(Deserialize, JsonSchema)]
struct Escalation {
    /// Why a person should look at it, in one sentence.
    reason: String,
    /// 1 (urgent) to 3 (whenever).
    #[schemars(range(min = 1, max = 3))]
    priority: u8,
}

/// Hand the ticket to a human agent.
#[tool(write, idempotent = false)]
async fn escalate(#[args] escalation: Escalation, ctx: &ToolContext) -> String {
    format!(
        "ticket for {} escalated at priority {}: {}",
        ctx.session_id, escalation.priority, escalation.reason
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&Escalate.spec().parameters)?
    );
    let agent = aidaemon_sdk::preset("support")
        .provider_from_env_or(scripted())
        .instructions("You triage support tickets. Look the order up, then escalate damage claims.")
        .tool(LookupOrder)
        .tool(Escalate)
        // Escalating opens a ticket and touches no files. Without this, a
        // tool that may change things and declares no paths waits for the
        // owner, since no scope check covers it.
        .path_free_tool("escalate")
        .build()
        .await?;
    let ticket = "Order ORD-1001 arrived with a cracked jar.";
    let outcome = agent.run("support:synthetic-ticket-1", ticket).await?;
    for receipt in outcome.receipts() {
        println!("[{}] {:?}", receipt.tool, receipt.status);
    }
    println!("{}", outcome.final_text.unwrap_or_default());
    Ok(())
}

fn scripted() -> ScriptedProvider {
    ScriptedProvider::new([
        tool_calls([call("c1", "lookup_order", json!({"order_id": "ORD-1001"}))]),
        tool_calls([call(
            "c2",
            "escalate",
            json!({"reason": "Delivered blender has a cracked jar.", "priority": 2}),
        )]),
        text("I've passed your cracked-jar claim to a colleague, who will be in touch."),
    ])
}
