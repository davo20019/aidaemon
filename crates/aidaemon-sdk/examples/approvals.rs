//! Approval and decline: refunds pause the run until the owner answers.
//! This example's owner is a rule (approve up to $100, decline above);
//! a real one answers on a terminal, a channel button or a web page. After
//! a decline the kernel refuses any call that would do the same thing.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example approvals --features testing
use aidaemon_sdk::policies::ApprovalMode;
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use serde_json::json;

/// Refund part or all of an order. Moves money (simulated here).
#[tool(destructive)]
async fn issue_refund(order_id: String, amount_cents: u32) -> String {
    format!("refunded {amount_cents} cents on {order_id} (simulated)")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = aidaemon_sdk::preset("refunds")
        .provider_from_env_or(scripted())
        .instructions("You process refund requests. Refund what the customer asks for.")
        .approvals(ApprovalMode::AllWrites)
        .tool(IssueRefund)
        .build()
        .await?;

    for (session, ticket) in [
        (
            "support:synthetic-1",
            "Refund $40.00 on ORD-1001, the jar was cracked.",
        ),
        (
            "support:synthetic-2",
            "Refund $900.00 on ORD-1002, I changed my mind.",
        ),
    ] {
        let mut outcome = agent.run(session, ticket).await?;
        while let RunStatus::Interrupted { interruption } = &outcome.status {
            let cents = interruption.call.arguments["amount_cents"]
                .as_u64()
                .unwrap_or(0);
            let resolution = if cents <= 10_000 {
                println!("owner approves {}", interruption.call.arguments);
                Resolution::Approve
            } else {
                println!("owner declines {}", interruption.call.arguments);
                Resolution::Deny {
                    reason: "Refunds over $100 need a manager.".into(),
                }
            };
            outcome = agent.resume(outcome.state, resolution).await?;
        }
        let succeeded = outcome.is_succeeded(); // false once the owner declined something
        println!(
            "succeeded: {succeeded}: {}\n",
            outcome.final_text.unwrap_or_default()
        );
    }
    Ok(())
}

fn scripted() -> ScriptedProvider {
    let refund = |id: &str, order: &str, cents: u32| {
        tool_calls([call(
            id,
            "issue_refund",
            json!({"order_id": order, "amount_cents": cents}),
        )])
    };
    ScriptedProvider::new([
        refund("c1", "ORD-1001", 4000),
        text("Done: $40.00 is on its way back to you."),
        refund("c2", "ORD-1002", 90000),
        text("I can't refund $900.00 myself; a manager will review it."),
    ])
}
