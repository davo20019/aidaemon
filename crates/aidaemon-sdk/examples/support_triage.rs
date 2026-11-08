//! A customer-support triage agent built only from SDK crates. It has no
//! file, shell or web tools: it reads an order, may issue a refund, and may
//! escalate a ticket to a person. Nothing it does is a coding task.
//!
//! The owner approves every write on the terminal: refunds and escalations
//! pause the run until someone answers `y`. Refunds in this example are
//! simulated; the order data is synthetic.
//!
//! ```text
//! ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example support_triage
//! OPENAI_API_KEY=...    cargo run -p aidaemon-sdk --example support_triage
//! ```
//! Optional: AIDAEMON_MODEL (model id), OPENAI_BASE_URL (any OpenAI-compatible
//! server), and a ticket as the first argument.

use std::io::{BufRead, Write};
use std::sync::Arc;

use aidaemon_sdk::policies::{
    ApprovalMode, ApprovalPolicy, EgressGuard, Redactor, ScopeGuard, UntrustedOutput,
};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::providers::{Anthropic, OpenAiCompatible};
use aidaemon_sdk::{Authority, AuthorityHook, FnTool, MemoryStore};
use serde_json::{json, Value};

const DEFAULT_TICKET: &str = "Hi, this is Alice Rivera (order ORD-1001). The Acme Blender \
    arrived with a cracked jar. I'd like a refund for the full amount, please.";

/// Synthetic order book. A real deployment would call the order system.
fn order(id: &str) -> Option<Value> {
    match id {
        "ORD-1001" => Some(json!({
            "order_id": "ORD-1001",
            "customer": "Alice Rivera",
            "item": "Acme Blender",
            "total_cents": 8900,
            "status": "delivered",
            "refunded_cents": 0
        })),
        "ORD-1002" => Some(json!({
            "order_id": "ORD-1002",
            "customer": "Jordan Lee",
            "item": "Acme Kettle",
            "total_cents": 4500,
            "status": "shipped",
            "refunded_cents": 0
        })),
        _ => None,
    }
}

fn tools() -> Vec<Arc<dyn Tool>> {
    let lookup_order = FnTool::new(
        ToolSpec {
            name: "lookup_order".into(),
            description: "Read an order: customer, item, total, status and refunds so far.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": false
            }),
        },
        ToolAnnotations::READ_ONLY,
        |args| async move {
            let id = args["order_id"].as_str().unwrap_or_default();
            Ok(match order(id) {
                Some(found) => ToolOutput::text(found.to_string()),
                None => ToolOutput::error(format!("no order {id}")),
            })
        },
    );

    let issue_refund = FnTool::new(
        ToolSpec {
            name: "issue_refund".into(),
            description: "Refund part or all of an order. Moves money: use only when the \
                          customer's claim is supported by the order."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount_cents": {"type": "integer", "minimum": 1}
                },
                "required": ["order_id", "amount_cents"],
                "additionalProperties": false
            }),
        },
        ToolAnnotations::DESTRUCTIVE,
        |args| async move {
            let id = args["order_id"].as_str().unwrap_or_default();
            let amount = args["amount_cents"].as_i64().unwrap_or_default();
            match order(id) {
                Some(found) if amount <= found["total_cents"].as_i64().unwrap_or(0) => {
                    // Simulated: a real tool would call the payment provider here.
                    Ok(ToolOutput::text(format!(
                        "refunded {amount} cents on {id} (simulated)"
                    )))
                }
                Some(_) => Ok(ToolOutput::error("refund exceeds the order total")),
                None => Ok(ToolOutput::error(format!("no order {id}"))),
            }
        },
    );

    let escalate_ticket = FnTool::new(
        ToolSpec {
            name: "escalate_ticket".into(),
            description: "Hand the ticket to a human agent with a short reason.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": false
            }),
        },
        ToolAnnotations::WRITE,
        |args| async move {
            let reason = args["reason"].as_str().unwrap_or_default();
            // Simulated: a real tool would open a ticket in the helpdesk.
            Ok(ToolOutput::text(format!(
                "escalated to a human agent: {reason}"
            )))
        },
    );

    vec![
        Arc::new(lookup_order),
        Arc::new(issue_refund),
        Arc::new(escalate_ticket),
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ticket = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_TICKET.to_string());

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

    let store = Arc::new(MemoryStore::new());
    let mut builder = provider(Agent::builder("support-triage"))
        .model(model)
        .instructions(
            "You triage customer-support tickets for Acme Corp. Look up the order before \
             deciding anything. Refund only when the order supports the claim, and never \
             more than the order total. Escalate to a human when the claim is unclear or \
             the amount is large. Reply to the customer in two or three plain sentences \
             that say what was done.",
        )
        // Every write (refund or escalation) asks the owner first.
        .hook(AuthorityHook::new(Authority::full()))
        .hook(ScopeGuard::new(Vec::<std::path::PathBuf>::new()))
        .hook(EgressGuard::new())
        .hook(ApprovalPolicy::new(ApprovalMode::AllWrites))
        .hook(Redactor::default())
        .hook(UntrustedOutput::default())
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
    for tool in tools() {
        builder = builder.tool_arc(tool);
    }
    let agent = builder.build()?;

    let mut outcome = agent.run("support:ticket-1", &ticket).await?;
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
                reason: "the owner declined".into(),
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
