//! `#[tool]`: the schema it derives, the annotations it declares, the calls
//! it makes, and the `Tool` conformance kit run on macro-made tools.

use aidaemon_sdk::conformance::check_tool;
use aidaemon_sdk::testing::tool_context;
use aidaemon_sdk::{
    tool, AccessManifest, CallEffects, EffectClass, EffectKey, SourceTrust, Tool, ToolAnnotations,
    ToolContext, ToolOutput,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

/// Look up an order: customer, item and total.
///
/// Read-only.
#[tool(read_only)]
async fn lookup_order(
    /// The order id, like ORD-1001.
    order_id: String,
    /// How many lines to show.
    limit: Option<u32>,
) -> String {
    format!("{order_id}: Acme Kettle (synthetic), limit {limit:?}")
}

/// Delete a note from the synthetic notes folder.
#[tool]
fn delete_note(path: String) -> Result<ToolOutput, String> {
    if path.contains("..") {
        return Err("path leaves the folder".into());
    }
    Ok(ToolOutput::text(format!("deleted {path}")))
}

/// Record a tag on the current session.
#[tool(write, idempotent, trusted, name = "tag-session", timeout_secs = 5)]
async fn tag(tag: String, ctx: &ToolContext) -> String {
    format!("{} tagged {tag}", ctx.session_id)
}

/// A refund request.
#[derive(Deserialize, JsonSchema)]
struct Refund {
    /// The order to refund.
    order_id: String,
    amount_cents: u64,
}

fn refund_effects(args: &Refund) -> CallEffects {
    CallEffects {
        keys: vec![EffectKey::named(EffectClass::Credential, &args.order_id)],
        complete: true,
    }
}

fn refund_access(args: &Refund) -> AccessManifest {
    AccessManifest {
        hosts: vec![format!("payments.example.com/{}", args.order_id)],
        ..Default::default()
    }
}

/// Refund part or all of an order (simulated).
#[tool(destructive, effects = refund_effects, access = refund_access)]
async fn issue_refund(ctx: &ToolContext, #[args] refund: Refund) -> Value {
    let _ = ctx;
    json!({"refunded": refund.amount_cents, "order_id": refund.order_id})
}

/// Fetch a public page.
#[tool(read_only, egress = true)]
async fn fetch_page(url: String) -> String {
    url
}

#[test]
fn the_spec_is_the_full_function_object() {
    let spec = LookupOrder.spec();
    assert_eq!(spec.name, "lookup_order");
    assert_eq!(
        spec.description,
        "Look up an order: customer, item and total.\n\nRead-only."
    );
    let params = &spec.parameters;
    assert_eq!(params["type"], "object");
    assert_eq!(params["additionalProperties"], false);
    assert_eq!(params["required"], json!(["order_id"]));
    assert_eq!(
        params["properties"]["order_id"]["description"],
        "The order id, like ORD-1001."
    );
    assert!(params["properties"]["limit"].is_object());
    assert!(params.get("title").is_none() && params.get("$schema").is_none());
}

#[test]
fn annotations_are_explicit_and_default_to_the_most_conservative() {
    assert_eq!(LookupOrder.annotations(), ToolAnnotations::READ_ONLY);
    // No preset: the same as a hand-written tool's default.
    assert_eq!(DeleteNote.annotations(), ToolAnnotations::default());
    assert_eq!(DeleteNote.annotations(), ToolAnnotations::DESTRUCTIVE);
    assert_eq!(
        Tag.annotations(),
        ToolAnnotations {
            idempotent: true,
            ..ToolAnnotations::WRITE
        }
    );
    assert!(FetchPage.annotations().egress && FetchPage.annotations().read_only);
    assert_eq!(
        FetchPage.effects(&json!({"url": "x"})),
        CallEffects::unknown()
    );
    assert_eq!(
        LookupOrder.effects(&json!({"order_id": "a"})),
        CallEffects::none()
    );

    assert_eq!(LookupOrder.source_trust(&json!({})), SourceTrust::Untrusted);
    assert_eq!(Tag.source_trust(&json!({})), SourceTrust::Trusted);
    assert_eq!(Tag.spec().name, "tag-session");
    assert_eq!(Tag.timeout(), Some(std::time::Duration::from_secs(5)));
    assert_eq!(LookupOrder.timeout(), None);
}

#[test]
fn effects_and_access_come_from_the_typed_arguments() {
    let args = json!({"order_id": "ORD-1001", "amount_cents": 500});
    assert_eq!(
        IssueRefund.effects(&args).keys,
        vec![EffectKey::named(EffectClass::Credential, "ORD-1001")]
    );
    assert_eq!(
        IssueRefund.access(&args).hosts,
        vec!["payments.example.com/ORD-1001".to_string()]
    );
    // Arguments that do not parse: undeclared, never "nothing".
    assert_eq!(IssueRefund.effects(&json!({})), CallEffects::unknown());
    assert!(IssueRefund.access(&json!({})).is_empty());
    assert_eq!(IssueRefund.spec().parameters["additionalProperties"], false);
}

#[test]
fn validate_refuses_what_the_schema_does_not_allow() {
    assert!(LookupOrder.validate(&json!({"order_id": "a"})).is_ok());
    assert!(LookupOrder.validate(&json!({})).is_err());
    assert!(LookupOrder.validate(&json!("a")).is_err());
    let err = LookupOrder
        .validate(&json!({"order_id": "a", "admin": true}))
        .unwrap_err();
    assert!(err.contains("admin"), "{err}");
    // `#[args]` types without `deny_unknown_fields` are just as closed.
    let err = IssueRefund
        .validate(&json!({"order_id": "a", "amount_cents": 1, "extra": 1}))
        .unwrap_err();
    assert!(err.contains("extra"), "{err}");
    assert!(IssueRefund
        .validate(&json!({"order_id": "a", "amount_cents": -1}))
        .is_err());
}

#[tokio::test]
async fn calls_pass_typed_arguments_and_the_context() {
    let ctx = tool_context();
    let out = LookupOrder
        .call(json!({"order_id": "ORD-1001", "limit": 3}), &ctx)
        .await
        .unwrap();
    assert_eq!(
        out.content,
        "ORD-1001: Acme Kettle (synthetic), limit Some(3)"
    );

    let out = Tag.call(json!({"tag": "vip"}), &ctx).await.unwrap();
    assert_eq!(out.content, "test:synthetic-session tagged vip");

    let out = IssueRefund
        .call(json!({"order_id": "ORD-1", "amount_cents": 250}), &ctx)
        .await
        .unwrap();
    assert_eq!(out.details["refunded"], 250);

    // A bad argument object is an error result the model can react to.
    let out = LookupOrder.call(json!({"oops": 1}), &ctx).await.unwrap();
    assert!(out.is_error, "{out:?}");

    // A returned error is a failed call.
    let err = DeleteNote
        .call(json!({"path": "../etc"}), &ctx)
        .await
        .unwrap_err();
    assert_eq!(err.0, "path leaves the folder");
    let out = DeleteNote
        .call(json!({"path": "a.txt"}), &ctx)
        .await
        .unwrap();
    assert_eq!(out.content, "deleted a.txt");
}

#[test]
fn macro_made_tools_pass_the_conformance_kit() {
    let tools: [&dyn Tool; 5] = [&LookupOrder, &DeleteNote, &Tag, &IssueRefund, &FetchPage];
    for tool in tools {
        check_tool(tool).assert_all_passed();
    }
}

#[tokio::test]
async fn a_macro_made_tool_runs_in_an_agent() {
    use aidaemon_sdk::testing::{call, text, tool_calls, ScriptedProvider};
    let agent = aidaemon_sdk::Agent::builder("macro-test")
        .allow_unguarded()
        .model("scripted")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "lookup_order", json!({"order_id": "ORD-7"}))]),
            text("done"),
        ]))
        .tool(LookupOrder)
        .build()
        .unwrap();
    let outcome = agent
        .run("test:synthetic-user", "look it up")
        .await
        .unwrap();
    assert!(outcome.is_completed());
    assert_eq!(outcome.receipts().len(), 1);
    assert_eq!(agent.tool_specs()[0], LookupOrder.spec());
}
