//! `track_requirements`: the model's own checklist, plus a stop hook that
//! refuses to finish while items are open.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use aidaemon_core::{
    CallEffects, Hook, HookContext, StopDecision, Tool, ToolAnnotations, ToolContext, ToolOutput,
    ToolResult, ToolSpec,
};

use crate::util::{fail, parse_args};

/// Most items a checklist may hold.
pub const MAX_ITEMS: usize = 100;

/// Where an item stands. The model sets `pending`, `in_progress`, `done`,
/// `deferred` and `superseded`; only the application sets `waived`
/// ([`ChecklistState::waive`]), because only the owner can drop scope.
///
/// Stopping and succeeding are separate: `deferred` lets the run stop, but
/// a deferred item is unmet, so the run does not count as succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    Pending,
    InProgress,
    Done,
    /// The model is not doing it in this run; needs a `reason`. Unmet.
    Deferred,
    /// Replaced by another item (`superseded_by`); met only once that item
    /// is done.
    Superseded,
    /// The owner dropped it from scope; set by the application only.
    Waived,
}

impl ItemStatus {
    fn marker(self) -> &'static str {
        match self {
            Self::Pending => "[ ]",
            Self::InProgress => "[~]",
            Self::Done => "[x]",
            Self::Deferred => "[-]",
            Self::Superseded => "[>]",
            Self::Waived => "[w]",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChecklistItem {
    pub text: String,
    pub status: ItemStatus,
    /// Why the item is deferred or waived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// For `superseded`: the 1-based number of the item that replaces it,
    /// in the same list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<usize>,
}

impl ChecklistItem {
    pub fn new(text: impl Into<String>, status: ItemStatus) -> Self {
        Self {
            text: text.into(),
            status,
            reason: None,
            superseded_by: None,
        }
    }
}

/// How one item resolves once the run stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resolution {
    /// Still being worked on (directly, or through the item replacing it).
    Open,
    Met,
    Unmet,
}

/// Follow `superseded_by` to the item that settles `index`. A superseded
/// item may only point at an item that is not itself superseded, so this is
/// one hop at most; anything else resolves as unmet.
fn resolve(items: &[ChecklistItem], index: usize) -> Resolution {
    let item = &items[index];
    match item.status {
        ItemStatus::Pending | ItemStatus::InProgress => Resolution::Open,
        ItemStatus::Done | ItemStatus::Waived => Resolution::Met,
        ItemStatus::Deferred => Resolution::Unmet,
        ItemStatus::Superseded => {
            let target = item
                .superseded_by
                .and_then(|n| n.checked_sub(1))
                .filter(|&t| t != index)
                .filter(|&t| {
                    items
                        .get(t)
                        .is_some_and(|i| i.status != ItemStatus::Superseded)
                });
            match target {
                Some(t) => resolve(items, t),
                None => Resolution::Unmet,
            }
        }
    }
}

/// Shared checklist state. Clone it to share between the tool, the gate and
/// your own code. Use one per run or session.
#[derive(Debug, Clone, Default)]
pub struct ChecklistState {
    items: Arc<Mutex<Vec<ChecklistItem>>>,
}

impl ChecklistState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn items(&self) -> Vec<ChecklistItem> {
        self.items.lock().expect("checklist poisoned").clone()
    }

    /// Items work is still expected on: pending or in progress, or
    /// superseded by one that is.
    pub fn open_items(&self) -> Vec<ChecklistItem> {
        let items = self.items();
        (0..items.len())
            .filter(|&i| resolve(&items, i) == Resolution::Open)
            .map(|i| items[i].clone())
            .collect()
    }

    /// Items that will not be met in this run: deferred, or superseded by
    /// an item that will not be met.
    pub fn unmet_items(&self) -> Vec<ChecklistItem> {
        let items = self.items();
        (0..items.len())
            .filter(|&i| resolve(&items, i) == Resolution::Unmet)
            .map(|i| items[i].clone())
            .collect()
    }

    /// Replace the list as the application (no provenance checks). The
    /// model's updates go through the tool, which checks them.
    pub fn set(&self, items: Vec<ChecklistItem>) {
        *self.items.lock().expect("checklist poisoned") = items;
    }

    /// The owner drops an item from scope. Returns false when no item has
    /// this text. Only the application can do this; the model can defer an
    /// item but not waive it.
    pub fn waive(&self, text: &str, reason: impl Into<String>) -> bool {
        let mut items = self.items.lock().expect("checklist poisoned");
        let Some(item) = items.iter_mut().find(|i| i.text == text.trim()) else {
            return false;
        };
        item.status = ItemStatus::Waived;
        item.reason = Some(reason.into());
        item.superseded_by = None;
        true
    }
}

fn render(items: &[ChecklistItem]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| format!("{}. {}", i + 1, describe(item)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn describe(item: &ChecklistItem) -> String {
    let mut line = format!("{} {}", item.status.marker(), item.text);
    match item.status {
        ItemStatus::Deferred | ItemStatus::Waived => {
            if let Some(reason) = &item.reason {
                line.push_str(&format!(" ({}: {reason})", status_word(item.status)));
            }
        }
        ItemStatus::Superseded => {
            if let Some(n) = item.superseded_by {
                line.push_str(&format!(" (superseded by item {n})"));
            }
        }
        _ => {}
    }
    line
}

fn status_word(status: ItemStatus) -> &'static str {
    match status {
        ItemStatus::Pending => "pending",
        ItemStatus::InProgress => "in progress",
        ItemStatus::Done => "done",
        ItemStatus::Deferred => "deferred",
        ItemStatus::Superseded => "superseded",
        ItemStatus::Waived => "waived",
    }
}

/// Check a model's update against the current list. Every item that is not
/// done or waived must stay visible: removing an open, deferred or
/// superseded item would erase the record that it was not accomplished.
/// Only the owner waives; a waived item the model keeps stays waived.
fn validate(previous: &[ChecklistItem], items: &[ChecklistItem]) -> Result<(), String> {
    for (i, item) in items.iter().enumerate() {
        let n = i + 1;
        let reason = item.reason.as_deref().map(str::trim).unwrap_or("");
        let was_waived = previous
            .iter()
            .any(|p| p.text == item.text && p.status == ItemStatus::Waived);
        match item.status {
            ItemStatus::Waived if !was_waived => {
                return Err(format!(
                    "item {n} (\"{}\"): only the owner can waive an item. Mark it deferred \
                     with a reason if you are not doing it.",
                    item.text
                ));
            }
            ItemStatus::Deferred if reason.is_empty() => {
                return Err(format!(
                    "item {n} (\"{}\"): a deferred item needs a `reason`.",
                    item.text
                ));
            }
            ItemStatus::Superseded => {
                let target = item
                    .superseded_by
                    .filter(|&t| t >= 1 && t != n)
                    .and_then(|t| items.get(t - 1));
                match target {
                    None => {
                        return Err(format!(
                            "item {n} (\"{}\"): a superseded item needs `superseded_by`, the \
                             number of another item in this list that replaces it.",
                            item.text
                        ));
                    }
                    Some(t) if t.status == ItemStatus::Superseded => {
                        return Err(format!(
                            "item {n} (\"{}\"): it is superseded by an item that is itself \
                             superseded; point it at the final replacement.",
                            item.text
                        ));
                    }
                    Some(_) => {}
                }
            }
            _ if was_waived => {
                return Err(format!(
                    "item {n} (\"{}\") was waived by the owner; keep its status `waived`.",
                    item.text
                ));
            }
            _ => {}
        }
        if item.status != ItemStatus::Superseded && item.superseded_by.is_some() {
            return Err(format!(
                "item {n} (\"{}\"): `superseded_by` is only for superseded items.",
                item.text
            ));
        }
    }
    let missing: Vec<&ChecklistItem> = previous
        .iter()
        .filter(|p| !matches!(p.status, ItemStatus::Done | ItemStatus::Waived))
        .filter(|p| !items.iter().any(|i| i.text == p.text))
        .collect();
    if !missing.is_empty() {
        let list = missing
            .iter()
            .map(|p| format!("- {}", describe(p)))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "these items are not done and cannot be removed:\n{list}\nKeep each one in the \
             list: mark it done once it is finished, deferred with a reason if you are not \
             doing it, or superseded with `superseded_by` pointing at the item that replaces \
             it (to reword an item, add the new wording and supersede the old one)."
        ));
    }
    Ok(())
}

/// The `track_requirements` tool. Each call replaces the whole list, so the
/// model always states the full current plan.
#[derive(Debug, Clone, Default)]
pub struct Checklist {
    state: ChecklistState,
}

impl Checklist {
    pub fn new(state: ChecklistState) -> Self {
        Self { state }
    }

    /// A tool and its matching stop gate over fresh shared state.
    pub fn with_gate() -> (Self, ChecklistGate) {
        let state = ChecklistState::new();
        (Self::new(state.clone()), ChecklistGate::new(state))
    }

    pub fn state(&self) -> &ChecklistState {
        &self.state
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChecklistArgs {
    items: Vec<ChecklistItem>,
}

#[async_trait]
impl Tool for Checklist {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "track_requirements".into(),
            description: "Declare or update your checklist for the current task. Send the \
                 complete list every time (it replaces the previous one). Use it for \
                 multi-step work: list every requirement up front, keep one item in_progress, \
                 and mark an item done only after it is actually finished and verified. You \
                 cannot finish while items are pending or in_progress. If you will not do an \
                 item, mark it deferred with a reason: the run can then end, but it is \
                 reported as not fully done. If another item replaces one, mark the old one \
                 superseded with superseded_by set to the new item's number. Items that are \
                 not done cannot be removed from the list."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "maxItems": MAX_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "What must be true when this item is done."},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "done", "deferred", "superseded"]},
                                "reason": {"type": "string", "description": "Required for deferred: why it is not being done."},
                                "superseded_by": {"type": "integer", "minimum": 1, "description": "Required for superseded: the number (1-based) of the item in this list that replaces it."}
                            },
                            "required": ["text", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        // Changes only agent-internal state, but two updates in one turn
        // must not race, so it is not read-only.
        ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
            egress: false,
        }
    }

    /// Only the agent's own list changes; nothing outside the agent.
    fn effects(&self, _arguments: &Value) -> CallEffects {
        CallEffects::none()
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let args: ChecklistArgs = match parse_args("track_requirements", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        if args.items.len() > MAX_ITEMS {
            return Ok(fail(format!(
                "too many items ({}); at most {MAX_ITEMS}",
                args.items.len()
            )));
        }
        let mut items = args.items;
        for item in &mut items {
            item.text = item.text.trim().to_string();
            item.reason = item
                .reason
                .take()
                .map(|r| r.trim().to_string())
                .filter(|r| !r.is_empty());
        }
        if items.iter().any(|i| i.text.is_empty()) {
            return Ok(fail("every item needs non-empty `text`"));
        }
        for (i, item) in items.iter().enumerate() {
            if items[..i].iter().any(|earlier| earlier.text == item.text) {
                return Ok(fail(format!(
                    "item {} repeats the text of an earlier item; each item needs its own text",
                    i + 1
                )));
            }
        }
        if let Err(e) = validate(&self.state.items(), &items) {
            return Ok(fail(e));
        }
        self.state.set(items.clone());
        let resolutions: Vec<Resolution> = (0..items.len()).map(|i| resolve(&items, i)).collect();
        let open = resolutions
            .iter()
            .filter(|r| **r == Resolution::Open)
            .count();
        let unmet = resolutions
            .iter()
            .filter(|r| **r == Resolution::Unmet)
            .count();
        let done = items
            .iter()
            .filter(|i| i.status == ItemStatus::Done)
            .count();
        let summary = if items.is_empty() {
            "Checklist cleared.".to_string()
        } else {
            format!(
                "Checklist updated: {done}/{} done.\n{}",
                items.len(),
                render(&items)
            )
        };
        Ok(ToolOutput::text(summary).with_details(json!({
            "total": items.len(),
            "done": done,
            "open": open,
            "unmet": unmet,
        })))
    }
}

/// Stop hook: while any item is pending or in progress, ask the model to
/// keep going, listing them. Once nothing is open it lets the run stop; if
/// any item is deferred (or superseded by one that is), it reports those as
/// unmet ([`StopDecision::AcceptUnmet`]) so the run is not counted as a
/// success. The agent's turn limits still bound how long this can continue.
#[derive(Debug, Clone)]
pub struct ChecklistGate {
    state: ChecklistState,
}

impl ChecklistGate {
    pub fn new(state: ChecklistState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl Hook for ChecklistGate {
    async fn on_stop(&self, _ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        let open = self.state.open_items();
        if !open.is_empty() {
            let list = open
                .iter()
                .map(|i| format!("- {}", describe(i)))
                .collect::<Vec<_>>()
                .join("\n");
            return StopDecision::Continue {
                message: format!(
                    "Your checklist still has {} open item(s):\n{list}\n\nFinish them and mark \
                     them done with track_requirements before giving your final answer. If you \
                     will not do one, mark it deferred with a reason; if another item replaces \
                     it, mark it superseded.",
                    open.len()
                ),
            };
        }
        let unmet = self.state.unmet_items();
        if unmet.is_empty() {
            return StopDecision::Accept;
        }
        StopDecision::AcceptUnmet {
            unmet: unmet
                .iter()
                .map(|i| format!("checklist: {}", describe(i)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::testing::{ScriptedProvider, call, text, tool_calls};
    use aidaemon_core::{Agent, Role};

    #[test]
    fn checklist_has_no_effects_outside_the_agent() {
        // Complete and empty, so a decline in the run never puts the
        // checklist to the owner.
        assert_eq!(
            Checklist::default().effects(&json!({"items": []})),
            aidaemon_core::CallEffects::none()
        );
    }

    #[tokio::test]
    async fn tool_replaces_list_and_validates() {
        let tool = Checklist::default();
        let out = tool
            .call(
                json!({"items": [
                    {"text": "write parser", "status": "done"},
                    {"text": "add tests", "status": "in_progress"}
                ]}),
                &crate::testutil::ctx(),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("1/2 done"));
        assert!(out.content.contains("2. [~] add tests"));
        assert_eq!(tool.state().open_items().len(), 1);

        let out = tool
            .call(
                json!({"items": [{"text": "x", "status": "finished"}]}),
                &crate::testutil::ctx(),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        // A rejected update leaves the previous list in place.
        assert_eq!(tool.state().items().len(), 2);
    }

    #[tokio::test]
    async fn gate_blocks_until_items_are_done() {
        let (tool, gate) = Checklist::with_gate();
        let provider = ScriptedProvider::new([
            tool_calls([call(
                "c1",
                "track_requirements",
                json!({"items": [
                    {"text": "update config", "status": "in_progress"},
                    {"text": "run tests", "status": "pending"}
                ]}),
            )]),
            // Tries to finish early; the gate pushes back.
            text("All done!"),
            tool_calls([call(
                "c2",
                "track_requirements",
                json!({"items": [
                    {"text": "update config", "status": "done"},
                    {"text": "run tests", "status": "done"}
                ]}),
            )]),
            text("Config updated and tests pass."),
        ]);
        let state = tool.state().clone();
        let agent = Agent::builder("checklist-test")
            .allow_unguarded()
            .model("scripted")
            .provider(provider)
            .tool(tool)
            .hook(gate)
            .build()
            .unwrap();
        let outcome = agent
            .run("test:synthetic-user-1", "Do the task")
            .await
            .unwrap();
        assert!(outcome.is_completed(), "{:?}", outcome.status);
        assert!(outcome.is_succeeded());
        assert_eq!(
            outcome.final_text.as_deref(),
            Some("Config updated and tests pass.")
        );
        let nudge = outcome
            .state
            .messages
            .iter()
            .find(|m| m.role == Role::User && m.content.contains("open item"))
            .expect("gate message in transcript");
        assert!(nudge.content.contains("update config") && nudge.content.contains("run tests"));
        assert!(state.open_items().is_empty());
    }

    async fn gate_decision(state: &ChecklistState) -> StopDecision {
        let ctx = HookContext {
            run_id: aidaemon_core::RunId::new(),
            agent_id: "a",
            session_id: "s",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: Default::default(),
            run_usage: Default::default(),
            untrusted_seen: false,
            declined: &[],
        };
        ChecklistGate::new(state.clone())
            .on_stop(&ctx, "done")
            .await
    }

    async fn update(tool: &Checklist, items: Value) -> ToolOutput {
        tool.call(json!({ "items": items }), &crate::testutil::ctx())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn deferring_lets_the_run_stop_but_not_succeed() {
        let (tool, gate) = Checklist::with_gate();
        let provider = ScriptedProvider::new([
            tool_calls([call(
                "c1",
                "track_requirements",
                json!({"items": [
                    {"text": "export the report", "status": "in_progress"},
                    {"text": "email the report", "status": "pending"}
                ]}),
            )]),
            tool_calls([call(
                "c2",
                "track_requirements",
                json!({"items": [
                    {"text": "export the report", "status": "done"},
                    {"text": "email the report", "status": "deferred",
                     "reason": "no mail tool is available"}
                ]}),
            )]),
            text("Exported. I could not email it."),
        ]);
        let agent = Agent::builder("checklist-test")
            .allow_unguarded()
            .model("scripted")
            .provider(provider)
            .tool(tool)
            .hook(gate)
            .build()
            .unwrap();
        let outcome = agent
            .run("test:synthetic-user-1", "Export and email")
            .await
            .unwrap();

        assert!(outcome.is_completed(), "deferral may stop the run");
        assert_eq!(outcome.turns(), 3, "the gate does not nag about a deferral");
        assert!(
            !outcome.is_succeeded(),
            "a deferred item is not accomplished"
        );
        assert_eq!(outcome.unmet().len(), 1);
        assert!(outcome.unmet()[0].contains("email the report"));
        assert!(outcome.unmet()[0].contains("no mail tool"));
    }

    #[tokio::test]
    async fn unfinished_items_cannot_be_removed_or_cleared() {
        let tool = Checklist::default();
        let out = update(
            &tool,
            json!([
                {"text": "a", "status": "done"},
                {"text": "b", "status": "pending"},
                {"text": "c", "status": "pending"}
            ]),
        )
        .await;
        assert!(!out.is_error);

        // Dropping an open item, or clearing the list, is refused and the
        // previous list stays.
        let out = update(&tool, json!([{"text": "a", "status": "done"}])).await;
        assert!(out.is_error);
        assert!(out.content.contains("cannot be removed") && out.content.contains("b"));
        assert!(update(&tool, json!([])).await.is_error);
        assert_eq!(tool.state().items().len(), 3);

        // A deferral needs a reason; once deferred it still cannot vanish.
        let out = update(
            &tool,
            json!([
                {"text": "a", "status": "done"},
                {"text": "b", "status": "deferred"},
                {"text": "c", "status": "done"}
            ]),
        )
        .await;
        assert!(out.is_error && out.content.contains("needs a `reason`"));
        let out = update(
            &tool,
            json!([
                {"text": "a", "status": "done"},
                {"text": "b", "status": "deferred", "reason": "blocked on access"},
                {"text": "c", "status": "done"}
            ]),
        )
        .await;
        assert!(!out.is_error, "{}", out.content);
        assert!(
            update(&tool, json!([{"text": "c", "status": "done"}]))
                .await
                .is_error
        );

        // Done items may be dropped.
        let out = update(
            &tool,
            json!([{"text": "b", "status": "deferred", "reason": "blocked on access"}]),
        )
        .await;
        assert!(!out.is_error, "{}", out.content);
    }

    #[tokio::test]
    async fn only_the_owner_waives() {
        let tool = Checklist::default();
        update(&tool, json!([{"text": "migrate db", "status": "pending"}])).await;
        let out = update(&tool, json!([{"text": "migrate db", "status": "waived"}])).await;
        assert!(out.is_error && out.content.contains("only the owner"));
        assert!(matches!(
            gate_decision(tool.state()).await,
            StopDecision::Continue { .. }
        ));

        assert!(tool.state().waive("migrate db", "owner: not this sprint"));
        assert_eq!(gate_decision(tool.state()).await, StopDecision::Accept);
        // The model cannot reopen or relabel what the owner waived.
        let out = update(
            &tool,
            json!([{"text": "migrate db", "status": "deferred",
                                         "reason": "x"}]),
        )
        .await;
        assert!(out.is_error && out.content.contains("waived by the owner"));
        assert!(!tool.state().waive("no such item", "x"));
    }

    #[tokio::test]
    async fn superseded_items_resolve_through_their_replacement() {
        let tool = Checklist::default();
        update(
            &tool,
            json!([{"text": "use library A", "status": "in_progress"}]),
        )
        .await;

        // Reword by superseding: the old item follows the new one.
        let out = update(
            &tool,
            json!([
                {"text": "use library A", "status": "superseded", "superseded_by": 2},
                {"text": "use library B", "status": "in_progress"}
            ]),
        )
        .await;
        assert!(!out.is_error, "{}", out.content);
        assert!(matches!(
            gate_decision(tool.state()).await,
            StopDecision::Continue { .. }
        ));

        // Replacement deferred: both are unmet.
        update(
            &tool,
            json!([
                {"text": "use library A", "status": "superseded", "superseded_by": 2},
                {"text": "use library B", "status": "deferred", "reason": "license review"}
            ]),
        )
        .await;
        let StopDecision::AcceptUnmet { unmet } = gate_decision(tool.state()).await else {
            panic!("expected unmet");
        };
        assert_eq!(unmet.len(), 2);

        // Replacement done: both are met.
        update(
            &tool,
            json!([
                {"text": "use library A", "status": "superseded", "superseded_by": 2},
                {"text": "use library B", "status": "done"}
            ]),
        )
        .await;
        assert_eq!(gate_decision(tool.state()).await, StopDecision::Accept);

        // Bad pointers are refused.
        for bad in [json!(null), json!(1), json!(9)] {
            let out = update(
                &tool,
                json!([
                    {"text": "use library A", "status": "superseded", "superseded_by": bad},
                    {"text": "use library B", "status": "done"}
                ]),
            )
            .await;
            assert!(out.is_error, "superseded_by {bad} must be refused");
        }
        let out = update(
            &tool,
            json!([
                {"text": "use library A", "status": "superseded", "superseded_by": 2},
                {"text": "use library B", "status": "superseded", "superseded_by": 3},
                {"text": "use library C", "status": "done"}
            ]),
        )
        .await;
        assert!(out.is_error && out.content.contains("itself superseded"));
    }

    /// The invariant below any wording: whatever sequence of updates the
    /// model sends, an item it declared and never marked done can only let
    /// the run stop as unmet, never as a plain success.
    #[tokio::test]
    async fn no_update_sequence_without_done_reaches_success() {
        const TEXTS: [&str; 3] = ["x", "y", "z"];
        const STATUSES: [&str; 4] = ["pending", "in_progress", "deferred", "superseded"];
        // Small xorshift so the test is deterministic without extra deps.
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let (mut accepted, mut stopped_unmet) = (0, 0);
        for _ in 0..300 {
            let tool = Checklist::default();
            for _ in 0..6 {
                let len = next(TEXTS.len() + 1);
                let items: Vec<Value> = (0..len)
                    .map(|i| {
                        let status = STATUSES[next(STATUSES.len())];
                        let mut item = json!({"text": TEXTS[i], "status": status});
                        if status == "deferred" && next(2) == 0 {
                            item["reason"] = json!("cannot");
                        }
                        if status == "superseded" {
                            item["superseded_by"] = json!(next(len + 1));
                        }
                        item
                    })
                    .collect();
                if !update(&tool, Value::Array(items)).await.is_error {
                    accepted += 1;
                }
                if !tool.state().items().is_empty() {
                    let decision = gate_decision(tool.state()).await;
                    assert_ne!(decision, StopDecision::Accept, "{:?}", tool.state().items());
                    if matches!(decision, StopDecision::AcceptUnmet { .. }) {
                        stopped_unmet += 1;
                    }
                }
            }
        }
        // The sequences exercise both accepted updates and unmet stops.
        assert!(
            accepted > 100 && stopped_unmet > 50,
            "{accepted} {stopped_unmet}"
        );
    }
}
