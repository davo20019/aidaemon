//! Approval and goal-confirmation prompts on any SDK channel.
//!
//! A prompt is sent with [`Channel::ask`] (buttons) and waits for an
//! [`ApprovalAnswer`]; the answer replaces the prompt with its outcome.
//! Who may answer is the caller's decision, made before [`Prompts::answer`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::{ApprovalChoice, ApprovalPrompt, Channel, Conversation, MessageRef, OutboundMessage};
use tokio::sync::{oneshot, Mutex};
use tracing::{info, warn};

use crate::approval_render;
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalResponse, GoalConfirmationStyle};

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);
const GOAL_TIMEOUT: Duration = Duration::from_secs(1800);
/// How long before an approval card expires the owner is reminded of it
/// (at most a fifth of its wait).
const APPROVAL_REMINDER_BEFORE: Duration = Duration::from_secs(60);
/// The reminder sent while an approval card still waits.
const APPROVAL_REMINDER: &str = "⏳ Still waiting for your answer on the card above. It \
expires in about a minute; then that step is skipped, not refused.";

#[derive(Clone, Copy)]
enum Kind {
    Approval,
    Goal(GoalConfirmationStyle),
}

struct Open {
    kind: Kind,
    answer: oneshot::Sender<ApprovalResponse>,
    /// The message carrying the buttons and its text, once sent.
    shown: Option<(MessageRef, String)>,
}

/// The prompts a channel has open, by id.
#[derive(Default)]
pub struct Prompts {
    open: Arc<Mutex<HashMap<String, Open>>>,
}

/// Closes a prompt that ends without an answer (it timed out, or whoever
/// asked stopped waiting and dropped the request): the prompt leaves the
/// open book and its message loses its buttons, so a stale card can never
/// be pressed as if it still decided something.
struct Unanswered {
    id: String,
    open: Arc<Mutex<HashMap<String, Open>>>,
    channel: Arc<dyn Channel>,
    done: bool,
}

impl Drop for Unanswered {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let (id, open, channel) = (self.id.clone(), self.open.clone(), self.channel.clone());
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        runtime.spawn(async move {
            let Some(prompt) = open.lock().await.remove(&id) else {
                return; // answered meanwhile
            };
            if let Some((message, text)) = prompt.shown {
                if let Err(error) = channel
                    .edit(&message, &expired_text(prompt.kind, &text))
                    .await
                {
                    warn!(%error, prompt = %id, "Failed to mark an unanswered prompt expired");
                }
            }
        });
    }
}

/// A prompt's text once it ended without an answer.
fn expired_text(kind: Kind, text: &str) -> String {
    match kind {
        Kind::Approval => format!(
            "{}\n\n⌛ Expired: no answer in time, so it was declined. Nothing was done.",
            approval_render::strip_approval_footer(text).trim_end()
        ),
        Kind::Goal(_) => format!(
            "{}\n\n⌛ Expired: no answer in time. Nothing was activated.",
            text.trim_end()
        ),
    }
}

/// What answering a prompt changed.
pub struct Answered {
    /// The message to replace, and its new text (without buttons).
    pub update: Option<(MessageRef, String)>,
    /// A short confirmation of the answer ("✅ Allowed this once").
    pub label: String,
    /// It was an approval (not a goal confirmation).
    pub approval: bool,
}

impl Prompts {
    /// Ask to approve `command`; waits up to five minutes. A timeout, a
    /// failed send or a closed prompt is an error (and declines).
    #[allow(clippy::too_many_arguments)]
    pub async fn approval(
        &self,
        channel: Arc<dyn Channel>,
        conversation: &Conversation,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
        one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        self.approval_ask(
            channel,
            conversation,
            &approval_render::ApprovalAsk {
                command,
                card: None,
                risk_level,
                warnings,
                permission_mode,
                one_time_only,
            },
        )
        .await
    }

    /// Ask for `ask`, rendered from its typed card when it has one; waits up
    /// to five minutes. A timeout, a failed send or a closed prompt is an
    /// error (and declines).
    pub async fn approval_ask(
        &self,
        channel: Arc<dyn Channel>,
        conversation: &Conversation,
        ask: &approval_render::ApprovalAsk<'_>,
    ) -> anyhow::Result<ApprovalResponse> {
        let session_button =
            approval_render::approval_use_session_button(ask.permission_mode, ask.risk_level);
        let choices = approval_choices(session_button, ask.one_time_only);
        let text = approval_render::build_approval_ask_markdown(ask);
        info!(
            risk = %ask.risk_level,
            mode = %ask.permission_mode,
            card = ask.card.is_some(),
            "Asking for an approval"
        );
        self.ask(
            channel,
            conversation,
            Kind::Approval,
            &[text],
            choices,
            APPROVAL_TIMEOUT,
        )
        .await
    }

    /// Ask to confirm a goal; waits up to thirty minutes. True to confirm.
    pub async fn goal(
        &self,
        channel: Arc<dyn Channel>,
        conversation: &Conversation,
        description: &str,
        details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        let pages = approval_render::build_goal_confirmation_pages(description, details, style);
        let choices = goal_choices(style);
        let response = self
            .ask(
                channel,
                conversation,
                Kind::Goal(style),
                &pages,
                choices,
                GOAL_TIMEOUT,
            )
            .await?;
        Ok(!matches!(response, ApprovalResponse::Deny))
    }

    async fn ask(
        &self,
        channel: Arc<dyn Channel>,
        conversation: &Conversation,
        kind: Kind,
        pages: &[String],
        choices: Vec<ApprovalChoice>,
        timeout: Duration,
    ) -> anyhow::Result<ApprovalResponse> {
        let Some((last, leading)) = pages.split_last() else {
            anyhow::bail!("empty prompt");
        };
        let id = uuid::Uuid::new_v4().to_string();
        let (answer, answered) = oneshot::channel();
        self.open.lock().await.insert(
            id.clone(),
            Open {
                kind,
                answer,
                shown: None,
            },
        );
        let mut unanswered = Unanswered {
            id: id.clone(),
            open: self.open.clone(),
            channel: channel.clone(),
            done: false,
        };
        let sent = async {
            for page in leading {
                channel
                    .send(conversation, &OutboundMessage::text(page.clone()))
                    .await?;
            }
            channel
                .ask(&ApprovalPrompt {
                    id: id.clone(),
                    conversation: conversation.clone(),
                    text: last.clone(),
                    choices,
                })
                .await
        }
        .await;
        match sent {
            Ok(message) => {
                if let Some(open) = self.open.lock().await.get_mut(&id) {
                    open.shown = Some((message, last.clone()));
                }
            }
            Err(error) => {
                unanswered.done = true;
                self.open.lock().await.remove(&id);
                warn!(%error, "Failed to send a prompt");
                anyhow::bail!("the prompt could not be delivered: {error}");
            }
        }
        let mut answered = answered;
        // An approval card nobody has answered gets one reminder shortly
        // before it expires: a helper working in the background may have
        // asked while the owner looked elsewhere.
        let remind_at = matches!(kind, Kind::Approval)
            .then(|| timeout.checked_sub(APPROVAL_REMINDER_BEFORE.min(timeout / 5)))
            .flatten()
            .filter(|at| !at.is_zero());
        let outcome = match remind_at {
            Some(at) => match tokio::time::timeout(at, &mut answered).await {
                Ok(answer) => Ok(answer),
                Err(_) => {
                    if let Err(error) = channel
                        .send(conversation, &OutboundMessage::text(APPROVAL_REMINDER))
                        .await
                    {
                        warn!(%error, prompt = %id, "Failed to remind about a waiting card");
                    }
                    tokio::time::timeout(timeout - at, &mut answered).await
                }
            },
            None => tokio::time::timeout(timeout, &mut answered).await,
        };
        // An answer already took the prompt out of the book and replaced its
        // message; anything else leaves it to `unanswered`, which expires it.
        unanswered.done = matches!(outcome, Ok(Ok(_)));
        drop(unanswered);
        match outcome {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => anyhow::bail!("the prompt was closed without an answer"),
            Err(_) => {
                warn!(prompt = %id, "Prompt timed out");
                anyhow::bail!("the prompt timed out")
            }
        }
    }

    /// Whether `prompt_id` is open (an answer to anything else is stale).
    pub async fn is_open(&self, prompt_id: &str) -> bool {
        self.open.lock().await.contains_key(prompt_id)
    }

    /// Answer prompt `prompt_id` with `choice_id`. `None` if the prompt is
    /// not open or the choice is not one of its own.
    pub async fn answer(&self, prompt_id: &str, choice_id: &str) -> Option<Answered> {
        let mut open = self.open.lock().await;
        let kind = open.get(prompt_id)?.kind;
        let response = response_for(kind, choice_id)?;
        let prompt = open.remove(prompt_id)?;
        let _ = prompt.answer.send(response.clone());
        let update = prompt.shown.map(|(message, text)| {
            let outcome = match kind {
                Kind::Approval => approval_render::finalize_approval_message(&text, &response),
                Kind::Goal(style) => goal_outcome(&text, style, choice_id, &response),
            };
            (message, outcome)
        });
        let label = match kind {
            Kind::Approval => {
                let (icon, status) = approval_render::response_status(&response);
                format!("{icon} {status}")
            }
            Kind::Goal(style) => goal_label(style, choice_id, &response).to_string(),
        };
        Some(Answered {
            update,
            label,
            approval: matches!(kind, Kind::Approval),
        })
    }
}

/// Allow (this once), then (unless the grant is one-time only) until a
/// restart or always, then deny. The labels say what each grant lasts, so
/// the card needs no footer explaining them.
fn approval_choices(session_button: bool, one_time_only: bool) -> Vec<ApprovalChoice> {
    let mut choices = vec![choice("once", "Allow")];
    if !one_time_only {
        choices.push(if session_button {
            choice("session", "Allow until restart")
        } else {
            choice("always", "Always allow")
        });
    }
    choices.push(choice("deny", "Deny"));
    choices
}

/// Confirm, (for Autopilot) edit, cancel.
fn goal_choices(style: GoalConfirmationStyle) -> Vec<ApprovalChoice> {
    let mut choices = vec![choice(
        "confirm",
        match style {
            GoalConfirmationStyle::Standard => "Approve & activate",
            GoalConfirmationStyle::Autopilot => "Enable Autopilot",
        },
    )];
    if matches!(style, GoalConfirmationStyle::Autopilot) {
        choices.push(choice("edit", "Edit permissions"));
    }
    choices.push(choice("cancel", "Cancel"));
    choices
}

fn choice(id: &str, label: &str) -> ApprovalChoice {
    ApprovalChoice {
        id: id.to_string(),
        label: label.to_string(),
    }
}

fn response_for(kind: Kind, choice_id: &str) -> Option<ApprovalResponse> {
    Some(match (kind, choice_id) {
        (Kind::Approval, "once") => ApprovalResponse::AllowOnce,
        (Kind::Approval, "session") => ApprovalResponse::AllowSession,
        (Kind::Approval, "always") => ApprovalResponse::AllowAlways,
        (Kind::Approval, "deny") => ApprovalResponse::Deny,
        (Kind::Goal(_), "confirm") => ApprovalResponse::AllowOnce,
        (Kind::Goal(_), "cancel" | "edit") => ApprovalResponse::Deny,
        _ => return None,
    })
}

fn goal_label(
    style: GoalConfirmationStyle,
    choice_id: &str,
    response: &ApprovalResponse,
) -> &'static str {
    let autopilot = matches!(style, GoalConfirmationStyle::Autopilot);
    match (autopilot, choice_id, response) {
        (true, "edit", _) => "Edit Autopilot permissions",
        (true, _, ApprovalResponse::Deny) => "Autopilot cancelled",
        (true, _, _) => "Autopilot enabled ✅",
        (false, _, ApprovalResponse::Deny) => "Goal cancelled",
        (false, _, _) => "Goal approved ✅",
    }
}

fn goal_outcome(
    text: &str,
    style: GoalConfirmationStyle,
    choice_id: &str,
    response: &ApprovalResponse,
) -> String {
    let autopilot = matches!(style, GoalConfirmationStyle::Autopilot);
    let status = match (autopilot, choice_id, response) {
        (true, "edit", _) => {
            "✏️ Autopilot not activated\nSend the accounts, actions, limits, or guardrails you want changed."
        }
        (true, _, ApprovalResponse::Deny) => "❌ Autopilot cancelled\nNo new autonomy was activated.",
        (true, _, _) => "✅ Autopilot enabled\nActivation is continuing now.",
        (false, _, ApprovalResponse::Deny) => "❌ Goal cancelled\nNothing was activated.",
        (false, _, _) => "✅ Goal approved\nActivation is continuing now.",
    };
    format!("{}\n\n{status}", text.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A card nobody answers loses its buttons: when it times out, and when
    /// whoever asked stops waiting and drops the request.
    #[tokio::test]
    async fn an_unanswered_card_expires() {
        use crate::testing::{direct, TestChannel};
        for dropped in [false, true] {
            let channel = Arc::new(
                TestChannel::new("synthetic")
                    .with_buttons(true)
                    .with_edits(true),
            );
            let prompts = Prompts::default();
            let conversation = direct("synthetic-user-1");
            let pages = ["Summarize the Acme forecast every Monday".to_string()];
            let ask = prompts.ask(
                channel.clone(),
                &conversation,
                Kind::Goal(GoalConfirmationStyle::Standard),
                &pages,
                goal_choices(GoalConfirmationStyle::Standard),
                Duration::from_millis(if dropped { 60_000 } else { 50 }),
            );
            if dropped {
                let _ = tokio::time::timeout(Duration::from_millis(50), ask).await;
            } else {
                assert!(ask.await.is_err(), "a timeout declines");
            }
            assert!(
                channel
                    .wait_for(Duration::from_secs(2), |c| !c.edits().is_empty())
                    .await,
                "dropped={dropped}: the card was not expired"
            );
            let edit = &channel.edits()[0];
            assert!(edit.text.contains("Expired"), "{}", edit.text);
            assert!(!prompts.is_open(&channel.prompts()[0].id).await);
        }
    }

    /// A waiting approval card gets one reminder before it expires.
    #[tokio::test]
    async fn a_waiting_approval_card_is_reminded_once_before_it_expires() {
        use crate::testing::{direct, TestChannel};
        let channel = Arc::new(
            TestChannel::new("synthetic")
                .with_buttons(true)
                .with_edits(true),
        );
        let prompts = Prompts::default();
        let conversation = direct("synthetic-user-1");
        let pages = ["Your assistant wants to run a command on your computer.".to_string()];
        let ask = prompts.ask(
            channel.clone(),
            &conversation,
            Kind::Approval,
            &pages,
            approval_choices(false, true),
            Duration::from_millis(300),
        );
        assert!(ask.await.is_err(), "nobody answered");
        let reminders: Vec<String> = channel
            .sent()
            .into_iter()
            .map(|sent| sent.message.text)
            .filter(|text| text.contains("Still waiting"))
            .collect();
        assert_eq!(reminders.len(), 1, "{reminders:?}");
    }

    fn labels(choices: Vec<ApprovalChoice>) -> Vec<String> {
        choices.into_iter().map(|c| c.label).collect()
    }

    #[test]
    fn approval_buttons_follow_the_grant_offered() {
        assert_eq!(
            labels(approval_choices(false, false)),
            ["Allow", "Always allow", "Deny"]
        );
        assert_eq!(
            labels(approval_choices(true, false)),
            ["Allow", "Allow until restart", "Deny"]
        );
        // A one-time grant offers nothing that persists.
        assert_eq!(labels(approval_choices(true, true)), ["Allow", "Deny"]);
    }

    #[test]
    fn autopilot_confirmation_can_be_sent_back_for_edits() {
        assert_eq!(
            labels(goal_choices(GoalConfirmationStyle::Standard)),
            ["Approve & activate", "Cancel"]
        );
        assert_eq!(
            labels(goal_choices(GoalConfirmationStyle::Autopilot)),
            ["Enable Autopilot", "Edit permissions", "Cancel"]
        );
    }

    #[test]
    fn choices_belong_to_their_kind() {
        assert!(response_for(Kind::Approval, "confirm").is_none());
        assert!(response_for(Kind::Goal(GoalConfirmationStyle::Standard), "always").is_none());
        assert!(matches!(
            response_for(Kind::Goal(GoalConfirmationStyle::Autopilot), "edit"),
            Some(ApprovalResponse::Deny)
        ));
    }

    #[test]
    fn a_goal_outcome_keeps_the_proposal() {
        let text = goal_outcome(
            "Post daily",
            GoalConfirmationStyle::Standard,
            "confirm",
            &ApprovalResponse::AllowOnce,
        );
        assert!(text.starts_with("Post daily\n\n✅ Goal approved"));
    }
}
