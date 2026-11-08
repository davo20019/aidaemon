//! Approval and goal-confirmation prompts on any SDK channel.
//!
//! A prompt is sent with [`Channel::ask`] (buttons) and waits for an
//! [`ApprovalAnswer`]; the answer replaces the prompt with its outcome.
//! Who may answer is the caller's decision, made before [`Prompts::answer`].

use std::collections::HashMap;
use std::time::Duration;

use aidaemon_channels::{
    ApprovalChoice, ApprovalPrompt, Channel, Conversation, MessageRef, OutboundMessage,
};
use tokio::sync::{oneshot, Mutex};
use tracing::{info, warn};

use super::approval_render;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::types::{ApprovalResponse, GoalConfirmationStyle};

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);
const GOAL_TIMEOUT: Duration = Duration::from_secs(1800);

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
pub(crate) struct Prompts {
    open: Mutex<HashMap<String, Open>>,
}

/// What answering a prompt changed.
pub(crate) struct Answered {
    /// The message to replace, and its new text (without buttons).
    pub update: Option<(MessageRef, String)>,
}

impl Prompts {
    /// Ask to approve `command`; waits up to five minutes. A timeout, a
    /// failed send or a closed prompt is an error (and declines).
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn approval(
        &self,
        channel: &dyn Channel,
        conversation: &Conversation,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
        one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        let session_button =
            approval_render::approval_use_session_button(permission_mode, risk_level);
        let mut choices = vec![choice("once", "Allow Once")];
        if !one_time_only {
            choices.push(if session_button {
                choice("session", "Allow Session")
            } else {
                choice("always", "Allow Always")
            });
        }
        choices.push(choice("deny", "Deny"));
        let text = approval_render::build_approval_message_markdown(
            command,
            risk_level,
            warnings,
            session_button,
            one_time_only,
        );
        info!(risk = %risk_level, mode = %permission_mode, "Asking for an approval");
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
    pub(crate) async fn goal(
        &self,
        channel: &dyn Channel,
        conversation: &Conversation,
        description: &str,
        details: &[String],
        style: GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        let pages = approval_render::build_goal_confirmation_pages(description, details, style);
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
        channel: &dyn Channel,
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
                self.open.lock().await.remove(&id);
                warn!(%error, "Failed to send a prompt");
                anyhow::bail!("the prompt could not be delivered: {error}");
            }
        }
        let outcome = tokio::time::timeout(timeout, answered).await;
        self.open.lock().await.remove(&id);
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
    pub(crate) async fn is_open(&self, prompt_id: &str) -> bool {
        self.open.lock().await.contains_key(prompt_id)
    }

    /// Answer prompt `prompt_id` with `choice_id`. `None` if the prompt is
    /// not open or the choice is not one of its own.
    pub(crate) async fn answer(&self, prompt_id: &str, choice_id: &str) -> Option<Answered> {
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
        Some(Answered { update })
    }
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
