//! Approvals as clients see them: the hub shows the gateway every card it
//! routes ([`ApprovalTap`]), the gateway publishes it, and the first answer
//! wins, from a client or from the chat the card is also shown in. A later
//! answer gets `conflict` naming the winner. The answer goes back through
//! the hub to whoever asked (the kernel's owner-approval resolver), so the
//! run resumes exactly as it would from the chat.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use aidaemon_channels::approval_render::{approval_use_session_button, card_headline, ApprovalAsk};
use aidaemon_channels::hub::{ApprovalTap, TapAnswer, TapClosed, TappedApproval};
use aidaemon_client_protocol as wire;
use aidaemon_client_protocol::{
    Approval, ApprovalDecision, ApprovalOutcome, ErrorCode, Event, ProtocolError, ResolvedBy, Risk,
};
use aidaemon_tools::command_risk::RiskLevel;
use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse};
use tokio::sync::oneshot;

use crate::bus::now_ms;
use crate::{Publisher, APP_CHANNEL};

/// How many settled approvals are remembered for `conflict` replies.
const SETTLED_KEPT: usize = 512;

struct Open {
    approval: Approval,
    /// Taken by the first client answer.
    answer: Option<oneshot::Sender<(ApprovalResponse, String)>>,
    /// The client answer that won, and the device that gave it.
    answered: Option<(ApprovalOutcome, Option<String>)>,
}

#[derive(Default)]
struct Book {
    open: HashMap<String, Open>,
    settled: HashMap<String, ApprovalOutcome>,
    settled_order: VecDeque<String>,
}

/// The approvals open now, and those recently settled.
pub struct Approvals {
    book: Mutex<Book>,
    publisher: Publisher,
    boot: String,
    next: AtomicU64,
}

/// Who a client answer was given by, as the chat's card says it.
fn from_label(device: Option<&str>) -> String {
    match device {
        Some(device) => format!("from {device} (app)"),
        None => "from the app".to_string(),
    }
}

impl Approvals {
    pub(crate) fn new(publisher: Publisher, boot: String) -> Self {
        Self {
            book: Mutex::default(),
            publisher,
            boot,
            next: AtomicU64::new(1),
        }
    }

    fn book(&self) -> std::sync::MutexGuard<'_, Book> {
        self.book.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Every approval waiting for an answer, oldest first.
    pub fn pending(&self) -> Vec<Approval> {
        let mut pending: Vec<Approval> = self
            .book()
            .open
            .values()
            .filter(|open| open.answered.is_none())
            .map(|open| open.approval.clone())
            .collect();
        pending.sort_by_key(|approval| approval.requested_at_ms);
        pending
    }

    /// Answer approval `id` from a client. The first answer wins; a `watch`
    /// may only deny above `ceiling`.
    pub fn resolve(
        &self,
        id: &str,
        decision: ApprovalDecision,
        device: Option<String>,
        ceiling: Option<Risk>,
    ) -> Result<ApprovalOutcome, ProtocolError> {
        let mut book = self.book();
        if let Some(outcome) = book.settled.get(id) {
            return Err(already(*outcome));
        }
        let Some(open) = book.open.get_mut(id) else {
            return Err(ProtocolError::new(
                ErrorCode::NotFound,
                format!("No approval {id} is open"),
            ));
        };
        if let Some((outcome, _)) = open.answered {
            return Err(already(outcome));
        }
        if !open.approval.options.contains(&decision) {
            return Err(ProtocolError::new(
                ErrorCode::Invalid,
                format!("{decision:?} is not offered for this approval"),
            ));
        }
        if let Some(ceiling) = ceiling {
            if decision != ApprovalDecision::Deny && open.approval.risk > ceiling {
                return Err(ProtocolError::new(
                    ErrorCode::Forbidden,
                    "This approval is above what this device may allow: open it on your phone \
                     or computer (denying it from here is fine)",
                ));
            }
        }
        let Some(answer) = open.answer.take() else {
            return Err(ProtocolError::new(
                ErrorCode::Conflict,
                "This approval is already being settled",
            ));
        };
        let outcome = ApprovalOutcome::from_decision(decision);
        if answer
            .send((response_of(decision), from_label(device.as_deref())))
            .is_err()
        {
            // The card closed meanwhile; its close names the outcome.
            return Err(ProtocolError::new(
                ErrorCode::Conflict,
                "This approval was settled before the answer arrived",
            ));
        }
        open.answered = Some((outcome, device));
        Ok(outcome)
    }

    /// Settle approvals a previous daemon process left open in the log:
    /// cards do not survive a restart, so they were withdrawn unanswered.
    pub(crate) fn withdraw_stale(&self, ids: Vec<(String, Option<String>)>) {
        for (id, session_id) in ids {
            self.publisher.publish(
                session_id,
                Event::ApprovalResolved {
                    approval_id: id,
                    outcome: ApprovalOutcome::Withdrawn,
                    resolved_by: None,
                },
            );
        }
    }
}

fn already(outcome: ApprovalOutcome) -> ProtocolError {
    let name = serde_json::to_value(outcome)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    ProtocolError::new(
        ErrorCode::Conflict,
        format!("This approval was already settled: {name}"),
    )
}

fn response_of(decision: ApprovalDecision) -> ApprovalResponse {
    match decision {
        ApprovalDecision::ApproveOnce => ApprovalResponse::AllowOnce,
        ApprovalDecision::ApproveSession => ApprovalResponse::AllowSession,
        ApprovalDecision::ApproveAlways => ApprovalResponse::AllowAlways,
        ApprovalDecision::Deny => ApprovalResponse::Deny,
    }
}

fn outcome_of(response: &ApprovalResponse) -> ApprovalOutcome {
    match response {
        ApprovalResponse::AllowOnce => ApprovalOutcome::ApprovedOnce,
        ApprovalResponse::AllowSession => ApprovalOutcome::ApprovedSession,
        ApprovalResponse::AllowAlways => ApprovalOutcome::ApprovedAlways,
        ApprovalResponse::Deny => ApprovalOutcome::Denied,
    }
}

pub(crate) fn risk_of(level: RiskLevel) -> Risk {
    match level {
        RiskLevel::Safe => Risk::Low,
        RiskLevel::Medium => Risk::Medium,
        RiskLevel::High => Risk::High,
        RiskLevel::Critical => Risk::Critical,
    }
}

/// The answers a card offers: the same buttons the chat shows.
fn options(kind: ApprovalKind, ask: &ApprovalAsk<'_>) -> Vec<ApprovalDecision> {
    use ApprovalDecision::*;
    match kind {
        ApprovalKind::GoalConfirmation | ApprovalKind::AutopilotConfirmation => {
            vec![ApproveOnce, Deny]
        }
        ApprovalKind::CommandOnce => vec![ApproveOnce, Deny],
        ApprovalKind::Command if ask.one_time_only => vec![ApproveOnce, Deny],
        ApprovalKind::Command => {
            if approval_use_session_button(ask.permission_mode, ask.risk_level) {
                vec![ApproveOnce, ApproveSession, Deny]
            } else {
                vec![ApproveOnce, ApproveAlways, Deny]
            }
        }
    }
}

fn card_of(card: &aidaemon_core::ApprovalCard) -> wire::ApprovalCard {
    use aidaemon_core::{AskReason, CardAction, HighImpact};
    wire::ApprovalCard {
        asked_by: card.asked_by.clone(),
        action: match &card.action {
            CardAction::Command {
                line,
                title,
                looks_inside,
            } => wire::CardAction::Command {
                line: line.clone(),
                title: title.clone(),
                looks_inside: looks_inside.clone(),
            },
            CardAction::OpenLocalSite { site, once_minutes } => wire::CardAction::OpenLocalSite {
                site: site.clone(),
                once_minutes: *once_minutes,
            },
            CardAction::Use { what, detail } => wire::CardAction::Use {
                what: what.clone(),
                detail: detail.clone(),
            },
            _ => wire::CardAction::Unknown,
        },
        reason: card.reason.as_ref().map(|reason| match reason {
            AskReason::ReadOutsideContent { check_says } => wire::AskReason::ReadOutsideContent {
                check_says: check_says.clone(),
            },
            AskReason::AfterADecline => wire::AskReason::AfterADecline,
            _ => wire::AskReason::Unknown,
        }),
        impact: card.impact.as_ref().map(|impact| match impact {
            HighImpact::Deletes { targets } => wire::Impact::Deletes {
                targets: targets.clone(),
            },
            HighImpact::Sends => wire::Impact::Sends,
            HighImpact::Spends => wire::Impact::Spends,
            HighImpact::WritesOutsideWorkArea => wire::Impact::WritesOutsideWorkArea,
            _ => wire::Impact::Unknown,
        }),
    }
}

/// The exact thing a card's action would run, for `detail`.
fn detail_of(card: &aidaemon_core::ApprovalCard) -> Option<String> {
    match &card.action {
        aidaemon_core::CardAction::Command { line, .. } => Some(line.clone()),
        aidaemon_core::CardAction::OpenLocalSite { site, .. } => Some(site.clone()),
        aidaemon_core::CardAction::Use { detail, .. } => detail.clone(),
        _ => None,
    }
}

fn approval_of(id: String, tapped: &TappedApproval<'_>) -> Approval {
    let ask = &tapped.ask;
    let (summary, detail, card) = match ask.card {
        Some(card) => (card_headline(card), detail_of(card), Some(card_of(card))),
        None => (ask.command.to_string(), None, None),
    };
    Approval {
        id,
        session_id: tapped.session_id.to_string(),
        kind: match tapped.kind {
            ApprovalKind::Command | ApprovalKind::CommandOnce => wire::ApprovalKind::Command,
            ApprovalKind::GoalConfirmation => wire::ApprovalKind::GoalConfirmation,
            ApprovalKind::AutopilotConfirmation => wire::ApprovalKind::AutopilotConfirmation,
        },
        risk: risk_of(ask.risk_level),
        summary,
        detail,
        card,
        warnings: ask.warnings.to_vec(),
        options: options(tapped.kind, ask),
        requested_at_ms: now_ms(),
        expires_at_ms: tapped.expires_at_ms,
        channel: (tapped.channel != APP_CHANNEL).then(|| tapped.channel.clone()),
    }
}

impl ApprovalTap for Approvals {
    fn opened(&self, tapped: &TappedApproval<'_>) -> (String, TapAnswer) {
        let id = format!(
            "{}-a{}",
            self.boot,
            self.next.fetch_add(1, Ordering::Relaxed)
        );
        let approval = approval_of(id.clone(), tapped);
        let (answer, answered) = oneshot::channel();
        self.book().open.insert(
            id.clone(),
            Open {
                approval: approval.clone(),
                answer: Some(answer),
                answered: None,
            },
        );
        self.publisher.publish(
            Some(approval.session_id.clone()),
            Event::ApprovalRequested { approval },
        );
        (id, answered)
    }

    fn closed(&self, id: &str, how: TapClosed) {
        let mut book = self.book();
        let Some(open) = book.open.remove(id) else {
            return;
        };
        let (outcome, resolved_by) = match how {
            TapClosed::Answered { response, by } => {
                let outcome = outcome_of(&response);
                let resolved_by = match &open.answered {
                    Some((_, device)) => ResolvedBy {
                        channel: APP_CHANNEL.to_string(),
                        device: device.clone(),
                    },
                    None => ResolvedBy {
                        channel: by,
                        device: None,
                    },
                };
                (outcome, Some(resolved_by))
            }
            TapClosed::Unanswered => {
                // Out of time, or the asker stopped waiting first.
                let expired = open
                    .approval
                    .expires_at_ms
                    .is_some_and(|at| now_ms().saturating_add(1_000) >= at);
                let outcome = if expired {
                    ApprovalOutcome::Expired
                } else {
                    ApprovalOutcome::Withdrawn
                };
                (outcome, None)
            }
        };
        book.settled.insert(id.to_string(), outcome);
        book.settled_order.push_back(id.to_string());
        while book.settled_order.len() > SETTLED_KEPT {
            if let Some(old) = book.settled_order.pop_front() {
                book.settled.remove(&old);
            }
        }
        drop(book);
        self.publisher.publish(
            Some(open.approval.session_id.clone()),
            Event::ApprovalResolved {
                approval_id: id.to_string(),
                outcome,
                resolved_by,
            },
        );
    }
}
