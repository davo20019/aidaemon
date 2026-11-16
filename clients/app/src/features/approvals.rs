//! Approval cards: what the agent wants to do, why it asks, what it would
//! touch, and every answer the daemon offers. The first answer anywhere
//! wins; a card answered elsewhere leaves by itself.

use aidaemon_client_protocol::{Approval, ApprovalDecision, AskReason, CardAction, Impact, Risk};
use nsh_core::prelude::*;

use crate::store::{Store, now_ms};
use crate::ui::{self, Glyph, Size, Tone, Variant, badge, btn, code, colors, radius, type_scale};

fn risk(risk: Risk) -> (&'static str, Tone) {
    match risk {
        Risk::Low => ("Low risk", Tone::Neutral),
        Risk::Medium => ("Medium risk", Tone::Warning),
        Risk::High => ("High risk", Tone::Danger),
        Risk::Critical => ("Critical", Tone::Danger),
    }
}

fn decision(d: ApprovalDecision) -> (&'static str, Variant) {
    match d {
        ApprovalDecision::ApproveOnce => ("Allow once", Variant::Primary),
        ApprovalDecision::ApproveSession => ("Allow this session", Variant::Outline),
        ApprovalDecision::ApproveAlways => ("Always allow", Variant::Outline),
        ApprovalDecision::Deny => ("Deny", Variant::Danger),
    }
}

/// How far an answer reaches.
fn scope(d: ApprovalDecision) -> &'static str {
    match d {
        ApprovalDecision::ApproveOnce => "Allow once: just this step",
        ApprovalDecision::ApproveSession => {
            "Allow this session: this kind of step, for the rest of this conversation"
        }
        ApprovalDecision::ApproveAlways => {
            "Always allow: this kind of step from now on, without asking"
        }
        ApprovalDecision::Deny => "Deny: skip just this step",
    }
}

fn impact(impact: &Impact) -> Option<String> {
    match impact {
        Impact::Deletes { targets } if targets.is_empty() => Some("Deletes files".into()),
        Impact::Deletes { targets } => Some(format!("Deletes {}", targets.join(", "))),
        Impact::Sends => Some("Sends something outside this machine".into()),
        Impact::Spends => Some("Spends money or credits".into()),
        Impact::WritesOutsideWorkArea => Some("Writes outside the work area".into()),
        Impact::Unknown => None,
    }
}

fn reason(reason: &AskReason) -> Option<String> {
    match reason {
        AskReason::ReadOutsideContent { check_says } => Some(match check_says {
            Some(why) => format!("It read content from outside: {why}"),
            None => "It read content from outside this conversation.".into(),
        }),
        AskReason::AfterADecline => Some("You declined something like this before.".into()),
        AskReason::Unknown => None,
    }
}

fn expires(at: Option<u64>) -> Option<String> {
    let left = at?.saturating_sub(now_ms()) / 1000;
    Some(if left >= 120 {
        format!("expires in {} min", left / 60)
    } else {
        format!("expires in {left}s")
    })
}

/// One approval, as a card.
pub fn card(store: Store, approval: Approval) -> View {
    let (risk_label, risk_tone) = risk(approval.risk);
    let mut facts: Vec<(Glyph, String, Tone)> = Vec::new();
    if let Some(card) = &approval.card {
        if let Some(text) = card.impact.as_ref().and_then(impact) {
            facts.push((Glyph::Warning, text, Tone::Warning));
        }
        if let Some(text) = card.reason.as_ref().and_then(reason) {
            facts.push((Glyph::Dot, text, Tone::Info));
        }
    }
    for warning in &approval.warnings {
        facts.push((Glyph::Warning, warning.clone(), Tone::Danger));
    }
    let (heading, body): (String, Option<View>) = match approval.card.as_ref().map(|c| &c.action) {
        Some(CardAction::Command {
            line,
            title,
            looks_inside,
        }) => (
            title.clone().unwrap_or_else(|| "Run a command".into()),
            Some(command_block(line, looks_inside)),
        ),
        Some(CardAction::OpenLocalSite { site, once_minutes }) => (
            format!("Open {site}"),
            Some(detail_text(format!("For {once_minutes} minutes."))),
        ),
        Some(CardAction::Use { what, detail }) => {
            (format!("Use {what}"), detail.clone().map(detail_text))
        }
        _ => (
            approval.summary.clone(),
            approval.detail.clone().map(detail_text),
        ),
    };
    let asked_by = approval.card.as_ref().and_then(|c| c.asked_by.clone());
    let teach = store.teach_approvals();
    if teach {
        store.approvals_taught();
    }
    let id = approval.id.clone();
    ui::surface()
        .border_color(risk_tone.color())
        // The card a view move brought up stands out until answered.
        .style_when(
            {
                let id = approval.id.clone();
                move || store.spotlight().as_deref() == Some(id.as_str())
            },
            |s| s.border_2().border_color(colors::ACCENT),
        )
        .p_4()
        .gap_3()
        .role(AccessibilityRole::Group)
        .label(format!("Approval: {heading}"))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(
                    ui::icon(Glyph::Approval)
                        .text_color(risk_tone.color())
                        .accessibility_elements_hidden(true),
                )
                .child(
                    text(heading)
                        .flex_1()
                        .role(AccessibilityRole::Header)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD)
                        .number_of_lines(2),
                )
                .child(badge(risk_label, risk_tone)),
        )
        .child(when(approval.card.is_some(), {
            let summary = approval.summary.clone();
            move || detail_text(summary.clone())
        }))
        .child(body)
        // The first card this device shows says how approvals work.
        .child(when(teach, || {
            detail_text(
                "Your agent asks before anything that sends, posts, deletes or spends. Allow \
                 it once, for this conversation, or always; Deny stops just this step. What it may \
                 do on its own is under Settings, Checks and limits."
                    .to_string(),
            )
        }))
        .children(facts.into_iter().map(|(glyph, text_, tone)| {
            view()
                .flex_row()
                .gap_2()
                .child(
                    ui::icon(glyph)
                        .text_color(tone.color())
                        .accessibility_elements_hidden(true),
                )
                .child(
                    text(text_)
                        .flex_1()
                        .text_size(px(type_scale::SM))
                        .text_color(colors::MUTED),
                )
        }))
        .child({
            let meta = text(
                [
                    asked_by.map(|who| format!("asked by {who}")),
                    expires(approval.expires_at_ms),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" · "),
            )
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT);
            let buttons = approval.options.iter().map(|d| {
                let (label, variant) = decision(*d);
                let (id, d) = (id.clone(), *d);
                // What an answer covers, in words and for screen readers.
                let button = btn(label, variant, Size::Sm, move || store.resolve(&id, d))
                    .accessibility_label(scope(d));
                (d, button)
            });
            if ui::touch() {
                // A phone stacks the answers full width, the main one
                // first: three side by side would wrap at a thumb's size.
                view()
                    .gap_2()
                    .children(buttons.map(|(_, b)| b.w_full()))
                    .child(meta)
            } else {
                view()
                    .flex_row()
                    .flex_wrap()
                    .items_center()
                    .gap_2()
                    // Pointing at an answer says how far it reaches.
                    .children(buttons.map(|(d, b)| ui::with_hint(scope(d), b)))
                    .child(view().flex_1())
                    .child(meta)
            }
        })
}

fn detail_text(value: String) -> View {
    view().child(
        text(value)
            .selectable(true)
            .text_size(px(type_scale::SM))
            .line_height(px(type_scale::LH_SM))
            .text_color(colors::MUTED),
    )
}

fn command_block(line: &str, looks_inside: &[String]) -> View {
    view()
        .gap_1()
        .child(
            view()
                .rounded(px(radius::CONTROL))
                .bg(colors::INSET)
                .px_3()
                .py_2()
                .child(code(line.to_owned()).selectable(true)),
        )
        .child(when(!looks_inside.is_empty(), {
            let inside = looks_inside.join(", ");
            move || detail_text(format!("Looks inside: {inside}"))
        }))
}
