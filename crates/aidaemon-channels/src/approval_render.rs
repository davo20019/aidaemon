//! Approval-message rendering shared across chat channels.
//!
//! Pure helpers that build inline keyboards and message text for command-approval
//! and scheduled-goal-confirmation prompts.

use crate::split::split_message;
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalResponse, GoalConfirmationStyle};

/// Truncate command display to fit Telegram's 4096 char limit.
/// Reserve ~300 chars for header, warnings, buttons, and footer.
const MAX_CMD_DISPLAY: usize = 3600;

/// Decide whether the "Allow Session" button is shown instead of "Allow Always".
///
/// - Default mode: Critical gets [Once, Session, Deny], others get [Once, Always, Deny]
/// - Cautious mode: All get [Once, Session, Deny]
/// - YOLO mode: All get [Once, Always, Deny]
pub fn approval_use_session_button(permission_mode: PermissionMode, risk_level: RiskLevel) -> bool {
    match permission_mode {
        PermissionMode::Cautious => true,
        PermissionMode::Default => risk_level >= RiskLevel::Critical,
        PermissionMode::Yolo => false,
    }
}

/// User-facing risk header: icon, title, and one-line context.
pub fn risk_header(risk_level: RiskLevel) -> (&'static str, &'static str, &'static str) {
    match risk_level {
        RiskLevel::Safe => (
            "ℹ️",
            "New command",
            "Your agent wants to run something it hasn't done before.",
        ),
        RiskLevel::Medium => (
            "⚠️",
            "Approval needed",
            "Your agent is waiting for your OK to continue.",
        ),
        RiskLevel::High => (
            "🔶",
            "Review carefully",
            "This action could have significant effects.",
        ),
        RiskLevel::Critical => (
            "🚨",
            "Dangerous action",
            "Review carefully before allowing.",
        ),
    }
}

/// Hint text explaining what the persistent-approval button does.
pub fn allow_button_hint(use_session_button: bool, one_time_only: bool) -> &'static str {
    if one_time_only {
        "This approval applies only to this action; you will be asked again for another sensitive action."
    } else if use_session_button {
        "\"Allow Session\" means you won't be asked again until the daemon restarts."
    } else {
        "\"Allow Always\" means you won't be asked again for this type of action."
    }
}

/// Short status shown after the user taps an approval button.
pub fn response_status(response: &ApprovalResponse) -> (&'static str, &'static str) {
    match response {
        ApprovalResponse::AllowOnce => ("✅", "Allowed this once"),
        ApprovalResponse::AllowSession => ("✅", "Allowed for this session"),
        ApprovalResponse::AllowAlways => ("✅", "Always allowed"),
        ApprovalResponse::Deny => ("❌", "Denied"),
    }
}

/// Optional follow-up detail after the status line.
pub fn response_status_detail(response: &ApprovalResponse) -> Option<&'static str> {
    match response {
        ApprovalResponse::AllowSession => Some("You'll be asked again after restart."),
        ApprovalResponse::AllowAlways => Some("You won't be asked again for this type of action."),
        _ => None,
    }
}

/// Strip the pre-decision footer (button hint) from an approval prompt message.
pub fn strip_approval_footer(text: &str) -> String {
    let mut body = text.trim_end().to_string();
    for marker in [
        "\n\n\"Allow Session\"",
        "\n\n\"Allow Always\"",
        "\n\n<i>\"Allow Session\"",
        "\n\n<i>\"Allow Always\"",
        "\n\n*\"Allow Session\"",
        "\n\n*\"Allow Always\"",
        "\n\n_\"Allow Session\"",
        "\n\n_\"Allow Always\"",
        "\n\nThis approval applies only",
        "\n\n<i>This approval applies only",
        "\n\n*This approval applies only",
        "\n\n_This approval applies only",
    ] {
        if let Some(idx) = body.find(marker) {
            body.truncate(idx);
            break;
        }
    }
    body.trim_end().to_string()
}

/// Build the message body shown after the user chooses Allow / Deny.
pub fn finalize_approval_message(original: &str, response: &ApprovalResponse) -> String {
    let body = strip_approval_footer(original);
    let (icon, status) = response_status(response);
    let mut text = format!("{body}\n\n{icon} {status}");
    if let Some(detail) = response_status_detail(response) {
        text.push_str(&format!("\n{detail}"));
    }
    text
}

fn truncate_command(command: &str) -> String {
    if command.len() > MAX_CMD_DISPLAY {
        let end = aidaemon_tools::text::floor_char_boundary(command, MAX_CMD_DISPLAY);
        format!(
            "{}...\n[truncated — {} chars total]",
            &command[..end],
            command.len()
        )
    } else {
        command.to_string()
    }
}

/// Build the Markdown body for a command-approval prompt.
pub fn build_approval_message_markdown(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
    use_session_button: bool,
    one_time_only: bool,
) -> String {
    let display_cmd = truncate_command(command);
    let (risk_icon, risk_label, risk_subtitle) = risk_header(risk_level);

    let mut text = format!(
        "{} **{}**\n{}\n\n**Requested action**\n```\n{display_cmd}\n```",
        risk_icon, risk_label, risk_subtitle
    );

    if !warnings.is_empty() {
        text.push('\n');
        for warning in warnings {
            text.push_str(&format!("\n• {warning}"));
        }
    }

    text.push_str(&format!(
        "\n\n*{}*",
        allow_button_hint(use_session_button, one_time_only)
    ));

    text
}

fn goal_detail_section(detail: &str) -> (&'static str, &str) {
    for (prefix, heading) in [
        ("Objective:", "🎯 Objective"),
        ("Constraints:", "🛡️ Rules it must follow"),
        ("Success criteria:", "✅ Success means"),
        ("Stop conditions:", "🛑 Stops when"),
        ("Guidance:", "🧭 Guidance"),
        ("Pinned strategy:", "🧭 Strategy"),
        ("Access summary:", "🔑 What it can do"),
        ("Observations allowed:", "👁️ Read access"),
        ("Allowed mutation effects:", "✍️ External changes"),
        ("Allowed targets:", "🎯 Allowed targets"),
        ("Exact operation scopes:", "🔧 Operation scope"),
        ("Mutation limits:", "📏 Activity limits"),
        ("Review interval:", "🔄 Check-in timing"),
        ("Expiration:", "⏱️ Ends"),
        ("Autonomy mode:", "🤖 How it works"),
        ("Owner checkpoints:", "🙋 Owner checkpoints"),
        ("Recovery policy:", "♻️ Recovery"),
        ("Confirmation binding:", "🔏 Confirmation binding"),
        ("Review effort:", "🧠 Review effort"),
        ("Review depth:", "🧠 Decision checks"),
        ("Policy reference:", "🔏 Approval scope"),
        ("Resolved token budgets:", "🧮 Compute budget"),
    ] {
        if let Some(value) = detail.strip_prefix(prefix) {
            return (heading, value.trim());
        }
    }
    ("ℹ️ Additional detail", detail.trim())
}

fn normalized_goal_text(text: &str) -> String {
    text.trim()
        .strip_prefix("Delegate mandate:")
        .unwrap_or(text.trim())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['.', ';'])
        .to_ascii_lowercase()
}

fn readable_policy(value: &str) -> String {
    value.replace(" | ", "\n")
}

/// A mandate's confirmation card: its five essentials one line each, then
/// everything else under one "Details" heading, so the owner reads what they
/// approve at a glance and the full policy is still there.
fn mandate_card_body(
    subject: &str,
    summary: &str,
    waiting_copy: &str,
    card: &aidaemon_core::mandate_card::MandateConfirmationCard,
) -> String {
    use aidaemon_core::mandate_card::Essential;
    let mut body = format!("{subject}\n{summary}\n");
    let summary_normalized = normalized_goal_text(summary);
    for (essential, text) in card.essentials() {
        // The summary already names what it does.
        if essential == Essential::What && summary_normalized.contains(&normalized_goal_text(text))
        {
            continue;
        }
        body.push_str(&format!("\n{}: {text}", essential.label()));
    }
    body.push_str(&format!("\n\n{waiting_copy}"));
    if !card.details.is_empty() {
        body.push_str("\n\nDetails");
        for detail in &card.details {
            body.push_str(&format!("\n• {}", readable_policy(detail.trim())));
        }
    }
    body
}

/// Build lossless plain-text pages for a goal-confirmation prompt.
///
/// Mandate confirmations can contain complete authority scopes and policy text,
/// which routinely exceed Telegram's 4096-character message limit. Keep enough
/// headroom for page labels and the final confirmation instruction, and put the
/// inline keyboard on the final page only. Plain text avoids splitting inside
/// generated HTML entities while preserving every disclosed detail.
pub fn build_goal_confirmation_pages(
    goal_description: &str,
    details: &[String],
    style: GoalConfirmationStyle,
) -> Vec<String> {
    // Fits Telegram, Discord, and Slack Block Kit after headings/footer are
    // added, so every supported inline channel can render the same lossless
    // policy card without truncating authority details.
    const PAGE_BODY_BYTES: usize = 1_500;

    let summary = goal_description
        .trim()
        .strip_prefix("Delegate mandate:")
        .unwrap_or(goal_description.trim())
        .trim();
    let summary_normalized = normalized_goal_text(summary);
    let (subject, waiting_copy) = match style {
        GoalConfirmationStyle::Standard => (
            "What you’re approving",
            "Nothing starts until you approve this goal.",
        ),
        GoalConfirmationStyle::Autopilot => (
            "What Autopilot will manage",
            "Nothing starts until you approve this exact policy.",
        ),
    };
    let body = match aidaemon_core::mandate_card::MandateConfirmationCard::from_lines(details) {
        Some(card) => mandate_card_body(subject, summary, waiting_copy, &card),
        None => {
            let mut body = format!("{subject}\n{summary}\n\n{waiting_copy}");
            for detail in details {
                let (heading, value) = goal_detail_section(detail);
                if heading == "🎯 Objective" && normalized_goal_text(value) == summary_normalized
                {
                    continue;
                }
                body.push_str(&format!("\n\n{heading}\n{}", readable_policy(value)));
            }
            body
        }
    };

    let chunks = split_message(&body, PAGE_BODY_BYTES);
    let page_count = chunks.len();
    chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let title = match style {
                GoalConfirmationStyle::Standard => "🔐 Review plan",
                GoalConfirmationStyle::Autopilot => "🚀 Enable Autopilot",
            };
            let mut page = if page_count == 1 {
                format!("{title} before activation\n\n{chunk}")
            } else {
                format!("{title} · {}/{page_count}\n\n{chunk}", index + 1)
            };
            if index + 1 == page_count {
                page.push_str(match style {
                    GoalConfirmationStyle::Standard => {
                        "\n\nReady to proceed? Approve and activate this plan, or cancel below."
                    }
                    GoalConfirmationStyle::Autopilot => {
                        "\n\nReady to proceed? Enable this exact Autopilot policy or cancel below."
                    }
                });
            }
            page
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_header_medium_is_approval_needed() {
        let (_, title, _) = risk_header(RiskLevel::Medium);
        assert_eq!(title, "Approval needed");
    }

    #[test]
    fn finalize_strips_footer_and_adds_status() {
        let prompt = build_approval_message_markdown(
            "Open website: https://example.com",
            RiskLevel::Medium,
            &[],
            false,
            false,
        );
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::AllowAlways);
        assert!(!final_msg.contains("Allow Always\" means"));
        assert!(final_msg.contains("Always allowed"));
        assert!(final_msg.contains("won't be asked again"));
        assert!(final_msg.contains("Open website"));
    }

    #[test]
    fn finalize_denied_shows_denied_status() {
        let prompt =
            build_approval_message_markdown("rm -rf /", RiskLevel::Critical, &[], true, false);
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::Deny);
        assert!(final_msg.contains("❌ Denied"));
    }

    #[test]
    fn one_time_only_prompt_has_no_persistent_choice() {
        let prompt = build_approval_message_markdown(
            "Run JavaScript on https://example.com (10 bytes)",
            RiskLevel::High,
            &[],
            false,
            true,
        );
        assert!(prompt.contains("applies only to this action"));
        assert!(!prompt.contains("Allow Always"));
    }

    #[test]
    fn goal_confirmation_pages_preserve_large_proposal_within_telegram_limit() {
        let final_marker = "FINAL_AUTHORITY_DETAIL";
        let details = vec![format!(
            "{}{}",
            "scope <unescaped> ".repeat(500),
            final_marker
        )];
        let pages = build_goal_confirmation_pages(
            "Delegate an ongoing mandate",
            &details,
            GoalConfirmationStyle::Standard,
        );

        assert!(pages.len() > 1);
        assert!(pages.iter().all(|page| page.len() <= 4_096));
        assert!(pages.last().unwrap().contains(final_marker));
        assert!(pages
            .last()
            .unwrap()
            .contains("Approve and activate this plan, or cancel"));
        assert!(pages.iter().any(|page| page.contains("<unescaped>")));
    }

    #[test]
    fn goal_confirmation_is_scannable_and_deduplicates_objective() {
        let description = "Delegate mandate: Steward the synthetic account for 24 hours.";
        let details = vec![
            "Objective: Steward the synthetic account for 24 hours.".to_string(),
            "Constraints: 1. Verify identity. | 2. Never retry an ambiguous mutation.".to_string(),
            "Expiration: 86400 seconds after actual activation".to_string(),
        ];

        let pages =
            build_goal_confirmation_pages(description, &details, GoalConfirmationStyle::Standard);
        let rendered = pages.join("\n");
        assert!(rendered.starts_with("🔐 Review plan before activation"));
        assert_eq!(rendered.matches("Steward the synthetic account").count(), 1);
        assert!(rendered.contains("🛡️ Rules it must follow\n1. Verify identity.\n2. Never retry"));
        assert!(rendered.contains("⏱️ Ends"));
        assert!(rendered.contains("Nothing starts until you approve this goal."));
    }

    #[test]
    fn a_mandate_card_shows_five_essentials_and_one_details_section() {
        let card = aidaemon_core::mandate_card::MandateConfirmationCard {
            what: "Every hour, check the synthetic status page".to_string(),
            how_often: "Every hour".to_string(),
            can_do: "Read only – makes no changes".to_string(),
            tells_you_when: "it finds something you asked to hear about".to_string(),
            ends: "Only when you stop it".to_string(),
            details: vec![
                "Owner checkpoints: It asks you before using a new account.".to_string(),
                "Policy reference: Your approval covers only these rules (policy version 2)."
                    .to_string(),
                "Timing note: It uses the length you gave.".to_string(),
            ],
        };
        let pages = build_goal_confirmation_pages(
            "Delegate mandate \"Status\": Every hour, check the synthetic status page",
            &card.into_lines(),
            GoalConfirmationStyle::Standard,
        );
        let rendered = pages.join("\n");
        assert_eq!(
            rendered.matches("check the synthetic status page").count(),
            1
        );
        for line in [
            "\nHow often: Every hour",
            "\nCan do: Read only – makes no changes",
            "\nTells you when: it finds something you asked to hear about",
            "\nEnds: Only when you stop it",
        ] {
            assert!(rendered.contains(line), "{line}: {rendered}");
        }
        assert_eq!(rendered.matches("\nDetails\n").count(), 1, "{rendered}");
        assert!(rendered.contains("• Timing note: It uses the length you gave."));
        assert!(!rendered.contains("Additional detail"), "{rendered}");
        let essentials_end = rendered.find("Ends:").unwrap();
        assert!(essentials_end < rendered.find("Details").unwrap());
    }

    #[test]
    fn autopilot_confirmation_is_structurally_distinct() {
        let pages = build_goal_confirmation_pages(
            "Maintain the synthetic account",
            &["Autonomy mode: Autopilot".to_string()],
            GoalConfirmationStyle::Autopilot,
        );
        let rendered = pages.join("\n");
        assert!(rendered.starts_with("🚀 Enable Autopilot"));
        assert!(rendered.contains("Enable this exact Autopilot policy"));
    }
}
