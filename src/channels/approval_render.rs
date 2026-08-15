//! Approval-message rendering shared across chat channels.
//!
//! Pure helpers that build inline keyboards and message text for command-approval
//! and scheduled-goal-confirmation prompts.

use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};

use super::formatting::html_escape;
use super::formatting::split_message;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::types::{ApprovalResponse, GoalConfirmationStyle};

/// Truncate command display to fit Telegram's 4096 char limit.
/// Reserve ~300 chars for header, warnings, buttons, and footer.
const MAX_CMD_DISPLAY: usize = 3600;

/// Decide whether the "Allow Session" button is shown instead of "Allow Always".
///
/// - Default mode: Critical gets [Once, Session, Deny], others get [Once, Always, Deny]
/// - Cautious mode: All get [Once, Session, Deny]
/// - YOLO mode: All get [Once, Always, Deny]
pub(super) fn approval_use_session_button(
    permission_mode: PermissionMode,
    risk_level: RiskLevel,
) -> bool {
    match permission_mode {
        PermissionMode::Cautious => true,
        PermissionMode::Default => risk_level >= RiskLevel::Critical,
        PermissionMode::Yolo => false,
    }
}

/// User-facing risk header: icon, title, and one-line context.
pub(super) fn risk_header(risk_level: RiskLevel) -> (&'static str, &'static str, &'static str) {
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
pub(super) fn allow_button_hint(use_session_button: bool, one_time_only: bool) -> &'static str {
    if one_time_only {
        "This approval applies only to this action; you will be asked again for another sensitive action."
    } else if use_session_button {
        "\"Allow Session\" means you won't be asked again until the daemon restarts."
    } else {
        "\"Allow Always\" means you won't be asked again for this type of action."
    }
}

/// Short status shown after the user taps an approval button.
pub(super) fn response_status(response: &ApprovalResponse) -> (&'static str, &'static str) {
    match response {
        ApprovalResponse::AllowOnce => ("✅", "Allowed this once"),
        ApprovalResponse::AllowSession => ("✅", "Allowed for this session"),
        ApprovalResponse::AllowAlways => ("✅", "Always allowed"),
        ApprovalResponse::Deny => ("❌", "Denied"),
    }
}

/// Optional follow-up detail after the status line.
pub(super) fn response_status_detail(response: &ApprovalResponse) -> Option<&'static str> {
    match response {
        ApprovalResponse::AllowSession => Some("You'll be asked again after restart."),
        ApprovalResponse::AllowAlways => Some("You won't be asked again for this type of action."),
        _ => None,
    }
}

/// Strip the pre-decision footer (button hint) from an approval prompt message.
fn strip_approval_footer(text: &str) -> String {
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
pub(super) fn finalize_approval_message(original: &str, response: &ApprovalResponse) -> String {
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
        let end = crate::utils::floor_char_boundary(command, MAX_CMD_DISPLAY);
        format!(
            "{}...\n[truncated — {} chars total]",
            &command[..end],
            command.len()
        )
    } else {
        command.to_string()
    }
}

/// Build the inline keyboard for a command-approval prompt.
pub(super) fn build_approval_keyboard(
    approval_id: &str,
    use_session_button: bool,
    one_time_only: bool,
) -> InlineKeyboardMarkup {
    if one_time_only {
        InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("Allow Once", format!("approve:once:{}", approval_id)),
            InlineKeyboardButton::callback("Deny", format!("approve:deny:{}", approval_id)),
        ]])
    } else if use_session_button {
        InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("Allow Once", format!("approve:once:{}", approval_id)),
            InlineKeyboardButton::callback(
                "Allow Session",
                format!("approve:session:{}", approval_id),
            ),
            InlineKeyboardButton::callback("Deny", format!("approve:deny:{}", approval_id)),
        ]])
    } else {
        InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback("Allow Once", format!("approve:once:{}", approval_id)),
            InlineKeyboardButton::callback(
                "Allow Always",
                format!("approve:always:{}", approval_id),
            ),
            InlineKeyboardButton::callback("Deny", format!("approve:deny:{}", approval_id)),
        ]])
    }
}

/// Build the HTML body for a Telegram command-approval prompt.
pub(super) fn build_approval_message_text(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
    use_session_button: bool,
    one_time_only: bool,
) -> String {
    let display_cmd = truncate_command(command);
    let escaped_cmd = html_escape(&display_cmd);
    let (risk_icon, risk_label, risk_subtitle) = risk_header(risk_level);

    let mut text = format!(
        "{} <b>{}</b>\n{}\n\n<b>Requested action</b>\n{}",
        risk_icon, risk_label, risk_subtitle, escaped_cmd
    );

    if !warnings.is_empty() {
        text.push('\n');
        for warning in warnings {
            text.push_str(&format!("\n• {}", html_escape(warning)));
        }
    }

    text.push_str(&format!(
        "\n\n<i>{}</i>",
        html_escape(allow_button_hint(use_session_button, one_time_only))
    ));

    text
}

/// Build the Markdown body for a Discord command-approval prompt.
pub(super) fn build_approval_message_discord(
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

/// Build the mrkdwn body for a Slack command-approval prompt.
pub(super) fn build_approval_message_slack(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
    use_session_button: bool,
    one_time_only: bool,
) -> String {
    let display_cmd = truncate_command(command);
    let (risk_icon, risk_label, risk_subtitle) = risk_header(risk_level);

    let mut text = format!(
        "{} *{}*\n{}\n\n*Requested action*\n```{display_cmd}```",
        risk_icon, risk_label, risk_subtitle
    );

    if !warnings.is_empty() {
        text.push('\n');
        for warning in warnings {
            text.push_str(&format!("\n• {warning}"));
        }
    }

    text.push_str(&format!(
        "\n\n_{}_",
        allow_button_hint(use_session_button, one_time_only)
    ));

    text
}

/// Build the inline keyboard for a scheduled-goal-confirmation prompt.
pub(super) fn build_goal_confirmation_keyboard(
    approval_id: &str,
    style: GoalConfirmationStyle,
) -> InlineKeyboardMarkup {
    match style {
        GoalConfirmationStyle::Standard => InlineKeyboardMarkup::new(vec![vec![
            InlineKeyboardButton::callback(
                "Approve goal ✅",
                format!("goal:confirm:{approval_id}"),
            ),
            InlineKeyboardButton::callback("Cancel", format!("goal:cancel:{approval_id}")),
        ]]),
        GoalConfirmationStyle::Autopilot => InlineKeyboardMarkup::new(vec![
            vec![InlineKeyboardButton::callback(
                "Enable Autopilot ✅",
                format!("autopilot:confirm:{approval_id}"),
            )],
            vec![
                InlineKeyboardButton::callback(
                    "Edit permissions",
                    format!("autopilot:edit:{approval_id}"),
                ),
                InlineKeyboardButton::callback("Cancel", format!("autopilot:cancel:{approval_id}")),
            ],
        ]),
    }
}

fn goal_detail_section(detail: &str) -> (&'static str, &str) {
    for (prefix, heading) in [
        ("Objective:", "🎯 Objective"),
        ("Constraints:", "🛡️ Guardrails"),
        ("Success criteria:", "✅ Success means"),
        ("Stop conditions:", "🛑 Stops when"),
        ("Pinned strategy:", "🧭 Strategy"),
        ("Observations allowed:", "👁️ Read access"),
        ("Allowed mutation effects:", "✍️ External changes"),
        ("Allowed targets:", "🎯 Allowed targets"),
        ("Exact operation scopes:", "🔧 Operation scope"),
        ("Mutation limits:", "📏 Action limits"),
        ("Review interval:", "🔄 Review timing"),
        ("Expiration:", "⏱️ Duration"),
        ("Autonomy mode:", "🚀 Operating mode"),
        ("Owner checkpoints:", "🙋 Owner checkpoints"),
        ("Recovery policy:", "♻️ Recovery"),
        ("Confirmation binding:", "🔏 Confirmation binding"),
        ("Review effort:", "🧠 Review effort"),
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

/// Build lossless plain-text pages for a goal-confirmation prompt.
///
/// Mandate confirmations can contain complete authority scopes and policy text,
/// which routinely exceed Telegram's 4096-character message limit. Keep enough
/// headroom for page labels and the final confirmation instruction, and put the
/// inline keyboard on the final page only. Plain text avoids splitting inside
/// generated HTML entities while preserving every disclosed detail.
pub(super) fn build_goal_confirmation_pages(
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
    let mut body = format!("{subject}\n{summary}\n\n{waiting_copy}");
    for detail in details {
        let (heading, value) = goal_detail_section(detail);
        if heading == "🎯 Objective" && normalized_goal_text(value) == summary_normalized {
            continue;
        }
        body.push_str(&format!("\n\n{heading}\n{}", readable_policy(value)));
    }

    let chunks = split_message(&body, PAGE_BODY_BYTES);
    let page_count = chunks.len();
    chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let title = match style {
                GoalConfirmationStyle::Standard => "🔐 Review goal",
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
                        "\n\nReady to proceed? Approve this exact goal or cancel it below."
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
        let prompt = build_approval_message_text(
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
        let prompt = build_approval_message_text("rm -rf /", RiskLevel::Critical, &[], true, false);
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::Deny);
        assert!(final_msg.contains("❌ Denied"));
    }

    #[test]
    fn one_time_only_prompt_has_no_persistent_choice() {
        let keyboard = build_approval_keyboard("approval-1", false, true);
        let serialized = serde_json::to_string(&keyboard).unwrap();
        assert!(serialized.contains("Allow Once"));
        assert!(serialized.contains("Deny"));
        assert!(!serialized.contains("Allow Always"));
        assert!(!serialized.contains("Allow Session"));

        let prompt = build_approval_message_text(
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
            .contains("Approve this exact goal or cancel it"));
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
        assert!(rendered.starts_with("🔐 Review goal before activation"));
        assert_eq!(rendered.matches("Steward the synthetic account").count(), 1);
        assert!(rendered.contains("🛡️ Guardrails\n1. Verify identity.\n2. Never retry"));
        assert!(rendered.contains("⏱️ Duration"));
        assert!(rendered.contains("Nothing starts until you approve this goal."));
    }

    #[test]
    fn autopilot_confirmation_is_structurally_distinct() {
        let keyboard =
            build_goal_confirmation_keyboard("approval-1", GoalConfirmationStyle::Autopilot);
        let serialized = serde_json::to_string(&keyboard).unwrap();
        assert!(serialized.contains("Enable Autopilot"));
        assert!(serialized.contains("Edit permissions"));
        assert!(serialized.contains("autopilot:confirm:approval-1"));
        assert!(!serialized.contains("goal:confirm"));

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
