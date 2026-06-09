//! Approval-message rendering shared across chat channels.
//!
//! Pure helpers that build inline keyboards and message text for command-approval
//! and scheduled-goal-confirmation prompts.

use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};

use super::formatting::html_escape;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::types::ApprovalResponse;

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
pub(super) fn allow_button_hint(use_session_button: bool) -> &'static str {
    if use_session_button {
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
) -> InlineKeyboardMarkup {
    if use_session_button {
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
        html_escape(allow_button_hint(use_session_button))
    ));

    text
}

/// Build the Markdown body for a Discord command-approval prompt.
pub(super) fn build_approval_message_discord(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
    use_session_button: bool,
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

    text.push_str(&format!("\n\n*{}*", allow_button_hint(use_session_button)));

    text
}

/// Build the mrkdwn body for a Slack command-approval prompt.
pub(super) fn build_approval_message_slack(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
    use_session_button: bool,
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

    text.push_str(&format!("\n\n_{}_", allow_button_hint(use_session_button)));

    text
}

/// Build the inline keyboard for a scheduled-goal-confirmation prompt.
pub(super) fn build_goal_confirmation_keyboard(approval_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("Confirm ✅", format!("goal:confirm:{}", approval_id)),
        InlineKeyboardButton::callback("Cancel ❌", format!("goal:cancel:{}", approval_id)),
    ]])
}

/// Build the HTML body for a scheduled-goal-confirmation prompt.
pub(super) fn build_goal_confirmation_text(goal_description: &str, details: &[String]) -> String {
    let escaped_desc = html_escape(goal_description);
    let mut text = format!(
        "📅 <b>Confirm scheduled goal</b>\n\n<code>{}</code>",
        escaped_desc
    );

    for detail in details {
        text.push_str(&format!("\n• {}", html_escape(detail)));
    }

    text
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
        );
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::AllowAlways);
        assert!(!final_msg.contains("Allow Always\" means"));
        assert!(final_msg.contains("Always allowed"));
        assert!(final_msg.contains("won't be asked again"));
        assert!(final_msg.contains("Open website"));
    }

    #[test]
    fn finalize_denied_shows_denied_status() {
        let prompt = build_approval_message_text("rm -rf /", RiskLevel::Critical, &[], true);
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::Deny);
        assert!(final_msg.contains("❌ Denied"));
    }
}
