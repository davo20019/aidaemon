//! Approval-message rendering for the Telegram channel.
//!
//! Pure helpers that build inline keyboards and HTML message text for
//! command-approval and scheduled-goal-confirmation prompts. Extracted
//! verbatim from `telegram.rs` (behavior-preserving move); the `Channel`
//! trait methods in `telegram.rs` call these to construct the UI before
//! sending it and awaiting the user's response.

use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};

use super::formatting::html_escape;
use crate::tools::command_risk::{PermissionMode, RiskLevel};

/// Truncate command display to fit Telegram's 4096 char limit.
/// Reserve ~200 chars for risk label, warnings, buttons, and footer.
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

/// Build the HTML body for a command-approval prompt.
pub(super) fn build_approval_message_text(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
    use_session_button: bool,
    short_id: &str,
) -> String {
    let display_cmd = if command.len() > MAX_CMD_DISPLAY {
        let end = crate::utils::floor_char_boundary(command, MAX_CMD_DISPLAY);
        format!(
            "{}...\n[truncated — {} chars total]",
            &command[..end],
            command.len()
        )
    } else {
        command.to_string()
    };
    let escaped_cmd = html_escape(&display_cmd);

    // Build message with risk info
    let (risk_icon, risk_label) = match risk_level {
        RiskLevel::Safe => ("ℹ️", "New command"),
        RiskLevel::Medium => ("⚠️", "Medium risk"),
        RiskLevel::High => ("🔶", "High risk"),
        RiskLevel::Critical => ("🚨", "Critical risk"),
    };

    let mut text = format!(
        "{} <b>{}</b>\n\n<code>{}</code>",
        risk_icon, risk_label, escaped_cmd
    );

    if !warnings.is_empty() {
        text.push('\n');
        for warning in warnings {
            text.push_str(&format!("\n• {}", html_escape(warning)));
        }
    }

    // Add explanation based on which button is shown
    if use_session_button {
        text.push_str("\n\n<i>\"Allow Session\" approves this command type until restart.</i>");
    } else {
        text.push_str("\n\n<i>\"Allow Always\" permanently approves this command type.</i>");
    }

    text.push_str(&format!("\n\n<i>[{}]</i>", short_id));

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
pub(super) fn build_goal_confirmation_text(
    goal_description: &str,
    details: &[String],
    short_id: &str,
) -> String {
    let escaped_desc = html_escape(goal_description);
    let mut text = format!(
        "📅 <b>Confirm scheduled goal</b>\n\n<code>{}</code>",
        escaped_desc
    );

    for detail in details {
        text.push_str(&format!("\n• {}", html_escape(detail)));
    }

    text.push_str(&format!("\n\n<i>[{}]</i>", short_id));

    text
}
