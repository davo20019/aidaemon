//! Approval-message rendering shared across chat channels.
//!
//! Pure helpers that build inline keyboards and message text for command-approval
//! and scheduled-goal-confirmation prompts.

use crate::split::split_message;
use aidaemon_core::{ApprovalCard, AskReason, CardAction, HighImpact};
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

/// Build the Markdown body for a command-approval prompt without typed card
/// facts: the risk header, the action in a code block and the warnings.
pub fn build_approval_message_markdown(
    command: &str,
    risk_level: RiskLevel,
    warnings: &[String],
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
    text
}

/// One approval to ask for: the request as the channel received it, and
/// its card's typed facts when the asker gave them.
#[derive(Debug, Clone, Copy)]
pub struct ApprovalAsk<'a> {
    /// The request's summary (what a channel without card rendering shows).
    pub command: &'a str,
    pub card: Option<&'a ApprovalCard>,
    pub risk_level: RiskLevel,
    pub warnings: &'a [String],
    pub permission_mode: PermissionMode,
    /// The grant covers this call only: no session or lasting choice.
    pub one_time_only: bool,
}

/// The info string of a fenced block whose content a channel shows
/// collapsed (Telegram: an expandable blockquote). Channels without it show
/// an ordinary code block.
pub const COLLAPSED_FENCE: &str = "details";

/// How much of a command the collapsed block keeps: its head and tail
/// around a marked cut.
const CARD_COMMAND_CHARS: usize = 1_200;

/// Build an approval prompt's Markdown: the typed card when there is one,
/// else the summary and warnings as they are.
pub fn build_approval_ask_markdown(ask: &ApprovalAsk<'_>) -> String {
    match ask.card {
        Some(card) => build_card_markdown(card, ask.risk_level, ask.warnings),
        None => build_approval_message_markdown(ask.command, ask.risk_level, ask.warnings),
    }
}

/// A card from typed facts: a warning only for a high-impact effect, one
/// plain sentence saying what will happen, one line on why the owner is
/// asked, and the exact command (with any notes) collapsed underneath.
pub fn build_card_markdown(card: &ApprovalCard, risk: RiskLevel, notes: &[String]) -> String {
    let mut lines = Vec::new();
    if let Some(warning) = impact_warning(card.impact.as_ref(), risk) {
        lines.push(warning);
    }
    lines.push(format!("**{}**", card_headline(card)));
    if let Some(why) = card_why(card) {
        lines.push(why);
    }
    let mut text = lines.join("\n");
    let detail = match &card.action {
        CardAction::Command { line, .. } => Some(shorten_middle(line, CARD_COMMAND_CHARS)),
        CardAction::Use { detail, .. } => detail
            .as_deref()
            .map(|detail| shorten_middle(detail, CARD_COMMAND_CHARS)),
        _ => None,
    };
    let notes: Vec<&str> = notes
        .iter()
        .map(|note| note.trim())
        .filter(|note| !note.is_empty())
        .collect();
    if detail.is_some() || !notes.is_empty() {
        let mut block = detail.unwrap_or_default();
        for note in notes {
            if !block.is_empty() {
                block.push_str("\n\n");
            }
            block.push_str(note);
        }
        // A fence inside the content would end the block early.
        let block = block.replace("```", "'''");
        text.push_str(&format!("\n\n```{COLLAPSED_FENCE}\n{block}\n```"));
    }
    text
}

/// The warning lines a channel without card rendering shows: the ask's
/// own, then the card's facts its summary does not carry (the effect, the
/// check's sentence, the exact command).
pub fn card_as_warnings(ask: &ApprovalAsk<'_>) -> Vec<String> {
    let mut warnings = ask.warnings.to_vec();
    let Some(card) = ask.card else {
        return warnings;
    };
    if let Some(warning) = impact_warning(card.impact.as_ref(), ask.risk_level) {
        warnings.push(warning.trim_start_matches("⚠️ ").replace("**", ""));
    }
    if let Some(AskReason::ReadOutsideContent {
        check_says: Some(says),
    }) = &card.reason
    {
        warnings.push(format!("Safety check: {says}"));
    }
    match &card.action {
        CardAction::Command { line, .. } => {
            warnings.push(format!(
                "Details: {}",
                shorten_middle(line, CARD_COMMAND_CHARS)
            ));
        }
        CardAction::Use {
            detail: Some(detail),
            ..
        } => warnings.push(format!(
            "Details: {}",
            shorten_middle(detail, CARD_COMMAND_CHARS)
        )),
        _ => {}
    }
    warnings
}

/// `text` with its middle cut at a marker when longer than `cap` chars.
fn shorten_middle(text: &str, cap: usize) -> String {
    let length = text.chars().count();
    if length <= cap {
        return text.to_string();
    }
    let tail_chars = cap / 4;
    let head: String = text.chars().take(cap - tail_chars).collect();
    let tail: String = text.chars().skip(length - tail_chars).collect();
    format!("{head} … ({} characters left out) … {tail}", length - cap)
}

/// The warning line, only for an effect worth one: deleting, sending,
/// spending or writing outside the work area, named. A critical request
/// with no named effect gets a plain caution; ordinary work gets none.
fn impact_warning(impact: Option<&HighImpact>, risk: RiskLevel) -> Option<String> {
    let line = match impact {
        Some(HighImpact::Deletes { targets }) if !targets.is_empty() => {
            format!("This deletes {}.", plain_list(targets, 3))
        }
        Some(HighImpact::Deletes { .. }) => "This deletes files.".to_string(),
        Some(HighImpact::Sends) => "This sends something to another service.".to_string(),
        Some(HighImpact::Spends) => "This spends money.".to_string(),
        Some(HighImpact::WritesOutsideWorkArea) => {
            "This writes outside your work area.".to_string()
        }
        Some(_) => "This has effects beyond your computer.".to_string(),
        None if risk >= RiskLevel::Critical => {
            "This can change a lot on your computer.".to_string()
        }
        None => return None,
    };
    Some(format!("⚠️ **{line}**"))
}

/// "a", "a and b", "a, b and c", "a, b, c and 2 more".
fn plain_list(items: &[String], shown: usize) -> String {
    let named: Vec<&str> = items.iter().take(shown).map(String::as_str).collect();
    let more = items.len().saturating_sub(shown);
    match (named.split_last(), more) {
        (None, _) => String::new(),
        (Some((last, [])), 0) => (*last).to_string(),
        (Some((last, rest)), 0) => format!("{} and {last}", rest.join(", ")),
        (Some(_), more) => format!("{} and {more} more", named.join(", ")),
    }
}

/// What will happen, in one sentence that names who asks.
pub fn card_headline(card: &ApprovalCard) -> String {
    let who = card
        .asked_by
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("Your assistant");
    let what = match &card.action {
        CardAction::Command {
            looks_inside,
            title,
            ..
        } => {
            let title = title
                .as_deref()
                .map(|title| title.trim().trim_end_matches(['.', ':']))
                .filter(|title| !title.is_empty());
            if !looks_inside.is_empty() {
                let noun = if looks_inside.len() == 1 {
                    "folder"
                } else {
                    "folders"
                };
                format!("look inside your {} {noun}", plain_list(looks_inside, 3))
            } else if let Some(title) = title {
                format!("run a command on your computer: {title}")
            } else {
                "run a command on your computer".to_string()
            }
        }
        CardAction::OpenLocalSite { site, .. } => {
            format!("open {site} in the browser, a site on this computer")
        }
        CardAction::Use { what, .. } => what.trim().trim_end_matches('.').to_string(),
        _ => "do something that needs your OK".to_string(),
    };
    format!("{who} wants to {what}.")
}

/// Why the owner is asked, in a line or two.
fn card_why(card: &ApprovalCard) -> Option<String> {
    let mut lines = Vec::new();
    if let Some(AskReason::ReadOutsideContent { check_says }) = &card.reason {
        lines.push(
            "It read content from outside, like a web page, so I'm checking with you \
             before it runs this."
                .to_string(),
        );
        if let Some(says) = check_says
            .as_deref()
            .map(str::trim)
            .filter(|says| !says.is_empty())
        {
            lines.push(format!("_Safety check: {says}_"));
        }
    }
    if let Some(AskReason::AfterADecline { declined }) = &card.reason {
        let what = declined
            .as_deref()
            .map(str::trim)
            .filter(|what| !what.is_empty())
            .unwrap_or("a step");
        lines.push(format!(
            "You said no to {what} earlier in this task, so I'm checking with you before \
             steps whose effects I can't fully tell."
        ));
    }
    if let CardAction::OpenLocalSite { once_minutes, .. } = &card.action {
        lines.push(format!(
            "Allow only if you asked for it. Allow covers the next {once_minutes} minutes; \
             the longer choice lasts until this browser session ends."
        ));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
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
        );
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::AllowAlways);
        assert!(final_msg.contains("Always allowed"));
        assert!(final_msg.contains("won't be asked again"));
        assert!(final_msg.contains("Open website"));
        // A card sent before footers were dropped still loses its footer.
        let old = format!("{prompt}\n\n*\"Allow Always\" means you won't be asked again.*");
        let final_msg = finalize_approval_message(&old, &ApprovalResponse::Deny);
        assert!(!final_msg.contains("Allow Always\" means"), "{final_msg}");
    }

    #[test]
    fn finalize_denied_shows_denied_status() {
        let prompt = build_approval_message_markdown("rm -rf /", RiskLevel::Critical, &[]);
        let final_msg = finalize_approval_message(&prompt, &ApprovalResponse::Deny);
        assert!(final_msg.contains("❌ Denied"));
    }

    const INSPECT: &str = "cd /Users/davidloor/projects && for d in claude-drupal \
        claude-wordpress; do echo \"== $d\"; ls -la $d | head -30; ls $d/.ddev 2>&1 | head; \
        (cd $d && ddev describe 2>&1 | head -25); done";

    fn ask<'a>(card: &'a ApprovalCard, risk: RiskLevel, warnings: &'a [String]) -> String {
        build_approval_ask_markdown(&ApprovalAsk {
            command: "summary a channel without cards would show",
            card: Some(card),
            risk_level: risk,
            warnings,
            permission_mode: PermissionMode::Default,
            one_time_only: true,
        })
    }

    /// The screenshot's card (live 2026-11-12), redone: a helper looking
    /// inside two folders gets one plain sentence, one line on why, the
    /// check's own short sentence, and the command collapsed below. No
    /// warning, no prose in a code block, no footer.
    #[test]
    fn a_low_risk_read_only_card_says_what_happens_without_alarm() {
        let card = ApprovalCard {
            asked_by: Some("Drupal install".to_string()),
            action: CardAction::Command {
                line: INSPECT.to_string(),
                title: Some("check claude site dirs and docker".to_string()),
                looks_inside: vec!["claude-drupal".to_string(), "claude-wordpress".to_string()],
            },
            reason: Some(AskReason::ReadOutsideContent {
                check_says: Some(
                    "Looking inside these folders doesn't look needed to install Drupal."
                        .to_string(),
                ),
            }),
            impact: None,
        };
        let text = ask(&card, RiskLevel::High, &[]);
        let expected = format!(
            "**Drupal install wants to look inside your claude-drupal and claude-wordpress \
             folders.**\n\
             It read content from outside, like a web page, so I'm checking with you before \
             it runs this.\n\
             _Safety check: Looking inside these folders doesn't look needed to install \
             Drupal._\n\n\
             ```{COLLAPSED_FENCE}\n{INSPECT}\n```"
        );
        assert_eq!(text, expected);
        for gone in [
            "Review carefully",
            "significant effects",
            "applies only",
            "⚠️",
            "owner",
        ] {
            assert!(!text.contains(gone), "{gone}: {text}");
        }
        // Prose never sits in a fenced block; only the command does.
        let fenced = text.split("```").nth(1).unwrap();
        assert!(!fenced.contains("wants to"), "{text}");
    }

    /// A high-impact effect is the one thing that earns a warning, and it
    /// says which effect; the title names the step when nothing typed does.
    #[test]
    fn a_high_risk_delete_card_names_the_effect() {
        let card = ApprovalCard {
            asked_by: None,
            action: CardAction::Command {
                line: "rm -rf build dist".to_string(),
                title: Some("clean old builds".to_string()),
                looks_inside: Vec::new(),
            },
            reason: None,
            impact: Some(HighImpact::Deletes {
                targets: vec!["build".to_string(), "dist".to_string()],
            }),
        };
        let notes = ["Execution target: host (local) workspace /srv/app".to_string()];
        let text = ask(&card, RiskLevel::High, &notes);
        assert!(
            text.starts_with(
                "⚠️ **This deletes build and dist.**\n\
                 **Your assistant wants to run a command on your computer: clean old builds.**"
            ),
            "{text}"
        );
        assert!(text.ends_with(&format!(
            "```{COLLAPSED_FENCE}\nrm -rf build dist\n\nExecution target: host (local) \
             workspace /srv/app\n```"
        )));
        // The same command with no effect worth a warning shows none, even
        // when its request is rated high.
        let plain = ApprovalCard {
            impact: None,
            ..card
        };
        assert!(!ask(&plain, RiskLevel::High, &[]).contains("⚠️"));
        assert!(ask(&plain, RiskLevel::Critical, &[]).starts_with("⚠️ **This can change"));
    }

    #[test]
    fn a_browser_card_names_the_site_and_how_long_allow_lasts() {
        let card = ApprovalCard {
            asked_by: Some("WordPress install".to_string()),
            ..ApprovalCard::new(CardAction::OpenLocalSite {
                site: "claude-wordpress.ddev.site".to_string(),
                once_minutes: 30,
            })
        };
        let text = ask(&card, RiskLevel::High, &[]);
        assert_eq!(
            text,
            "**WordPress install wants to open claude-wordpress.ddev.site in the browser, a \
             site on this computer.**\n\
             Allow only if you asked for it. Allow covers the next 30 minutes; the longer \
             choice lasts until this browser session ends."
        );
    }

    #[test]
    fn a_long_command_is_cut_in_the_middle_and_a_fence_cannot_end_the_block() {
        let line = format!("echo ```; {} tail-marker", "x".repeat(5_000));
        let card = ApprovalCard::new(CardAction::Command {
            line,
            title: None,
            looks_inside: Vec::new(),
        });
        let text = ask(&card, RiskLevel::Medium, &[]);
        assert!(text.starts_with("**Your assistant wants to run a command on your computer.**"));
        assert!(text.contains("characters left out"));
        assert!(text.contains("tail-marker"));
        assert_eq!(text.matches("```").count(), 2, "{text}");
        assert!(text.len() < 2_000);
    }

    /// A channel without card rendering still gets the card's facts as
    /// warning lines under its summary.
    #[test]
    fn a_card_flattens_for_channels_without_cards() {
        let card = ApprovalCard {
            reason: Some(AskReason::ReadOutsideContent {
                check_says: Some("This isn't part of your request.".to_string()),
            }),
            impact: Some(HighImpact::Deletes {
                targets: Vec::new(),
            }),
            ..ApprovalCard::new(CardAction::Command {
                line: "rm -rf x".to_string(),
                title: None,
                looks_inside: Vec::new(),
            })
        };
        let notes = ["note".to_string()];
        let warnings = card_as_warnings(&ApprovalAsk {
            command: "Run a command on your computer?",
            card: Some(&card),
            risk_level: RiskLevel::High,
            warnings: &notes,
            permission_mode: PermissionMode::Default,
            one_time_only: true,
        });
        assert_eq!(
            warnings,
            [
                "note",
                "This deletes files.",
                "Safety check: This isn't part of your request.",
                "Details: rm -rf x"
            ]
        );
    }

    #[test]
    fn lists_read_plainly() {
        let names = |n: usize| (1..=n).map(|i| format!("f{i}")).collect::<Vec<_>>();
        assert_eq!(plain_list(&names(1), 3), "f1");
        assert_eq!(plain_list(&names(2), 3), "f1 and f2");
        assert_eq!(plain_list(&names(3), 3), "f1, f2 and f3");
        assert_eq!(plain_list(&names(5), 3), "f1, f2, f3 and 2 more");
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
        assert!(
            pages
                .last()
                .unwrap()
                .contains("Approve and activate this plan, or cancel")
        );
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
