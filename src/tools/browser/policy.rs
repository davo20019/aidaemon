//! Browser action risk classification.
//!
//! This module provides a lightweight, self-contained classifier that assigns a
//! [`BrowserActionRisk`] to every browser action *before* it executes.  Task 7
//! (approval enforcement) consumes this output — this module has NO side effects
//! and makes NO approval decisions itself.
//!
//! ## Class definitions
//!
//! | Class            | Actions                                 | Notes                                   |
//! |------------------|-----------------------------------------|-----------------------------------------|
//! | `Observation`    | get_text, screenshot, scroll, wait, list_tabs | Read-only; no page mutations       |
//! | `Navigation`     | navigate, new_tab, switch_tab           | Changes current URL / active tab        |
//! | `Mutation`       | click, fill, execute_js                 | Alters page state; `execute_js` always sensitive |
//! | `Administrative` | close, close_tab, set_mode              | Lifecycle ops; not in the plan's 3 buckets but needed for completeness |
//!
//! Unknown actions (not in the schema) default to `Mutation` with
//! `sensitive = false` and `consequential = false`.  This is the *most
//! restrictive sane default*: Task 7 will prompt for anything Mutation-or-above,
//! so an unrecognised action is never silently treated as a free observation.
//!
//! ## Consequential signals
//!
//! A consequential flag is set when the action's primary text context (script for
//! `execute_js`; selector for `click`/`fill`) contains a standalone consequential
//! keyword (word-boundary match via [`crate::agent::keyword_match`]).
//!
//! The keyword scan on selectors is intentionally low-recall: real selectors are
//! usually opaque (`#btn-3`, `.checkout-cta`) and rarely contain plain English
//! verbs.  The primary consequential signals are:
//! - the action type (`execute_js` → always sensitive), and
//! - the script text for `execute_js`.
//!
//! Do NOT rely on selector keywords as the main guard — they are a bonus signal
//! only.

// `classify` is intentionally public so Task 7 can call it.  It has no callers
// yet (Task 7 is a separate commit), so `#[allow(dead_code)]` prevents a spurious
// Clippy warning until Task 7 wires approval enforcement.
#![allow(dead_code)]

use crate::agent::keyword_match;
use crate::tools::web_fetch::{classify_blocked_host, BlockedHostClass};

// ─── Network policy glue (SSRF) ─────────────────────────────────────────────────
//
// ONE source of truth for "what is a blocked host" lives in
// `web_fetch::classify_blocked_host` / `validate_url_for_ssrf`. This module only
// adapts that shared decision into a SECRET-SAFE, structured error for the
// browser tool: the returned message names ONLY the host class — never the URL,
// path, query, or any embedded credentials. (The classifier never receives the
// chance to echo caller data: it returns a fixed enum, and we render only its
// `.label()`.)

/// A request that the private-network policy refuses to allow. Carries only the
/// host CLASS — by construction it cannot leak the URL or any secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockedRequest {
    pub class: BlockedHostClass,
}

impl BlockedRequest {
    /// A user/LLM-facing, secret-free message: only the fixed host-class label.
    pub fn message(self) -> String {
        format!("Navigation blocked: target is a {}", self.class.label())
    }
}

/// Validate a URL against the shared private-network policy.
///
/// Returns `Ok(())` for an allowed public http(s) URL, or `Err(BlockedRequest)`
/// naming only the host class for anything the shared policy blocks (loopback,
/// RFC1918/unique-local private ranges, link-local/metadata, IPv4-mapped IPv6,
/// disallowed schemes, malformed URLs).
///
/// This is the single seam every browser code path (tool-initiated navigation,
/// final-URL revalidation, and — if/when CDP interception lands — per-request
/// interception) should call, so the policy never diverges between surfaces.
pub fn validate_network_url(url: &str) -> Result<(), BlockedRequest> {
    match classify_blocked_host(url) {
        None => Ok(()),
        Some(class) => Err(BlockedRequest { class }),
    }
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// Coarse risk category for a browser action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrowserRiskClass {
    /// Read-only: get_text, screenshot, wait, list_tabs.
    Observation,
    /// Changes the active URL or tab: navigate, new_tab, switch_tab.
    Navigation,
    /// Modifies page state: click, fill, execute_js.
    Mutation,
    /// Lifecycle operations: close, close_tab, set_mode.
    ///
    /// Not listed in the task's three named buckets but required for exhaustive
    /// coverage.  Task 7 can treat these as low-risk (session teardown / mode
    /// switch) or prompt at its discretion.
    Administrative,
}

/// Full risk descriptor for a single browser action invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrowserActionRisk {
    /// Coarse class (see [`BrowserRiskClass`]).
    pub class: BrowserRiskClass,
    /// `true` when the action has elevated inherent risk regardless of context.
    /// `execute_js` is **always** sensitive because arbitrary JavaScript can
    /// exfiltrate credentials, submit forms, or perform account actions.
    pub sensitive: bool,
    /// `true` when the action's text context (script or selector) contains a
    /// consequential keyword (word-boundary match).
    pub consequential: bool,
}

// ─── Keyword list ─────────────────────────────────────────────────────────────

/// Keywords whose standalone presence in script text or a selector signals a
/// potentially high-impact operation.
///
/// Sourced from the task spec: submit, purchase, delete, send, publish,
/// permission, authentication, account-management.  Extended with close
/// financial/session synonyms judged in-scope: buy, checkout, pay, transfer,
/// "sign out", "log out".
///
/// Word-boundary matching (via [`keyword_match`]) means derived forms like
/// "submitted", "purchasing", "deleted", "sender" do NOT match — which is the
/// desired low-false-positive behaviour.
const CONSEQUENTIAL_KEYWORDS: &[&str] = &[
    "submit",
    "purchase",
    "delete",
    "send",
    "publish",
    "permission",
    "authentication",
    "account",
    // Extended synonyms
    "buy",
    "checkout",
    "pay",
    "transfer",
    "sign out",
    "log out",
];

/// Return `true` if `text` contains any [`CONSEQUENTIAL_KEYWORDS`] as a
/// complete word (or phrase).
fn is_consequential(text: &str) -> bool {
    CONSEQUENTIAL_KEYWORDS
        .iter()
        .any(|kw| keyword_match(text, kw))
}

// ─── Classifier ───────────────────────────────────────────────────────────────

/// Classify the risk of a browser action.
///
/// # Parameters
/// - `action`   — the action string from the tool call (`"click"`, `"navigate"`, …).
/// - `selector` — CSS selector, if the action accepts one (`click`, `fill`, `get_text`, `wait`).
/// - `script`   — JavaScript source, if the action accepts one (`execute_js`).
///
/// # Returns
/// A [`BrowserActionRisk`] that describes the class, sensitivity, and
/// whether a consequential keyword was found in the relevant context.
pub fn classify(action: &str, selector: Option<&str>, script: Option<&str>) -> BrowserActionRisk {
    match action {
        // ── Observation ───────────────────────────────────────────────────────
        "get_text" | "screenshot" | "scroll" | "wait" | "list_tabs" | "get_console_logs"
        | "get_network_errors" => BrowserActionRisk {
            class: BrowserRiskClass::Observation,
            sensitive: false,
            consequential: false,
        },

        // ── Navigation ────────────────────────────────────────────────────────
        "navigate" | "new_tab" | "switch_tab" => BrowserActionRisk {
            class: BrowserRiskClass::Navigation,
            sensitive: false,
            consequential: false,
        },

        // ── Mutation: click / fill ────────────────────────────────────────────
        "click" | "fill" => BrowserActionRisk {
            class: BrowserRiskClass::Mutation,
            sensitive: false,
            // Scan the selector for consequential keywords (low-recall — see
            // module-level doc comment).
            consequential: selector.map(is_consequential).unwrap_or(false),
        },

        // ── Mutation: execute_js ──────────────────────────────────────────────
        "execute_js" => BrowserActionRisk {
            class: BrowserRiskClass::Mutation,
            // Always sensitive: arbitrary JS can exfiltrate credentials, submit
            // forms, or perform account actions regardless of script content.
            sensitive: true,
            // Consequential scan on script text; if no script provided, default
            // to false — but Task 7 should still prompt because sensitive=true.
            consequential: script.map(is_consequential).unwrap_or(false),
        },

        // ── Administrative ────────────────────────────────────────────────────
        "close" | "close_tab" | "set_mode" => BrowserActionRisk {
            class: BrowserRiskClass::Administrative,
            sensitive: false,
            consequential: false,
        },

        // ── Unknown ───────────────────────────────────────────────────────────
        // Default to the most restrictive sane class so that a future action
        // added to the schema is never accidentally treated as a free observation.
        _ => BrowserActionRisk {
            class: BrowserRiskClass::Mutation,
            sensitive: false,
            consequential: false,
        },
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Action-to-class mapping (table-driven) ────────────────────────────────

    #[test]
    fn observation_actions_map_correctly() {
        for action in &[
            "get_text",
            "screenshot",
            "scroll",
            "wait",
            "list_tabs",
            "get_console_logs",
            "get_network_errors",
        ] {
            let r = classify(action, None, None);
            assert_eq!(
                r.class,
                BrowserRiskClass::Observation,
                "action={action} should be Observation"
            );
            assert!(!r.sensitive, "action={action} should not be sensitive");
            assert!(
                !r.consequential,
                "action={action} should not be consequential"
            );
        }
    }

    #[test]
    fn navigation_actions_map_correctly() {
        for action in &["navigate", "new_tab", "switch_tab"] {
            let r = classify(action, None, None);
            assert_eq!(
                r.class,
                BrowserRiskClass::Navigation,
                "action={action} should be Navigation"
            );
            assert!(!r.sensitive, "action={action} should not be sensitive");
            assert!(
                !r.consequential,
                "action={action} should not be consequential"
            );
        }
    }

    #[test]
    fn mutation_actions_click_fill_map_correctly() {
        for action in &["click", "fill"] {
            let r = classify(action, None, None);
            assert_eq!(
                r.class,
                BrowserRiskClass::Mutation,
                "action={action} should be Mutation"
            );
            assert!(!r.sensitive, "action={action} should not be sensitive");
            assert!(
                !r.consequential,
                "action={action} should not be consequential"
            );
        }
    }

    #[test]
    fn execute_js_is_always_mutation_and_sensitive() {
        // Benign script: no consequential keyword, but still sensitive.
        let r = classify("execute_js", None, Some("1 + 1"));
        assert_eq!(r.class, BrowserRiskClass::Mutation);
        assert!(r.sensitive, "execute_js must always be sensitive");
        assert!(!r.consequential);

        // No script at all: still sensitive.
        let r2 = classify("execute_js", None, None);
        assert!(
            r2.sensitive,
            "execute_js with no script must still be sensitive"
        );
    }

    #[test]
    fn administrative_actions_map_correctly() {
        for action in &["close", "close_tab", "set_mode"] {
            let r = classify(action, None, None);
            assert_eq!(
                r.class,
                BrowserRiskClass::Administrative,
                "action={action} should be Administrative"
            );
            assert!(!r.sensitive, "action={action} should not be sensitive");
            assert!(
                !r.consequential,
                "action={action} should not be consequential"
            );
        }
    }

    #[test]
    fn unknown_action_defaults_to_mutation_not_sensitive_not_consequential() {
        let r = classify("teleport_browser", None, None);
        assert_eq!(
            r.class,
            BrowserRiskClass::Mutation,
            "unknown actions should default to Mutation (most restrictive sane class)"
        );
        assert!(!r.sensitive);
        assert!(!r.consequential);
    }

    // ── execute_js is ALWAYS sensitive ───────────────────────────────────────

    #[test]
    fn execute_js_always_sensitive_even_with_benign_script() {
        // A clearly benign expression must still be sensitive.
        let benign_scripts = [
            "document.title",
            "1 + 1",
            "window.location.href",
            "Array.from(document.querySelectorAll('a')).map(a => a.href)",
        ];
        for script in &benign_scripts {
            let r = classify("execute_js", None, Some(script));
            assert!(
                r.sensitive,
                "execute_js with benign script '{script}' must still be sensitive=true"
            );
        }
    }

    // ── Word-boundary consequential scan ─────────────────────────────────────
    //
    // The key contract: derived / compound forms must NOT trigger a false positive.
    // This exercises the `contains_keyword_as_words` (re-exported as `keyword_match`)
    // word-boundary guarantee.

    #[test]
    fn derived_forms_do_not_false_positive() {
        // "deleted" should NOT match keyword "delete"
        let r = classify("click", Some("#deleted-items"), None);
        assert!(
            !r.consequential,
            "'deleted' must not match 'delete' (word-boundary check)"
        );

        // "sender" should NOT match keyword "send"
        let r2 = classify("click", Some(".sender-info"), None);
        assert!(
            !r2.consequential,
            "'sender' must not match 'send' (word-boundary check)"
        );

        // "submitted" should NOT match keyword "submit"
        let r3 = classify("execute_js", None, Some("form.submitted = true;"));
        assert!(
            !r3.consequential,
            "'submitted' must not match 'submit' (word-boundary check)"
        );

        // "purchasing" should NOT match keyword "purchase"
        let r4 = classify("execute_js", None, Some("// this is a purchasing flow"));
        assert!(
            !r4.consequential,
            "'purchasing' must not match 'purchase' (word-boundary check)"
        );

        // "publisher" should NOT match keyword "publish"
        let r5 = classify("click", Some(".publisher-name"), None);
        assert!(
            !r5.consequential,
            "'publisher' must not match 'publish' (word-boundary check)"
        );
    }

    #[test]
    fn standalone_keyword_in_script_triggers_consequential() {
        // "submit" as a standalone whitespace-delimited token in a script → consequential.
        // The word-boundary splitter splits on WHITESPACE then trims edge punctuation.
        // "submit" (with trailing semicolon stripped) is a standalone token.
        let r = classify("execute_js", None, Some("// call submit; await result"));
        assert!(
            r.consequential,
            "'submit' as standalone word in script comment should be consequential"
        );

        // Explicit standalone word: "delete"
        let r2 = classify("execute_js", None, Some("// delete this record"));
        assert!(
            r2.consequential,
            "'delete' as standalone word should be consequential"
        );

        // "send" as standalone word in script
        // "send(message)" is one token; trimming ")" gives "send(message" which still has
        // inner punctuation → does NOT match "send". Use a comment form instead.
        let r3 = classify("execute_js", None, Some("// send email to user"));
        assert!(
            r3.consequential,
            "'send' standalone in comment should be consequential"
        );

        // "purchase" as standalone word
        let r4 = classify("execute_js", None, Some("initiating purchase flow"));
        assert!(
            r4.consequential,
            "'purchase' standalone should be consequential"
        );

        // Low-recall: "form.submit()" is a single token "form.submit" after stripping
        // trailing parens — NOT equal to keyword "submit" — this is INTENTIONAL.
        // The primary guard is sensitive=true (always set for execute_js), not
        // consequential detection in method calls.
        let r_low_recall = classify(
            "execute_js",
            None,
            Some("document.querySelector('form').submit()"),
        );
        // We do NOT assert consequential here — the low-recall behavior is documented.
        // Task 7 still prompts because sensitive=true.
        assert!(
            r_low_recall.sensitive,
            "execute_js is always sensitive even when consequential is false"
        );
    }

    #[test]
    fn opaque_selector_is_not_consequential() {
        // Typical opaque selectors used in real apps must NOT trigger.
        let opaque_selectors = [
            "#btn-3",
            ".checkout-cta",
            "[data-id='42']",
            "#submit-btn-container", // hyphenated — not a standalone word
            ".pay-later-link",       // hyphenated
        ];
        for sel in &opaque_selectors {
            let r = classify("click", Some(sel), None);
            assert!(
                !r.consequential,
                "opaque selector '{sel}' must not be consequential"
            );
        }
    }

    #[test]
    fn consequential_click_with_standalone_keyword_in_selector() {
        // A selector that literally contains a standalone consequential word.
        // word-boundary split on whitespace: "delete" is a token after trimming.
        // Note: "#delete" trims punctuation → "delete" → matches.
        let r = classify("click", Some("#delete"), None);
        assert!(
            r.consequential,
            "selector '#delete' (→ token 'delete') must be consequential"
        );

        // A selector with bare "submit" after punctuation trim
        let r2 = classify("click", Some(".submit"), None);
        assert!(
            r2.consequential,
            "selector '.submit' (→ token 'submit') must be consequential"
        );

        // A selector containing "account" as a standalone token
        let r3 = classify("fill", Some("#account-email"), None);
        // "#account-email" is one whitespace token, trimmed to "account-email" — hyphenated,
        // so it does NOT equal standalone "account". This is the documented low-recall behavior.
        assert!(
            !r3.consequential,
            "'#account-email' is hyphenated → does not match 'account' standalone"
        );
    }

    #[test]
    fn execute_js_consequential_scan_uses_script_not_selector() {
        // Consequential keyword only in selector: irrelevant for execute_js.
        let r = classify("execute_js", Some("delete"), Some("1 + 1"));
        // "1 + 1" has no consequential keyword — should NOT be consequential
        // (selector is ignored for execute_js per the spec).
        assert!(
            !r.consequential,
            "execute_js consequential scan uses script, not selector"
        );
        // But still sensitive.
        assert!(r.sensitive);
    }

    #[test]
    fn all_schema_actions_covered() {
        // Ensure every action in the browser schema has an explicit classification
        // (i.e. none fall through to the unknown-action default unexpectedly).
        let schema_actions = [
            "navigate",
            "screenshot",
            "click",
            "fill",
            "get_text",
            "execute_js",
            "wait",
            "list_tabs",
            "new_tab",
            "switch_tab",
            "close_tab",
            "set_mode",
            "close",
        ];

        let expected_classes = [
            BrowserRiskClass::Navigation,     // navigate
            BrowserRiskClass::Observation,    // screenshot
            BrowserRiskClass::Mutation,       // click
            BrowserRiskClass::Mutation,       // fill
            BrowserRiskClass::Observation,    // get_text
            BrowserRiskClass::Mutation,       // execute_js
            BrowserRiskClass::Observation,    // wait
            BrowserRiskClass::Observation,    // list_tabs
            BrowserRiskClass::Navigation,     // new_tab
            BrowserRiskClass::Navigation,     // switch_tab
            BrowserRiskClass::Administrative, // close_tab
            BrowserRiskClass::Administrative, // set_mode
            BrowserRiskClass::Administrative, // close
        ];

        for (action, expected) in schema_actions.iter().zip(expected_classes.iter()) {
            let r = classify(action, None, None);
            assert_eq!(
                r.class, *expected,
                "action={action}: expected {expected:?}, got {:?}",
                r.class
            );
        }
    }
}
