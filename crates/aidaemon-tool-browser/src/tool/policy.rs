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
//! | `Administrative` | render_pdf, close, close_tab, set_mode  | Local rendering/lifecycle operations |
//!
//! Unknown actions (not in the schema) default to `Mutation` with
//! `sensitive = false`.  This is the *most
//! restrictive sane default*: Task 7 will prompt for anything Mutation-or-above,
//! so an unrecognised action is never silently treated as a free observation.
//!
//! ## Point of action
//!
//! Whether a click or fill needs the owner every time is decided from typed
//! DOM facts about the element it targets ([`ElementFacts`], read from the
//! live page by a constant script), never from words in the selector, the
//! script or the page: an opaque selector (`#btn-3`) names nothing. See
//! [`click_needs_owner`] and [`fill_needs_owner`]; facts that cannot be read
//! count as point of action.

// `classify` is intentionally public so Task 7 can call it.  It has no callers
// yet (Task 7 is a separate commit), so `#[allow(dead_code)]` prevents a spurious
// Clippy warning until Task 7 wires approval enforcement.
#![allow(dead_code)]

pub use aidaemon_tools::web::local_site_key;
use aidaemon_tools::web::{BlockedHostClass, classify_blocked_host};

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

    /// Only a loopback target can be opened with the owner's approval.
    pub fn is_loopback(self) -> bool {
        self.class == BlockedHostClass::Loopback
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
    /// Local rendering/lifecycle operations: render_pdf, close, close_tab,
    /// set_mode.
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
}

// ─── Element facts ────────────────────────────────────────────────────────────

/// Typed DOM facts about the element a click or fill targets, read from the
/// live page by [`ELEMENT_FACTS_FN`]. The facts describe the nearest
/// interactive element at or above the selector's match (a label resolves to
/// its control). Text on the page (labels, accessible names) is never part
/// of the decision.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(default)]
pub struct ElementFacts {
    /// Lowercase tag name.
    pub tag: String,
    /// The explicit ARIA `role` attribute, lowercased.
    pub role: Option<String>,
    /// For an `<input>`, its `type` property, lowercased.
    pub input_type: Option<String>,
    /// The `autocomplete` attribute, lowercased (a token list).
    pub autocomplete: Option<String>,
    /// Scheme of the nearest `a[href]` ancestor-or-self, if any
    /// (`https`, `javascript`, ...).
    pub link_scheme: Option<String>,
    /// That link carries a `download` attribute.
    pub link_download: bool,
    /// The element is a submit control of a form (`button` of type submit,
    /// `input` of type submit or image).
    pub submits_form: bool,
    /// The effective method of the element's form (`formmethod` on the
    /// submitter, else the form's `method`), lowercased; `None` outside a form.
    pub form_method: Option<String>,
    /// The form is marked as a search (`role=search`, a `<search>` ancestor,
    /// or an `input type=search` inside it).
    pub search_form: bool,
    /// A disclosure control: `<summary>`, or `aria-expanded` /
    /// `aria-haspopup` present.
    pub disclosure: bool,
    /// A text-entry target: a text-like `<input>`, a `<textarea>`, or
    /// contenteditable content.
    pub editable: bool,
}

/// Constant JS function bound to the resolved element (`this`) through
/// `call_js_fn`; the selector is never interpolated into script source. It
/// returns the JSON shape of [`ElementFacts`].
pub const ELEMENT_FACTS_FN: &str = r#"function() {
  const INTERACTIVE = 'a[href],button,input,select,textarea,summary,label,option,'
    + '[contenteditable=""],[contenteditable="true"],[role="button"],[role="link"],'
    + '[role="tab"],[role="menuitem"],[role="menuitemcheckbox"],[role="menuitemradio"],'
    + '[role="option"],[role="checkbox"],[role="radio"],[role="switch"],[role="treeitem"],'
    + '[role="combobox"],[role="textbox"],[role="searchbox"]';
  let el = (this.closest && this.closest(INTERACTIVE)) || this;
  if (el.tagName === 'LABEL' && el.control) el = el.control;
  const tag = String(el.tagName || '').toLowerCase();
  const attr = (name) => {
    const value = el.getAttribute ? el.getAttribute(name) : null;
    return value === null ? null : value.trim().toLowerCase();
  };
  const link = el.closest ? el.closest('a[href]') : null;
  let linkScheme = null;
  if (link) {
    try { linkScheme = new URL(link.href, document.baseURI).protocol.replace(':', '').toLowerCase(); }
    catch (_) { linkScheme = 'invalid'; }
  }
  const type = typeof el.type === 'string' ? el.type.toLowerCase() : null;
  const form = el.form || (el.closest ? el.closest('form') : null);
  const submits = !!form && ((tag === 'button' && type === 'submit')
    || (tag === 'input' && (type === 'submit' || type === 'image')));
  let method = null;
  if (form) {
    method = ((submits && el.getAttribute('formmethod')) || form.getAttribute('method') || 'get')
      .trim().toLowerCase();
  }
  const search = !!form && ((form.getAttribute('role') || '').toLowerCase() === 'search'
    || !!form.closest('search,[role="search"]')
    || !!form.querySelector('input[type="search"]'));
  const nonText = ['button', 'submit', 'reset', 'image', 'checkbox', 'radio', 'file',
    'hidden', 'range', 'color'];
  const editable = !!el.isContentEditable || tag === 'textarea'
    || (tag === 'input' && !nonText.includes(type));
  return {
    tag,
    role: attr('role'),
    input_type: tag === 'input' ? type : null,
    autocomplete: attr('autocomplete'),
    link_scheme: linkScheme,
    link_download: !!link && link.hasAttribute('download'),
    submits_form: submits,
    form_method: method,
    search_form: search,
    disclosure: tag === 'summary' || (el.hasAttribute && (el.hasAttribute('aria-expanded')
      || el.hasAttribute('aria-haspopup'))),
    editable,
  };
}"#;

/// What reading a target's facts found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetFacts {
    /// The session has no page yet: the call runs on a fresh blank page,
    /// where nothing can be submitted.
    NoPage,
    /// No element matches the selector.
    Missing,
    /// The element's facts.
    Element(ElementFacts),
    /// The facts could not be read; treated as point of action.
    Unreadable,
}

impl TargetFacts {
    /// Whether `action` on this target needs the owner for this exact call.
    /// A missing element or a blank page cannot be acted on, so they do not;
    /// facts that could not be read do (fail safe).
    pub fn needs_owner(&self, action: &str) -> bool {
        match self {
            Self::NoPage | Self::Missing => false,
            Self::Unreadable => true,
            Self::Element(facts) => match action {
                "click" => click_needs_owner(facts),
                "fill" => fill_needs_owner(facts),
                _ => true,
            },
        }
    }
}

fn role_is(facts: &ElementFacts, roles: &[&str]) -> bool {
    facts
        .role
        .as_deref()
        .is_some_and(|role| roles.contains(&role))
}

/// Whether clicking this element needs the owner every time. A click is
/// ordinary only when its effect is typed as navigation or local page state:
/// a web link, a search form's (non-POST) submit, a disclosure, a selection
/// control, or focusing a text field. Submitting any other form, and any
/// other button or handler-driven element, whose effect the DOM cannot show,
/// needs the owner.
pub fn click_needs_owner(facts: &ElementFacts) -> bool {
    if facts.submits_form {
        let posts = !matches!(facts.form_method.as_deref(), None | Some("get"));
        return !facts.search_form || posts;
    }
    let web_link = matches!(facts.link_scheme.as_deref(), Some("http" | "https"));
    if (web_link && !facts.link_download) || role_is(facts, &["link"]) {
        return false;
    }
    if facts.disclosure || facts.tag == "summary" {
        return false;
    }
    let selection = matches!(facts.tag.as_str(), "select" | "option")
        || matches!(facts.input_type.as_deref(), Some("checkbox" | "radio"))
        || role_is(facts, &["tab", "option", "treeitem", "combobox"]);
    // Anything else (a plain or role=button, a handler-driven element) does
    // what a script decides, which the DOM cannot show.
    !(selection || facts.editable)
}

/// Whether filling this element needs the owner every time: a password,
/// payment or one-time-code field, or a target that is not a text field
/// (what a fill does to it is unknown). `autocomplete` is a typed token
/// list (HTML autofill field names), matched exactly per token.
pub fn fill_needs_owner(facts: &ElementFacts) -> bool {
    if !facts.editable || facts.input_type.as_deref() == Some("password") {
        return true;
    }
    facts.autocomplete.as_deref().is_some_and(|tokens| {
        tokens.split_ascii_whitespace().any(|token| {
            token.starts_with("cc-")
                || matches!(
                    token,
                    "current-password" | "new-password" | "one-time-code" | "transaction-amount"
                )
        })
    })
}

// ─── Classifier ───────────────────────────────────────────────────────────────

/// Classify the risk of a browser action.
///
/// # Parameters
/// - `action`   — the action string from the tool call (`"click"`, `"navigate"`, …).
///
/// # Returns
/// A [`BrowserActionRisk`] that describes the class and sensitivity. Whether a
/// click or fill is point of action comes from its target's [`ElementFacts`].
pub fn classify(action: &str) -> BrowserActionRisk {
    match action {
        // ── Observation ───────────────────────────────────────────────────────
        "get_text" | "inspect_page" | "screenshot" | "scroll" | "wait" | "list_tabs"
        | "get_console_logs" | "get_network_errors" => BrowserActionRisk {
            class: BrowserRiskClass::Observation,
            sensitive: false,
        },

        // ── Navigation ────────────────────────────────────────────────────────
        "navigate" | "new_tab" | "switch_tab" => BrowserActionRisk {
            class: BrowserRiskClass::Navigation,
            sensitive: false,
        },

        // ── Mutation: click / fill ────────────────────────────────────────────
        "click" | "fill" => BrowserActionRisk {
            class: BrowserRiskClass::Mutation,
            sensitive: false,
        },

        // ── Mutation: execute_js ──────────────────────────────────────────────
        "execute_js" => BrowserActionRisk {
            class: BrowserRiskClass::Mutation,
            // Always sensitive: arbitrary JS can exfiltrate credentials, submit
            // forms, or perform account actions regardless of script content.
            sensitive: true,
        },

        // ── Administrative ────────────────────────────────────────────────────
        "render_pdf" | "close" | "close_tab" | "set_mode" => BrowserActionRisk {
            class: BrowserRiskClass::Administrative,
            sensitive: false,
        },

        // ── Unknown ───────────────────────────────────────────────────────────
        // Default to the most restrictive sane class so that a future action
        // added to the schema is never accidentally treated as a free observation.
        _ => BrowserActionRisk {
            class: BrowserRiskClass::Mutation,
            sensitive: false,
        },
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_map_to_their_class() {
        let cases = [
            ("navigate", BrowserRiskClass::Navigation),
            ("screenshot", BrowserRiskClass::Observation),
            ("render_pdf", BrowserRiskClass::Administrative),
            ("click", BrowserRiskClass::Mutation),
            ("fill", BrowserRiskClass::Mutation),
            ("get_text", BrowserRiskClass::Observation),
            ("inspect_page", BrowserRiskClass::Observation),
            ("scroll", BrowserRiskClass::Observation),
            ("execute_js", BrowserRiskClass::Mutation),
            ("wait", BrowserRiskClass::Observation),
            ("list_tabs", BrowserRiskClass::Observation),
            ("get_console_logs", BrowserRiskClass::Observation),
            ("get_network_errors", BrowserRiskClass::Observation),
            ("new_tab", BrowserRiskClass::Navigation),
            ("switch_tab", BrowserRiskClass::Navigation),
            ("close_tab", BrowserRiskClass::Administrative),
            ("set_mode", BrowserRiskClass::Administrative),
            ("close", BrowserRiskClass::Administrative),
            ("teleport_browser", BrowserRiskClass::Mutation),
        ];
        for (action, class) in cases {
            let risk = classify(action);
            assert_eq!(risk.class, class, "{action}");
            assert_eq!(risk.sensitive, action == "execute_js", "{action}");
        }
    }

    fn element(tag: &str) -> ElementFacts {
        ElementFacts {
            tag: tag.to_string(),
            ..ElementFacts::default()
        }
    }

    fn submit(method: Option<&str>, search: bool) -> ElementFacts {
        ElementFacts {
            input_type: None,
            submits_form: true,
            form_method: method.map(str::to_string),
            search_form: search,
            ..element("button")
        }
    }

    /// The selector says nothing: the same opaque selector is ordinary on a
    /// link and point of action on a form's submit button.
    #[test]
    fn opaque_selectors_are_judged_by_their_element() {
        let link = ElementFacts {
            link_scheme: Some("https".into()),
            ..element("a")
        };
        assert!(!click_needs_owner(&link));
        assert!(click_needs_owner(&submit(Some("post"), false)));
        // A form with no method is GET by default, yet a script may post it.
        assert!(click_needs_owner(&submit(None, false)));
        assert!(click_needs_owner(&submit(Some("get"), false)));
    }

    #[test]
    fn a_search_forms_get_submit_is_ordinary_but_a_posted_one_is_not() {
        assert!(!click_needs_owner(&submit(Some("get"), true)));
        assert!(!click_needs_owner(&submit(None, true)));
        assert!(click_needs_owner(&submit(Some("post"), true)));
    }

    /// A plain button's effect is a script the DOM cannot show (a single-page
    /// app's "place order"), so it needs the owner.
    #[test]
    fn a_handler_driven_button_needs_the_owner() {
        assert!(click_needs_owner(&element("button")));
        assert!(click_needs_owner(&ElementFacts {
            role: Some("button".into()),
            ..element("div")
        }));
        assert!(click_needs_owner(&ElementFacts {
            role: Some("switch".into()),
            ..element("div")
        }));
        assert!(click_needs_owner(&ElementFacts {
            link_scheme: Some("javascript".into()),
            ..element("a")
        }));
        assert!(click_needs_owner(&ElementFacts {
            link_scheme: Some("https".into()),
            link_download: true,
            ..element("a")
        }));
    }

    #[test]
    fn local_ui_state_clicks_are_ordinary() {
        let ordinary = [
            ElementFacts {
                disclosure: true,
                ..element("button")
            },
            element("summary"),
            element("select"),
            ElementFacts {
                input_type: Some("checkbox".into()),
                ..element("input")
            },
            ElementFacts {
                role: Some("tab".into()),
                ..element("div")
            },
            ElementFacts {
                input_type: Some("text".into()),
                editable: true,
                ..element("input")
            },
        ];
        for facts in ordinary {
            assert!(!click_needs_owner(&facts), "{facts:?}");
        }
    }

    #[test]
    fn filling_a_secret_or_payment_field_needs_the_owner() {
        let text = ElementFacts {
            input_type: Some("text".into()),
            editable: true,
            ..element("input")
        };
        assert!(!fill_needs_owner(&text));
        assert!(fill_needs_owner(&ElementFacts {
            input_type: Some("password".into()),
            ..text.clone()
        }));
        for autocomplete in [
            "cc-number",
            "billing cc-csc",
            "one-time-code",
            "new-password",
        ] {
            assert!(
                fill_needs_owner(&ElementFacts {
                    autocomplete: Some(autocomplete.into()),
                    ..text.clone()
                }),
                "{autocomplete}"
            );
        }
        assert!(!fill_needs_owner(&ElementFacts {
            autocomplete: Some("email".into()),
            ..text.clone()
        }));
        assert!(fill_needs_owner(&element("button")));
    }

    #[test]
    fn unreadable_facts_fail_safe() {
        assert!(TargetFacts::Unreadable.needs_owner("click"));
        assert!(TargetFacts::Unreadable.needs_owner("fill"));
        assert!(!TargetFacts::Missing.needs_owner("click"));
        assert!(!TargetFacts::NoPage.needs_owner("fill"));
    }

    #[test]
    fn facts_deserialize_from_the_script_shape() {
        let facts: ElementFacts = serde_json::from_value(serde_json::json!({
            "tag": "button",
            "role": null,
            "input_type": null,
            "autocomplete": null,
            "link_scheme": null,
            "link_download": false,
            "submits_form": true,
            "form_method": "post",
            "search_form": false,
            "disclosure": false,
            "editable": false
        }))
        .unwrap();
        assert!(facts.submits_form);
        assert!(click_needs_owner(&facts));
    }
}
