#![allow(dead_code)]

use super::types::{IndexedElement, PointTarget};

/// Action risk class for computer-use policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionClass {
    Observation,
    LocalMutation,
    Consequential,
    Prohibited,
}

/// Parsed computer-use tool action name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputerActionKind {
    ListApps,
    LaunchApp,
    GetAppState,
    Screenshot,
    ActivateApp,
    Click,
    TypeText,
    PressKey,
    Scroll,
    SetValue,
}

impl ComputerActionKind {
    pub fn parse(action: &str) -> Result<Self, String> {
        match action {
            "list_apps" => Ok(Self::ListApps),
            "launch_app" => Ok(Self::LaunchApp),
            "get_app_state" => Ok(Self::GetAppState),
            "screenshot" => Ok(Self::Screenshot),
            "activate_app" => Ok(Self::ActivateApp),
            "click" => Ok(Self::Click),
            "type_text" => Ok(Self::TypeText),
            "press_key" => Ok(Self::PressKey),
            "scroll" => Ok(Self::Scroll),
            "set_value" => Ok(Self::SetValue),
            other => Err(format!(
                "Unknown computer_use action '{other}'. Valid: list_apps, launch_app, get_app_state, screenshot, activate_app, click, type_text, press_key, scroll, set_value"
            )),
        }
    }

    pub fn base_class(self) -> ActionClass {
        match self {
            Self::ListApps | Self::GetAppState | Self::Screenshot => ActionClass::Observation,
            Self::LaunchApp
            | Self::ActivateApp
            | Self::Click
            | Self::TypeText
            | Self::PressKey
            | Self::Scroll => ActionClass::LocalMutation,
            Self::SetValue => ActionClass::LocalMutation,
        }
    }
}

/// Actionable error for when a target app is not currently running. The tool can
/// only attach to live processes, so an installed-but-closed app is invisible;
/// the message states the literal next step (launch it) instead of dead-ending.
pub fn no_running_app_message(app: &str) -> String {
    format!(
        "No running app matching '{app}'. It may be installed but not running — computer_use \
         only controls apps that are already open. Start it with action \"launch_app\" \
         (app: \"{app}\"), or call list_apps to see what is currently running."
    )
}

/// Bundle identifiers that must never receive input events.
pub const PROHIBITED_BUNDLE_IDS: &[&str] = &["com.apple.Terminal", "com.apple.loginwindow"];

pub fn is_prohibited_bundle(bundle_id: &str) -> bool {
    PROHIBITED_BUNDLE_IDS
        .iter()
        .any(|blocked| bundle_id.eq_ignore_ascii_case(blocked))
}

/// Whether an element is an editable text input. Used to decide whether to
/// select-all before typing into it (so the new text replaces leftover content,
/// e.g. a stale URL in an address bar) — typing into a non-text control should
/// not trigger a select-all.
pub fn is_text_input_element(element: &IndexedElement) -> bool {
    let role = element.role.to_ascii_lowercase();
    let subrole = element
        .subrole
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase();
    role.contains("textfield")
        || role.contains("textarea")
        || role.contains("combobox")
        || role.contains("searchfield")
        || subrole.contains("searchfield")
}

pub fn is_secure_element(element: &IndexedElement) -> bool {
    let role = element.role.to_ascii_lowercase();
    let subrole = element
        .subrole
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase();
    role.contains("securetextfield") || subrole.contains("secure") || subrole.contains("password")
}

/// The class of `action` on `element`, from typed accessibility facts only
/// (role, subrole, enabled, default/cancel button, dialog membership); the
/// element's title and any typed text are never read. A secure text field is
/// prohibited. A click is consequential when it confirms a window or sheet
/// (its `AXDefaultButton`), presses any other button of a sheet or dialog
/// except its cancel button, or has a target whose facts are unknown: no
/// element (a coordinate click is judged by [`classify_point_click`]
/// instead) or an element with no role. Other clicks, and typing or setting
/// a value (data, not an action), are local: the per-app approval covers
/// them.
pub fn classify_target(
    action: ComputerActionKind,
    element: Option<&IndexedElement>,
) -> ActionClass {
    if element.is_some_and(is_secure_element) {
        return ActionClass::Prohibited;
    }
    match (action, element) {
        (ComputerActionKind::Click, None) => ActionClass::Consequential,
        (ComputerActionKind::Click, Some(el)) => click_class(el),
        _ => action.base_class(),
    }
}

/// The class of a coordinate click from what lies under the point. A point
/// whose app cannot be identified is consequential (asks every time); a
/// point in the named app is judged by the element there like an element
/// click, except that an element with no role is an ordinary click in that
/// app (the app is known, and its approval covers it).
pub fn classify_point_click(target: &PointTarget) -> ActionClass {
    match target {
        PointTarget::Unidentified => ActionClass::Consequential,
        PointTarget::InApp(element) if element.role.trim().is_empty() => ActionClass::LocalMutation,
        PointTarget::InApp(element) => classify_target(ComputerActionKind::Click, Some(element)),
    }
}

fn click_class(el: &IndexedElement) -> ActionClass {
    if el.role.trim().is_empty() {
        return ActionClass::Consequential;
    }
    if !el.enabled || el.cancel_button {
        return ActionClass::LocalMutation;
    }
    if el.default_button || (el.in_dialog && el.role == "AXButton") {
        return ActionClass::Consequential;
    }
    ActionClass::LocalMutation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_fields_are_prohibited() {
        let el = IndexedElement {
            index: 1,
            role: "AXTextField".to_string(),
            title: "Password".to_string(),
            enabled: true,
            bounds: None,
            subrole: Some("AXSecureTextField".to_string()),
            interactive: true,
            ..Default::default()
        };
        assert_eq!(
            classify_target(ComputerActionKind::TypeText, Some(&el)),
            ActionClass::Prohibited
        );
    }

    fn button(title: &str) -> IndexedElement {
        IndexedElement {
            index: 2,
            role: "AXButton".to_string(),
            title: title.to_string(),
            enabled: true,
            interactive: true,
            ..Default::default()
        }
    }

    /// The title says nothing either way: a "Send Message" toolbar button is
    /// a local click, an unlabeled default button confirms.
    #[test]
    fn titles_do_not_decide_the_class() {
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&button("Send Message"))),
            ActionClass::LocalMutation
        );
        let confirm = IndexedElement {
            default_button: true,
            ..button("")
        };
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&confirm)),
            ActionClass::Consequential
        );
    }

    #[test]
    fn dialog_buttons_are_consequential_except_cancel() {
        let in_sheet = IndexedElement {
            in_dialog: true,
            ..button("7")
        };
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&in_sheet)),
            ActionClass::Consequential
        );
        let cancel = IndexedElement {
            cancel_button: true,
            ..in_sheet.clone()
        };
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&cancel)),
            ActionClass::LocalMutation
        );
        let checkbox = IndexedElement {
            role: "AXCheckBox".to_string(),
            ..in_sheet
        };
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&checkbox)),
            ActionClass::LocalMutation
        );
    }

    /// A target whose facts are unknown is consequential (fail safe).
    #[test]
    fn unknown_targets_are_consequential() {
        assert_eq!(
            classify_target(ComputerActionKind::Click, None),
            ActionClass::Consequential
        );
        let no_role = IndexedElement {
            role: String::new(),
            ..button("OK")
        };
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&no_role)),
            ActionClass::Consequential
        );
        // Typed text is data, never a decision.
        assert_eq!(
            classify_target(ComputerActionKind::TypeText, None),
            ActionClass::LocalMutation
        );
    }

    /// A point in the named app is app-scoped unless it lands on a
    /// confirming button; a point whose app is unknown always asks.
    #[test]
    fn point_clicks_are_judged_by_what_lies_under_them() {
        assert_eq!(
            classify_point_click(&PointTarget::Unidentified),
            ActionClass::Consequential
        );
        let group = IndexedElement {
            role: "AXGroup".to_string(),
            enabled: true,
            ..Default::default()
        };
        assert_eq!(
            classify_point_click(&PointTarget::InApp(group)),
            ActionClass::LocalMutation
        );
        assert_eq!(
            classify_point_click(&PointTarget::InApp(IndexedElement::default())),
            ActionClass::LocalMutation
        );
        let confirm = IndexedElement {
            default_button: true,
            in_dialog: true,
            ..button("")
        };
        assert_eq!(
            classify_point_click(&PointTarget::InApp(confirm)),
            ActionClass::Consequential
        );
    }

    #[test]
    fn terminal_bundle_is_prohibited() {
        assert!(is_prohibited_bundle("com.apple.Terminal"));
    }

    #[test]
    fn text_inputs_are_detected() {
        let field = IndexedElement {
            index: 1,
            role: "AXTextField".to_string(),
            title: "Address and search bar".to_string(),
            enabled: true,
            bounds: None,
            subrole: None,
            interactive: true,
            ..Default::default()
        };
        assert!(is_text_input_element(&field));

        let button = IndexedElement {
            index: 2,
            role: "AXButton".to_string(),
            title: "Like".to_string(),
            enabled: true,
            bounds: None,
            subrole: None,
            interactive: true,
            ..Default::default()
        };
        assert!(!is_text_input_element(&button));
    }
}
