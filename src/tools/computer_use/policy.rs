#![allow(dead_code)]

use super::types::IndexedElement;

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
    GetAppState,
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
            "get_app_state" => Ok(Self::GetAppState),
            "activate_app" => Ok(Self::ActivateApp),
            "click" => Ok(Self::Click),
            "type_text" => Ok(Self::TypeText),
            "press_key" => Ok(Self::PressKey),
            "scroll" => Ok(Self::Scroll),
            "set_value" => Ok(Self::SetValue),
            other => Err(format!(
                "Unknown computer_use action '{other}'. Valid: list_apps, get_app_state, activate_app, click, type_text, press_key, scroll, set_value"
            )),
        }
    }

    pub fn action_name(self) -> &'static str {
        match self {
            Self::ListApps => "list_apps",
            Self::GetAppState => "get_app_state",
            Self::ActivateApp => "activate_app",
            Self::Click => "click",
            Self::TypeText => "type_text",
            Self::PressKey => "press_key",
            Self::Scroll => "scroll",
            Self::SetValue => "set_value",
        }
    }

    pub fn base_class(self) -> ActionClass {
        match self {
            Self::ListApps | Self::GetAppState => ActionClass::Observation,
            Self::ActivateApp | Self::Click | Self::TypeText | Self::PressKey | Self::Scroll => {
                ActionClass::LocalMutation
            }
            Self::SetValue => ActionClass::LocalMutation,
        }
    }

    pub fn requires_snapshot_generation(self) -> bool {
        !matches!(self, Self::ListApps | Self::GetAppState)
    }
}

/// Bundle identifiers that must never receive input events.
pub const PROHIBITED_BUNDLE_IDS: &[&str] = &["com.apple.Terminal", "com.apple.loginwindow"];

pub fn is_prohibited_bundle(bundle_id: &str) -> bool {
    PROHIBITED_BUNDLE_IDS
        .iter()
        .any(|blocked| bundle_id.eq_ignore_ascii_case(blocked))
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

const CONSEQUENTIAL_KEYWORDS: &[&str] = &[
    "send",
    "submit",
    "delete",
    "remove",
    "buy",
    "purchase",
    "publish",
    "post",
    "pay",
    "confirm",
    "authorize",
    "permission",
];

pub fn classify_target(
    action: ComputerActionKind,
    element: Option<&IndexedElement>,
    typed_text: Option<&str>,
) -> ActionClass {
    if let Some(el) = element {
        if is_secure_element(el) {
            return ActionClass::Prohibited;
        }
        let label = format!(
            "{} {} {}",
            el.title,
            el.role,
            el.subrole.as_deref().unwrap_or("")
        )
        .to_ascii_lowercase();
        if CONSEQUENTIAL_KEYWORDS
            .iter()
            .any(|kw| contains_keyword_as_words(&label, kw))
        {
            return ActionClass::Consequential;
        }
    }
    if let Some(text) = typed_text {
        let lower = text.to_ascii_lowercase();
        if CONSEQUENTIAL_KEYWORDS
            .iter()
            .any(|kw| contains_keyword_as_words(&lower, kw))
        {
            return ActionClass::Consequential;
        }
    }
    action.base_class()
}

fn contains_keyword_as_words(text: &str, keyword: &str) -> bool {
    let text_words: Vec<&str> = text.split_whitespace().collect();
    let kw_words: Vec<&str> = keyword.split_whitespace().collect();
    if kw_words.is_empty() {
        return false;
    }
    text_words
        .windows(kw_words.len())
        .any(|window| window.iter().zip(kw_words.iter()).all(|(a, b)| a == b))
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
        };
        assert_eq!(
            classify_target(ComputerActionKind::TypeText, Some(&el), None),
            ActionClass::Prohibited
        );
    }

    #[test]
    fn send_button_is_consequential() {
        let el = IndexedElement {
            index: 2,
            role: "AXButton".to_string(),
            title: "Send Message".to_string(),
            enabled: true,
            bounds: None,
            subrole: None,
            interactive: true,
        };
        assert_eq!(
            classify_target(ComputerActionKind::Click, Some(&el), None),
            ActionClass::Consequential
        );
    }

    #[test]
    fn terminal_bundle_is_prohibited() {
        assert!(is_prohibited_bundle("com.apple.Terminal"));
    }
}
