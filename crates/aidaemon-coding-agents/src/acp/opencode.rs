//! OpenCode's sign-ins: one per model provider it is logged in to, each an
//! API key or a subscription plan's OAuth sign-in, kept in its `auth.json`.
//!
//! A plan's terms may not allow another AI agent to hand it work (a Claude
//! plan's do not), so when work is handed to OpenCode its launch turns off
//! every provider it is signed in to with OAuth (`disabled_providers`, in
//! the config it reads from `OPENCODE_CONFIG_CONTENT`): it then works only
//! on its API-key providers. The owner's own prompts use them all.
//!
//! OpenCode lets itself edit files and run commands unasked by default, so
//! every launch sets its permissions to ask (`permission`): each action
//! then comes to AIdaemon's host, which judges it in the checked mode and
//! answers yes in full access.

/// The variable OpenCode reads extra config from, merged over its own.
const CONFIG_VARIABLE: &str = "OPENCODE_CONFIG_CONTENT";

/// What its own sign-in adds to a launch whose work another agent handed
/// it: its OAuth providers turned off, and its actions asked.
pub(super) fn delegated() -> Result<crate::SignedIn, String> {
    let auth = match std::fs::read_to_string(auth_path()?) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("its sign-ins could not be read: {error}")),
    };
    let existing = std::env::var(CONFIG_VARIABLE).ok();
    let config = without_plans(auth.as_deref(), existing.as_deref())?;
    let config = asking(config.as_deref().or(existing.as_deref()));
    Ok(signed(config))
}

/// What the owner's own launch of it adds: its actions asked.
pub(super) fn own() -> Result<crate::SignedIn, String> {
    let existing = std::env::var(CONFIG_VARIABLE).ok();
    Ok(signed(asking(existing.as_deref())))
}

fn signed(config: String) -> crate::SignedIn {
    crate::SignedIn {
        env: vec![(CONFIG_VARIABLE.to_string(), config)],
        ..crate::SignedIn::default()
    }
}

/// The permissions OpenCode asks about (its config's `permission`).
const ASKED: [&str; 3] = ["edit", "bash", "webfetch"];

/// `existing` (a config, if any) with each of [`ASKED`] set to ask, unless
/// it already denies it.
fn asking(existing: Option<&str>) -> String {
    let mut config = existing
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    let permission = &mut config["permission"];
    if !permission.is_object() {
        *permission = serde_json::json!({});
    }
    for key in ASKED {
        // Machine format: its config's permission values ("allow" | "ask" | "deny").
        if permission[key].as_str() != Some("deny") {
            permission[key] = serde_json::json!("ask");
        }
    }
    config.to_string()
}

/// Where OpenCode keeps its sign-ins: `$XDG_DATA_HOME/opencode/auth.json`,
/// else `~/.local/share/opencode/auth.json` (on every OS).
fn auth_path() -> Result<std::path::PathBuf, String> {
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".local").join("share")))
        .ok_or("there is no home folder to find its sign-ins in")?;
    Ok(data.join("opencode").join("auth.json"))
}

/// The config that turns off each provider `auth` (its `auth.json`) signs
/// in to with OAuth, merged into `existing` (the config variable already
/// set, if any). `None`: there is none to turn off, so nothing changes.
fn without_plans(auth: Option<&str>, existing: Option<&str>) -> Result<Option<String>, String> {
    let Some(auth) = auth else {
        return Ok(None);
    };
    // Machine format: its auth file, `{ "<provider>": { "type": "oauth" | "api" | ... } }`.
    let auth: serde_json::Value = serde_json::from_str(auth)
        .map_err(|error| format!("its sign-ins are unreadable: {error}"))?;
    let mut plans: Vec<&str> = auth
        .as_object()
        .ok_or("its sign-ins are unreadable: not an object")?
        .iter()
        .filter(|(_, entry)| entry.get("type").and_then(serde_json::Value::as_str) == Some("oauth"))
        .map(|(provider, _)| provider.as_str())
        .collect();
    if plans.is_empty() {
        return Ok(None);
    }
    plans.sort_unstable();
    let mut config = existing
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    let disabled = &mut config["disabled_providers"];
    if !disabled.is_array() {
        *disabled = serde_json::json!([]);
    }
    if let Some(list) = disabled.as_array_mut() {
        for provider in plans {
            if !list.iter().any(|known| known.as_str() == Some(provider)) {
                list.push(serde_json::json!(provider));
            }
        }
    }
    Ok(Some(config.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disabled(config: &str) -> Vec<String> {
        serde_json::from_str::<serde_json::Value>(config).unwrap()["disabled_providers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn plan_sign_ins_are_turned_off_and_api_keys_kept() {
        let auth = r#"{"anthropic":{"type":"oauth","refresh":"r","access":"a","expires":1},
                       "openrouter":{"type":"api","key":"synthetic-key"},
                       "github-copilot":{"type":"oauth"}}"#;
        let config = without_plans(Some(auth), None).unwrap().unwrap();
        assert_eq!(disabled(&config), ["anthropic", "github-copilot"]);
        // Never carries a credential.
        assert!(!config.contains("synthetic-key"));
    }

    #[test]
    fn nothing_changes_without_a_plan_sign_in() {
        assert_eq!(without_plans(None, None).unwrap(), None);
        assert_eq!(
            without_plans(Some(r#"{"openrouter":{"type":"api","key":"k"}}"#), None).unwrap(),
            None
        );
        assert!(without_plans(Some("not json"), None).is_err());
    }

    #[test]
    fn every_launch_asks_and_keeps_a_deny() {
        let value: serde_json::Value = serde_json::from_str(&asking(None)).unwrap();
        for key in ASKED {
            assert_eq!(value["permission"][key], "ask");
        }
        let value: serde_json::Value = serde_json::from_str(&asking(Some(
            r#"{"model":"openrouter/synthetic","permission":{"bash":"deny","edit":"allow"}}"#,
        )))
        .unwrap();
        assert_eq!(value["permission"]["bash"], "deny");
        assert_eq!(value["permission"]["edit"], "ask");
        assert_eq!(value["model"], "openrouter/synthetic");
        // A plain "allow" for everything becomes asking.
        let value: serde_json::Value =
            serde_json::from_str(&asking(Some(r#"{"permission":"allow"}"#))).unwrap();
        assert_eq!(value["permission"]["webfetch"], "ask");
        // Plans turned off and actions asked, together.
        let config = without_plans(Some(r#"{"anthropic":{"type":"oauth"}}"#), None).unwrap();
        let value: serde_json::Value = serde_json::from_str(&asking(config.as_deref())).unwrap();
        assert_eq!(value["disabled_providers"][0], "anthropic");
        assert_eq!(value["permission"]["bash"], "ask");
    }

    #[test]
    fn config_already_set_is_kept() {
        let config = without_plans(
            Some(r#"{"anthropic":{"type":"oauth"}}"#),
            Some(r#"{"model":"openrouter/synthetic","disabled_providers":["groq","anthropic"]}"#),
        )
        .unwrap()
        .unwrap();
        assert_eq!(disabled(&config), ["groq", "anthropic"]);
        let value: serde_json::Value = serde_json::from_str(&config).unwrap();
        assert_eq!(value["model"], "openrouter/synthetic");
    }
}
