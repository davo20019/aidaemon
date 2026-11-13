//! `manage_config` lives in `aidaemon_runtime::tools::config_manager`; its
//! tests on the daemon's own config check stay here.

pub use aidaemon_runtime::tools::config_manager::ConfigManagerTool;

#[cfg(test)]
mod tests {
    use crate::config::AppConfig;
    use crate::traits::ToolCalls as _;
    use crate::traits::{OwnerApprovalNeed, Tool, ToolCallContext};
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;

    fn write_temp_config(contents: &str) -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, contents).expect("write temp config");
        (dir, path)
    }

    const FAILOVER_CONFIG: &str = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "first-key"

[provider.fallbacks.models]
default = "claude-sonnet-4-20250514"
"#;

    #[tokio::test]
    async fn sensitive_changes_ask_with_the_old_wording_and_no_secret_in_the_subject() {
        let (_dir, path) = write_temp_config(FAILOVER_CONFIG);
        // The owner already stored the new failover's key themselves.
        let secrets = aidaemon_secrets::MemorySecretStore::new();
        aidaemon_secrets::SecretStore::set(
            &secrets,
            "provider_fallback_1_api_key",
            "sk-synthetic-test-key",
        )
        .unwrap();
        let tool = ConfigManagerTool::new(crate::config::config_file(path), Arc::new(secrets));

        // A literal secret value is refused before any card.
        let literal =
            json!({"action": "set", "key": "provider.api_key", "value": "synthetic-secret"});
        assert!(matches!(
            tool.owner_approval(&literal).await,
            OwnerApprovalNeed::NotNeeded
        ));

        let set = json!({"action": "set", "key": "provider.api_key", "value": "keychain"});
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&set).await else {
            panic!("a secret change must ask");
        };
        assert_eq!(
            request.summary,
            "Config change: provider.api_key = [REDACTED]"
        );
        assert_eq!(request.risk, aidaemon_core::ApprovalRisk::High);
        assert_eq!(
            request.warnings,
            vec!["Modifying security-sensitive configuration".to_string()]
        );
        let mut other = set.clone();
        other["key"] = json!("provider.fallbacks[0].api_key");
        let OwnerApprovalNeed::Ask(changed) = tool.owner_approval(&other).await else {
            panic!("a secret change must ask");
        };
        assert_ne!(request.subject, changed.subject);

        let remove = json!({"action": "remove_failover_provider", "failover_index": 0});
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&remove).await else {
            panic!("a failover removal must ask");
        };
        assert!(
            request.summary.starts_with("Remove failover provider [0]"),
            "{}",
            request.summary
        );

        let add = json!({"action": "add_failover_provider", "provider": "anthropic"});
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&add).await else {
            panic!("a failover addition must ask");
        };
        assert!(
            request.summary.starts_with("Add failover provider #1: "),
            "{}",
            request.summary
        );
        assert!(!request.subject.contains("sk-synthetic-test-key"));

        for no_card in [
            json!({"action": "set", "key": "provider.models.default", "value": "\"other\""}),
            json!({"action": "read"}),
            json!({"action": "get", "key": "provider.kind"}),
            json!({"action": "list_failover_providers"}),
            json!({"action": "remove_failover_provider", "failover_index": 5}),
            json!({"action": "switch_provider", "provider": "unknown-provider"}),
            json!({"action": "restore"}),
        ] {
            assert!(
                matches!(
                    tool.owner_approval(&no_card).await,
                    OwnerApprovalNeed::NotNeeded
                ),
                "{no_card}"
            );
        }
    }

    const TWO_FALLBACKS: &str = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "first-key"

[provider.fallbacks.models]
default = "first-model"

[[provider.fallbacks]]
kind = "google_genai"
api_key = "second-key"

[provider.fallbacks.models]
default = "old-model"
fallback = ["old-a", "old-b"]
"#;

    #[tokio::test]
    async fn a_list_position_edits_that_fallback_entry_and_is_checked_on_disk() {
        let (_dir, path) = write_temp_config(TWO_FALLBACKS);
        let tool = ConfigManagerTool::new(
            crate::config::config_file(path.clone()),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        );
        for (key, value) in [
            ("provider.fallbacks[1].models.default", "new-model"),
            ("provider.fallbacks.1.models.smart", "\"new-smart\""),
            ("provider.fallbacks[1].models.fallback", "[\"new-a\"]"),
        ] {
            let args = json!({"action": "set", "key": key, "value": value}).to_string();
            let outcome = tool
                .execute(&args, &ToolCallContext::default())
                .await
                .unwrap();
            assert!(outcome.output.contains("Checked"), "{}", outcome.output);
            assert_eq!(
                outcome.metadata.outcome_status,
                Some(crate::traits::ToolOutcomeStatus::Succeeded)
            );
        }
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("fallbacks["), "{text}");
        let config: AppConfig = toml::from_str(&text).unwrap();
        let google = &config.provider.fallbacks[1].models;
        assert_eq!(google.default_model, "new-model");
        assert_eq!(google.smart, "new-smart");
        assert_eq!(google.fallback_models, vec!["new-a".to_string()]);
        assert_eq!(
            config.provider.fallbacks[0].models.default_model,
            "first-model"
        );
    }

    #[tokio::test]
    async fn paths_the_config_does_not_have_or_read_are_refused_unchanged() {
        let (_dir, path) = write_temp_config(TWO_FALLBACKS);
        let tool = ConfigManagerTool::new(
            crate::config::config_file(path.clone()),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        );
        for (key, expect) in [
            ("provider.fallbacks[2].models.default", "has 2 entries"),
            (
                "provider.fallbacks.models.default",
                "is a list of 2 entries",
            ),
            ("provider.\"fallbacks[2]\".models", "is not a config key"),
            ("provider.models.no_such_setting", "does not read a setting"),
            ("no_such_section.value", "does not read a setting"),
        ] {
            let args = json!({"action": "set", "key": key, "value": "x"}).to_string();
            let outcome = tool
                .execute(&args, &ToolCallContext::default())
                .await
                .unwrap();
            assert!(outcome.output.contains(expect), "{key}: {}", outcome.output);
            assert!(
                outcome.output.contains("NOT modified"),
                "{}",
                outcome.output
            );
            assert_eq!(
                outcome.metadata.outcome_status,
                Some(crate::traits::ToolOutcomeStatus::FailedPermanent)
            );
            assert_eq!(fs::read_to_string(&path).unwrap(), TWO_FALLBACKS);
        }
    }

    #[tokio::test]
    async fn unset_removes_a_setting_and_read_is_a_typed_success() {
        let (_dir, path) = write_temp_config(TWO_FALLBACKS);
        let tool = ConfigManagerTool::new(
            crate::config::config_file(path.clone()),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        );
        let args =
            json!({"action": "unset", "key": "provider.fallbacks[1].models.fallback"}).to_string();
        let output = tool.call(&args).await.unwrap();
        assert!(
            output.starts_with("Removed provider.fallbacks[1].models.fallback"),
            "{output}"
        );
        assert!(!fs::read_to_string(&path).unwrap().contains("old-a"));

        let read = tool
            .execute(r#"{"action":"read"}"#, &ToolCallContext::default())
            .await
            .unwrap();
        assert_eq!(
            read.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Succeeded)
        );
        assert!(read.output.contains("old-model"));
        assert!(!read.output.contains("second-key"), "{}", read.output);
        let get = tool
            .call(r#"{"action":"get","key":"provider.fallbacks[1]"}"#)
            .await
            .unwrap();
        assert!(
            get.contains("old-model") && !get.contains("second-key"),
            "{get}"
        );
    }

    #[tokio::test]
    async fn sensitive_change_without_the_kernel_is_refused() {
        let (_dir, path) = write_temp_config(FAILOVER_CONFIG);
        let tool = ConfigManagerTool::new(
            crate::config::config_file(path.clone()),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        );
        let before = fs::read_to_string(&path).unwrap();
        let args = json!({"action": "set", "key": "provider.base_url", "value": "\"https://h.example.invalid/v1\""})
            .to_string();
        let err = tool.call(&args).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("requires approval through the agent kernel"),
            "{err}"
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), before);

        let remove = json!({"action": "remove_failover_provider", "failover_index": 0}).to_string();
        assert!(tool.call(&remove).await.is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[test]
    fn changes_declare_the_config_path_and_reads_declare_nothing() {
        let tool = ConfigManagerTool::new(
            crate::config::config_file(PathBuf::from("/tmp/aidaemon-synthetic/config.toml")),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        );
        let effects = tool.effects_for(r#"{"action":"set","key":"provider.kind","value":"x"}"#);
        assert_eq!(
            effects.keys,
            vec![
                aidaemon_core::EffectKey::path(
                    aidaemon_core::EffectClass::Configuration,
                    "/tmp/aidaemon-synthetic/config.toml"
                )
                .unwrap()
            ]
        );
        assert!(!effects.complete);
        assert_eq!(
            tool.effects_for(r#"{"action":"read"}"#),
            aidaemon_core::CallEffects::none()
        );
    }

    fn gateway_config(dir: &std::path::Path, extra: &str) -> String {
        format!(
            r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gpt-4o"

[client_gateway]
token_file = "{}"
{extra}
"#,
            dir.join("client-gateway-token.sha256").display()
        )
    }

    #[tokio::test]
    async fn client_gateway_changes_ask_the_owner() {
        let tokens = tempfile::tempdir().unwrap();
        let (_dir, path) = write_temp_config(&gateway_config(tokens.path(), ""));
        let tool = ConfigManagerTool::new(
            crate::config::config_file(path),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        );
        for call in [
            json!({"action": "set", "key": "client_gateway.enabled", "value": "true"}),
            json!({"action": "set", "key": "client_gateway.port", "value": "9000"}),
            json!({"action": "unset", "key": "client_gateway"}),
            json!({"action": "enable_client_gateway"}),
        ] {
            assert!(
                matches!(tool.owner_approval(&call).await, OwnerApprovalNeed::Ask(_)),
                "{call} must ask"
            );
        }
        let other = json!({"action": "set", "key": "provider.models.primary", "value": "\"m\""});
        assert!(matches!(
            tool.owner_approval(&other).await,
            OwnerApprovalNeed::NotNeeded
        ));
    }
}
