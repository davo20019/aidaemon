//! The model and effort an app conversation answers with (protocol 1.16):
//! what the runtime can pin, and the per-session setting the turn reads.

use aidaemon_client_protocol::{
    Billing, Effort, ModelChoice, ModelOption, ModelOptions, ModelProvider, SessionModel,
};
use aidaemon_providers::routing::ProviderKind;
use aidaemon_runtime::llm::SharedLlmRuntime;
use aidaemon_runtime::session_model::{
    self, PickableProvider, PinnedModel, SessionModelSetting, efforts_for, pickable,
};
use aidaemon_tools::services::SettingsStore;

fn effort_of(effort: &str) -> Effort {
    match effort {
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        _ => Effort::Auto,
    }
}

fn billing_of(kind: ProviderKind) -> Billing {
    match kind {
        ProviderKind::OpenaiChatgpt => Billing::Plan,
        _ => Billing::PerToken,
    }
}

fn option_of(provider: PickableProvider) -> ModelProvider {
    let efforts: Vec<Effort> = efforts_for(provider.kind)
        .iter()
        .map(|effort| effort_of(effort))
        .collect();
    ModelProvider {
        name: session_model::provider_name(provider.kind).to_string(),
        billing: billing_of(provider.kind),
        models: provider
            .models
            .into_iter()
            .map(|id| ModelOption {
                id,
                efforts: efforts.clone(),
            })
            .collect(),
        id: provider.id,
    }
}

/// Every provider of the live runtime with a model to pin, main first.
pub(crate) fn options(llm: &SharedLlmRuntime) -> ModelOptions {
    ModelOptions {
        providers: pickable(&llm.snapshot())
            .into_iter()
            .filter(|provider| !provider.models.is_empty())
            .map(option_of)
            .collect(),
    }
}

fn available_models(options: &ModelOptions) -> String {
    let models: Vec<&str> = options
        .providers
        .iter()
        .flat_map(|provider| provider.models.iter().map(|model| model.id.as_str()))
        .collect();
    if models.is_empty() {
        "No models are available.".to_string()
    } else {
        format!("Available models: {}", models.join(", "))
    }
}

/// Handle the app's per-conversation model commands. Other slash commands
/// continue through the channel command dispatcher.
pub(crate) async fn slash(
    settings: &dyn SettingsStore,
    llm: &SharedLlmRuntime,
    session_id: &str,
    command: &str,
    args: &str,
) -> Option<String> {
    match command {
        "model" => {
            let requested = args.trim();
            if requested.is_empty() {
                let current = current(settings, llm, session_id).await;
                return Some(match current.model {
                    ModelChoice::Pinned { provider, model } => {
                        format!("This conversation uses `{model}` ({provider}).")
                    }
                    _ => "This conversation uses automatic model routing.".to_string(),
                });
            }
            let choices = options(llm);
            let selected = choices.providers.iter().find_map(|provider| {
                provider
                    .models
                    .iter()
                    .any(|model| model.id == requested)
                    .then(|| ModelChoice::Pinned {
                        provider: provider.id.clone(),
                        model: requested.to_string(),
                    })
            });
            let Some(selected) = selected else {
                return Some(format!(
                    "Unknown model `{requested}`. {}",
                    available_models(&choices)
                ));
            };
            Some(
                match set(settings, llm, session_id, Some(selected), None).await {
                    Ok(model) => match model.model {
                        ModelChoice::Pinned { provider, model } => {
                            format!("This conversation now uses `{model}` ({provider}).")
                        }
                        _ => "This conversation uses automatic model routing.".to_string(),
                    },
                    Err(error) => error,
                },
            )
        }
        "auto" => Some(
            match set(settings, llm, session_id, Some(ModelChoice::Auto), None).await {
                Ok(_) => "This conversation now uses automatic model routing.".to_string(),
                Err(error) => error,
            },
        ),
        _ => None,
    }
}

/// What `session_id` answers with now: a pin the runtime no longer has, or
/// an effort its provider does not take, reads as automatic, as the turn
/// treats it.
pub(crate) async fn current(
    settings: &dyn SettingsStore,
    llm: &SharedLlmRuntime,
    session_id: &str,
) -> SessionModel {
    let setting = session_model::get(settings, session_id).await;
    let snapshot = llm.snapshot();
    let pinned = setting
        .pinned
        .as_ref()
        .and_then(|pin| session_model::resolve(&snapshot, pin).map(|provider| (pin, provider)));
    let kind = pinned
        .as_ref()
        .map_or(snapshot.provider_kind(), |(_, provider)| provider.kind);
    SessionModel {
        session_id: session_id.to_string(),
        model: match pinned {
            Some((pin, _)) => ModelChoice::Pinned {
                provider: pin.provider.clone(),
                model: pin.model.clone(),
            },
            None => ModelChoice::Auto,
        },
        effort: setting
            .effort
            .as_deref()
            .filter(|effort| session_model::effort_valid(kind, effort))
            .map_or(Effort::Auto, effort_of),
    }
}

/// Apply `model` and `effort` (a `None` stays as it is) to `session_id`.
/// An effort the conversation's provider does not take is refused; moving
/// to a provider that takes none drops the effort.
pub(crate) async fn set(
    settings: &dyn SettingsStore,
    llm: &SharedLlmRuntime,
    session_id: &str,
    model: Option<ModelChoice>,
    effort: Option<Effort>,
) -> Result<SessionModel, String> {
    let snapshot = llm.snapshot();
    let mut setting: SessionModelSetting = session_model::get(settings, session_id).await;
    match model {
        None => {}
        Some(ModelChoice::Auto) => setting.pinned = None,
        Some(ModelChoice::Pinned { provider, model }) => {
            let pin = PinnedModel { provider, model };
            if session_model::resolve(&snapshot, &pin).is_none() {
                return Err(format!(
                    "{} is not a model this daemon is set up with",
                    pin.model
                ));
            }
            setting.pinned = Some(pin);
        }
        Some(ModelChoice::Unknown) => return Err("Unknown model choice".to_string()),
    }
    let kind = setting
        .pinned
        .as_ref()
        .and_then(|pin| session_model::resolve(&snapshot, pin))
        .map_or(snapshot.provider_kind(), |provider| provider.kind);
    match effort {
        None => {
            if setting
                .effort
                .as_deref()
                .is_some_and(|effort| !session_model::effort_valid(kind, effort))
            {
                setting.effort = None;
            }
        }
        Some(Effort::Auto) => setting.effort = None,
        Some(level @ (Effort::Low | Effort::Medium | Effort::High)) => {
            let level = match level {
                Effort::Low => "low",
                Effort::Medium => "medium",
                _ => "high",
            };
            if !session_model::effort_valid(kind, level) {
                return Err(format!(
                    "{} takes no effort setting",
                    session_model::provider_name(kind)
                ));
            }
            setting.effort = Some(level.to_string());
        }
        Some(Effort::Unknown) => return Err("Unknown effort".to_string()),
    }
    session_model::set(settings, session_id, &setting)
        .await
        .map_err(|error| format!("Could not save the conversation's model: {error}"))?;
    Ok(current(settings, llm, session_id).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::testing::ScriptedProvider;
    use aidaemon_runtime::llm::ProviderRuntimeTarget;
    use std::sync::Arc;

    fn runtime() -> SharedLlmRuntime {
        let backend = || Arc::new(ScriptedProvider::new([])) as aidaemon_runtime::llm::ModelBackend;
        SharedLlmRuntime::new_with_failovers(
            backend(),
            None,
            ProviderKind::Anthropic,
            "synthetic-claude".into(),
            vec![ProviderRuntimeTarget::new(
                backend(),
                None,
                ProviderKind::GoogleGenai,
                "synthetic-gemini".into(),
            )],
        )
    }

    fn pin(provider: &str, model: &str) -> Option<ModelChoice> {
        Some(ModelChoice::Pinned {
            provider: provider.into(),
            model: model.into(),
        })
    }

    #[tokio::test]
    async fn efforts_follow_the_provider_and_unknown_models_are_refused() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let settings = harness.state.as_ref();
        let llm = runtime();

        let options = options(&llm);
        assert_eq!(options.providers.len(), 2);
        assert_eq!(options.providers[0].models[0].efforts.len(), 3);
        assert!(options.providers[1].models[0].efforts.is_empty());

        let set_high = set(
            settings,
            &llm,
            "app:s1",
            pin("main", "synthetic-claude"),
            Some(Effort::High),
        )
        .await
        .unwrap();
        assert_eq!(set_high.effort, Effort::High);

        // Gemini takes no effort: asking for one is refused, and moving
        // there drops the one set.
        assert!(
            set(
                settings,
                &llm,
                "app:s1",
                pin("failover-0", "synthetic-gemini"),
                Some(Effort::Low)
            )
            .await
            .is_err()
        );
        let moved = set(
            settings,
            &llm,
            "app:s1",
            pin("failover-0", "synthetic-gemini"),
            None,
        )
        .await
        .unwrap();
        assert_eq!(moved.effort, Effort::Auto);
        assert_eq!(moved.model, pin("failover-0", "synthetic-gemini").unwrap());

        assert!(
            set(
                settings,
                &llm,
                "app:s1",
                pin("main", "synthetic-gemini"),
                None
            )
            .await
            .is_err()
        );
        let auto = set(settings, &llm, "app:s1", Some(ModelChoice::Auto), None)
            .await
            .unwrap();
        assert_eq!(auto.model, ModelChoice::Auto);
    }

    #[tokio::test]
    async fn slash_model_isolated_by_conversation_resolves_provider_and_auto_clears() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let settings = harness.state.as_ref();
        let llm = runtime();

        let selected = slash(
            settings,
            &llm,
            "app:synthetic-one",
            "model",
            "synthetic-gemini",
        )
        .await
        .unwrap();
        assert!(selected.contains("failover-0"), "{selected}");
        assert_eq!(
            current(settings, &llm, "app:synthetic-one").await.model,
            pin("failover-0", "synthetic-gemini").unwrap()
        );
        assert_eq!(
            current(settings, &llm, "app:synthetic-two").await.model,
            ModelChoice::Auto
        );

        let report = slash(settings, &llm, "app:synthetic-one", "model", "")
            .await
            .unwrap();
        assert!(report.contains("synthetic-gemini"), "{report}");

        let unknown = slash(
            settings,
            &llm,
            "app:synthetic-one",
            "model",
            "synthetic-missing",
        )
        .await
        .unwrap();
        assert!(unknown.contains("Unknown model `synthetic-missing`"));
        assert!(unknown.contains("synthetic-claude, synthetic-gemini"));
        assert_eq!(
            current(settings, &llm, "app:synthetic-one").await.model,
            pin("failover-0", "synthetic-gemini").unwrap(),
            "an unknown model must not change the existing pin"
        );
        assert_eq!(
            slash(settings, &llm, "app:synthetic-one", "help", "").await,
            None
        );

        let auto = slash(settings, &llm, "app:synthetic-one", "auto", "")
            .await
            .unwrap();
        assert!(auto.contains("automatic model routing"));
        assert_eq!(
            current(settings, &llm, "app:synthetic-one").await.model,
            ModelChoice::Auto
        );
        let report = slash(settings, &llm, "app:synthetic-one", "model", "")
            .await
            .unwrap();
        assert!(report.contains("automatic model routing"), "{report}");
    }
}
