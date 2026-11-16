//! `aidaemon setup` with no config: the model step in this terminal, for a
//! scripted or app-less install. The same steps as the app's, through the
//! same host: found models first, then a ChatGPT sign-in or a provider key,
//! each tested before anything is saved.

use std::path::Path;

use aidaemon_client_gateway::setup::SetupHost;
use aidaemon_client_protocol::{ProviderChoice, SecretText, SetupAction, SetupStep};
use dialoguer::{Input, Password, Select};

pub(crate) async fn run(config_path: &Path) -> anyhow::Result<()> {
    let stop = tokio_util::sync::CancellationToken::new();
    let model = super::model::ModelSetup::new(config_path.to_path_buf(), stop.clone());
    let host = super::model::Host(model.clone());
    println!();
    println!("  AIdaemon setup: connect a model. Nothing is saved until it answers a test.");
    println!("  Looking for models on this computer...");
    model.detect().await;
    loop {
        let state = host.state(true).await;
        if state.step == SetupStep::Starting {
            break;
        }
        let mut items: Vec<String> = Vec::new();
        let mut actions: Vec<Option<SetupAction>> = Vec::new();
        for (index, found) in state.found.iter().enumerate() {
            if found.works {
                items.push(format!("Use {} (found: {})", found.name, found.source));
                actions.push(Some(SetupAction::UseFound {
                    index: index as u32,
                }));
            }
        }
        for option in &state.providers {
            let billing = match option.billing {
                aidaemon_client_protocol::Billing::Plan => "uses your plan, has limits",
                aidaemon_client_protocol::Billing::Local => "runs on this machine, free",
                _ => "pay per token",
            };
            items.push(format!("{} ({billing})", option.name));
            actions.push(None);
        }
        let choice = Select::new()
            .with_prompt("Which model should AIdaemon use?")
            .items(&items)
            .default(0)
            .interact()?;
        let action = match actions[choice].clone() {
            Some(action) => action,
            None => {
                let option =
                    &state.providers[choice - actions.iter().filter(|a| a.is_some()).count()];
                if !option.needs_key && option.preset == "openai_chatgpt" {
                    let paste = !dialoguer::Confirm::new()
                        .with_prompt("Is there a browser on this computer?")
                        .default(true)
                        .interact()?;
                    let state = host.step(SetupAction::SignIn { paste }, true).await;
                    let state = match state {
                        Ok(state) => state,
                        Err(reason) => {
                            println!("  {reason}");
                            continue;
                        }
                    };
                    let Some(sign_in) = state.sign_in else {
                        continue;
                    };
                    println!();
                    println!("  Open this address and sign in:");
                    println!("  {}", sign_in.url);
                    if sign_in.paste {
                        let redirect: String = Input::new()
                            .with_prompt("Paste the address the browser ended on")
                            .interact_text()?;
                        SetupAction::FinishSignIn { redirect }
                    } else {
                        println!("  Waiting for the sign-in to finish...");
                        loop {
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            let state = host.state(true).await;
                            if state.step == SetupStep::Starting || state.sign_in.is_none() {
                                for notice in state.notices {
                                    println!("  {notice}");
                                }
                                break;
                            }
                        }
                        continue;
                    }
                } else {
                    if !option.key_url.is_empty() {
                        println!("  Get a key at {}", option.key_url);
                    }
                    let base_url = if option.needs_base_url {
                        Input::new()
                            .with_prompt("Address (OpenAI-compatible, ending in /v1)")
                            .interact_text()?
                    } else {
                        String::new()
                    };
                    // A server on this machine takes no key; the owner's
                    // own endpoint may take none.
                    let local = option.billing == aidaemon_client_protocol::Billing::Local;
                    let key = if local {
                        String::new()
                    } else if option.needs_base_url {
                        Password::new()
                            .with_prompt(format!(
                                "{} API key (empty if it needs none)",
                                option.name
                            ))
                            .allow_empty_password(true)
                            .interact()?
                    } else {
                        Password::new()
                            .with_prompt(format!("{} API key", option.name))
                            .interact()?
                    };
                    // The account's own models, from the provider.
                    println!(
                        "  {}",
                        if local {
                            "Looking for its models..."
                        } else {
                            "Checking the key..."
                        }
                    );
                    let checked = host
                        .step(
                            SetupAction::CheckKey {
                                provider: ProviderChoice {
                                    preset: option.preset.clone(),
                                    base_url: base_url.clone(),
                                    ..ProviderChoice::default()
                                },
                                key: SecretText(key.clone()),
                            },
                            true,
                        )
                        .await;
                    let models =
                        match checked {
                            Ok(state) => state.checked.unwrap_or_else(|| {
                                aidaemon_client_protocol::CheckedKey {
                                    preset: option.preset.clone(),
                                    models: Vec::new(),
                                    suggested: None,
                                }
                            }),
                            Err(reason) => {
                                println!("  {reason}");
                                continue;
                            }
                        };
                    let primary_model = if models.models.is_empty() {
                        Input::new().with_prompt("Model").interact_text()?
                    } else {
                        let default = models
                            .suggested
                            .as_ref()
                            .and_then(|s| models.models.iter().position(|m| m == s))
                            .unwrap_or(0);
                        let at = Select::new()
                            .with_prompt("Model")
                            .items(&models.models)
                            .default(default)
                            .interact()?;
                        models.models[at].clone()
                    };
                    SetupAction::ConnectModel {
                        provider: ProviderChoice {
                            preset: option.preset.clone(),
                            base_url,
                            primary_model,
                            ..ProviderChoice::default()
                        },
                        key: Some(SecretText(key)),
                    }
                }
            }
        };
        println!("  Testing...");
        match host.step(action, true).await {
            Ok(state) => {
                for notice in &state.notices {
                    println!("  {notice}");
                }
            }
            Err(reason) => println!("  {reason}"),
        }
    }
    let state = host.state(true).await;
    println!();
    println!(
        "  Set up with {}. Config saved to {}.",
        state.model.unwrap_or_default(),
        config_path.display()
    );
    println!("  Start it now with `aidaemon`, or at login with `aidaemon service install`.");
    println!("  Open the app (`aidaemon app`) to connect a chat and say hello.");
    Ok(())
}
