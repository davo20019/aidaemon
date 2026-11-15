//! The composer's model and effort: what this conversation (or the next
//! new one) answers with. Only the app's own conversations show it; a chat
//! bot's conversations answer with that chat's model.

use aidaemon_client_protocol::{Effort, ModelChoice, ModelOptions};
use nsh_core::prelude::*;

use crate::store::{Read, Store};
use crate::ui;

/// A model choice as a list value: `auto`, or provider and model.
fn encode(model: &ModelChoice) -> String {
    match model {
        ModelChoice::Pinned { provider, model } => format!("{provider}\n{model}"),
        _ => "auto".to_owned(),
    }
}

fn decode(value: &str) -> ModelChoice {
    match value.split_once('\n') {
        Some((provider, model)) => ModelChoice::Pinned {
            provider: provider.to_owned(),
            model: model.to_owned(),
        },
        None => ModelChoice::Auto,
    }
}

fn effort_value(effort: Effort) -> &'static str {
    match effort {
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        _ => "auto",
    }
}

fn effort_of(value: &str) -> Effort {
    match value {
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        _ => Effort::Auto,
    }
}

fn effort_title(effort: Effort) -> &'static str {
    match effort {
        Effort::Low => "Low effort",
        Effort::Medium => "Medium effort",
        Effort::High => "High effort",
        _ => "Auto effort",
    }
}

/// `Auto`, then every model; the provider is named when there is more
/// than one.
fn model_items(options: &ModelOptions) -> Vec<(String, String)> {
    let named = options.providers.len() > 1;
    let mut items = vec![("auto".to_owned(), "Auto model".to_owned())];
    for provider in &options.providers {
        for model in &provider.models {
            let title = if named {
                format!("{} · {}", model.id, provider.name)
            } else {
                model.id.clone()
            };
            items.push((
                encode(&ModelChoice::Pinned {
                    provider: provider.id.clone(),
                    model: model.id.clone(),
                }),
                title,
            ));
        }
    }
    items
}

pub fn model_picker(store: Store, session: Option<String>) -> View {
    if !untracked(|| store.can_pick_model(session.as_deref())) {
        return view();
    }
    // Read the lists once online (again after a reconnect: the daemon's
    // providers may have changed).
    {
        let session = session.clone();
        effect(move || {
            if store.is_online() {
                untracked(|| {
                    store.refresh_models();
                    if let Some(id) = &session {
                        store.load_session_model(id);
                    }
                });
            }
        });
    }
    let current = {
        let session = session.clone();
        computed(move || store.model_of(session.as_deref()))
    };
    let models = {
        let session = session.clone();
        dynamic(move || {
            let Read::Ready(options) = store.model_options() else {
                return Some(view());
            };
            // Only one model and nothing to choose between: no picker.
            if options
                .providers
                .iter()
                .map(|p| p.models.len())
                .sum::<usize>()
                < 2
            {
                return Some(view());
            }
            let session = session.clone();
            Some(untracked(move || {
                view().child(ui::choice_compact(
                    "Model",
                    "Auto model",
                    model_items(&options),
                    move || Some(encode(&current.get().0)),
                    move |chosen| {
                        let model = chosen.as_deref().map_or(ModelChoice::Auto, decode);
                        store.set_session_model(session.as_deref(), Some(model), None);
                    },
                ))
            }))
        })
    };
    let efforts = dynamic(move || {
        let (model, _) = current.get();
        let efforts = store.efforts_of(&model);
        if efforts.is_empty() {
            return Some(view());
        }
        let session = session.clone();
        Some(untracked(move || {
            let mut items = vec![("auto".to_owned(), effort_title(Effort::Auto).to_owned())];
            items.extend(
                efforts
                    .into_iter()
                    .map(|e| (effort_value(e).to_owned(), effort_title(e).to_owned())),
            );
            view().child(ui::choice_compact(
                "Effort",
                "Auto effort",
                items,
                move || Some(effort_value(current.get().1).to_owned()),
                move |chosen| {
                    let effort = chosen.as_deref().map_or(Effort::Auto, effort_of);
                    store.set_session_model(session.as_deref(), None, Some(effort));
                },
            ))
        }))
    });
    view()
        .test_id("model-picker")
        .flex_row()
        .flex_wrap()
        .items_center()
        .gap_1()
        .child(models)
        .child(efforts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_choice_round_trips_through_its_list_value() {
        let pinned = ModelChoice::Pinned {
            provider: "failover-0".into(),
            model: "synthetic-model".into(),
        };
        assert_eq!(decode(&encode(&pinned)), pinned);
        assert_eq!(decode(&encode(&ModelChoice::Auto)), ModelChoice::Auto);
        for effort in [Effort::Auto, Effort::Low, Effort::Medium, Effort::High] {
            assert_eq!(effort_of(effort_value(effort)), effort);
        }
    }
}
