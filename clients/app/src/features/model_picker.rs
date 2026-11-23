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

/// One chip in the composer's row: the model (Auto or the one pinned) and
/// the effort when it is not Auto. A press opens one picker with the
/// models (their provider under each when there are several) and, for a
/// model that takes one, the efforts, each with what is in effect checked.
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
    let label = computed(move || {
        let (model, effort) = current.get();
        let model = match model {
            ModelChoice::Pinned { model, .. } => model,
            _ => "Auto".to_owned(),
        };
        match effort {
            Effort::Auto => model,
            effort => format!("{model} · {}", effort_word(effort)),
        }
    });
    let shown = computed(move || match store.model_options() {
        Read::Ready(options) => {
            options
                .providers
                .iter()
                .map(|p| p.models.len())
                .sum::<usize>()
                >= 2
                || !store.efforts_of(&current.get().0).is_empty()
        }
        _ => false,
    });
    let chip = view()
        .h(px(if ui::touch() { 44.0 } else { 26.0 }))
        .flex_row()
        .items_center()
        .gap(px(4.0))
        .px(px(8.0))
        .child(
            text(move || label.get())
                .number_of_lines(1)
                .max_w(px(180.0))
                .text_size(px(ui::type_scale::XS))
                .text_color(ui::colors::MUTED),
        )
        .child(
            ui::icon(ui::Glyph::Down)
                .text_color(ui::colors::FAINT)
                .accessibility_elements_hidden(true),
        );
    let pick_session = session.clone();
    view().test_id("model-picker").child(when(
        move || shown.get(),
        move || {
            let session = pick_session.clone();
            ui::pick_list(
                ui::Picker {
                    title: "Model and effort",
                    items: move || choices(store, current.get_untracked()),
                    pick: move |id: String| {
                        if let Some(model) = id.strip_prefix("model:") {
                            store.set_session_model(session.as_deref(), Some(decode(model)), None);
                        } else if let Some(effort) = id.strip_prefix("effort:") {
                            store.set_session_model(
                                session.as_deref(),
                                None,
                                Some(effort_of(effort)),
                            );
                        }
                    },
                    action: None,
                },
                move || format!("Model and effort: {}", label.get()),
                chip.clone(),
                false,
            )
        },
    ))
}

fn effort_word(effort: Effort) -> &'static str {
    match effort {
        Effort::Low => "Low",
        Effort::Medium => "Medium",
        Effort::High => "High",
        _ => "Auto",
    }
}

/// The models, then the efforts the current model takes.
fn choices(store: Store, (model, effort): (ModelChoice, Effort)) -> Vec<ui::PickItem> {
    let mut items = Vec::new();
    if let Read::Ready(options) = untracked(|| store.model_options()) {
        let current = encode(&model);
        for (value, title) in model_items(&options) {
            let (title, detail) = match title.split_once(" · ") {
                Some((model, provider)) => (model.to_owned(), provider.to_owned()),
                None if value == "auto" => (
                    "Auto".to_owned(),
                    "AIdaemon picks its hard-work or quick-jobs model for each message".to_owned(),
                ),
                None => (title, String::new()),
            };
            items.push(
                ui::PickItem::new(format!("model:{value}"), title)
                    .detail(detail)
                    .group("Model")
                    .checked(value == current),
            );
        }
    }
    let efforts = untracked(|| store.efforts_of(&model));
    if !efforts.is_empty() {
        for e in std::iter::once(Effort::Auto).chain(efforts) {
            items.push(
                ui::PickItem::new(format!("effort:{}", effort_value(e)), effort_title(e))
                    .group("Effort")
                    .checked(e == effort),
            );
        }
    }
    items
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
