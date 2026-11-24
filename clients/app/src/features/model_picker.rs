//! The composer's model and effort: what this conversation (or the next
//! new one) answers with. Only the app's own conversations show it; a chat
//! bot's conversations answer with that chat's model. A conversation on
//! code picks who does the work instead: AIDaemon, or a coding agent that
//! takes each message straight, with no model call.

use aidaemon_client_protocol::{Effort, ModelChoice, ModelOptions};
use nsh_core::prelude::*;

use crate::store::{Read, Store};
use crate::ui;

/// A model choice as a list value: `auto`, provider and model, or
/// `agent` and the agent.
fn encode(model: &ModelChoice) -> String {
    match model {
        ModelChoice::Pinned { provider, model } => format!("{provider}\n{model}"),
        ModelChoice::Agent { agent } => format!("{AGENT}\t{agent}"),
        _ => "auto".to_owned(),
    }
}

fn decode(value: &str) -> ModelChoice {
    if let Some(agent) = value.strip_prefix(AGENT).and_then(|v| v.strip_prefix('\t')) {
        return ModelChoice::Agent {
            agent: agent.to_owned(),
        };
    }
    match value.split_once('\n') {
        Some((provider, model)) => ModelChoice::Pinned {
            provider: provider.to_owned(),
            model: model.to_owned(),
        },
        None => ModelChoice::Auto,
    }
}

/// The list value's mark for a coding agent (a provider id never has a tab).
const AGENT: &str = "agent";

/// The name the picker shows for coding agent `id` (the id when unknown).
fn agent_name(options: &ModelOptions, id: &str) -> String {
    options
        .agents
        .iter()
        .find(|agent| agent.id == id)
        .map_or_else(|| id.to_owned(), |agent| agent.name.clone())
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
    // On code, who does the work: AIDaemon or a coding agent.
    let code = {
        let session = session.clone();
        computed(move || match &session {
            Some(id) => store.is_coding(id),
            None => store.new_is_code(),
        })
    };
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
    // A new code conversation starts with who did the work in this
    // device's last one.
    if session.is_none() {
        effect(move || {
            if code.get()
                && matches!(current.get().0, ModelChoice::Auto)
                && let Some(agent) = store.last_code_agent()
            {
                untracked(|| {
                    store.set_session_model(None, Some(ModelChoice::Agent { agent }), None)
                });
            }
        });
    }
    let label = computed(move || {
        let (model, effort) = current.get();
        let is_code = code.get();
        let model = match model {
            ModelChoice::Pinned { model, .. } if is_code => format!("AIDaemon · {model}"),
            ModelChoice::Pinned { model, .. } => model,
            ModelChoice::Agent { agent } => match store.model_options() {
                Read::Ready(options) => agent_name(&options, &agent),
                _ => agent,
            },
            _ if is_code => "AIDaemon".to_owned(),
            _ => "Auto".to_owned(),
        };
        match effort {
            Effort::Auto => model,
            effort => format!("{model} · {}", effort_word(effort)),
        }
    });
    let shown = computed(move || match store.model_options() {
        Read::Ready(options) if code.get() => {
            !options.agents.is_empty()
                || matches!(current.get().0, ModelChoice::Agent { .. })
                || options
                    .providers
                    .iter()
                    .map(|p| p.models.len())
                    .sum::<usize>()
                    >= 2
                || !store.efforts_of(&current.get().0).is_empty()
        }
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
                    title: if code.get_untracked() {
                        "Who does the work"
                    } else {
                        "Model and effort"
                    },
                    items: move || {
                        if code.get_untracked() {
                            code_choices(store, current.get_untracked())
                        } else {
                            choices(store, current.get_untracked())
                        }
                    },
                    pick: move |id: String| {
                        if let Some(model) = id.strip_prefix("model:") {
                            let model = decode(model);
                            if code.get_untracked() {
                                store.remember_code_worker(&model);
                            }
                            store.set_session_model(session.as_deref(), Some(model), None);
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
                move || {
                    if code.get() {
                        format!("Who does the work: {}", label.get())
                    } else {
                        format!("Model and effort: {}", label.get())
                    }
                },
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

/// In a code conversation: the `Agent` group (AIDaemon and each ready coding
/// agent), plus AIDaemon's `Model` and `Effort` groups whenever AIDaemon is
/// selected so the owner can still pin a model or effort for AIDaemon in Code.
fn code_choices(store: Store, (model, effort): (ModelChoice, Effort)) -> Vec<ui::PickItem> {
    let is_agent = matches!(model, ModelChoice::Agent { .. });
    let mut items = agent_choices(store, model.clone());
    if !is_agent {
        let model_and_effort = choices(store, (model, effort));
        let has_models = matches!(
            untracked(|| store.model_options()),
            Read::Ready(options)
                if options
                    .providers
                    .iter()
                    .map(|p| p.models.len())
                    .sum::<usize>()
                    >= 2
        );
        let has_efforts = !untracked(|| store.efforts_of(&ModelChoice::Auto)).is_empty()
            || model_and_effort
                .iter()
                .any(|item| item.id.starts_with("effort:"));
        if has_models || has_efforts {
            items.extend(model_and_effort);
        }
    }
    items
}

/// AIDaemon (its own model, picked for each message), then each coding
/// agent ready on the daemon's computer.
fn agent_choices(store: Store, model: ModelChoice) -> Vec<ui::PickItem> {
    let mut items = vec![
        ui::PickItem::new("model:auto", "AIDaemon")
            .detail("Plans and works with its own model")
            .group("Agent")
            .checked(!matches!(model, ModelChoice::Agent { .. })),
    ];
    if let Read::Ready(options) = untracked(|| store.model_options()) {
        let current = encode(&model);
        for agent in &options.agents {
            let value = encode(&ModelChoice::Agent {
                agent: agent.id.clone(),
            });
            items.push(
                ui::PickItem::new(format!("model:{value}"), agent.name.clone())
                    .detail("Each message goes straight to it")
                    .group("Agent")
                    .checked(value == current),
            );
        }
    }
    items
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
        let agent = ModelChoice::Agent {
            agent: "codex".into(),
        };
        assert_eq!(decode(&encode(&agent)), agent);
        for effort in [Effort::Auto, Effort::Low, Effort::Medium, Effort::High] {
            assert_eq!(effort_of(effort_value(effort)), effort);
        }
    }
}
