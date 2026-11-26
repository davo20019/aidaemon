//! The composer's recipient, model and effort: who this message goes to
//! (a specific coding-agent session with its live state, a fresh coding-agent
//! session, or AIDaemon with its model and effort). Only the app's own
//! conversations show it; a chat bot's conversations answer with that chat's
//! model.

use aidaemon_client_core::ComposerTarget;
use aidaemon_client_protocol::{Effort, ModelChoice, ModelOptions, RunPhase};
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

/// One chip in the composer's row: who the message goes to (a specific agent
/// session with its state, a new agent session, or AIDaemon with its model
/// and effort).
pub fn model_picker(store: Store, session: Option<String>) -> View {
    if !untracked(|| store.can_pick_model(session.as_deref())) {
        return view();
    }
    // On code, who does the work: a session, a new coding agent, or AIDaemon.
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
    let target_view = {
        let session = session.clone();
        computed(move || store.composer_target_view(session.as_deref(), code.get()))
    };
    let label = computed(move || target_view.get().chip);
    let shown = computed(move || {
        if target_view.get().shows_recipient {
            return true;
        }
        match store.model_options() {
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
        }
    });
    let chip = view()
        .h(px(if ui::touch() { 44.0 } else { 26.0 }))
        .flex_row()
        .items_center()
        .min_w(px(0.0))
        .max_w(percentage(1.0))
        .gap(px(4.0))
        .px(px(if ui::touch() { 6.0 } else { 8.0 }))
        .child(
            text(move || label.get())
                .truncate()
                .flex_shrink(1.0)
                .min_w(px(0.0))
                .max_w(px(240.0))
                .text_size(px(ui::type_scale::XS))
                .text_color(ui::colors::MUTED),
        )
        .child(
            ui::icon_inline(ui::Glyph::Down)
                .flex_shrink_0()
                .text_color(ui::colors::FAINT)
                .accessibility_elements_hidden(true),
        );
    let pick_session = session.clone();
    view()
        .test_id("model-picker")
        .flex_shrink(1.0)
        .min_w(px(0.0))
        .child(when(
            move || shown.get(),
            move || {
                let session = pick_session.clone();
                let has_session = session.is_some();
                ui::pick_list(
                    ui::Picker {
                        title: if target_view.get_untracked().shows_recipient {
                            "Send to"
                        } else {
                            "Model and effort"
                        },
                        items: move || {
                            let view = target_view.get_untracked();
                            if view.shows_recipient {
                                recipient_choices(
                                    store,
                                    has_session,
                                    view.target,
                                    current.get_untracked(),
                                )
                            } else {
                                choices(store, current.get_untracked())
                            }
                        },
                        pick: move |id: String| {
                            if let Some(thread) = id.strip_prefix("session:") {
                                if let Some(session_id) = session.as_deref() {
                                    store.set_composer_target(
                                        session_id,
                                        ComposerTarget::Session {
                                            thread: thread.to_owned(),
                                        },
                                    );
                                }
                            } else if let Some(agent) = id.strip_prefix("new_agent:") {
                                let agent_model = ModelChoice::Agent {
                                    agent: agent.to_owned(),
                                };
                                if code.get_untracked() {
                                    store.remember_code_worker(&agent_model);
                                }
                                match session.as_deref() {
                                    Some(session_id) => store.set_composer_target(
                                        session_id,
                                        ComposerTarget::NewAgent {
                                            agent: agent.to_owned(),
                                        },
                                    ),
                                    None => store.set_session_model(None, Some(agent_model), None),
                                }
                            } else if id == "aidaemon" {
                                if code.get_untracked() {
                                    store.remember_code_worker(&ModelChoice::Auto);
                                }
                                match session.as_deref() {
                                    Some(session_id) => store
                                        .set_composer_target(session_id, ComposerTarget::Aidaemon),
                                    None => {
                                        store.set_session_model(None, Some(ModelChoice::Auto), None)
                                    }
                                }
                            } else if let Some(model) = id.strip_prefix("model:") {
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
                        let current_label = label.get();
                        if target_view.get().shows_recipient {
                            let clean =
                                current_label.strip_prefix("To: ").unwrap_or(&current_label);
                            format!("Send to: {clean}")
                        } else {
                            format!("Model and effort: {current_label}")
                        }
                    },
                    chip.clone(),
                    false,
                )
            },
        ))
}

/// Recipient picker for conversations with coding agents (or Code mode):
/// 1. Every addressable session in the conversation, working/waiting first;
/// 2. `"New <Agent>"` for each ready coding agent;
/// 3. AIDaemon (plus its model & effort options when AIDaemon is selected).
fn recipient_choices(
    store: Store,
    has_session: bool,
    target: ComposerTarget,
    (model, effort): (ModelChoice, Effort),
) -> Vec<ui::PickItem> {
    let mut items = Vec::new();
    if has_session {
        for p in untracked(|| store.participants())
            .into_iter()
            .filter(|p| p.can_address())
        {
            let state = if p.waiting {
                "needs approval".to_owned()
            } else if p.working() {
                if !p.latest.step.is_empty() {
                    format!("working · {}", p.latest.step)
                } else {
                    "working".to_owned()
                }
            } else {
                match p.latest.phase {
                    RunPhase::Failed => "idle · last run failed".to_owned(),
                    RunPhase::Stopped => "idle · stopped".to_owned(),
                    _ if p.runs > 1 => format!("idle · {} runs", p.runs),
                    _ => "idle".to_owned(),
                }
            };
            let checked =
                matches!(&target, ComposerTarget::Session { thread } if *thread == p.thread);
            items.push(
                ui::PickItem::new(format!("session:{}", p.thread), p.title())
                    .detail(state)
                    .group("Sessions")
                    .checked(checked),
            );
        }
    }
    if let Read::Ready(options) = untracked(|| store.model_options()) {
        for agent in &options.agents {
            let checked =
                matches!(&target, ComposerTarget::NewAgent { agent: id } if *id == agent.id);
            items.push(
                ui::PickItem::new(
                    format!("new_agent:{}", agent.id),
                    format!("New {}", agent.name),
                )
                .detail("Start a fresh session")
                .group("New session")
                .checked(checked),
            );
        }
    }
    items.push(
        ui::PickItem::new("aidaemon", "AIDaemon")
            .detail("Plans and works with its own model")
            .group("Assistant")
            .checked(target == ComposerTarget::Aidaemon),
    );
    if target == ComposerTarget::Aidaemon {
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
