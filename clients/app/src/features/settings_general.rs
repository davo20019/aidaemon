//! The Settings page's everyday parts: the General tab (a short list of the
//! settings most owners change, each with a real control) and the decision
//! model on the Models tab. Every change is a `SettingsChange::Set` of the
//! config key, so it takes the same path, approvals and refusals as the
//! full list on the Advanced tab.

use aidaemon_client_protocol::{SecretText, Setting, Settings, SettingsChange};
use nsh_core::prelude::*;

use crate::store::Store;
use crate::ui::{self, Size, Tone, Variant, colors, section_label, type_scale};

fn muted(words: impl Into<SharedString>) -> Text {
    text(words.into())
        .text_size(px(ui::fit::small()))
        .text_color(colors::MUTED)
}

fn faint(words: impl Into<SharedString>) -> Text {
    text(words.into())
        .text_size(px(type_scale::XS))
        .text_color(colors::FAINT)
}

/// The setting at `key`, as the daemon sent it.
fn find<'a>(settings: &'a Settings, key: &str) -> Option<&'a Setting> {
    settings
        .groups
        .iter()
        .flat_map(|g| &g.settings)
        .find(|s| s.key == key)
}

/// A TOML value as plain text: a quoted string without its quotes.
fn plain(toml: &str) -> String {
    let toml = toml.trim();
    toml.strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .map(|t| t.replace("\\\"", "\"").replace("\\\\", "\\"))
        .unwrap_or_else(|| toml.to_string())
}

/// `text` as a TOML string.
fn quoted(text: &str) -> String {
    format!(
        "\"{}\"",
        text.trim().replace('\\', "\\\\").replace('"', "\\\"")
    )
}

/// What `key` is now: its value, else its documented default, else
/// `fallback`.
fn current(settings: &Settings, key: &str, fallback: &str) -> String {
    find(settings, key)
        .and_then(|s| s.value.clone().or_else(|| s.example.clone()))
        .map(|v| plain(&v))
        .unwrap_or_else(|| fallback.to_string())
}

fn set(store: Store, key: &str, toml: String) {
    store.change_settings(SettingsChange::Set {
        key: key.to_string(),
        value: Some(toml),
    });
}

/// How a General setting is changed.
#[derive(Clone, Copy)]
enum Control {
    Switch,
    /// One of (value, title).
    Pick(&'static [(&'static str, &'static str)]),
    Text(&'static str),
    Number(&'static str),
}

struct Item {
    key: &'static str,
    title: &'static str,
    help: &'static str,
    control: Control,
    /// What it is when neither the file nor the example says.
    default: &'static str,
}

const fn item(
    key: &'static str,
    title: &'static str,
    help: &'static str,
    control: Control,
    default: &'static str,
) -> Item {
    Item {
        key,
        title,
        help,
        control,
        default,
    }
}

const TRUST: &[(&str, &str)] = &[
    ("auto", "Decide by model"),
    ("guided", "Guided: checks its work more"),
    ("autonomous", "Autonomous: fewer checks"),
];

/// The General tab, by section.
const GENERAL: &[(&str, &[Item])] = &[
    (
        "Assistant",
        &[
            item(
                "agent.name",
                "Name",
                "What the assistant calls itself.",
                Control::Text("aidaemon"),
                "aidaemon",
            ),
            item(
                "policy.trust_tier",
                "How closely it checks itself",
                "Decide by model picks for you; frontier models get fewer checks.",
                Control::Pick(TRUST),
                "auto",
            ),
        ],
    ),
    (
        "What it can do",
        &[
            item(
                "browser.enabled",
                "Browse websites",
                "Open pages, click and fill forms in its own browser.",
                Control::Switch,
                "true",
            ),
            item(
                "browser.headless",
                "Keep the browser hidden",
                "Off shows its browser window on the daemon's computer.",
                Control::Switch,
                "true",
            ),
            item(
                "computer_use.enabled",
                "Use this computer's screen",
                "Operate desktop apps on macOS (needs Accessibility and Screen Recording).",
                Control::Switch,
                "false",
            ),
            item(
                "cli_agents.enabled",
                "Coding agents",
                "Hand coding work to Claude Code, Codex or Gemini CLI when installed.",
                Control::Switch,
                "false",
            ),
            item(
                "subagents.enabled",
                "Helper agents",
                "Split big tasks between helper agents that work in parallel.",
                Control::Switch,
                "true",
            ),
            item(
                "skills.enabled",
                "Skills",
                "Reusable instructions it learns and follows.",
                Control::Switch,
                "true",
            ),
        ],
    ),
    (
        "Files and history",
        &[
            item(
                "execution.workspace_root",
                "Workspace folder",
                "Where relative paths and commands start.",
                Control::Text("/path/to/folder"),
                "",
            ),
            item(
                "execution.allow_outside_workspace",
                "Reach outside the workspace",
                "Off keeps file tools and commands inside the workspace folder.",
                Control::Switch,
                "true",
            ),
            item(
                "state.retention.messages_days",
                "Keep chat history (days)",
                "0 keeps it forever.",
                Control::Number("90"),
                "90",
            ),
        ],
    ),
];

/// The General tab.
pub fn general(store: Store, settings: &Settings) -> View {
    let mut page = view().gap_5().child(muted(
        "The settings most people change. Every setting is on the Advanced tab.",
    ));
    for (title, items) in GENERAL {
        let mut list = ui::surface().gap_0();
        for (at, item) in items.iter().enumerate() {
            if at > 0 {
                list = list.child(ui::separator());
            }
            list = list.child(row(store, settings, item));
        }
        page = page.child(view().gap_2().child(section_label(*title)).child(list));
    }
    page.child(this_device(store))
}

/// How this device's composer behaves: kept on the device, not the daemon.
fn this_device(store: Store) -> View {
    view().gap_2().child(section_label("On this device")).child(
        ui::surface()
            .p_3()
            .gap_2()
            .child(ui::labeled(
                "While the agent works, Enter",
                ui::choice(
                    "While the agent works, Enter",
                    "Tells it now",
                    vec![
                        ("now".to_owned(), "Tells it now (⌘↩ sends after)".to_owned()),
                        (
                            "after".to_owned(),
                            "Sends after it (⌘↩ tells it now)".to_owned(),
                        ),
                    ],
                    move || Some(if store.enter_queues() { "after" } else { "now" }.to_owned()),
                    move |chosen| store.set_enter_queues(chosen.as_deref() == Some("after")),
                ),
            ))
            .child(faint(
                "Telling it now: the running work reads your message at its next step. Sending \
                 after: it waits under \"Waiting to send\" and runs as its own turn.",
            )),
    )
}

fn row(store: Store, settings: &Settings, item: &Item) -> View {
    let key = item.key;
    let now = current(settings, key, item.default);
    let setting = find(settings, key);
    let asks = setting.is_some_and(|s| s.guarded);
    let later = match setting.map(|s| s.applies) {
        Some(aidaemon_client_protocol::Applies::Restart) => Some("Takes effect on a restart."),
        Some(aidaemon_client_protocol::Applies::Reload) => Some("Takes effect on a reload."),
        _ => None,
    };
    let mut words = view()
        .flex_1()
        .min_w(px(0.0))
        .gap_0p5()
        .child(text(item.title).text_size(px(ui::fit::body())))
        .child(faint(item.help).number_of_lines(3));
    if let Some(later) = later {
        words = words.child(faint(later));
    }
    let head = view()
        .flex_row()
        .items_center()
        .gap_3()
        .child(words)
        .child(when(asks, || ui::badge("asks", Tone::Warning)));
    let body = match item.control {
        Control::Switch => {
            let on = now == "true";
            return view().p_3().child(head.child(ui::switch_only(
                item.title,
                move || on,
                move |next: bool| set(store, key, next.to_string()),
            )));
        }
        Control::Pick(options) => {
            let chosen = signal(now);
            let items = options
                .iter()
                .map(|(v, t)| (v.to_string(), t.to_string()))
                .collect();
            view().flex_row().child(ui::choice(
                item.title,
                "Choose",
                items,
                move || Some(chosen.get()),
                move |next: Option<String>| {
                    if let Some(next) = next {
                        chosen.set(next.clone());
                        set(store, key, quoted(&next));
                    }
                },
            ))
        }
        Control::Text(placeholder) | Control::Number(placeholder) => {
            let number = matches!(item.control, Control::Number(_));
            let draft = signal(now.clone());
            let saved = now;
            let save = move || {
                let value = draft.get_untracked();
                let value = value.trim();
                if number {
                    if value.parse::<i64>().is_err() {
                        return;
                    }
                    set(store, key, value.to_string());
                } else {
                    set(store, key, quoted(value));
                }
            };
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(
                    ui::field(placeholder, draft)
                        .label(item.title)
                        .flex_1()
                        .min_w(px(0.0)),
                )
                .child(
                    ui::btn("Save", Variant::Secondary, Size::Sm, save).disabled(move || {
                        draft.with(|d| {
                            d.trim() == saved.trim()
                                || d.trim().is_empty()
                                || (number && d.trim().parse::<i64>().is_err())
                        })
                    }),
                )
        }
    };
    view().p_3().gap_2().child(head).child(body)
}

const BACKENDS: &[(&str, &str)] = &[
    ("off", "Off"),
    ("provider", "Your chat model"),
    ("gemini", "Google Gemini decisions"),
    ("typesafe", "TypeSafe Jev"),
];

const MODES: &[(&str, &str)] = &[("off", "Off"), ("shadow", "Watch only"), ("on", "On")];

/// Each decision: its key under `decisions.mode`, what it decides, and its
/// mode when unset.
const DECISIONS: &[(&str, &str, &str)] = &[
    (
        "turn_digest",
        "How a message fits the conversation",
        "shadow",
    ),
    ("resume", "Whether to pick up unfinished work", "shadow"),
    (
        "recall_filter",
        "Which memories matter to a message",
        "shadow",
    ),
    ("model_route", "Which model tier a request needs", "off"),
    (
        "command_risk_second",
        "A second opinion on risky commands",
        "off",
    ),
];

/// What a decision backend is, in a sentence.
fn backend_words(backend: &str) -> &'static str {
    match backend {
        "provider" => {
            "Your main provider's fast model answers. Its confidence is its own guess, so \
             decisions that need an honest one stay on Watch only."
        }
        "gemini" => {
            "Google's Decisions API: fast, with calibrated confidence. What it decides on is \
             sent to Google. Empty key: your Google AI Studio provider's key."
        }
        "typesafe" => {
            "TypeSafe's Jev, a System One model: typed answers with calibrated probabilities \
             in well under a second. What it decides on is sent to TypeSafe (early access)."
        }
        _ => "The assistant decides these itself, as it would without a decision model.",
    }
}

/// The decision model, for the Models tab.
pub fn decisions(store: Store, settings: &Settings) -> View {
    let backend = signal(current(settings, "decisions.backend", "off"));
    let model_now = current(settings, "decisions.model", "");
    let model = signal(model_now.clone());
    let key = signal(String::new());
    let key_set = find(settings, "decisions.api_key").is_some_and(|s| s.secret_set);
    let mut section = view()
        .gap_2()
        .child(section_label("Decision model"))
        .child(muted(
            "A fast model that answers small yes-or-which questions with a confidence, so \
             the assistant routes and recalls without a full model call.",
        ));
    let mut card = ui::surface().p_3().gap_3().child(
        view()
            .flex_row()
            .items_center()
            .gap_3()
            .child(text("Backend").flex_1().text_size(px(ui::fit::body())))
            .child(ui::choice(
                "Decision backend",
                "Choose",
                BACKENDS
                    .iter()
                    .map(|(v, t)| (v.to_string(), t.to_string()))
                    .collect(),
                move || Some(backend.get()),
                move |next: Option<String>| {
                    if let Some(next) = next {
                        backend.set(next.clone());
                        set(store, "decisions.backend", quoted(&next));
                    }
                },
            )),
    );
    card = card.child(dynamic(move || {
        let chosen = backend.get();
        let model_now = model_now.clone();
        Some(untracked(move || {
            let mut more = view().gap_3().child(faint(backend_words(&chosen)));
            if chosen == "off" {
                return more;
            }
            let placeholder = match chosen.as_str() {
                "gemini" => "gemini-decision-1",
                "typesafe" => "jev-latest",
                _ => "Your fast model",
            };
            let saved = model_now.clone();
            more = more
                .child(faint(format!("Model (empty: {placeholder})")))
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .child(
                            ui::field(placeholder, model)
                                .label("Decision model name")
                                .flex_1()
                                .min_w(px(0.0)),
                        )
                        .child(
                            ui::btn("Save", Variant::Secondary, Size::Sm, move || {
                                let value = model.get_untracked();
                                if value.trim().is_empty() {
                                    store.change_settings(SettingsChange::Set {
                                        key: "decisions.model".into(),
                                        value: None,
                                    });
                                } else {
                                    set(store, "decisions.model", quoted(&value));
                                }
                            })
                            .disabled(move || model.with(|m| m.trim() == saved.trim())),
                        ),
                );
            if chosen != "provider" {
                let status = if key_set {
                    ui::status("Key in the keychain", Tone::Success)
                } else if chosen == "gemini" {
                    ui::status("Uses your Google AI Studio provider's key", Tone::Neutral)
                } else {
                    ui::status("No key: it cannot answer", Tone::Danger)
                };
                let mut keys = view().flex_row().flex_wrap().items_center().gap_2().child(
                    ui::btn("Save key", Variant::Primary, Size::Sm, move || {
                        let value = key.get_untracked();
                        if value.trim().is_empty() {
                            return;
                        }
                        store.change_settings(SettingsChange::Secret {
                            key: "decisions.api_key".into(),
                            value: SecretText(value.trim().to_string()),
                        });
                        key.set(String::new());
                    })
                    .disabled(move || key.with(|k| k.trim().is_empty())),
                );
                let url = match chosen.as_str() {
                    "gemini" => "https://aistudio.google.com/apikey",
                    _ => "https://console.typesafe.ai/",
                };
                keys = keys.child(ui::btn("Get a key", Variant::Ghost, Size::Sm, move || {
                    drop(open_url(url));
                }));
                more = more
                    .child(status)
                    .child(
                        ui::secret_field("Paste its API key", key)
                            .label("Decision API key")
                            .w_full(),
                    )
                    .child(keys);
            }
            more
        }))
    }));
    // What each decision may do.
    let rows: Vec<(String, &'static str, Signal<String>)> = DECISIONS
        .iter()
        .map(|(name, words, default)| {
            let full = format!("decisions.mode.{name}");
            let chosen = signal(current(settings, &full, default));
            (full, *words, chosen)
        })
        .collect();
    card = card.child(when(
        move || backend.get() != "off",
        move || {
            let mut modes = view().gap_0();
            for (at, (full, words, chosen)) in rows.iter().cloned().enumerate() {
                if at > 0 {
                    modes = modes.child(ui::separator());
                }
                modes = modes.child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_3()
                        .py_2()
                        .child(
                            text(words)
                                .flex_1()
                                .min_w(px(0.0))
                                .number_of_lines(2)
                                .text_size(px(ui::fit::small())),
                        )
                        .child(ui::choice(
                            words,
                            "Choose",
                            MODES
                                .iter()
                                .map(|(v, t)| (v.to_string(), t.to_string()))
                                .collect(),
                            move || Some(chosen.get()),
                            move |next: Option<String>| {
                                if let Some(next) = next {
                                    chosen.set(next.clone());
                                    set(store, &full, quoted(&next));
                                }
                            },
                        )),
                );
            }
            view()
                .gap_1()
                .child(text("What it decides").text_size(px(ui::fit::body())))
                .child(faint(
                    "Watch only asks and records the answer without acting on it, to see \
                     how it does first.",
                ))
                .child(modes)
        },
    ));
    section = section.child(card);
    section
}

/// A config table's name in plain words.
pub fn group_title(name: &str) -> String {
    let known = match name {
        "agent" => "Assistant",
        "provider" => "Model providers",
        "decisions" => "Decision model",
        "execution" => "Files and commands",
        "terminal" => "Terminal",
        "state" => "Storage and history",
        "memory" => "Memory",
        "browser" => "Browser",
        "computer_use" => "Using this computer",
        "subagents" => "Helper agents",
        "cli_agents" => "Coding agents",
        "skills" => "Skills",
        "policy" => "Checks and limits",
        "telegram" | "telegram_bots" => "Telegram",
        "slack" | "slack_bots" => "Slack",
        "discord" | "discord_bots" => "Discord",
        "client_gateway" => "Apps (gateway)",
        "daemon" => "Daemon",
        "files" => "Files and voice notes",
        "search" => "Web search",
        "nodes" => "Devices",
        "mcp" => "MCP servers",
        "diagnostics" => "Diagnostics",
        "checkpoints" => "Checkpoints",
        "triggers" => "Triggers",
        "health" => "Health checks",
        "tools" => "Tools",
        "behavior" => "Behavior",
        _ => "",
    };
    if known.is_empty() {
        human(name)
    } else {
        known.to_string()
    }
}

/// `snake_case` as words, first letter upper-case.
pub fn human(name: &str) -> String {
    let words = name.replace(['_', '-'], " ");
    let mut chars = words.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// A setting's own name in plain words: its last part of the key.
pub fn setting_title(key: &str) -> String {
    let last = key.rsplit('.').next().unwrap_or(key);
    // A list entry (`fallbacks[0]`) reads as its table.
    human(last.split('[').next().unwrap_or(last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_text_reads_plain_and_writes_back_quoted() {
        assert_eq!(plain("\"gpt-x\""), "gpt-x");
        assert_eq!(plain("90"), "90");
        assert_eq!(quoted("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(plain(&quoted("a\\b \"c\"")), "a\\b \"c\"");
    }

    #[test]
    fn keys_read_as_words() {
        assert_eq!(
            setting_title("state.retention.messages_days"),
            "Messages days"
        );
        assert_eq!(group_title("computer_use"), "Using this computer");
        assert_eq!(group_title("some_table"), "Some table");
    }
}
