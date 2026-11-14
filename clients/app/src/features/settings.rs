//! The Settings page: every setting of the daemon, by config table, each
//! editable in place (secrets only stored, never shown); providers and
//! failover; runtime bots and MCP servers; the database; and Reload or
//! Restart for saved changes. Read on open, on Refresh and when a change
//! lands. Guarded changes wait for an approval, shown like any other.

use aidaemon_client_protocol::{
    Applies, ApplyHow, BotInfo, DatabaseAction, McpAction, McpServerInfo, ProviderChoice,
    SecretText, Setting, SettingGroup, SettingKind, Settings, SettingsChange,
};
use nsh_core::prelude::*;

use crate::store::{SettingsRead, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, section_label, size, type_scale};

pub fn page(store: Store) -> View {
    let filter = signal(String::new());
    view().flex_1().min_w(px(0.0)).child(header(store)).child(
        scroll().flex_1().child(
            view().items_center().child(
                view()
                    .w_full()
                    .max_w(px(size::COLUMN))
                    .px_4()
                    .py_4()
                    .gap_3()
                    // What the last change did (or why it did not), until
                    // dismissed: an error here is worth reading.
                    .child(when(
                        move || store.toast().is_some(),
                        move || {
                            ui::banner(move || store.toast().unwrap_or_default(), Tone::Neutral)
                                .child(ui::btn("Dismiss", Variant::Ghost, Size::Sm, move || {
                                    store.dismiss_toast()
                                }))
                        },
                    ))
                    .child(
                        ui::field("Filter settings", filter)
                            .label("Filter settings")
                            .w_full(),
                    )
                    .child(dynamic(move || {
                        let read = store.settings();
                        let query = filter.get().trim().to_lowercase();
                        Some(untracked(move || body(store, read, &query)))
                    })),
            ),
        ),
    )
}

fn header(store: Store) -> View {
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .border_b_1()
        .border_color(colors::LINE)
        .child(when(
            move || store.is_compact() || !store.sidebar_open(),
            move || {
                ui::icon_btn(Glyph::Sidebar, "Conversations", Size::Md, move || {
                    store.open_sidebar()
                })
            },
        ))
        .child(
            text("Settings")
                .flex_1()
                .px_2()
                .role(AccessibilityRole::Header)
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::SEMIBOLD),
        )
        .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Md, move || {
            store.refresh_settings()
        }))
}

fn body(store: Store, read: SettingsRead, query: &str) -> View {
    match read {
        SettingsRead::NotYet | SettingsRead::Loading => {
            view().child(ui::working("Reading settings…"))
        }
        SettingsRead::Failed(reason) => view().child(
            ui::banner(
                if reason.contains("Unknown command") {
                    "This daemon is too old to show its settings here. Update aidaemon, then \
                     try again."
                        .to_string()
                } else {
                    reason
                },
                Tone::Danger,
            )
            .child(ui::btn(
                "Try again",
                Variant::Secondary,
                Size::Sm,
                move || store.refresh_settings(),
            )),
        ),
        SettingsRead::Ready(settings) => ready(store, *settings, query),
    }
}

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

fn ready(store: Store, settings: Settings, query: &str) -> View {
    let mut page = view().gap_3().child(muted(format!(
        "Saved to {}. Changes that widen what the assistant may do ask first.",
        settings.config_path
    )));
    if settings.reload_pending {
        page = page.child(
            ui::banner("Some saved changes take effect on a reload.", Tone::Info).child(ui::btn(
                "Reload",
                Variant::Primary,
                Size::Sm,
                move || apply(store, ApplyHow::Reload),
            )),
        );
    }
    if settings.restart_pending {
        page = page.child(
            ui::banner(
                "Some saved changes take effect after a restart.",
                Tone::Warning,
            )
            .child(ui::btn("Restart", Variant::Primary, Size::Sm, move || {
                apply(store, ApplyHow::Restart)
            })),
        );
    }
    page = page.child(dynamic(move || {
        let waiting = store.settings_waiting();
        (waiting > 0).then(|| {
            untracked(move || {
                ui::banner(
                    format!(
                        "{waiting} {} for approval.",
                        if waiting == 1 {
                            "change waits"
                        } else {
                            "changes wait"
                        }
                    ),
                    Tone::Info,
                )
            })
        })
    }));
    for notice in &settings.notices {
        page = page.child(ui::banner(notice.clone(), Tone::Neutral));
    }

    let filtering = !query.is_empty();
    for group in &settings.groups {
        let shown: Vec<Setting> = group
            .settings
            .iter()
            .filter(|s| {
                !filtering
                    || format!("{} {}", s.key, s.description)
                        .to_lowercase()
                        .contains(query)
            })
            .cloned()
            .collect();
        if shown.is_empty() {
            continue;
        }
        page = page.child(group_section(store, group, shown, filtering));
    }
    if !filtering {
        page = page
            .child(other_setting(store))
            .child(providers(store, &settings))
            .child(bots(store, &settings.bots))
            .child(mcp_servers(store, &settings.mcp_servers))
            .child(database(store, &settings))
            .child(apply_section(store));
    }
    page
}

fn apply(store: Store, how: ApplyHow) {
    store.change_settings(SettingsChange::Apply { how });
}

fn group_section(store: Store, group: &SettingGroup, shown: Vec<Setting>, open: bool) -> View {
    let set = shown
        .iter()
        .filter(|s| s.value.is_some() || s.secret_set)
        .count();
    let label = format!("{} · {} set of {}", group.name, set, shown.len());
    let open = signal(open);
    view()
        .gap_1()
        .child(ui::disclosure(label, open))
        .child(when(
            move || open.get(),
            move || {
                let mut list = view().gap_1();
                for setting in shown.clone() {
                    list = list.child(setting_row(store, setting));
                }
                list
            },
        ))
}

/// What a setting shows when closed: its value, "default", or for a secret
/// whether one is stored.
fn shown_value(setting: &Setting) -> String {
    match (&setting.value, setting.kind) {
        (_, SettingKind::Secret) if setting.secret_set => "stored".to_string(),
        (_, SettingKind::Secret) => "not set".to_string(),
        (Some(value), _) => value.clone(),
        (None, _) => match &setting.example {
            Some(example) => format!("default (example: {example})"),
            None => "default".to_string(),
        },
    }
}

fn applies_label(applies: Applies) -> Option<&'static str> {
    match applies {
        Applies::Reload => Some("on reload"),
        Applies::Restart => Some("on restart"),
        _ => None,
    }
}

fn setting_row(store: Store, setting: Setting) -> View {
    let editing = signal(false);
    let key = setting.key.clone();
    let mut meta = Vec::new();
    if let Some(label) = applies_label(setting.applies) {
        meta.push(label.to_string());
    }
    let summary = shown_value(&setting);
    let guarded = setting.guarded;
    let description = setting.description.clone();
    let editor_setting = setting.clone();
    ui::surface()
        .px_3()
        .py_2()
        .gap_1()
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .gap(px(2.0))
                        .child(ui::code(key.clone()).selectable(true))
                        .child(
                            text(summary)
                                .number_of_lines(2)
                                .selectable(true)
                                .text_size(px(ui::fit::small()))
                                .text_color(if setting.value.is_some() || setting.secret_set {
                                    colors::TEXT
                                } else {
                                    colors::FAINT
                                }),
                        ),
                )
                .child(when(move || guarded, || ui::badge("asks", Tone::Warning)))
                .child(
                    ui::btn(
                        if setting.kind == SettingKind::Secret {
                            "Set"
                        } else {
                            "Edit"
                        },
                        Variant::Ghost,
                        Size::Sm,
                        move || editing.update(|e| *e = !*e),
                    )
                    .accessibility_label(format!("Edit {key}")),
                ),
        )
        .child(when(move || !description.is_empty(), {
            let description = setting.description.clone();
            let meta = meta.join(" · ");
            move || {
                let line = if meta.is_empty() {
                    description.clone()
                } else {
                    format!("{description} ({meta})")
                };
                faint(line).number_of_lines(3)
            }
        }))
        .child(when(
            move || editing.get(),
            move || editor(store, editor_setting.clone(), editing),
        ))
}

fn editor(store: Store, setting: Setting, editing: Signal<bool>) -> View {
    let key = setting.key.clone();
    if setting.kind == SettingKind::Secret {
        let secret = signal(String::new());
        let k = key.clone();
        return view()
            .gap_2()
            .pt_1()
            .child(muted(
                "Stored in this computer's keychain. It is never shown again.",
            ))
            .child(
                ui::secret_field("Paste the secret", secret)
                    .label(format!("Secret for {key}"))
                    .w_full(),
            )
            .child(
                view()
                    .flex_row()
                    .gap_2()
                    .child(ui::btn("Store", Variant::Primary, Size::Sm, move || {
                        let value = secret.get_untracked();
                        if value.trim().is_empty() {
                            return;
                        }
                        store.change_settings(SettingsChange::Secret {
                            key: k.clone(),
                            value: SecretText(value),
                        });
                        secret.set(String::new());
                        editing.set(false);
                    }))
                    .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                        secret.set(String::new());
                        editing.set(false);
                    })),
            );
    }
    let start = setting
        .value
        .clone()
        .or_else(|| setting.example.clone())
        .unwrap_or_default();
    let draft = signal(start);
    let (save_key, reset_key) = (key.clone(), key.clone());
    let can_reset = setting.value.is_some();
    let mut actions = view().flex_row().flex_wrap().gap_2();
    if setting.kind == SettingKind::Bool {
        for (label, value) in [("Turn on", "true"), ("Turn off", "false")] {
            let k = key.clone();
            actions = actions.child(ui::btn(label, Variant::Secondary, Size::Sm, move || {
                store.change_settings(SettingsChange::Set {
                    key: k.clone(),
                    value: Some(value.to_string()),
                });
                editing.set(false);
            }));
        }
    } else {
        actions = actions.child(ui::btn("Save", Variant::Primary, Size::Sm, move || {
            let value = draft.get_untracked();
            if value.trim().is_empty() {
                return;
            }
            store.change_settings(SettingsChange::Set {
                key: save_key.clone(),
                value: Some(value.trim().to_string()),
            });
            editing.set(false);
        }));
    }
    if can_reset {
        actions = actions.child(ui::btn(
            "Use the default",
            Variant::Ghost,
            Size::Sm,
            move || {
                store.change_settings(SettingsChange::Set {
                    key: reset_key.clone(),
                    value: None,
                });
                editing.set(false);
            },
        ));
    }
    actions = actions.child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
        editing.set(false)
    }));
    let mut editor = view().gap_2().pt_1();
    if setting.kind != SettingKind::Bool {
        editor = editor
            .child(faint(
                "As TOML: text in quotes (\"gpt-x\"), numbers and true/false as they are, \
                 lists in brackets.",
            ))
            .child(
                ui::field("Value", draft)
                    .label(format!("Value of {key}"))
                    .w_full(),
            );
    }
    editor.child(actions)
}

/// Any setting by its key, for one the page does not list.
fn other_setting(store: Store) -> View {
    let key = signal(String::new());
    let value = signal(String::new());
    view()
        .gap_2()
        .child(section_label("Another setting"))
        .child(muted(
            "Set any key the daemon reads, by its dotted path (a list entry as name[0]).",
        ))
        .child(
            ui::field("Key, like memory.retention_days", key)
                .label("Setting key")
                .w_full(),
        )
        .child(
            ui::field("Value as TOML", value)
                .label("Setting value")
                .w_full(),
        )
        .child(
            view()
                .flex_row()
                .child(ui::btn("Save", Variant::Secondary, Size::Sm, move || {
                    let (k, v) = (key.get_untracked(), value.get_untracked());
                    if k.trim().is_empty() || v.trim().is_empty() {
                        return;
                    }
                    store.change_settings(SettingsChange::Set {
                        key: k.trim().to_string(),
                        value: Some(v.trim().to_string()),
                    });
                    key.set(String::new());
                    value.set(String::new());
                })),
        )
}

/// The failover entries' indexes, from their settings' keys.
fn failover_indexes(settings: &Settings) -> Vec<u32> {
    let mut indexes: Vec<u32> = settings
        .groups
        .iter()
        .flat_map(|g| &g.settings)
        .filter_map(|s| {
            let rest = s.key.strip_prefix("provider.fallbacks[")?;
            rest.split(']').next()?.parse().ok()
        })
        .collect();
    indexes.sort_unstable();
    indexes.dedup();
    indexes
}

fn providers(store: Store, settings: &Settings) -> View {
    let preset = signal(String::new());
    let base_url = signal(String::new());
    let primary = signal(String::new());
    let fast = signal(String::new());
    let smart = signal(String::new());
    let choice = move || ProviderChoice {
        preset: preset.get_untracked().trim().to_string(),
        base_url: base_url.get_untracked().trim().to_string(),
        primary_model: primary.get_untracked().trim().to_string(),
        fast_model: fast.get_untracked().trim().to_string(),
        smart_model: smart.get_untracked().trim().to_string(),
    };
    let mut failovers = view().flex_row().flex_wrap().gap_2();
    for index in failover_indexes(settings) {
        failovers = failovers.child(ui::btn(
            format!("Remove failover {}", index + 1),
            Variant::Outline,
            Size::Sm,
            move || store.change_settings(SettingsChange::RemoveFailover { index }),
        ));
    }
    view()
        .gap_2()
        .child(section_label("Providers"))
        .child(muted(
            "Pick a preset (openai, anthropic, openrouter, google, ollama, chatgpt…). Empty \
             fields take the preset's own. Store its key under provider.api_key (or the \
             failover's) above.",
        ))
        .child(
            ui::field("Preset", preset)
                .label("Provider preset")
                .w_full(),
        )
        .child(
            ui::field("Base URL (optional)", base_url)
                .label("Base URL")
                .w_full(),
        )
        .child(
            ui::field("Main model (optional)", primary)
                .label("Main model")
                .w_full(),
        )
        .child(
            ui::field("Fast model (optional)", fast)
                .label("Fast model")
                .w_full(),
        )
        .child(
            ui::field("Smart model (optional)", smart)
                .label("Smart model")
                .w_full(),
        )
        .child(
            view()
                .flex_row()
                .flex_wrap()
                .gap_2()
                .child(ui::btn(
                    "Make it the main provider",
                    Variant::Secondary,
                    Size::Sm,
                    move || {
                        let provider = choice();
                        if !provider.preset.is_empty() {
                            store.change_settings(SettingsChange::SwitchProvider { provider });
                        }
                    },
                ))
                .child(ui::btn(
                    "Add as failover",
                    Variant::Secondary,
                    Size::Sm,
                    move || {
                        let provider = choice();
                        if !provider.preset.is_empty() {
                            store.change_settings(SettingsChange::AddFailover { provider });
                        }
                    },
                ))
                .child(ui::btn(
                    "Undo the last config change",
                    Variant::Ghost,
                    Size::Sm,
                    move || store.change_settings(SettingsChange::RestoreBackup),
                )),
        )
        .child(failovers)
}

fn bots(store: Store, bots: &[BotInfo]) -> View {
    let mut section = view().gap_2().child(section_label("Connected bots"));
    if bots.is_empty() {
        return section.child(muted(
            "No bots were connected at runtime. Bots in config.toml are settings above.",
        ));
    }
    for bot in bots.iter().cloned() {
        let id = bot.id;
        let who = if bot.allowed_user_ids.is_empty() {
            "anyone allowed by its platform".to_string()
        } else {
            bot.allowed_user_ids.join(", ")
        };
        section = section.child(
            ui::surface()
                .flex_row()
                .items_center()
                .gap_3()
                .px_3()
                .py_2()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .gap(px(2.0))
                        .child(text(format!("{} bot {}", bot.platform, bot.id)))
                        .child(faint(format!("Talks to {who}"))),
                )
                .child(ui::btn("Remove", Variant::Danger, Size::Sm, move || {
                    store.change_settings(SettingsChange::RemoveBot { bot_id: id })
                })),
        );
    }
    section
}

fn mcp_servers(store: Store, servers: &[McpServerInfo]) -> View {
    let mut section = view().gap_2().child(section_label("MCP servers"));
    if servers.is_empty() {
        return section.child(muted("No MCP servers."));
    }
    for server in servers.iter().cloned() {
        let mcp = move |name: String, action: McpAction| {
            move || {
                store.change_settings(SettingsChange::Mcp {
                    name: name.clone(),
                    action,
                })
            }
        };
        let name = server.name.clone();
        let tools = if server.tools.is_empty() {
            "no tools running".to_string()
        } else {
            format!("{} tools", server.tools.len())
        };
        let mut actions = view().flex_row().gap_1();
        actions = if server.enabled {
            actions
                .child(ui::btn(
                    "Restart",
                    Variant::Ghost,
                    Size::Sm,
                    mcp(name.clone(), McpAction::Restart),
                ))
                .child(ui::btn(
                    "Disable",
                    Variant::Ghost,
                    Size::Sm,
                    mcp(name.clone(), McpAction::Disable),
                ))
        } else {
            actions.child(ui::btn(
                "Enable",
                Variant::Secondary,
                Size::Sm,
                mcp(name.clone(), McpAction::Enable),
            ))
        };
        if !server.from_config {
            actions = actions.child(ui::btn(
                "Remove",
                Variant::Danger,
                Size::Sm,
                mcp(name.clone(), McpAction::Remove),
            ));
        }
        section = section.child(
            ui::surface()
                .flex_row()
                .items_center()
                .gap_3()
                .px_3()
                .py_2()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .gap(px(2.0))
                        .child(text(server.name.clone()))
                        .child(ui::code(server.command.clone()).number_of_lines(1))
                        .child(faint(format!(
                            "{} · {tools}{}",
                            if server.enabled { "on" } else { "off" },
                            if server.from_config {
                                " · from config.toml"
                            } else {
                                ""
                            }
                        ))),
                )
                .child(actions),
        );
    }
    section
}

fn database(store: Store, settings: &Settings) -> View {
    let to = signal(String::new());
    let mut actions = view().flex_row().flex_wrap().gap_2().child(ui::btn(
        "Move at next restart",
        Variant::Secondary,
        Size::Sm,
        move || {
            let path = to.get_untracked();
            if path.trim().is_empty() {
                return;
            }
            store.change_settings(SettingsChange::Database {
                action: DatabaseAction::MoveTo {
                    path: path.trim().to_string(),
                },
            });
            to.set(String::new());
        },
    ));
    if settings.database_encrypted {
        actions = actions.child(ui::btn(
            "Rotate its key at next restart",
            Variant::Secondary,
            Size::Sm,
            move || {
                store.change_settings(SettingsChange::Database {
                    action: DatabaseAction::RotateKey,
                })
            },
        ));
    }
    if !settings.notices.is_empty() {
        actions = actions.child(ui::btn(
            "Cancel what is scheduled",
            Variant::Ghost,
            Size::Sm,
            move || {
                store.change_settings(SettingsChange::Database {
                    action: DatabaseAction::Cancel,
                })
            },
        ));
    }
    view()
        .gap_2()
        .child(section_label("Database"))
        .child(muted(format!(
            "{}{}",
            settings.database_path,
            if settings.database_encrypted {
                " · encrypted"
            } else {
                ""
            }
        )))
        .child(
            ui::field("New path for the database file", to)
                .label("New database path")
                .w_full(),
        )
        .child(actions)
}

fn apply_section(store: Store) -> View {
    view()
        .gap_2()
        .child(section_label("Apply"))
        .child(muted(
            "Reload puts provider and model changes into effect now. Restart applies the \
             rest; apps reconnect on their own.",
        ))
        .child(
            view()
                .flex_row()
                .gap_2()
                .child(ui::btn("Reload", Variant::Secondary, Size::Sm, move || {
                    apply(store, ApplyHow::Reload)
                }))
                .child(ui::btn("Restart", Variant::Outline, Size::Sm, move || {
                    apply(store, ApplyHow::Restart)
                })),
        )
}
