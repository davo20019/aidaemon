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

/// The page's tabs.
const TABS: [&str; 4] = ["Settings", "Providers", "Connections", "Database"];

pub fn page(store: Store) -> View {
    let filter = signal(String::new());
    let tab = signal(0_usize);
    // Open groups survive a refresh (a change landing rereads the page).
    let open = signal(Vec::<SharedString>::new());
    view().flex_1().min_w(px(0.0)).child(header(store)).child(
        scroll().flex_1().child(
            view().items_center().child(
                view()
                    .w_full()
                    .max_w(px(size::COLUMN))
                    .px_4()
                    .py_4()
                    .gap_3()
                    // Why the last change was refused, until dismissed: an
                    // error here is worth reading. What a change did is a
                    // toast.
                    .child(when(
                        move || store.settings_error().is_some(),
                        move || {
                            ui::banner(
                                move || store.settings_error().unwrap_or_default(),
                                Tone::Danger,
                            )
                            .child(ui::btn(
                                "Dismiss",
                                Variant::Ghost,
                                Size::Sm,
                                move || store.dismiss_settings_error(),
                            ))
                        },
                    ))
                    .child(ui::tab_bar(tab, &TABS))
                    .child(when(
                        move || tab.get() == 0,
                        move || {
                            ui::field("Filter settings", filter)
                                .label("Filter settings")
                                .w_full()
                        },
                    ))
                    .child(dynamic(move || {
                        let read = store.settings();
                        let query = filter.get().trim().to_lowercase();
                        let shown = tab.get();
                        Some(untracked(move || body(store, read, &query, shown, open)))
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

fn body(
    store: Store,
    read: SettingsRead,
    query: &str,
    tab: usize,
    open: Signal<Vec<SharedString>>,
) -> View {
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
        SettingsRead::Ready(settings) => ready(store, *settings, query, tab, open),
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

fn ready(
    store: Store,
    settings: Settings,
    query: &str,
    tab: usize,
    open: Signal<Vec<SharedString>>,
) -> View {
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
                restart(store)
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

    match tab {
        1 => return page.child(providers(store, &settings)),
        2 => {
            return page
                .child(bots(store, &settings.bots))
                .child(mcp_servers(store, &settings.mcp_servers));
        }
        3 => {
            return page
                .child(database(store, &settings))
                .child(apply_section(store));
        }
        _ => {}
    }
    let filtering = !query.is_empty();
    let mut groups = Vec::new();
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
        groups.push(group_section(store, group, shown));
    }
    // Filtering opens every group that matched.
    let open = if filtering {
        signal(
            groups
                .iter()
                .map(|(id, _, _)| SharedString::from(id.clone()))
                .collect(),
        )
    } else {
        open
    };
    page = page.child(ui::sections(open, groups));
    if !filtering {
        page = page.child(other_setting(store));
    }
    page
}

/// Restart, after asking: every app reconnects.
fn restart(store: Store) {
    ui::confirm(
        "Restart the daemon?",
        "Running work stops. Apps reconnect on their own.",
        "Restart",
        move || apply(store, ApplyHow::Restart),
    );
}

fn apply(store: Store, how: ApplyHow) {
    store.change_settings(SettingsChange::Apply { how });
}

/// A group as a section: its id, its heading and its rows.
fn group_section(
    store: Store,
    group: &SettingGroup,
    shown: Vec<Setting>,
) -> (String, String, View) {
    let set = shown
        .iter()
        .filter(|s| s.value.is_some() || s.secret_set)
        .count();
    let label = format!("{} · {} set of {}", group.name, set, shown.len());
    let mut list = view().gap_1().pb_2();
    for setting in shown {
        list = list.child(setting_row(store, setting));
    }
    (group.name.clone(), label, list)
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
    let is_bool = setting.kind == SettingKind::Bool;
    let on = setting.value.as_deref() == Some("true");
    let can_reset = setting.value.is_some();
    let (toggle_key, reset_key) = (setting.key.clone(), setting.key.clone());
    let kind = setting.kind;
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
                .child(when(is_bool && can_reset, move || {
                    let k = reset_key.clone();
                    ui::btn("Use the default", Variant::Ghost, Size::Sm, move || {
                        store.change_settings(SettingsChange::Set {
                            key: k.clone(),
                            value: None,
                        })
                    })
                }))
                .child(when(is_bool, move || {
                    let k = toggle_key.clone();
                    ui::toggle(
                        if on { "On" } else { "Off" },
                        move || on,
                        move |next: bool| {
                            store.change_settings(SettingsChange::Set {
                                key: k.clone(),
                                value: Some(next.to_string()),
                            })
                        },
                    )
                    .accessibility_label(format!("Turn {toggle_key} on or off"))
                }))
                .child(when(!is_bool, move || {
                    ui::btn(
                        if kind == SettingKind::Secret {
                            "Set"
                        } else {
                            "Edit"
                        },
                        Variant::Ghost,
                        Size::Sm,
                        move || editing.update(|e| *e = !*e),
                    )
                    .accessibility_label(format!("Edit {key}"))
                })),
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
    // A daemon before protocol 1.6 sends no list: the preset is typed.
    let options: Vec<(String, String)> = settings
        .providers
        .iter()
        .map(|p| (p.preset.clone(), p.name.clone()))
        .collect();
    let picker: View = if options.is_empty() {
        view().child(
            ui::field("Preset", preset)
                .label("Provider preset")
                .w_full(),
        )
    } else {
        view().flex_row().child(ui::choice(
            "Provider",
            "Choose a provider",
            options,
            move || Some(preset.get()).filter(|p| !p.is_empty()),
            move |next: Option<String>| preset.set(next.unwrap_or_default()),
        ))
    };
    view()
        .gap_2()
        .child(section_label("Providers"))
        .child(muted(
            "Choose a provider. Empty fields take its own. Store its key under \
             provider.api_key (or the failover's) on the Settings tab.",
        ))
        .child(picker)
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
    let mut rows = Vec::new();
    for bot in bots.iter().cloned() {
        let id = bot.id;
        let who = if bot.allowed_user_ids.is_empty() {
            "anyone allowed by its platform".to_string()
        } else {
            bot.allowed_user_ids.join(", ")
        };
        let name = format!("{} bot {}", bot.platform, bot.id);
        let asked = name.clone();
        rows.push(vec![
            view().child(text(name).number_of_lines(1)),
            view().child(muted(who).number_of_lines(2)),
            view().child(
                ui::btn("Remove", Variant::Ghost, Size::Sm, move || {
                    ui::confirm(
                        format!("Remove {asked}?"),
                        "It stops answering until it is connected again.",
                        "Remove",
                        move || store.change_settings(SettingsChange::RemoveBot { bot_id: id }),
                    )
                })
                .accessibility_label(format!("Remove bot {id}")),
            ),
        ]);
    }
    section = section.child(ui::data_table(
        "Connected bots",
        &["Bot", "Talks to", ""],
        rows,
    ));
    section
}

fn mcp_servers(store: Store, servers: &[McpServerInfo]) -> View {
    let section = view().gap_2().child(section_label("MCP servers"));
    if servers.is_empty() {
        return section.child(muted("No MCP servers."));
    }
    let mut rows = Vec::new();
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
            let remove = mcp(name.clone(), McpAction::Remove);
            let asked = name.clone();
            actions = actions.child(
                ui::btn("Remove", Variant::Ghost, Size::Sm, move || {
                    let remove = remove.clone();
                    ui::confirm(
                        format!("Remove {asked}?"),
                        "Its tools go away. Adding it back needs its command again.",
                        "Remove",
                        remove,
                    )
                })
                .accessibility_label(format!("Remove {name}")),
            );
        }
        rows.push(vec![
            view()
                .gap(px(2.0))
                .child(text(server.name.clone()).number_of_lines(1))
                .child(ui::code(server.command.clone()).number_of_lines(1)),
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(ui::status(
                    if server.enabled { "On" } else { "Off" },
                    if server.enabled {
                        Tone::Success
                    } else {
                        Tone::Neutral
                    },
                ))
                .child(faint(format!(
                    "{tools}{}",
                    if server.from_config {
                        " · config.toml"
                    } else {
                        ""
                    }
                ))),
            actions,
        ]);
    }
    section.child(ui::data_table(
        "MCP servers",
        &["Server", "State", ""],
        rows,
    ))
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
                    restart(store)
                })),
        )
}
