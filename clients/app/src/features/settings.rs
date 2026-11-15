//! The Settings page: every setting of the daemon, by config table, each
//! editable in place (secrets only stored, never shown); providers and
//! failover; runtime bots and MCP servers; the database; and Reload or
//! Restart for saved changes. Read on open, on Refresh and when a change
//! lands. Guarded changes wait for an approval, shown like any other.

use aidaemon_client_protocol::{
    Applies, ApplyHow, Billing, BotInfo, ConnectedProvider, CredentialState, DatabaseAction,
    McpAction, McpServerInfo, ProviderChoice, ProviderOption, SecretText, Setting, SettingGroup,
    SettingKind, Settings, SettingsChange,
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
    // The Providers tab's form, kept while the settings are read again.
    let form = ProviderForm::new();
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
                        Some(untracked(move || {
                            body(store, read, &query, shown, open, form)
                        }))
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
    form: ProviderForm,
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
        SettingsRead::Ready(settings) => ready(store, *settings, query, tab, open, form),
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
    form: ProviderForm,
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
        1 => return page.child(providers(store, &settings, form)),
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

/// What the Providers tab's add form holds.
#[derive(Clone, Copy)]
struct ProviderForm {
    preset: Signal<String>,
    key: Signal<String>,
    base_url: Signal<String>,
    primary: Signal<String>,
    fast: Signal<String>,
    smart: Signal<String>,
    more: Signal<bool>,
    redirect: Signal<String>,
    /// The connected provider whose key is being replaced, by its setting.
    replacing: Signal<Option<String>>,
    replacement: Signal<String>,
}

impl ProviderForm {
    fn new() -> Self {
        Self {
            preset: signal(String::new()),
            key: signal(String::new()),
            base_url: signal(String::new()),
            primary: signal(String::new()),
            fast: signal(String::new()),
            smart: signal(String::new()),
            more: signal(false),
            redirect: signal(String::new()),
            replacing: signal(None),
            replacement: signal(String::new()),
        }
    }

    /// Start over with `preset`.
    fn pick(self, preset: String) {
        self.preset.set(preset);
        for field in [self.key, self.base_url, self.primary, self.fast, self.smart] {
            field.set(String::new());
        }
    }

    fn choice(self) -> ProviderChoice {
        ProviderChoice {
            preset: self.preset.get_untracked().trim().to_string(),
            base_url: self.base_url.get_untracked().trim().to_string(),
            primary_model: self.primary.get_untracked().trim().to_string(),
            fast_model: self.fast.get_untracked().trim().to_string(),
            smart_model: self.smart.get_untracked().trim().to_string(),
        }
    }

    /// The key typed, if any.
    fn key(self) -> Option<SecretText> {
        let key = self.key.get_untracked();
        (!key.trim().is_empty()).then(|| SecretText(key.trim().to_string()))
    }
}

fn credential_words(credential: CredentialState) -> (&'static str, Tone) {
    match credential {
        CredentialState::KeyStored => ("Key stored in the keychain", Tone::Success),
        CredentialState::KeyInFile => ("Key written in config.toml", Tone::Warning),
        CredentialState::KeyMissing => ("No key: it cannot answer", Tone::Danger),
        CredentialState::SignedIn => ("Signed in to ChatGPT", Tone::Success),
        CredentialState::SignedOut => ("Not signed in to ChatGPT", Tone::Danger),
        CredentialState::NotNeeded => ("Needs no key", Tone::Neutral),
        CredentialState::Unknown => ("Unknown", Tone::Neutral),
    }
}

fn billing_words(billing: Billing) -> &'static str {
    match billing {
        Billing::Plan => "Uses your subscription",
        _ => "Billed per use by the provider",
    }
}

fn providers(store: Store, settings: &Settings, form: ProviderForm) -> View {
    let mut page = view().gap_4();
    page = page.child(in_use(store, settings, form));
    if let Some(sign_in) = settings.sign_in.clone() {
        page = page.child(sign_in_card(store, sign_in, form));
    }
    page.child(add_provider(store, settings, form))
        .child(view().flex_row().child(ui::btn(
            "Undo the last config change",
            Variant::Ghost,
            Size::Sm,
            move || store.change_settings(SettingsChange::RestoreBackup),
        )))
}

/// The providers the daemon uses, main first, each saying how it signs in.
fn in_use(store: Store, settings: &Settings, form: ProviderForm) -> View {
    let mut section = view().gap_2().child(section_label("In use"));
    // A daemon before protocol 1.8 sends no list.
    if settings.connected.is_empty() {
        return section.child(muted(
            "This daemon does not list its providers. Update aidaemon to see them here.",
        ));
    }
    section = section.child(muted(
        "The main provider answers first. When it fails, each failover is tried in turn.",
    ));
    let mut list = ui::surface().gap_0();
    let count = settings.connected.len();
    for (at, provider) in settings.connected.iter().cloned().enumerate() {
        list = list.child(connected_row(
            store,
            provider,
            form,
            settings.chatgpt_signed_in,
        ));
        if at + 1 < count {
            list = list.child(ui::separator());
        }
    }
    section.child(list)
}

fn connected_row(
    store: Store,
    provider: ConnectedProvider,
    form: ProviderForm,
    signed_in: bool,
) -> View {
    let role = match provider.failover {
        None => "Main".to_string(),
        Some(index) => format!("Failover {}", index + 1),
    };
    let (credential, tone) = credential_words(provider.credential);
    let models = if provider.models.is_empty() {
        "The provider's own models".to_string()
    } else {
        provider.models.join(", ")
    };
    let mut actions = view().flex_row().flex_wrap().gap_2();
    let setting = provider.key_setting.clone();
    if !setting.is_empty() {
        let replacing = setting.clone();
        actions = actions.child(
            ui::btn(
                if provider.credential == CredentialState::KeyMissing {
                    "Add key"
                } else {
                    "Replace key"
                },
                Variant::Outline,
                Size::Sm,
                move || {
                    form.replacement.set(String::new());
                    form.replacing.update(|r| {
                        *r = if r.as_deref() == Some(replacing.as_str()) {
                            None
                        } else {
                            Some(replacing.clone())
                        }
                    });
                },
            )
            .accessibility_label(format!("Key for {}", provider.name)),
        );
    }
    if provider.credential == CredentialState::SignedOut && !signed_in {
        actions = actions.child(ui::btn("Sign in", Variant::Primary, Size::Sm, move || {
            store.change_settings(SettingsChange::SignIn { paste: false })
        }));
    }
    if let Some(index) = provider.failover {
        let asked = format!("Remove {} ({role})?", provider.name);
        actions = actions.child(
            ui::btn("Remove", Variant::Ghost, Size::Sm, move || {
                ui::confirm(
                    asked.clone(),
                    "It is no longer tried when the providers before it fail.",
                    "Remove",
                    move || store.change_settings(SettingsChange::RemoveFailover { index }),
                )
            })
            .accessibility_label(format!("Remove {} as failover", provider.name)),
        );
    }
    let editing = {
        let setting = setting.clone();
        move || {
            form.replacing
                .with(|r| r.as_deref() == Some(setting.as_str()))
        }
    };
    view()
        .p_3()
        .gap_1p5()
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(
                    text(provider.name.clone())
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::MEDIUM),
                )
                .child(ui::badge(
                    role,
                    if provider.failover.is_none() {
                        Tone::Accent
                    } else {
                        Tone::Neutral
                    },
                )),
        )
        .child(muted(models).number_of_lines(2))
        .child(ui::status(credential, tone))
        .child(actions)
        .child(when(editing, move || {
            let setting = setting.clone();
            let store_it = move || {
                let value = form.replacement.get_untracked();
                if value.trim().is_empty() {
                    return;
                }
                store.change_settings(SettingsChange::Secret {
                    key: setting.clone(),
                    value: SecretText(value.trim().to_string()),
                });
                form.replacement.set(String::new());
                form.replacing.set(None);
            };
            view()
                .gap_2()
                .child(muted(
                    "Stored in the daemon computer's keychain. It is never shown again.",
                ))
                .child(
                    ui::secret_field("Paste the new key", form.replacement)
                        .label("New API key")
                        .w_full(),
                )
                .child(
                    view()
                        .flex_row()
                        .gap_2()
                        .child(
                            ui::btn("Store key", Variant::Primary, Size::Sm, store_it)
                                .disabled(move || form.replacement.with(|k| k.trim().is_empty())),
                        )
                        .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                            form.replacement.set(String::new());
                            form.replacing.set(None);
                        })),
                )
        }))
}

/// A ChatGPT sign-in under way.
fn sign_in_card(
    store: Store,
    sign_in: aidaemon_client_protocol::SignIn,
    form: ProviderForm,
) -> View {
    let url = sign_in.url.clone();
    let mut card = ui::surface()
        .p_4()
        .gap_2()
        .child(
            text("Sign in to ChatGPT")
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::MEDIUM),
        )
        .child(
            view()
                .flex_row()
                .flex_wrap()
                .gap_2()
                .child(ui::btn(
                    "Open the sign-in page",
                    Variant::Primary,
                    Size::Sm,
                    move || {
                        drop(open_url(url.clone()));
                    },
                ))
                .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                    form.redirect.set(String::new());
                    store.change_settings(SettingsChange::CancelSignIn);
                })),
        )
        .child(ui::code(sign_in.url.clone()).number_of_lines(2));
    if sign_in.paste {
        card = card
            .child(muted(
                "After you sign in, the browser lands on a page it cannot open (a localhost \
                 address). Copy that page's whole address and paste it here.",
            ))
            .child(
                ui::field("http://localhost:1455/auth/callback?code=…", form.redirect)
                    .label("Address the browser ended on")
                    .w_full(),
            )
            .child(
                view().flex_row().child(
                    ui::btn("Finish sign-in", Variant::Secondary, Size::Sm, move || {
                        let redirect = form.redirect.get_untracked();
                        if redirect.trim().is_empty() {
                            return;
                        }
                        store.change_settings(SettingsChange::FinishSignIn { redirect });
                        form.redirect.set(String::new());
                    })
                    .disabled(move || form.redirect.with(|r| r.trim().is_empty())),
                ),
            );
    } else {
        card = card.child(ui::working("Waiting for you to finish in the browser"));
    }
    card
}

/// Connect another provider: as the main one or a failover.
fn add_provider(store: Store, settings: &Settings, form: ProviderForm) -> View {
    let options = settings.providers.clone();
    let signed_in = settings.chatgpt_signed_in;
    let checked = settings.checked.clone();
    let mut section = view()
        .gap_2()
        .child(section_label("Add a provider"))
        .child(muted(
            "Pick a provider, sign in or paste its key, then make it the main provider or a \
             failover.",
        ));
    // A daemon before protocol 1.6 sends no list: the preset is typed.
    if options.is_empty() {
        return section
            .child(
                ui::field("Preset", form.preset)
                    .label("Provider preset")
                    .w_full(),
            )
            .child(submit_buttons(store, form, true));
    }
    let items: Vec<(String, String)> = options
        .iter()
        .map(|p| (p.preset.clone(), p.name.clone()))
        .collect();
    section = section.child(view().flex_row().child(ui::choice(
        "Provider",
        "Choose a provider",
        items,
        move || Some(form.preset.get()).filter(|p| !p.is_empty()),
        move |next: Option<String>| form.pick(next.unwrap_or_default()),
    )));
    section.child(dynamic(move || {
        let preset = form.preset.get();
        let option = options.iter().find(|o| o.preset == preset).cloned()?;
        let checked = checked.clone().filter(|c| c.preset == preset);
        Some(untracked(move || {
            provider_form(store, option, form, signed_in, checked)
        }))
    }))
}

fn provider_form(
    store: Store,
    option: ProviderOption,
    form: ProviderForm,
    signed_in: bool,
    checked: Option<aidaemon_client_protocol::CheckedKey>,
) -> View {
    let mut body = view().gap_2().child(muted(billing_words(option.billing)));
    let ready = if option.needs_key {
        if !option.key_url.is_empty() {
            let url = option.key_url.clone();
            body = body.child(view().flex_row().child(ui::btn(
                "Get a key",
                Variant::Ghost,
                Size::Sm,
                move || {
                    drop(open_url(url.clone()));
                },
            )));
        }
        if option.needs_base_url {
            body = body.child(
                ui::field("Address, like https://api.example.com/v1", form.base_url)
                    .label("Provider address")
                    .w_full(),
            );
        }
        let preset = option.preset.clone();
        let check = move || {
            let key = form.key.get_untracked();
            if key.trim().is_empty() {
                return;
            }
            store.change_settings(SettingsChange::CheckKey {
                provider: ProviderChoice {
                    preset: preset.clone(),
                    base_url: form.base_url.get_untracked().trim().to_string(),
                    ..ProviderChoice::default()
                },
                key: SecretText(key.trim().to_string()),
            });
        };
        body = body
            .child(
                ui::secret_field("Paste the API key", form.key)
                    .label("API key")
                    .w_full(),
            )
            .child(
                view().flex_row().child(
                    ui::btn("Check key", Variant::Secondary, Size::Sm, check)
                        .disabled(move || form.key.with(|k| k.trim().is_empty())),
                ),
            );
        match checked {
            Some(found) => {
                if form.primary.with_untracked(String::is_empty)
                    && let Some(suggested) = found.suggested.clone()
                {
                    form.primary.set(suggested);
                }
                let models: Vec<(String, String)> = found
                    .models
                    .iter()
                    .map(|m| (m.clone(), m.clone()))
                    .collect();
                body = body
                    .child(ui::status(
                        format!("The key works: {} models", found.models.len()),
                        Tone::Success,
                    ))
                    .child(view().flex_row().child(ui::choice(
                        "Main model",
                        "Choose a model",
                        models,
                        move || Some(form.primary.get()).filter(|m| !m.is_empty()),
                        move |next: Option<String>| form.primary.set(next.unwrap_or_default()),
                    )));
            }
            None => {
                body = body.child(faint(
                    "Check the key to pick from the models its account offers. Saving checks \
                     it too.",
                ));
            }
        }
        true
    } else {
        // The ChatGPT subscription: a sign-in, not a key.
        if signed_in {
            body = body.child(ui::status("Signed in to ChatGPT", Tone::Success));
        } else {
            body = body
                .child(ui::status("Not signed in yet", Tone::Warning))
                .child(view().flex_row().child(ui::btn(
                    "Sign in with ChatGPT",
                    Variant::Primary,
                    Size::Sm,
                    move || store.change_settings(SettingsChange::SignIn { paste: false }),
                )));
        }
        signed_in
    };
    let more_label = move || {
        if form.more.get() {
            "Fewer options"
        } else {
            "More options"
        }
    };
    let needs_base_url = option.needs_base_url;
    let default_model = option.model.clone();
    body = body
        .child(ui::disclosure(more_label, form.more))
        .child(when(
            move || form.more.get(),
            move || {
                let mut more = view().gap_2();
                if !needs_base_url {
                    more = more.child(
                        ui::field(
                            "Address (leave empty for the provider's own)",
                            form.base_url,
                        )
                        .label("Base URL")
                        .w_full(),
                    );
                }
                if !default_model.is_empty() {
                    more = more.child(faint(format!(
                        "Empty model fields take the provider's own (main: {default_model})."
                    )));
                }
                more.child(
                    ui::field("Main model (optional)", form.primary)
                        .label("Main model")
                        .w_full(),
                )
                .child(
                    ui::field("Fast model (optional)", form.fast)
                        .label("Fast model")
                        .w_full(),
                )
                .child(
                    ui::field("Smart model (optional)", form.smart)
                        .label("Smart model")
                        .w_full(),
                )
            },
        ));
    body.child(submit_buttons(store, form, ready))
}

fn submit_buttons(store: Store, form: ProviderForm, ready: bool) -> View {
    let send = move |failover: bool| {
        let provider = form.choice();
        if provider.preset.is_empty() {
            return;
        }
        let key = form.key();
        store.change_settings(if failover {
            SettingsChange::AddFailover { provider, key }
        } else {
            SettingsChange::SwitchProvider { provider, key }
        });
        // The key went to the daemon: it is not kept here.
        form.key.set(String::new());
    };
    view()
        .flex_row()
        .flex_wrap()
        .gap_2()
        .child(
            ui::btn(
                "Make it the main provider",
                Variant::Primary,
                Size::Sm,
                move || send(false),
            )
            .disabled(move || !ready),
        )
        .child(
            ui::btn("Add as failover", Variant::Secondary, Size::Sm, move || {
                send(true)
            })
            .disabled(move || !ready),
        )
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
