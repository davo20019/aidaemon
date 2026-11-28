//! The Settings page: every setting of the daemon, by config table, each
//! editable in place (secrets only stored, never shown); providers and
//! failover; runtime bots and MCP servers; the database; and Reload or
//! Restart for saved changes. Read on open, on Refresh and when a change
//! lands. Guarded changes wait for an approval, shown like any other.

use aidaemon_client_protocol::{
    Applies, ApplyHow, ApprovalDecision, Billing, BotInfo, ConnectedProvider, CredentialState,
    DatabaseAction, McpAction, McpServerInfo, ProviderChoice, ProviderOption, ProviderUsage,
    SecretText, Setting, SettingGroup, SettingKind, Settings, SettingsChange, UsageBalance,
};
use std::rc::Rc;

use nsh_core::prelude::*;

use crate::features::{chatgpt_sign_in, settings_general};
use crate::store::{AnswerSurface, SettingsRead, SettingsTab, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, section_label, size, type_scale};

/// The page's tabs.
const TABS: [&str; 4] = ["General", "Models", "Connections", "Advanced"];
/// Bots, paired devices and MCP servers.
const CONNECTIONS: usize = 2;
/// The tab with every setting, the database and Reload or Restart.
const ADVANCED: usize = 3;
/// The terminal's one page: the models and this device, no tabs. The
/// rest is the desktop and web app's.
const TERMINAL: usize = usize::MAX;

fn in_terminal() -> bool {
    nsh_core::element::idiom() == nsh_core::platform::Idiom::Terminal
}

pub fn page(store: Store) -> View {
    let filter = signal(String::new());
    let terminal = in_terminal();
    let tab = signal(if terminal { TERMINAL } else { 0_usize });
    // A view move (the agent showing the paired devices) picks the tab,
    // also while the page is open. The terminal's one page has no tabs.
    effect(move || {
        if let Some(asked) = store.take_settings_tab()
            && !terminal
        {
            tab.set(match asked {
                SettingsTab::Connections => CONNECTIONS,
            });
        }
    });
    // Open groups survive a refresh (a change landing rereads the page).
    let open = signal(Vec::<SharedString>::new());
    // The Providers tab's form, kept while the settings are read again.
    let form = ProviderForm::new();
    form.follow(store);
    offer_restart_when_needed(store);
    ask_approvals_in_place(store);
    view()
        .flex_1()
        .min_w(px(0.0))
        .child(header(store))
        .child(page_body(store, filter, tab, open, form, terminal))
        .child(pending_bar(store))
}

/// What a saved change still needs, if anything: (reload, restart).
fn pending(store: Store) -> (bool, bool) {
    match store.settings() {
        SettingsRead::Ready(s) => (s.reload_pending, s.restart_pending),
        _ => (false, false),
    }
}

/// When a change lands that only a restart puts into effect, ask right
/// away: whoever flipped the switch should not have to find a button.
/// Only a change seen while the page is open asks; one already waiting
/// when it opens is left to the bar.
/// A change the daemon asks this device to approve is asked here, where it
/// was made, in the same alert as the app's own questions: the answer goes
/// to the daemon's approval, so its card in the conversation list is
/// answered too and nothing is asked twice. Each is asked once; one
/// dismissed stays waiting (the banner above counts it).
fn ask_approvals_in_place(store: Store) {
    let asked = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
    effect(move || {
        for approval in store.settings_asking_here() {
            if asked.borrow().contains(&approval.id) {
                continue;
            }
            asked.borrow_mut().push(approval.id.clone());
            let mut detail = approval.summary.clone();
            for warning in &approval.warnings {
                detail.push_str("\n\n");
                detail.push_str(warning);
            }
            let id = approval.id.clone();
            untracked(move || ask_approval(store, id, detail));
        }
    });
}

/// One approval's dialog. Allow pressed before the dialog has been up for
/// the arming delay (it may have opened under a click or key aimed at
/// something else) answers nothing and asks again.
fn ask_approval(store: Store, id: String, detail: String) {
    store.answer_shown(AnswerSurface::Dialog, &id);
    let again = detail.clone();
    ui::yes_or_no("Allow this change?", detail, "Allow", "Deny", move |yes| {
        let decision = if yes {
            ApprovalDecision::ApproveOnce
        } else {
            ApprovalDecision::Deny
        };
        if !store.resolve_clicked(AnswerSurface::Dialog, &id, decision) {
            ask_approval(store, id, again);
        }
    })
}

fn offer_restart_when_needed(store: Store) {
    let before = std::cell::Cell::new(None::<bool>);
    effect(move || {
        let SettingsRead::Ready(settings) = store.settings() else {
            return;
        };
        let now = settings.restart_pending;
        if before.replace(Some(now)) == Some(false) && now {
            untracked(move || {
                ui::now_or_later(
                    "Restart to finish?",
                    "This change takes effect after aidaemon restarts, which takes a few \
                     seconds. Work in progress stops; apps reconnect on their own.",
                    "Restart now",
                    "Later",
                    move || apply(store, ApplyHow::Restart),
                )
            });
        }
    });
}

/// Pinned under the page, so it shows however far down the change was.
fn pending_bar(store: Store) -> View {
    view().child(dynamic(move || {
        let (reload, needs_restart) = pending(store);
        (reload || needs_restart).then(|| {
            untracked(move || {
                let bar = if needs_restart {
                    ui::banner("Some changes need a restart to take effect.", Tone::Warning).child(
                        ui::btn("Restart now", Variant::Primary, Size::Sm, move || {
                            restart(store)
                        }),
                    )
                } else {
                    ui::banner("Some changes need a reload to take effect.", Tone::Info).child(
                        ui::btn("Reload", Variant::Primary, Size::Sm, move || {
                            apply(store, ApplyHow::Reload)
                        }),
                    )
                };
                view()
                    .items_center()
                    .px_4()
                    .pb_2()
                    .border_t_1()
                    .border_color(colors::LINE)
                    .child(bar)
            })
        })
    }))
}

fn page_body(
    store: Store,
    filter: Signal<String>,
    tab: Signal<usize>,
    open: Signal<Vec<SharedString>>,
    form: ProviderForm,
    terminal: bool,
) -> View {
    view().flex_1().min_h(px(0.0)).child(ui::refreshable(
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
                    .children((!terminal).then(|| ui::tab_bar(tab, &TABS)))
                    .child(when(
                        move || tab.get() == ADVANCED,
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
        move || store.pull_refreshing(),
        move || store.pull_refresh(),
    ))
}

fn header(store: Store) -> View {
    crate::features::config_nav::header(store)
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
                if store.settings_unsupported() {
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
    let mut page = view().gap_3();
    if tab == ADVANCED {
        page = page.child(muted(format!(
            "Saved to {}. Changes that widen what the assistant may do ask first.",
            settings.config_path
        )));
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
        TERMINAL => {
            return page
                .child(providers(store, &settings, form))
                .child(settings_general::this_device(store))
                .child(faint(
                    "Bots, MCP servers and every other setting are in the desktop and web app.",
                ));
        }
        0 => return page.child(settings_general::general(store, &settings)),
        // Grouped by who works: AIdaemon itself, the coding agents it
        // hands work to, then what runs behind the scenes.
        1 => {
            let mut page = page.child(providers(store, &settings, form));
            if settings.coding_agents.iter().any(|agent| agent.takes_model) {
                page = page.child(settings_general::agent_models(
                    store,
                    &settings.coding_agents,
                ));
            }
            return page.child(settings_general::decisions(store, &settings));
        }
        CONNECTIONS => {
            return page
                .child(bots(store, &settings.bots))
                .child(eink_access(store, &settings))
                .child(devices(store))
                .child(mcp_servers(store, &settings.mcp_servers));
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
        page = page
            .child(other_setting(
                store,
                settings
                    .groups
                    .iter()
                    .flat_map(|g| g.settings.iter().map(|s| s.key.clone()))
                    .collect(),
            ))
            .child(database(store, &settings))
            .child(apply_section(store))
            .child(
                view().flex_row().child(
                    ui::btn(
                        "Undo the last config change",
                        Variant::Ghost,
                        Size::Sm,
                        move || store.change_settings(SettingsChange::RestoreBackup),
                    )
                    .disabled(move || store.settings_busy()),
                ),
            );
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
    let title = settings_general::group_title(&group.name);
    let label = if set == 0 {
        title
    } else {
        format!("{title} · {set} changed")
    };
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
                        .child(
                            text(settings_general::setting_title(&key))
                                .number_of_lines(1)
                                .text_size(px(ui::fit::body())),
                        )
                        .child(faint(key.clone()).selectable(true))
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
            .child(ui::code_editor(
                draft,
                "toml",
                format!("Value of {key}"),
                12,
            ));
    }
    editor.child(actions)
}

/// Any setting by its key, for one the page does not list.
fn other_setting(store: Store, keys: Vec<String>) -> View {
    let key = signal(String::new());
    let value = signal(String::new());
    view()
        .gap_2()
        .child(section_label("Another setting"))
        .child(muted(
            "Set any key the daemon reads, by its dotted path (a list entry as name[0]).",
        ))
        .child(ui::suggest_field(
            "Setting key",
            "Key, like memory.retention_days",
            key,
            keys,
        ))
        .child(ui::code_editor(value, "toml", "Setting value", 8))
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

/// Where the Providers tab's form puts the provider it connects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Adding {
    Main,
    Failover,
}

/// What the Providers tab's form holds: kept while the settings are read
/// again, so a refresh does not wipe what was typed.
#[derive(Clone, Copy)]
struct ProviderForm {
    /// The form is open, and for which place.
    adding: Signal<Option<Adding>>,
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
    /// The connected row whose models per job are being chosen.
    modeling: Signal<Option<usize>>,
    hard: Signal<String>,
    quick: Signal<String>,
    /// The form's change was sent: it closes once the daemon takes it.
    sent: Signal<bool>,
    /// "Sign in" was pressed here (holding the sign-in page shown then, if
    /// any): the new page opens once the daemon names it.
    open_sign_in: Signal<bool>,
    /// Where the form opened: under connected row `n`, or (`BELOW`) under
    /// the list.
    at: Signal<usize>,
}

/// The form's place under the list of providers in use.
const BELOW: usize = usize::MAX;

impl ProviderForm {
    fn new() -> Self {
        Self {
            adding: signal(None),
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
            modeling: signal(None),
            hard: signal(String::new()),
            quick: signal(String::new()),
            sent: signal(false),
            open_sign_in: signal(false),
            at: signal(BELOW),
        }
    }

    /// Close once the daemon has answered the change sent from the form,
    /// unless it refused it (the banner says why and the form stays).
    fn follow(self, store: Store) {
        effect(move || {
            if !self.sent.get() || store.settings_busy() {
                return;
            }
            self.sent.set(false);
            if store.settings_error().is_none() {
                self.close();
            }
        });
        // The device that pressed "Sign in" opens the page by itself once
        // the daemon has answered: a new sign-in's, or the one under way
        // that it joined. The card's button stays for opening it again.
        effect(move || {
            if !self.open_sign_in.get() || store.settings_busy() {
                return;
            }
            if store.settings_error().is_some() {
                self.open_sign_in.set(false);
            } else if let Some(url) = sign_in_url(store) {
                self.open_sign_in.set(false);
                crate::features::links::open_https(url);
            }
        });
    }

    /// Start a ChatGPT sign-in and open its page once the daemon names it.
    fn sign_in(self, store: Store) {
        self.open_sign_in.set(true);
        store.change_settings(SettingsChange::SignIn { paste: false });
    }

    /// Start over with `preset`.
    fn pick(self, preset: String) {
        self.preset.set(preset);
        self.more.set(false);
        for field in [self.key, self.base_url, self.primary, self.fast, self.smart] {
            field.set(String::new());
        }
    }

    /// Open the form for `adding`, with `preset` picked (or none).
    fn open(self, adding: Adding, preset: String, at: usize) {
        self.at.set(at);
        self.pick(preset);
        self.replacing.set(None);
        self.adding.set(Some(adding));
    }

    fn close(self) {
        self.pick(String::new());
        self.adding.set(None);
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

/// The page of the sign-in under way, if any.
fn sign_in_url(store: Store) -> Option<String> {
    match store.settings() {
        SettingsRead::Ready(s) => s.sign_in.map(|s| s.url),
        _ => None,
    }
}

fn credential_words(credential: CredentialState) -> (&'static str, Tone) {
    match credential {
        CredentialState::KeyStored => ("Key in the keychain", Tone::Success),
        CredentialState::KeyInFile => ("Key in config.toml", Tone::Warning),
        CredentialState::KeyMissing => ("No key: it cannot answer", Tone::Danger),
        CredentialState::SignedIn => ("Signed in to ChatGPT", Tone::Success),
        CredentialState::SignedOut => ("Not signed in: it cannot answer", Tone::Danger),
        CredentialState::NotNeeded => ("Needs no key", Tone::Neutral),
        CredentialState::Unknown => ("Unknown", Tone::Neutral),
    }
}

fn billing_words(billing: Billing) -> &'static str {
    match billing {
        Billing::Plan => "Uses your subscription",
        Billing::Local => "Runs on your own machine",
        _ => "Billed per use",
    }
}

fn providers(store: Store, settings: &Settings, form: ProviderForm) -> View {
    let mut page = view().gap_4();
    if let Some(sign_in) = settings.sign_in.clone() {
        page = page.child(sign_in_card(store, sign_in, form));
    }
    let known = Known {
        options: settings.providers.clone(),
        signed_in: settings.chatgpt_signed_in,
        checked: settings.checked.clone(),
        main_name: settings
            .connected
            .first()
            .map(|p| p.name.clone())
            .unwrap_or_default(),
        keyed: settings
            .connected
            .iter()
            .filter(|p| p.failover.is_some() && p.credential == CredentialState::KeyStored)
            .map(|p| p.preset.clone())
            .collect(),
        connected: settings
            .connected
            .iter()
            .filter(|p| !p.preset.is_empty())
            .map(|p| p.preset.clone())
            .collect(),
    };
    page.child(in_use(store, settings, form, known))
}

/// The add form, when it is open at `at`.
fn form_at(store: Store, form: ProviderForm, known: Known, at: usize) -> View {
    view().child(dynamic(move || {
        let adding = form.adding.get()?;
        (form.at.get() == at).then(|| {
            let known = known.clone();
            untracked(move || add_form(store, adding, form, known))
        })
    }))
}

/// The providers the daemon uses, main first, each saying how it signs in.
fn in_use(store: Store, settings: &Settings, form: ProviderForm, known: Known) -> View {
    let mut section = view().gap_2().child(section_label("AIdaemon"));
    // A daemon before protocol 1.8 sends no list.
    if settings.connected.is_empty() {
        return section
            .child(muted(
                "This daemon does not list its providers. Update aidaemon to see them here.",
            ))
            .child(view().flex_row().gap_2().child(ui::btn(
                "Change the main provider",
                Variant::Secondary,
                Size::Sm,
                move || form.open(Adding::Main, String::new(), BELOW),
            )))
            .child(form_at(store, form, known, BELOW));
    }
    section = section.child(muted(
        "The coordinator: it answers you, and hands coding work to the coding agents \
         below. It uses a strong model for hard work and a fast one for quick jobs; a \
         conversation can pick another of these models from the composer.",
    ));
    let count = settings.connected.len();
    let row = |at: usize, provider: ConnectedProvider| {
        view()
            .child(connected_row(
                store,
                provider,
                form,
                settings.chatgpt_signed_in,
                settings.checked.clone(),
                at,
                count - 1,
            ))
            .child(view().px_3().child(form_at(store, form, known.clone(), at)))
    };
    // The main provider, then its backups under a heading of their own:
    // they answer only when it can't.
    let mut main = ui::surface().gap_0();
    let mut backups = ui::surface().gap_0();
    for (at, provider) in settings.connected.iter().cloned().enumerate() {
        if at == 0 {
            main = main.child(row(at, provider));
            continue;
        }
        if at > 1 {
            backups = backups.child(ui::separator());
        }
        backups = backups.child(row(at, provider));
    }
    section = section.child(main);
    if count > 1 {
        section = section
            .child(
                view()
                    .gap_1()
                    .pt_2()
                    .child(
                        text("If the main provider can't answer")
                            .text_size(px(ui::fit::small()))
                            .font_weight(FontWeight::MEDIUM),
                    )
                    .child(faint(
                        "Each backup is tried in order, with the same kind of model (hard \
                         work stays on a strong model).",
                    )),
            )
            .child(backups);
    }
    section
        .child(when(
            move || form.adding.get() != Some(Adding::Failover),
            move || {
                view().flex_row().child(
                    ui::btn("Add a backup", Variant::Secondary, Size::Sm, move || {
                        form.open(Adding::Failover, String::new(), BELOW)
                    })
                    .disabled(move || store.settings_busy()),
                )
            },
        ))
        .child(form_at(store, form, known, BELOW))
}

/// Which model does which job: one key reaches every model of its
/// provider, so the card names jobs, not a list to add to. Each is a
/// label and a model, shown as a row so a long id never breaks mid-name.
fn models_by_job(provider: &ConnectedProvider) -> Vec<(&'static str, String)> {
    let (hard, quick) = (provider.hard_model.as_str(), provider.quick_model.as_str());
    match (hard.is_empty(), quick.is_empty()) {
        (false, false) if hard == quick => vec![("All work", hard.to_string())],
        (false, false) => vec![
            ("Hard work", hard.to_string()),
            ("Quick jobs", quick.to_string()),
        ],
        (false, true) => vec![("Hard work", hard.to_string())],
        (true, false) => vec![("Quick jobs", quick.to_string())],
        // A daemon before protocol 1.20, or an address that decides.
        (true, true) if provider.models.is_empty() => {
            vec![("Models", "The provider's own".to_string())]
        }
        (true, true) => vec![("Models", provider.models.join(", "))],
    }
}

/// A label and its value on one line: the label in a fixed column.
fn labeled_row(label: &'static str, value: String) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(
            text(label)
                .w(px(84.0))
                .flex_shrink_0()
                .text_size(px(ui::fit::small()))
                .text_color(colors::MUTED),
        )
        .child(
            text(value)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .text_size(px(ui::fit::small()))
                .text_color(colors::TEXT),
        )
}

/// A usage window's name: "5-hour limit", "Weekly limit".
fn window_name(minutes: u32) -> String {
    match minutes {
        10_080 => "Weekly limit".to_string(),
        1_440 => "Daily limit".to_string(),
        m if m >= 1_440 && m % 1_440 == 0 => format!("{}-day limit", m / 1_440),
        m if m >= 60 && m % 60 == 0 => format!("{}-hour limit", m / 60),
        m => format!("{m}-minute limit"),
    }
}

/// When a window starts over, in the viewer's time: "resets 1:40 PM",
/// "resets tomorrow 9:05 AM", "resets Monday 9:05 AM".
fn resets_words(at_ms: u64, now_ms: u64) -> String {
    let day = ui::time::day_label(at_ms, now_ms);
    let clock = ui::time::clock(at_ms);
    match day.as_str() {
        "Today" => format!("resets {clock}"),
        "Tomorrow" => format!("resets tomorrow {clock}"),
        _ => format!("resets {day} {clock}"),
    }
}

/// Dollars from cents: "$6.50".
fn dollars(cents: u64) -> String {
    format!("${}.{:02}", cents / 100, cents % 100)
}

/// What a balance shows: the words, and how full its meter is (none
/// without a cap).
fn balance_words(balance: &UsageBalance) -> (String, Option<f32>) {
    match (balance.limit_cents, balance.remaining_cents) {
        (Some(limit), Some(left)) if limit > 0 => (
            format!("{} left of {}", dollars(left), dollars(limit)),
            Some(limit.saturating_sub(left) as f32 / limit as f32),
        ),
        _ => (
            format!("{} spent · no cap set", dollars(balance.used_cents)),
            None,
        ),
    }
}

/// A plan's name as people write it: "plus" → "Plus".
fn plan_name(plan: &str) -> String {
    let mut chars = plan.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// One usage line: its name, a figure on the right, and a meter.
fn usage_line(name: String, figure: String, share: Option<f32>) -> View {
    let mut line = view().gap_1().child(
        view()
            .flex_row()
            .items_center()
            .gap_2()
            .child(
                text(name.clone())
                    .flex_1()
                    .min_w(px(0.0))
                    .number_of_lines(1)
                    .text_size(px(ui::fit::small()))
                    .text_color(colors::TEXT),
            )
            .child(
                text(figure)
                    .number_of_lines(1)
                    .text_size(px(ui::fit::small()))
                    .text_color(colors::MUTED),
            ),
    );
    if let Some(share) = share {
        line = line.child(ui::meter(name, share));
    }
    line
}

/// What the provider's account has used and has left.
fn usage_view(usage: &ProviderUsage) -> Option<View> {
    if usage.windows.is_empty() && usage.balance.is_none() {
        return None;
    }
    let now = crate::store::now_ms();
    let mut block = view().gap_2().pt_1();
    for window in &usage.windows {
        let mut figure = format!("{}% used", window.used_percent);
        if let Some(at) = window.resets_at_ms {
            figure = format!("{figure} · {}", resets_words(at, now));
        }
        block = block.child(usage_line(
            window_name(window.minutes),
            figure,
            Some(f32::from(window.used_percent) / 100.0),
        ));
    }
    if let Some(balance) = &usage.balance {
        let (figure, share) = balance_words(balance);
        block = block.child(usage_line("Balance".to_string(), figure, share));
    }
    Some(block)
}

/// A TOML string holding `id`.
fn toml_string(id: &str) -> String {
    format!("\"{}\"", id.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Pick the model a provider uses for hard work and for quick jobs: any
/// id may be typed (the provider decides whether it works), with the
/// models it is known to have suggested. Empty takes the catalog's.
/// A model id typed in full, or one of `known` tapped in: the field
/// shows what is chosen, and keeps any id typed.
fn model_pick(job: &'static str, value: Signal<String>, known: &[String]) -> View {
    let mut picks = view().flex_row().flex_wrap().gap_1();
    for model in known.iter().take(12) {
        let model = model.clone();
        let label = format!("Use {model} for {}", job.to_lowercase());
        picks = picks.child(
            ui::btn(model.clone(), Variant::Ghost, Size::Sm, move || {
                value.set(model.clone())
            })
            .accessibility_label(label),
        );
    }
    ui::labeled(
        job,
        view()
            .gap_1()
            .child(
                ui::field("Model id", value)
                    .label(format!("{job} model"))
                    .w_full(),
            )
            .child(picks),
    )
}

fn job_models_editor(
    store: Store,
    provider: &ConnectedProvider,
    form: ProviderForm,
    checked: Option<aidaemon_client_protocol::CheckedKey>,
    at: usize,
) -> View {
    let mut known: Vec<String> = Vec::new();
    let account = checked
        .filter(|c| c.preset == provider.preset && !provider.preset.is_empty())
        .map(|c| c.models)
        .unwrap_or_default();
    for model in [provider.hard_model.clone(), provider.quick_model.clone()]
        .into_iter()
        .chain(provider.models.iter().cloned())
        .chain(provider.offered_models.iter().cloned())
        .chain(account)
    {
        if !model.is_empty() && !known.contains(&model) {
            known.push(model);
        }
    }
    let prefix = match provider.failover {
        None => "provider.models".to_string(),
        Some(index) => format!("provider.fallbacks[{index}].models"),
    };
    let current = (provider.hard_model.clone(), provider.quick_model.clone());
    view().child(when(
        move || form.modeling.get() == Some(at),
        move || {
            let (prefix, current) = (prefix.clone(), current.clone());
            let save = {
                let (prefix, current) = (prefix.clone(), current.clone());
                move || {
                    for (job, typed, was) in [
                        ("smart", form.hard.get_untracked(), &current.0),
                        ("fast", form.quick.get_untracked(), &current.1),
                    ] {
                        let typed = typed.trim();
                        if typed == was.as_str() {
                            continue;
                        }
                        store.change_settings(SettingsChange::Set {
                            key: format!("{prefix}.{job}"),
                            value: (!typed.is_empty()).then(|| toml_string(typed)),
                        });
                    }
                    form.modeling.set(None);
                }
            };
            let defaults = move || {
                for job in ["smart", "fast"] {
                    store.change_settings(SettingsChange::Set {
                        key: format!("{prefix}.{job}"),
                        value: None,
                    });
                }
                form.modeling.set(None);
            };
            view()
                .gap_2()
                .child(model_pick("Hard work", form.hard, &known))
                .child(model_pick("Quick jobs", form.quick, &known))
                .child(faint(
                    "Any model id this provider offers. Empty uses its usual one. \
                     It takes effect on a reload.",
                ))
                .child(
                    view()
                        .flex_row()
                        .flex_wrap()
                        .gap_2()
                        .child(
                            ui::btn("Save models", Variant::Primary, Size::Sm, save)
                                .disabled(move || store.settings_busy()),
                        )
                        .child(
                            ui::btn("Use defaults", Variant::Ghost, Size::Sm, defaults)
                                .disabled(move || store.settings_busy()),
                        )
                        .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                            form.modeling.set(None)
                        })),
                )
        },
    ))
}

fn connected_row(
    store: Store,
    provider: ConnectedProvider,
    form: ProviderForm,
    signed_in: bool,
    checked: Option<aidaemon_client_protocol::CheckedKey>,
    at: usize,
    // How many failovers there are: the last one cannot move down.
    failovers: usize,
) -> View {
    let main = provider.failover.is_none();
    let role = match provider.failover {
        None => "Main".to_string(),
        Some(index) => format!("Backup {}", index + 1),
    };
    let (credential, tone) = credential_words(provider.credential);
    let credential: String = match provider.usage.as_ref().map(|u| u.plan.as_str()) {
        Some(plan) if !plan.is_empty() => format!("{credential} · {} plan", plan_name(plan)),
        _ => credential.to_string(),
    };
    let models_rows = models_by_job(&provider);
    let usage = provider.usage.as_ref().and_then(usage_view);
    let mut actions = view().flex_row().flex_wrap().gap_2();
    if main {
        actions = actions.child(
            ui::btn("Change provider", Variant::Outline, Size::Sm, move || {
                form.open(Adding::Main, String::new(), at)
            })
            .disabled(move || store.settings_busy())
            .accessibility_label("Change the main provider"),
        );
    } else if !provider.preset.is_empty() {
        // A failover that can already answer is promoted as it is: its key
        // (or sign-in) and models carry over, and the daemon's approval card
        // is the one question asked. Only one that cannot (no key here)
        // opens the form to get one.
        let ready = matches!(
            provider.credential,
            CredentialState::KeyStored | CredentialState::NotNeeded | CredentialState::SignedIn
        );
        let choice = ProviderChoice {
            preset: provider.preset.clone(),
            base_url: provider.base_url.clone(),
            primary_model: provider.hard_model.clone(),
            fast_model: provider.quick_model.clone(),
            smart_model: provider.hard_model.clone(),
        };
        actions = actions.child(
            ui::btn("Make main", Variant::Outline, Size::Sm, move || {
                if !ready {
                    form.open(Adding::Main, choice.preset.clone(), at);
                    return;
                }
                store.change_settings(SettingsChange::SwitchProvider {
                    provider: choice.clone(),
                    key: None,
                });
            })
            .disabled(move || store.settings_busy())
            .accessibility_label(format!("Make {} the main provider", provider.name)),
        );
    }
    if provider.credential == CredentialState::SignedOut && !signed_in {
        actions = actions.child(
            ui::btn("Sign in", Variant::Primary, Size::Sm, move || {
                form.sign_in(store)
            })
            .disabled(move || store.settings_busy()),
        );
    }
    {
        let (hard, quick) = (provider.hard_model.clone(), provider.quick_model.clone());
        actions = actions.child(
            ui::btn("Change models", Variant::Outline, Size::Sm, move || {
                if form.modeling.get_untracked() == Some(at) {
                    form.modeling.set(None);
                    return;
                }
                form.hard.set(hard.clone());
                form.quick.set(quick.clone());
                form.modeling.set(Some(at));
            })
            .disabled(move || store.settings_busy())
            .accessibility_label(format!("Change the models {} uses", provider.name)),
        );
    }
    let job_models = job_models_editor(store, &provider, form, checked, at);
    let setting = provider.key_setting.clone();
    if !setting.is_empty() {
        let replacing = setting.clone();
        let missing = provider.credential == CredentialState::KeyMissing;
        let in_file = provider.credential == CredentialState::KeyInFile;
        actions = actions.child(
            ui::btn(
                if missing {
                    "Add key"
                } else if in_file {
                    "Move to keychain"
                } else {
                    "Replace key"
                },
                if missing {
                    Variant::Primary
                } else {
                    Variant::Outline
                },
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
    // Failovers are tried in order: each moves up or down a place.
    let order: Option<View> = provider.failover.map(|index| {
        let mut moves = view().flex_row().gap_1();
        let name = provider.name.clone();
        // What this page shows there: the daemon refuses if it changed.
        let up_name = name.clone();
        if index > 0 {
            moves = moves.child(
                ui::icon_btn(Glyph::Up, format!("Move {name} up"), Size::Sm, move || {
                    store.change_settings(SettingsChange::MoveFailover {
                        index,
                        to: index - 1,
                        expect: up_name.clone(),
                    })
                })
                .disabled(move || store.settings_busy()),
            );
        }
        let name = provider.name.clone();
        let down_name = name.clone();
        if (index as usize) + 1 < failovers {
            moves = moves.child(
                ui::icon_btn(
                    Glyph::Down,
                    format!("Move {name} down"),
                    Size::Sm,
                    move || {
                        store.change_settings(SettingsChange::MoveFailover {
                            index,
                            to: index + 1,
                            expect: down_name.clone(),
                        })
                    },
                )
                .disabled(move || store.settings_busy()),
            );
        }
        moves
    });
    // Removing a failover sits by its name, away from the everyday actions.
    let remove: Option<View> = provider.failover.map(|index| {
        let asked = format!("Remove {} ({role})?", provider.name);
        let removed_name = provider.name.clone();
        view().child(ui::icon_btn(
            Glyph::Close,
            "Remove this backup",
            Size::Sm,
            move || {
                ui::confirm(
                    asked.clone(),
                    "It is no longer tried when the providers before it fail.",
                    "Remove",
                    {
                        let removed_name = removed_name.clone();
                        move || {
                            store.change_settings(SettingsChange::RemoveFailover {
                                index,
                                expect: removed_name.clone(),
                            })
                        }
                    },
                )
            },
        ))
    });
    let editing = {
        let setting = setting.clone();
        move || {
            form.replacing
                .with(|r| r.as_deref() == Some(setting.as_str()))
        }
    };
    let mut models = view().gap_1();
    for (label, model) in models_rows {
        models = models.child(labeled_row(label, model));
    }
    view()
        .p_3()
        .gap_2()
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
                    if main { Tone::Accent } else { Tone::Neutral },
                ))
                .children(order)
                .children(remove),
        )
        .child(models)
        .child(ui::status(credential, tone))
        .children(usage)
        .child(actions)
        .child(job_models)
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
                .child(
                    ui::secret_field("Paste the new key", form.replacement)
                        .label("New API key")
                        .w_full(),
                )
                .child(faint(
                    "Kept in the daemon computer's keychain and never shown again. It takes \
                     effect on a reload.",
                ))
                .child(
                    view()
                        .flex_row()
                        .gap_2()
                        .child(
                            ui::btn("Save key", Variant::Primary, Size::Sm, store_it).disabled(
                                move || {
                                    store.settings_busy()
                                        || form.replacement.with(|k| k.trim().is_empty())
                                },
                            ),
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
        .gap_3()
        .child(
            text("Sign in to ChatGPT")
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::MEDIUM),
        )
        .child(muted(if sign_in.paste {
            chatgpt_sign_in::intro()
        } else {
            "Open the sign-in page and sign in with your ChatGPT account. This page \
             updates when you are done."
        }))
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
                        crate::features::links::open_https(url.clone());
                    },
                ))
                .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                    form.redirect.set(String::new());
                    store.change_settings(SettingsChange::CancelSignIn);
                })),
        );
    if sign_in.paste {
        card = card
            .child(muted(chatgpt_sign_in::paste_step()))
            .child(
                ui::field("http://127.0.0.1:…/auth/callback?code=…", form.redirect)
                    .label("Address the browser ended on")
                    .w_full(),
            )
            .child(
                view()
                    .flex_row()
                    .flex_wrap()
                    .gap_2()
                    .child(
                        ui::btn(
                            "Paste and finish",
                            Variant::Secondary,
                            Size::Sm,
                            move || {
                                chatgpt_sign_in::paste_and_finish(
                                    store,
                                    form.redirect,
                                    move |redirect| {
                                        store.change_settings(SettingsChange::FinishSignIn {
                                            redirect,
                                        });
                                        form.redirect.set(String::new());
                                    },
                                );
                            },
                        )
                        .disabled(move || store.settings_busy()),
                    )
                    .child(
                        ui::btn("Finish sign-in", Variant::Outline, Size::Sm, move || {
                            let redirect = form.redirect.get_untracked();
                            if redirect.trim().is_empty() {
                                return;
                            }
                            store.change_settings(SettingsChange::FinishSignIn { redirect });
                            form.redirect.set(String::new());
                        })
                        .disabled(move || {
                            store.settings_busy() || form.redirect.with(|r| r.trim().is_empty())
                        }),
                    ),
            );
    } else {
        card = card.child(ui::working("Waiting for you to finish in the browser"));
    }
    card
}

/// What the add form reads from the settings.
#[derive(Clone)]
struct Known {
    options: Vec<ProviderOption>,
    signed_in: bool,
    checked: Option<aidaemon_client_protocol::CheckedKey>,
    /// The main provider's name.
    main_name: String,
    /// Presets a failover holds a stored key for: made main, they use it.
    keyed: Vec<String>,
    /// Presets already in use: one key reaches all of a provider's models,
    /// so a failover never adds the same provider again.
    connected: Vec<String>,
}

/// Connect a provider as the main one or a failover: pick it, then sign
/// in or paste its key.
fn add_form(store: Store, adding: Adding, form: ProviderForm, known: Known) -> View {
    let Known {
        options,
        signed_in,
        checked,
        main_name,
        keyed,
        connected,
    } = known;
    let (title, lead) = match adding {
        Adding::Main => (
            "Change the main provider",
            if main_name.is_empty() {
                "The provider that answers first.".to_string()
            } else {
                format!("Replaces {main_name} as the provider that answers first.")
            },
        ),
        Adding::Failover => (
            "Add a backup",
            "Tried when the providers before it fail.".to_string(),
        ),
    };
    let card = ui::surface()
        .p_4()
        .gap_3()
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(
                    text(title)
                        .flex_1()
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::MEDIUM),
                )
                .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                    form.close()
                })),
        )
        .child(muted(lead));
    // A daemon before protocol 1.6 sends no list: the preset is typed.
    if options.is_empty() {
        return card
            .child(
                ui::field("Preset", form.preset)
                    .label("Provider preset")
                    .w_full(),
            )
            .child(submit(store, adding, form, Rc::new(|| true)));
    }
    // A custom endpoint differs by address, so it may appear twice.
    let options: Vec<ProviderOption> = match adding {
        Adding::Main => options,
        Adding::Failover => options
            .into_iter()
            .filter(|o| o.needs_base_url || !connected.contains(&o.preset))
            .collect(),
    };
    card.child(dynamic(move || {
        let preset = form.preset.get();
        let options = options.clone();
        let checked = checked.clone();
        let keyed = keyed.clone();
        Some(untracked(move || {
            match options.iter().find(|o| o.preset == preset).cloned() {
                None => provider_list(options, form, signed_in),
                Some(option) => {
                    let checked = checked.filter(|c| c.preset == option.preset);
                    let borrows = adding == Adding::Main && keyed.contains(&option.preset);
                    view()
                        .gap_3()
                        .child(
                            view()
                                .flex_row()
                                .items_center()
                                .gap_2()
                                .child(
                                    view()
                                        .flex_1()
                                        .min_w(px(0.0))
                                        .child(
                                            text(option.name.clone())
                                                .number_of_lines(1)
                                                .text_size(px(ui::fit::body())),
                                        )
                                        .child(faint(billing_words(option.billing))),
                                )
                                .child(ui::btn(
                                    "Pick another",
                                    Variant::Ghost,
                                    Size::Sm,
                                    move || form.pick(String::new()),
                                )),
                        )
                        .child(provider_form(
                            store, adding, option, form, signed_in, checked, borrows,
                        ))
                }
            }
        }))
    }))
}

/// Every provider the daemon can connect, each saying how it bills and
/// signs in.
fn provider_list(options: Vec<ProviderOption>, form: ProviderForm, signed_in: bool) -> View {
    if options.is_empty() {
        return view().child(muted("Every provider here is already in use."));
    }
    let mut list = view().gap_0p5();
    for option in options {
        let preset = option.preset.clone();
        let how = if !option.needs_key {
            if signed_in { "Signed in" } else { "Sign in" }
        } else {
            "API key"
        };
        list = list.child(
            ui::row(|| false, move || form.pick(preset.clone()))
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .py_1()
                        .child(
                            text(option.name.clone())
                                .number_of_lines(2)
                                .text_size(px(ui::fit::body())),
                        )
                        .child(faint(billing_words(option.billing))),
                )
                .child(faint(how))
                .child(
                    ui::icon(Glyph::Chevron)
                        .text_color(colors::FAINT)
                        .accessibility_elements_hidden(true),
                )
                .accessibility_label(format!("{}: {}", option.name, how)),
        );
    }
    list
}

fn provider_form(
    store: Store,
    adding: Adding,
    option: ProviderOption,
    form: ProviderForm,
    signed_in: bool,
    checked: Option<aidaemon_client_protocol::CheckedKey>,
    // A failover of this provider has a stored key it can use.
    borrows: bool,
) -> View {
    let mut body = view().gap_2();
    let ready: Rc<dyn Fn() -> bool> = if option.needs_key {
        if option.needs_base_url {
            body = body.child(
                ui::field("https://api.example.com/v1", form.base_url)
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
        let mut key_row = view().flex_row().flex_wrap().items_center().gap_2().child(
            ui::btn("Check key", Variant::Secondary, Size::Sm, check)
                .disabled(move || store.settings_busy() || form.key.with(|k| k.trim().is_empty())),
        );
        if !option.key_url.is_empty() {
            let url = option.key_url.clone();
            key_row = key_row.child(ui::btn("Get a key", Variant::Ghost, Size::Sm, move || {
                crate::features::links::open_https(url.clone());
            }));
        }
        body = body
            .child(
                ui::secret_field("Paste the API key", form.key)
                    .label("API key")
                    .w_full(),
            )
            .child(key_row);
        if borrows {
            body = body.child(faint(
                "Leave it empty to use the key this backup already has.",
            ));
        }
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
                        format!("The key works. {} models to pick from.", found.models.len()),
                        Tone::Success,
                    ))
                    .child(view().flex_row().child(ui::choice(
                        "Model",
                        "Choose a model",
                        models,
                        move || Some(form.primary.get()).filter(|m| !m.is_empty()),
                        move |next: Option<String>| form.primary.set(next.unwrap_or_default()),
                    )));
            }
            None if !option.model.is_empty() => {
                body = body.child(faint(format!(
                    "Uses {} unless you pick another. Check the key to see what its \
                     account offers.",
                    option.model
                )));
            }
            None => {}
        }
        // A key typed, or one its failover has.
        Rc::new(move || borrows || form.key.with(|k| !k.trim().is_empty()))
    } else if signed_in {
        body = body.child(ui::status("Signed in to ChatGPT", Tone::Success));
        Rc::new(|| true)
    } else {
        body = body
            .child(muted("Sign in first: it uses your ChatGPT plan."))
            .child(
                view().flex_row().child(
                    ui::btn(
                        "Sign in with ChatGPT",
                        Variant::Primary,
                        Size::Sm,
                        move || form.sign_in(store),
                    )
                    .disabled(move || store.settings_busy()),
                ),
            );
        Rc::new(|| false)
    };
    let more_label = move || {
        if form.more.get() {
            "Fewer options"
        } else {
            "More options"
        }
    };
    let needs_base_url = option.needs_base_url;
    body = body
        .child(ui::disclosure(more_label, form.more))
        .child(ui::unfold(form.more, move || {
            let mut more = view().gap_2().child(faint(
                "Empty fields take the provider's own address and models.",
            ));
            if !needs_base_url {
                more = more.child(
                    ui::field("Address", form.base_url)
                        .label("Base URL")
                        .w_full(),
                );
            }
            more.child(
                ui::field("Main model", form.primary)
                    .label("Main model")
                    .w_full(),
            )
            .child(
                ui::field("Fast model", form.fast)
                    .label("Fast model")
                    .w_full(),
            )
            .child(
                ui::field("Smart model", form.smart)
                    .label("Smart model")
                    .w_full(),
            )
        }));
    body.child(submit(store, adding, form, ready))
}

fn submit(store: Store, adding: Adding, form: ProviderForm, ready: Rc<dyn Fn() -> bool>) -> View {
    let send = move || {
        let provider = form.choice();
        if provider.preset.is_empty() {
            return;
        }
        let key = form.key();
        store.change_settings(match adding {
            Adding::Main => SettingsChange::SwitchProvider { provider, key },
            Adding::Failover => SettingsChange::AddFailover { provider, key },
        });
        // The form closes once the daemon takes it (`ProviderForm::follow`).
        form.sent.set(true);
    };
    let label = match adding {
        Adding::Main => "Make it the main provider",
        Adding::Failover => "Add as backup",
    };
    view()
        .flex_row()
        .items_center()
        .gap_3()
        .pt_1()
        .child(
            ui::btn(label, Variant::Primary, Size::Md, send)
                .disabled(move || !ready() || store.settings_busy()),
        )
        .child(when(
            move || store.settings_busy(),
            || ui::working("Working"),
        ))
}

fn bots(store: Store, bots: &[BotInfo]) -> View {
    let mut section =
        view()
            .gap_2()
            .child(section_label("Connected bots"))
            .child(view().flex_row().child(ui::btn(
                "Reach it from your phone",
                Variant::Outline,
                Size::Sm,
                move || crate::features::setup::reach_from_phone(store),
            )));
    if bots.is_empty() {
        return section.child(muted(
            "No bots yet. Connect a Telegram, Slack or Discord bot to talk to your agent from \
             your phone. Bots in config.toml are settings above.",
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
            {
                // The platform by its mark, then which bot.
                let mut cell = view().flex_row().items_center().gap_1();
                if let Some(glyph) = ui::platform_glyph(&bot.platform) {
                    cell = cell.child(
                        ui::icon(glyph)
                            .text_color(colors::MUTED)
                            .accessibility_elements_hidden(true),
                    );
                }
                cell.child(text(name).number_of_lines(1))
            },
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

/// The paired devices, each with whether it may watch and control the
/// agent's screens. Only the daemon's own computer changes it: a phone
/// never widens what it may do itself.
fn devices(store: Store) -> View {
    store.refresh_devices();
    view()
        .gap_2()
        .child(section_label("Paired devices"))
        // Outside the list: a devices read (after pairing one) rebuilds the
        // list, and this keeps where its pairing stands.
        .child(crate::features::pairing::pair_eink(store))
        .child(dynamic(move || {
            let read = store.devices();
            Some(untracked(move || match read {
                aidaemon_client_core::Read::Ready((devices, can_change)) => {
                    let devices: Vec<_> = devices.into_iter().filter(|d| !d.revoked).collect();
                    let this_device = store.this_device();
                    let mut list = view().gap_2();
                    if can_change && crate::link::owner_pairing::available() {
                        list = list.child(crate::features::pairing::pair_phone(store));
                    }
                    if devices.is_empty() {
                        return list.child(muted("No paired devices.")).into_element();
                    }
                    if !can_change {
                        list = list.child(muted(
                            "Change what a device may do in the app on the computer AIDaemon \
                             runs on.",
                        ));
                    }
                    for device in devices {
                        let on = device
                            .scopes
                            .contains(&aidaemon_client_protocol::Scope::Control);
                        let id = device.device_id.clone();
                        let name = device.name.clone();
                        let control: View = if let Some(level) = device.eink_access {
                            eink_level(store, device.device_id.clone(), device.name.clone(), level)
                        } else if can_change {
                            view().child(
                                ui::toggle(
                                    "Can watch and control the agent's screens",
                                    move || on,
                                    move |next: bool| store.set_device_control(&id, next),
                                )
                                .accessibility_label(format!(
                                    "Let {name} watch and control the agent's screens"
                                )),
                            )
                        } else {
                            view().child(ui::badge(
                                if on {
                                    "Can watch and control screens"
                                } else {
                                    "Cannot watch screens"
                                },
                                if on { Tone::Info } else { Tone::Neutral },
                            ))
                        };
                        let this = this_device.as_deref() == Some(device.device_id.as_str());
                        let mut title = view().flex_row().items_center().gap_2().child(
                            text(device.name.clone())
                                .font_weight(FontWeight::MEDIUM)
                                .flex_1(),
                        );
                        if this {
                            title = title.child(ui::badge("This device", Tone::Info));
                        } else {
                            let id = device.device_id.clone();
                            let asked = device.name.clone();
                            title = title.child(
                                ui::btn("Remove", Variant::Ghost, Size::Sm, move || {
                                    let id = id.clone();
                                    ui::confirm(
                                        format!("Remove {asked}?"),
                                        "It loses access at once and must pair again to come \
                                         back.",
                                        "Remove",
                                        move || store.revoke_device(&id),
                                    )
                                })
                                .accessibility_label(format!("Remove {}", device.name)),
                            );
                        }
                        let mut row = ui::inset().gap_1().child(title);
                        if let Some(kind) = device_kind_label(device.kind) {
                            row = row.child(muted(kind));
                        }
                        let (used, stale) = ui::time::device_use(
                            device.created_at_ms,
                            device.last_seen_ms,
                            crate::store::now_ms(),
                        );
                        row = row.child(if stale {
                            view().flex_row().child(ui::badge(used, Tone::Warning))
                        } else {
                            view().child(muted(used))
                        });
                        if this {
                            row = row.child(muted(
                                "To remove this device, use another device or the computer \
                                 AIDaemon runs on.",
                            ));
                        }
                        list = list.child(row.child(control));
                    }
                    list.into_element()
                }
                aidaemon_client_core::Read::Failed(why) => muted(why).into_element(),
                _ => muted("Reading the devices…").into_element(),
            }))
        }))
        .child(dynamic(move || {
            let why = store.devices_error()?;
            Some(untracked(move || {
                ui::banner(why, Tone::Danger).into_element()
            }))
        }))
}

/// The setting that names the e-ink bridge, and the project's bridge.
const EINK_BRIDGE_KEY: &str = "client_gateway.eink.bridge_url";
const PROJECT_BRIDGE: &str = "wss://bridge.aidaemon.ai/v1/daemon";

/// E-ink access: off, through the project's bridge, or through the owner's
/// own. Turning it on is asked on another device (it widens who reaches
/// the daemon); the daemon starts or stops it at once.
fn eink_access(store: Store, settings: &Settings) -> View {
    // A TOML basic string, its quotes taken off (machine format).
    let saved: Option<String> = settings
        .groups
        .iter()
        .flat_map(|g| &g.settings)
        .find(|s| s.key == EINK_BRIDGE_KEY)
        .and_then(|s| s.value.as_deref())
        .map(|v| v.trim().trim_matches('"').trim().to_owned())
        .filter(|v| !v.is_empty());
    let mode = signal(match saved.as_deref() {
        None => 0_usize,
        Some(PROJECT_BRIDGE) => 1,
        Some(_) => 2,
    });
    let custom = signal(
        saved
            .clone()
            .filter(|url| url != PROJECT_BRIDGE)
            .unwrap_or_default(),
    );
    let set = move |url: Option<String>| {
        store.change_settings(SettingsChange::Set {
            key: EINK_BRIDGE_KEY.into(),
            value: url.map(|url| {
                let escaped = url.replace('\\', "\\\\").replace('"', "\\\"");
                format!("\"{escaped}\"")
            }),
        });
    };
    let choose = move |at: usize| {
        mode.set(at);
        match at {
            0 => set(None),
            1 => set(Some(PROJECT_BRIDGE.to_owned())),
            // Their own bridge waits for its address and Save.
            _ => {}
        }
    };
    view()
        .gap_2()
        .child(section_label("E-ink access"))
        .child(muted(
            "Lets an e-ink reader like a Kindle reach this daemon through a bridge, end-to-end \
             encrypted. Off unless you turn it on. Then pair it under Paired devices below.",
        ))
        .child(view().flex_row().child(ui::segmented(
            "E-ink access",
            &["Off", "AIDaemon's bridge", "My own bridge"],
            move || mode.get(),
            choose,
        )))
        .child(when(
            move || mode.get() == 2,
            move || {
                view()
                    .flex_row()
                    .items_end()
                    .gap_2()
                    .child(
                        view().flex_1().child(ui::labeled(
                            "Bridge address",
                            ui::field("wss://bridge.example.com/v1/daemon", custom)
                                .auto_correct(false)
                                .label("E-ink bridge address"),
                        )),
                    )
                    .child(
                        ui::btn("Save", Variant::Primary, Size::Sm, move || {
                            set(Some(custom.get_untracked().trim().to_owned()))
                        })
                        .disabled(move || {
                            // Machine format: the bridge's URL scheme (the
                            // daemon checks it again).
                            custom.with(|url| {
                                let url = url.trim();
                                !["wss://", "ws://127.0.0.1", "ws://localhost"]
                                    .iter()
                                    .any(|scheme| url.starts_with(scheme))
                            })
                        }),
                    )
            },
        ))
}

/// What an e-ink level allows, in one line.
fn eink_level_line(level: aidaemon_client_protocol::EinkAccess) -> &'static str {
    use aidaemon_client_protocol::EinkAccess;
    match level {
        EinkAccess::ViewOnly => "Reads conversations, runs and changes. Sends nothing.",
        EinkAccess::Limited => {
            "Chats and answers medium-risk agent approvals once; runs it starts stay capped at \
             medium risk."
        }
        _ => {
            "Chats and answers agent approvals up to high risk once; its runs follow your \
             settings. Never a shell, settings, pairing or lasting grants."
        }
    }
}

/// The warning under Full: the reader's browser is old.
const EINK_FULL_WARNING: &str =
    "Its browser is no longer updated; a harmful site opened in it could act with this access.";

/// An e-ink reader's access level, as a choice with what each allows. A
/// raise is confirmed here first; lowering applies at once.
fn eink_level(
    store: Store,
    device_id: String,
    name: String,
    level: aidaemon_client_protocol::EinkAccess,
) -> View {
    use aidaemon_client_protocol::EinkAccess;
    const LEVELS: [EinkAccess; 3] = [EinkAccess::ViewOnly, EinkAccess::Limited, EinkAccess::Full];
    const NAMES: [&str; 3] = ["View only", "Limited", "Full"];
    let at = LEVELS.iter().position(|l| *l == level).unwrap_or(1);
    let mut column = view()
        .gap_1()
        .child(view().flex_row().child(ui::segmented(
            "E-ink access level",
            &NAMES,
            move || at,
            move |chosen: usize| {
                let next = LEVELS[chosen];
                if chosen < at {
                    store.set_eink_access(&device_id, next);
                } else if chosen > at {
                    let mut detail = eink_level_line(next).to_owned();
                    if next == EinkAccess::Full {
                        detail.push_str("\n\n");
                        detail.push_str(EINK_FULL_WARNING);
                    }
                    let id = device_id.clone();
                    ui::confirm(
                        format!("Give {name} {} access?", NAMES[chosen]),
                        detail,
                        "Confirm",
                        move || store.set_eink_access(&id, next),
                    );
                }
            },
        )))
        .child(muted(eink_level_line(level)));
    if level == EinkAccess::Full {
        column = column.child(ui::banner(EINK_FULL_WARNING, Tone::Warning));
    }
    column.child(dynamic(move || {
        let note = store.eink_access_note()?;
        Some(untracked(move || muted(note)))
    }))
}

/// What kind of device a paired one is, where the name alone may not say.
fn device_kind_label(kind: aidaemon_client_protocol::ClientKind) -> Option<&'static str> {
    use aidaemon_client_protocol::ClientKind;
    match kind {
        ClientKind::Eink => Some("E-ink reader"),
        ClientKind::Mobile => Some("Phone"),
        ClientKind::Watch => Some("Watch"),
        ClientKind::Desktop => Some("Computer"),
        ClientKind::Web => Some("Browser"),
        ClientKind::Tui => Some("Terminal"),
        ClientKind::Script => Some("Script"),
        _ => None,
    }
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

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn windows_are_named_by_their_length() {
        assert_eq!(window_name(300), "5-hour limit");
        assert_eq!(window_name(10_080), "Weekly limit");
        assert_eq!(window_name(1_440), "Daily limit");
        assert_eq!(window_name(2_880), "2-day limit");
        assert_eq!(window_name(45), "45-minute limit");
    }

    #[test]
    fn a_balance_with_a_cap_fills_its_meter_and_one_without_does_not() {
        let capped = UsageBalance {
            used_cents: 350,
            limit_cents: Some(1_000),
            remaining_cents: Some(650),
        };
        let (words, share) = balance_words(&capped);
        assert_eq!(words, "$6.50 left of $10.00");
        assert!((share.unwrap() - 0.35).abs() < 1e-6);
        let open = UsageBalance {
            used_cents: 346,
            limit_cents: None,
            remaining_cents: None,
        };
        assert_eq!(
            balance_words(&open),
            ("$3.46 spent · no cap set".to_string(), None)
        );
    }

    #[test]
    fn plans_read_as_people_write_them() {
        assert_eq!(plan_name("plus"), "Plus");
        assert_eq!(plan_name(""), "");
    }
}
