//! First run. With no daemon to sign in to, the desktop app starts one on
//! this computer at once (pairing with a machine elsewhere is the other way
//! in; a browser or terminal asks). Then the daemon's own setup step:
//! connect a model, one click when one is found here. The first
//! conversation opens next; reaching the agent from a phone waits until the
//! owner asks for it in Settings, and ends when their first message there
//! has been answered.

use aidaemon_client_protocol::{Billing, ProviderChoice, SecretText, SetupAction, SetupStep};
use nsh_core::prelude::*;

use crate::link;
use crate::store::Store;
use crate::ui::{self, Size, Tone, Variant, btn, colors, field, secret_field, status, type_scale};

fn heading(label: &'static str) -> Text {
    text(label)
        .role(AccessibilityRole::Header)
        .text_size(px(ui::fit::title()))
        .font_weight(FontWeight::SEMIBOLD)
}

fn lead(label: impl IntoReactive<SharedString>) -> Text {
    text(label)
        .text_size(px(type_scale::SM))
        .line_height(px(type_scale::LH_SM))
        .text_color(colors::MUTED)
}

/// A provider or its model by name, behind the provider's mark when it has
/// one.
fn provider_name(preset: &str, name: String, weight: FontWeight) -> View {
    let mut row = view().flex_row().items_center().gap_2();
    // One without a mark keeps its name in line with the others.
    row = row.child(match ui::provider_glyph(preset) {
        Some(glyph) => ui::icon(glyph).accessibility_elements_hidden(true),
        None => ui::icon(ui::Glyph::Dot)
            .text_color(colors::FAINT)
            .accessibility_elements_hidden(true),
    });
    row.child(
        text(name)
            .flex_1()
            .min_w(px(0.0))
            .text_size(px(type_scale::BASE))
            .font_weight(weight),
    )
}

fn frame(content: View) -> View {
    // Centred while it fits; a long step scrolls from its top.
    view().flex_1().child(
        scroll().flex_1().child(
            view()
                .min_h(percentage(1.0))
                .items_center()
                .justify_center()
                .p_6()
                .child(
                    ui::surface()
                        .w(px(520.0))
                        .max_w(percentage(1.0))
                        .p_6()
                        .gap_4()
                        .child(content),
                ),
        ),
    )
}

fn billing_words(billing: Billing) -> &'static str {
    match billing {
        Billing::Plan => "uses your plan · has limits",
        Billing::Local => "runs on your own machine · free",
        _ => "pay per token",
    }
}

/// The desktop app was opened to run the agent here: its first launch on
/// this device starts it rather than ask. Once only, so an owner who later
/// removes the service is asked, not overruled.
const STARTED_HERE_KEY: &str = "aidaemon.started_here";

/// The owner asked for the phone step (from Settings). First run skips it,
/// so the first conversation comes before any bot token.
static PHONE_ASKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Open the phone step: talk to the agent from Telegram, Slack or Discord.
pub fn reach_from_phone(store: Store) {
    PHONE_ASKED.store(true, std::sync::atomic::Ordering::Relaxed);
    store.setup_step_later(SetupAction::Detect);
}

/// No daemon yet. The desktop app starts one here at once; pairing with a
/// machine elsewhere is the other way in. A browser or a terminal asks.
pub fn where_page(store: Store) -> View {
    let terminal = idiom() == nsh_core::platform::Idiom::Terminal;
    let auto = link::RUNS_DAEMON && !terminal;
    let first_launch = nsh_core::storage::Storage::get_item(STARTED_HERE_KEY).is_none();
    let auto = auto && (first_launch || store.starting_daemon().is_some());
    if auto && first_launch {
        let _ = nsh_core::storage::Storage::set_item(STARTED_HERE_KEY, "1");
        spawn(async move { link::install_here(store) });
    }
    let elsewhere = signal(!link::RUNS_DAEMON);
    frame(
        view()
            .gap_4()
            .child(heading(if auto {
                "Welcome to AIDaemon"
            } else if link::MOBILE {
                "Connect to your agent"
            } else {
                "Where should your agent run?"
            }))
            .child(lead(if link::MOBILE {
                "AIDaemon is an agent that works for you around the clock. It runs on a computer \
                 or server you own; this app is how you talk to it from your phone."
            } else {
                "AIDaemon is an agent that works for you around the clock. It runs on a computer \
                 you choose; this app is how you talk to it."
            }))
            .child(when(link::RUNS_DAEMON, move || run_here(store, auto)))
            .child(when(
                move || auto && !elsewhere.get(),
                move || {
                    view().flex_row().child(btn(
                        "Run it on another machine instead",
                        Variant::Ghost,
                        Size::Sm,
                        move || elsewhere.set(true),
                    ))
                },
            ))
            .child(when(
                move || !auto || elsewhere.get(),
                move || pair_elsewhere(store),
            )),
    )
}

/// This computer: started already (`auto`), or a button.
fn run_here(store: Store, auto: bool) -> View {
    ui::surface()
        .p_4()
        .gap_2()
        .child(
            text("On this computer")
                .text_size(px(type_scale::BASE))
                .font_weight(FontWeight::MEDIUM),
        )
        .child(lead(
            "Starts now and whenever you log in. No administrator password. It is offline while \
             this computer sleeps.",
        ))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_3()
                .child(when(
                    move || !auto || store.starting_daemon().is_some_and(|(_, failed)| failed),
                    move || {
                        btn(
                            if auto { "Try again" } else { "Run it here" },
                            Variant::Primary,
                            Size::Md,
                            move || link::install_here(store),
                        )
                        .disabled(move || {
                            store.starting_daemon().is_some_and(|(_, failed)| !failed)
                        })
                    },
                ))
                .child(dynamic(move || {
                    let (what, failed) = store.starting_daemon()?;
                    Some(untracked(move || {
                        if failed {
                            status(what, Tone::Danger).flex_1().min_w(px(0.0))
                        } else {
                            ui::working(what).flex_1().min_w(px(0.0))
                        }
                    }))
                })),
        )
}

/// A daemon elsewhere, paired with the link it printed.
fn pair_elsewhere(store: Store) -> View {
    let link_text = signal(String::new());
    let name = signal(String::from(if link::MOBILE { "Phone" } else { "Desktop" }));
    let pairing = signal(None::<(String, Tone)>);
    // One pairing at a time: a second (Connect tapped again, the link
    // opened twice) would redeem a used code and be refused.
    let busy = stored(false);
    let owner = Owner::current();
    let pair = move || {
        if busy.replace(true) {
            return;
        }
        pairing.set(Some((
            "Pairing… If your computer asks, approve this phone there.".into(),
            Tone::Accent,
        )));
        owner.run(|| {
            spawn(async move {
                let result =
                    link::pair_remote(store, link_text.get_untracked(), name.get_untracked()).await;
                let _ = busy.try_with(|b| *b = false);
                if let Err(why) = result {
                    let _ = pairing.try_set(Some((why, Tone::Danger)));
                }
            })
        });
    };
    // An `aidaemon-pair:` link opened on this device (a QR code, a tapped
    // link; the shell listens for them) fills the field and pairs at once.
    effect(move || {
        if store.pairing_link().is_none() {
            return;
        }
        if let Some(opened) = store.take_pairing_link()
            && link_text.try_set(opened)
        {
            pair();
        }
    });
    // A pairing link the owner copied (from a message, a mail) fills the
    // field; they still tap Connect.
    if link::MOBILE {
        spawn(async move {
            if !clipboard_has_string().await {
                return;
            }
            let copied = clipboard_string().await;
            if aidaemon_client_core::signin::PairingLink::parse(&copied).is_some()
                && link_text.with_untracked(|l| l.trim().is_empty())
                && link_text.try_set(copied.trim().to_owned())
            {
                let _ = pairing.try_set(Some((
                    "Found the pairing link you copied. Tap Connect.".into(),
                    Tone::Accent,
                )));
            }
        });
    }
    ui::surface()
        .p_4()
        .gap_2()
        .child(
            text(if link::MOBILE {
                "Pair this phone"
            } else {
                "On another machine"
            })
            .text_size(px(type_scale::BASE))
            .font_weight(FontWeight::MEDIUM),
        )
        .child(lead(if link::MOBILE {
            "On the computer AIDaemon runs on, open the AIDaemon app, go to Settings → Paired \
             devices → Pair a phone and scan the code with this phone's camera. Or send \
             /pair_device to your agent in Telegram, Slack or Discord and tap the link it \
             replies with. You can also paste the link here."
        } else {
            "A server or another computer that stays on. Start `aidaemon` there; it prints a link \
             and a code. Paste the link here."
        }))
        .child(ui::labeled(
            "Pairing link",
            field("https://…/client/v1/pair#…", link_text),
        ))
        .child(ui::labeled(
            "This device's name",
            field(if link::MOBILE { "Phone" } else { "Desktop" }, name),
        ))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_3()
                .child(
                    btn("Connect", Variant::Secondary, Size::Md, pair)
                        .disabled(move || link_text.with(|l| l.trim().is_empty())),
                )
                .child(dynamic(move || {
                    let (message, tone) = pairing.get()?;
                    Some(untracked(move || {
                        status(message, tone).flex_1().min_w(px(0.0))
                    }))
                })),
        )
}

/// The daemon's setup steps.
pub fn page(store: Store) -> View {
    let step = computed(move || store.setup().map(|s| s.step));
    view().flex_1().child(dynamic(move || {
        let step = step.get();
        Some(untracked(move || match step {
            Some(SetupStep::Channel) if PHONE_ASKED.load(std::sync::atomic::Ordering::Relaxed) => {
                channel_step(store).key("setup-channel")
            }
            // First run on the daemon's own computer: one question, the
            // phone. Chat bots wait in Settings.
            Some(SetupStep::Channel) if link::owner_pairing::available() => {
                phone_step(store).key("setup-phone")
            }
            // Elsewhere: the first conversation before any bot token.
            Some(SetupStep::Channel) => {
                spawn(async move {
                    if !store.setup_busy() {
                        store.setup_step(SetupAction::SkipChannel);
                    }
                });
                opening(store).key("setup-opening")
            }
            Some(SetupStep::Starting) => starting(store).key("setup-starting"),
            _ => model_step(store).key("setup-model"),
        }))
    }))
}

fn problem(store: Store) -> View {
    view()
        .gap_2()
        .child(dynamic(move || {
            let error = store.setup_error()?;
            Some(untracked(move || ui::banner(error, Tone::Danger)))
        }))
        .child(dynamic(move || {
            let notices = store.setup().map(|s| s.notices).unwrap_or_default();
            (!notices.is_empty()).then(|| {
                untracked(move || {
                    let mut list = view().gap_1();
                    for notice in notices {
                        list = list.child(ui::banner(notice, Tone::Info));
                    }
                    list
                })
            })
        }))
}

fn opening(store: Store) -> View {
    frame(
        view()
            .gap_4()
            .child(heading("Ready"))
            .child(ui::working("Opening your first conversation"))
            .child(problem(store)),
    )
}

fn starting(store: Store) -> View {
    frame(
        view()
            .gap_4()
            .child(heading("Connected"))
            .child(lead(move || {
                format!(
                    "{} answered its test. AIDaemon is starting with it…",
                    store
                        .setup()
                        .and_then(|s| s.model)
                        .unwrap_or_else(|| "The model".into())
                )
            }))
            .child(ui::working("Starting"))
            .child(problem(store)),
    )
}

fn model_step(store: Store) -> View {
    // The provider being connected with a key, by preset.
    let chosen = signal(None::<String>);
    let key = signal(String::new());
    let model = signal(String::new());
    let base_url = signal(String::new());
    let more = signal(false);
    let has_found = computed(move || {
        store
            .setup()
            .is_some_and(|s| s.found.iter().any(|m| m.works))
    });
    let connect = move || {
        let Some(preset) = chosen.get_untracked() else {
            return;
        };
        store.setup_step(SetupAction::ConnectModel {
            provider: ProviderChoice {
                preset,
                base_url: base_url.get_untracked().trim().to_owned(),
                primary_model: model.get_untracked().trim().to_owned(),
                ..ProviderChoice::default()
            },
            key: Some(SecretText(key.get_untracked())).filter(|k| !k.0.trim().is_empty()),
        });
    };
    frame(
        view()
            .gap_4()
            .child(heading("Connect a model"))
            .child(lead(
                "Your agent thinks with an AI model. Nothing is saved until it answers a test.",
            ))
            // A tested model found here is one click; the rest wait under
            // "Use something else".
            .child(dynamic(move || {
                let found = store.setup().map(|s| s.found).unwrap_or_default();
                let (index, best) = found.iter().enumerate().find(|(_, m)| m.works)?;
                let (index, best) = (index, best.clone());
                Some(untracked(move || {
                    ui::surface()
                        .p_4()
                        .gap_2()
                        .child(provider_name(
                            &best.preset,
                            format!("Found: {}", best.name),
                            FontWeight::MEDIUM,
                        ))
                        .child(lead(format!(
                            "{} · {}. It answered a test just now.",
                            best.source,
                            billing_words(best.billing)
                        )))
                        .child(
                            view()
                                .flex_row()
                                .items_center()
                                .gap_3()
                                .child(
                                    btn("Continue", Variant::Primary, Size::Md, move || {
                                        store.setup_step(SetupAction::UseFound {
                                            index: index as u32,
                                        })
                                    })
                                    .disabled(move || store.setup_busy()),
                                )
                                .child(when(
                                    move || store.setup_busy(),
                                    || ui::working("Connecting"),
                                )),
                        )
                }))
            }))
            .child(when(
                move || has_found.get() && !more.get(),
                move || {
                    view().flex_row().child(btn(
                        "Use something else",
                        Variant::Ghost,
                        Size::Sm,
                        move || more.set(true),
                    ))
                },
            ))
            .child(when(
                move || more.get() || !has_found.get(),
                move || choices(store, chosen, key, model, base_url, connect),
            ))
            .child(lead(
                "Your conversations and memory stay on the machine AIDaemon runs on. There is no \
                 account and no usage data is sent; it talks to the model you pick and the sites \
                 you ask about.",
            ))
            .child(problem(store)),
    )
}

/// Every other way in: other models found here, a sign-in, a provider key.
fn choices(
    store: Store,
    chosen: Signal<Option<String>>,
    key: Signal<String>,
    model: Signal<String>,
    base_url: Signal<String>,
    connect: impl Fn() + Copy + 'static,
) -> View {
    let redirect = signal(String::new());
    // "Sign in" was pressed here (holding the page shown then, if any): the
    // new page opens once the daemon names it.
    let open_sign_in = signal(None::<Option<String>>);
    let shown = move || store.setup().and_then(|s| s.sign_in).map(|s| s.url);
    effect(move || {
        let Some(before) = open_sign_in.get() else {
            return;
        };
        if let Some(url) = shown()
            && Some(&url) != before.as_ref()
        {
            open_sign_in.set(None);
            crate::features::links::open_https(url);
        }
    });
    view()
        .gap_4()
        // Found on this machine, tested.
        .child(dynamic(move || {
            let found = store.setup().map(|s| s.found).unwrap_or_default();
            // The first that works is the card above.
            let above = found.iter().position(|m| m.works);
            (found.len() > usize::from(above.is_some())).then(|| {
                untracked(move || {
                    let mut list = view().gap_2().child(ui::section_label("Also found here"));
                    for (index, model) in found.into_iter().enumerate() {
                        if Some(index) == above {
                            continue;
                        }
                        let detail = if model.works {
                            format!("{} · {}", model.source, billing_words(model.billing))
                        } else {
                            format!("{} · did not answer: {}", model.source, model.detail)
                        };
                        list = list.child(
                            view()
                                .flex_row()
                                .items_center()
                                .gap_3()
                                .child(
                                    view()
                                        .flex_1()
                                        .min_w(px(0.0))
                                        .child(provider_name(
                                            &model.preset,
                                            model.name,
                                            FontWeight::NORMAL,
                                        ))
                                        .child(lead(detail)),
                                )
                                .child(when(model.works, move || {
                                    btn("Use it", Variant::Primary, Size::Sm, move || {
                                        store.setup_step(SetupAction::UseFound {
                                            index: index as u32,
                                        })
                                    })
                                    .disabled(move || store.setup_busy())
                                })),
                        );
                    }
                    list
                })
            })
        }))
        // A sign-in under way.
        .child(dynamic(move || {
            let sign_in = store.setup().and_then(|s| s.sign_in)?;
            Some(untracked(move || {
                let url = sign_in.url.clone();
                let mut card = ui::surface()
                    .p_4()
                    .gap_2()
                    .child(
                        text("Sign in to ChatGPT")
                            .text_size(px(type_scale::BASE))
                            .font_weight(FontWeight::MEDIUM),
                    )
                    .child(btn(
                        "Open the sign-in page",
                        Variant::Primary,
                        Size::Sm,
                        move || {
                            crate::features::links::open_https(url.clone());
                        },
                    ))
                    .child(ui::code(sign_in.url.clone()).number_of_lines(2));
                if sign_in.paste {
                    card = card
                        .child(lead(
                            "Signed in on another device? When its browser shows it cannot \
                                 reach the page it ends on, copy that page's address and paste \
                                 it here.",
                        ))
                        .child(field("http://127.0.0.1:…/auth/callback?code=…", redirect))
                        .child(
                            btn("Finish sign-in", Variant::Secondary, Size::Sm, move || {
                                store.setup_step(SetupAction::FinishSignIn {
                                    redirect: redirect.get_untracked(),
                                })
                            })
                            .disabled(move || {
                                store.setup_busy() || redirect.with(|r| r.trim().is_empty())
                            }),
                        );
                } else {
                    card = card.child(ui::working("Waiting for you to finish in the browser"));
                }
                card
            }))
        }))
        // Every provider, each saying how it bills.
        .child(ui::section_label(
            if store
                .setup()
                .is_some_and(|s| s.found.iter().any(|m| m.works))
            {
                "Or connect"
            } else {
                "Connect"
            },
        ))
        .child(dynamic(move || {
            let providers = store.setup().map(|s| s.providers).unwrap_or_default();
            Some(untracked(move || {
                let mut list = view().gap_2();
                for option in providers {
                    let preset = option.preset.clone();
                    // A model server on the owner's machine takes no key
                    // and no sign-in.
                    let local = option.billing == Billing::Local;
                    let sign_in = !option.needs_key && !local;
                    let action = move || {
                        if sign_in {
                            open_sign_in.set(Some(shown()));
                            store.setup_step(SetupAction::SignIn { paste: false });
                        } else {
                            chosen.set(Some(preset.clone()));
                            model.set(String::new());
                            key.set(String::new());
                            base_url.set(String::new());
                        }
                    };
                    let open = {
                        let preset = option.preset.clone();
                        move || chosen.with(|c| c.as_deref() == Some(preset.as_str()))
                    };
                    let key_url = option.key_url.clone();
                    let needs_base_url = option.needs_base_url;
                    let preset_id = option.preset.clone();
                    list = list.child(
                        view()
                            .gap_2()
                            .child(
                                view()
                                    .flex_row()
                                    .items_center()
                                    .gap_3()
                                    .child(
                                        view()
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .child(provider_name(
                                                &option.preset,
                                                option.name.clone(),
                                                FontWeight::NORMAL,
                                            ))
                                            .child(lead(billing_words(option.billing))),
                                    )
                                    .child(
                                        btn(
                                            if sign_in {
                                                "Sign in"
                                            } else if local {
                                                "Use it"
                                            } else {
                                                "Use a key"
                                            },
                                            if sign_in {
                                                Variant::Primary
                                            } else {
                                                Variant::Outline
                                            },
                                            Size::Sm,
                                            action,
                                        )
                                        .disabled(move || store.setup_busy()),
                                    ),
                            )
                            .child(when(open, move || {
                                let key_url = key_url.clone();
                                let mut form = view().gap_2().pl_3();
                                if !key_url.is_empty() {
                                    let url = key_url.clone();
                                    form = form.child(btn(
                                        "Get a key",
                                        Variant::Ghost,
                                        Size::Sm,
                                        move || {
                                            crate::features::links::open_https(url.clone());
                                        },
                                    ));
                                }
                                if needs_base_url && local {
                                    // Its usual address unless another is named.
                                    form = form.child(ui::labeled(
                                        "Address (leave empty for its usual one)",
                                        field("http://localhost:8081/v1", base_url),
                                    ));
                                } else if needs_base_url {
                                    form = form.child(ui::labeled(
                                        "Address",
                                        field("https://api.example.com/v1", base_url),
                                    ));
                                }
                                let preset = preset_id.clone();
                                let check = {
                                    let preset = preset.clone();
                                    move || {
                                        model.set(String::new());
                                        store.setup_step(SetupAction::CheckKey {
                                            provider: ProviderChoice {
                                                preset: preset.clone(),
                                                base_url: base_url
                                                    .get_untracked()
                                                    .trim()
                                                    .to_owned(),
                                                ..ProviderChoice::default()
                                            },
                                            key: SecretText(key.get_untracked()),
                                        })
                                    }
                                };
                                // The account's own models, once the key checks out.
                                let checked = move || {
                                    store
                                        .setup()
                                        .and_then(|s| s.checked)
                                        .filter(|c| c.preset == preset)
                                };
                                let checked_models = checked.clone();
                                if needs_base_url || local {
                                    form = form.child(lead(
                                        "A model you run yourself costs nothing per message. \
                                         Small ones chat well but often stumble on tasks with \
                                         many steps; larger ones do better.",
                                    ));
                                }
                                if needs_base_url && !local {
                                    form = form.child(ui::labeled(
                                        "API key (leave empty if your server needs none)",
                                        secret_field("Paste the key", key),
                                    ));
                                } else if !local {
                                    form = form.child(ui::labeled(
                                        "API key",
                                        secret_field("Paste the key", key),
                                    ));
                                }
                                form.child(
                                    view()
                                        .flex_row()
                                        .items_center()
                                        .gap_3()
                                        .child(
                                            btn(
                                                if local {
                                                    "Find its models"
                                                } else if needs_base_url {
                                                    "Find models"
                                                } else {
                                                    "Check key"
                                                },
                                                Variant::Secondary,
                                                Size::Sm,
                                                check,
                                            )
                                            .disabled(
                                                move || {
                                                    store.setup_busy()
                                                        || if local {
                                                            false
                                                        } else if needs_base_url {
                                                            base_url.with(|b| b.trim().is_empty())
                                                        } else {
                                                            !local
                                                                && key.with(|k| k.trim().is_empty())
                                                        }
                                                },
                                            ),
                                        )
                                        .child(when(
                                            move || store.setup_busy(),
                                            || ui::working("Asking the provider"),
                                        )),
                                )
                                .child(dynamic(move || {
                                    let found = checked_models()?;
                                    // Preselect the usual model when the account has it.
                                    if model.with_untracked(String::is_empty)
                                        && let Some(suggested) = found.suggested.clone()
                                    {
                                        model.set(suggested);
                                    }
                                    Some(untracked(move || {
                                        let options: Vec<(String, String)> = found
                                            .models
                                            .iter()
                                            .map(|m| (m.clone(), m.clone()))
                                            .collect();
                                        view()
                                            .gap_2()
                                            .child(lead(if local || needs_base_url {
                                                format!(
                                                    "It answered. It has {} models.",
                                                    found.models.len()
                                                )
                                            } else {
                                                format!(
                                                    "The key works. Its account offers {} \
                                                         models.",
                                                    found.models.len()
                                                )
                                            }))
                                            .child(ui::labeled(
                                                "Model",
                                                view().flex_row().child(ui::choice(
                                                    "Model",
                                                    "Choose a model",
                                                    options,
                                                    move || {
                                                        Some(model.get()).filter(|m| !m.is_empty())
                                                    },
                                                    move |next: Option<String>| {
                                                        model.set(next.unwrap_or_default())
                                                    },
                                                )),
                                            ))
                                    }))
                                }))
                                .child(when(
                                    move || checked().is_some(),
                                    move || {
                                        view()
                                            .flex_row()
                                            .items_center()
                                            .gap_3()
                                            .child(
                                                btn(
                                                    "Test and connect",
                                                    Variant::Primary,
                                                    Size::Sm,
                                                    connect,
                                                )
                                                .disabled(move || {
                                                    store.setup_busy()
                                                        || model.with(|m| m.trim().is_empty())
                                                }),
                                            )
                                            .child(when(
                                                move || store.setup_busy(),
                                                || ui::working("Testing"),
                                            ))
                                    },
                                ))
                            }))
                            .child(ui::separator()),
                    );
                }
                list
            }))
        }))
}

/// Where a self-hoster reads about their own relay, web app or tunnel.
const OWN_SETUP_URL: &str = "https://github.com/davo20019/aidaemon/blob/master/deploy/README.md";

/// First run: use AIDaemon from a phone, by scanning one code. The phone
/// opens the web app (or the AIDaemon app, when it has it), already paired;
/// pairing it continues setup.
fn phone_step(store: Store) -> View {
    let done = move || store.setup_step_later(SetupAction::SkipChannel);
    let advanced = signal(false);
    frame(
        view()
            .gap_4()
            .child(heading("Use AIDaemon from your phone?"))
            .child(lead(
                "Scan this code with your phone's camera. AIDaemon opens on your phone, \
                 already paired: no install, no account.",
            ))
            .child(super::pairing::pair_phone_now(
                store,
                std::rc::Rc::new(done),
            ))
            .child(hint_text(
                "It connects through AIDaemon's relay, encrypted end to end: the relay \
                 cannot read your messages.",
            ))
            .child(
                view()
                    .flex_row()
                    .flex_wrap()
                    .gap_2()
                    .child(
                        btn("Skip for now", Variant::Ghost, Size::Sm, done)
                            .disabled(move || store.setup_busy()),
                    )
                    .child(btn("Advanced", Variant::Ghost, Size::Sm, move || {
                        advanced.update(|open| *open = !*open)
                    })),
            )
            .child(dynamic(move || {
                advanced.get().then(|| {
                    untracked(move || {
                        view()
                            .gap_2()
                            .child(hint_text(
                                "Your own relay, your own copy of the web app, or your own \
                                 tunnel (Tailscale, Cloudflare): set them in config.toml.",
                            ))
                            .child(view().flex_row().child(btn(
                                "How to set them up",
                                Variant::Ghost,
                                Size::Sm,
                                move || drop(open_url(OWN_SETUP_URL.to_string())),
                            )))
                    })
                })
            })),
    )
}

fn hint_text(words: &'static str) -> Text {
    text(words)
        .text_size(px(type_scale::SM))
        .line_height(px(type_scale::LH_SM))
        .text_color(colors::MUTED)
}

fn channel_step(store: Store) -> View {
    let platform = signal(None::<String>);
    let token = signal(String::new());
    let app_token = signal(String::new());
    let connect = move || {
        let Some(chosen) = platform.get_untracked() else {
            return;
        };
        let app = app_token.get_untracked();
        store.setup_step(SetupAction::ConnectChannel {
            platform: chosen,
            token: SecretText(token.get_untracked()),
            app_token: (!app.trim().is_empty()).then_some(SecretText(app)),
        });
    };
    frame(
        view()
            .gap_4()
            .child(heading("Reach it from your phone"))
            .child(lead(
                "Talk to your agent from Telegram, Slack or Discord too: make a bot there, paste \
                 its token, then send it a message. You can come back to this from Settings.",
            ))
            .child(dynamic(move || {
                let setup = store.setup()?;
                Some(untracked(move || match setup.channel {
                    // Connected: wait for the owner's first message.
                    Some(wait) if wait.connected => view()
                        .gap_3()
                        .child(status(
                            format!("Connected: your agent answered you as {}.", wait.bot),
                            Tone::Success,
                        ))
                        .child(btn("Continue", Variant::Primary, Size::Md, move || {
                            PHONE_ASKED.store(false, std::sync::atomic::Ordering::Relaxed);
                            store.setup_step_later(SetupAction::Finish)
                        })),
                    Some(wait) => view()
                        .gap_3()
                        .child(lead(match &wait.claim_code {
                            Some(code) => format!(
                                "Now send this message to {} from your phone to make it yours: \
                                 /claim {code}. Only this code claims it. Its answer arrives there.",
                                wait.bot
                            ),
                            None => format!(
                                "Now send any message to {} from your phone. Its answer arrives there.",
                                wait.bot
                            ),
                        }))
                        .child(ui::working("Waiting for your first message")),
                    None => {
                        let mut choices = view().flex_row().gap_2();
                        for option in setup.channels.clone() {
                            let id = option.platform.clone();
                            choices = choices.child(
                                btn(option.name.clone(), Variant::Outline, Size::Sm, move || {
                                    platform.set(Some(id.clone()))
                                })
                                .style_when(
                                    {
                                        let id = option.platform.clone();
                                        move || platform.with(|p| p.as_deref() == Some(id.as_str()))
                                    },
                                    |s| s.border_color(colors::ACCENT),
                                ),
                            );
                        }
                        let channels = setup.channels.clone();
                        view().gap_3().child(choices).child(dynamic(move || {
                            let chosen = platform.get()?;
                            let option = channels.iter().find(|c| c.platform == chosen)?.clone();
                            Some(untracked(move || {
                                let help = option.help_url.clone();
                                let mut form = view()
                                    .gap_2()
                                    .child(btn(
                                        format!("Make a {} bot", option.name),
                                        Variant::Ghost,
                                        Size::Sm,
                                        move || {
                                            crate::features::links::open_https(help.clone());
                                        },
                                    ))
                                    .child(ui::labeled(
                                        "Bot token",
                                        secret_field("Paste the bot's token", token),
                                    ));
                                if option.needs_app_token {
                                    form = form.child(ui::labeled(
                                        "App-level token",
                                        secret_field("xapp-…", app_token),
                                    ));
                                }
                                form.child(
                                    btn("Connect", Variant::Primary, Size::Sm, connect).disabled(
                                        move || {
                                            store.setup_busy()
                                                || token.with(|t| t.trim().is_empty())
                                        },
                                    ),
                                )
                            }))
                        }))
                    }
                }))
            }))
            .child(problem(store))
            .child(
                btn("Not now", Variant::Ghost, Size::Sm, move || {
                    PHONE_ASKED.store(false, std::sync::atomic::Ordering::Relaxed);
                    store.setup_step_later(SetupAction::SkipChannel)
                })
                .disabled(move || store.setup_busy()),
            ),
    )
}
