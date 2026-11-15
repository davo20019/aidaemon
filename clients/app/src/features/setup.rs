//! First run. With no daemon to sign in to: where the agent runs (this
//! computer, or a machine elsewhere paired with its printed link). Then the
//! daemon's own setup steps: connect a model (the one that must pass; each
//! provider says how it bills), and optionally reach the agent from a
//! phone, which ends when the owner's first message there has been
//! answered.

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

fn labeled(label: &'static str, control: impl IntoElement) -> View {
    view()
        .gap_1p5()
        .child(
            text(label)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::MUTED),
        )
        .child(control)
}

fn frame(content: View) -> View {
    view().flex_1().items_center().justify_center().p_6().child(
        ui::surface()
            .w(px(520.0))
            .max_w(percentage(1.0))
            .p_6()
            .gap_4()
            .child(content),
    )
}

fn billing_words(billing: Billing) -> &'static str {
    match billing {
        Billing::Plan => "uses your plan · has limits",
        _ => "pay per token",
    }
}

/// No daemon yet: run one here, or connect to one elsewhere.
pub fn where_page(store: Store) -> View {
    let link_text = signal(String::new());
    let name = signal(String::from("Desktop"));
    let pairing = signal(None::<(String, Tone)>);
    let pair = move || {
        pairing.set(Some(("Pairing…".into(), Tone::Accent)));
        spawn(async move {
            let result =
                link::pair_remote(store, link_text.get_untracked(), name.get_untracked()).await;
            if let Err(why) = result {
                let _ = pairing.try_set(Some((why, Tone::Danger)));
            }
        });
    };
    frame(
        view()
            .gap_4()
            .child(heading("Where should your agent run?"))
            .child(lead(
                "AIdaemon is an agent that works for you around the clock. It runs on a computer \
                 you choose; this app is how you talk to it.",
            ))
            .child(when(link::RUNS_DAEMON, move || {
                ui::surface()
                    .p_4()
                    .gap_2()
                    .child(
                        text("On this computer")
                            .text_size(px(type_scale::BASE))
                            .font_weight(FontWeight::MEDIUM),
                    )
                    .child(lead(
                        "Starts now and whenever you log in. No administrator password. It is \
                         offline while this computer sleeps.",
                    ))
                    .child(
                        view()
                            .flex_row()
                            .items_center()
                            .gap_3()
                            .child(
                                btn("Run it here", Variant::Primary, Size::Md, move || {
                                    link::install_here(store)
                                })
                                .disabled(move || {
                                    store.starting_daemon().is_some_and(|(_, failed)| !failed)
                                }),
                            )
                            .child(dynamic(move || {
                                let (what, failed) = store.starting_daemon()?;
                                Some(untracked(move || {
                                    status(what, if failed { Tone::Danger } else { Tone::Accent })
                                        .flex_1()
                                        .min_w(px(0.0))
                                }))
                            })),
                    )
            }))
            .child(
                ui::surface()
                    .p_4()
                    .gap_2()
                    .child(
                        text("On another machine")
                            .text_size(px(type_scale::BASE))
                            .font_weight(FontWeight::MEDIUM),
                    )
                    .child(lead(
                        "A server or another computer that stays on. Start `aidaemon` there; \
                         it prints a link and a code. Paste the link here.",
                    ))
                    .child(labeled(
                        "Pairing link",
                        field("http://…/client/v1/next/#offer=…", link_text),
                    ))
                    .child(labeled("This device's name", field("Desktop", name)))
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
                    ),
            ),
    )
}

/// The daemon's setup steps.
pub fn page(store: Store) -> View {
    let step = computed(move || store.setup().map(|s| s.step));
    view().flex_1().child(dynamic(move || {
        let step = step.get();
        Some(untracked(move || match step {
            Some(SetupStep::Channel) => channel_step(store).key("setup-channel"),
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

fn starting(store: Store) -> View {
    frame(
        view()
            .gap_4()
            .child(heading("Connected"))
            .child(lead(move || {
                format!(
                    "{} answered its test. AIdaemon is starting with it…",
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
    let redirect = signal(String::new());
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
                "Your agent thinks with an AI model. Pick one you already have; nothing is saved \
                 until it answers a test.",
            ))
            // Found on this machine, tested.
            .child(dynamic(move || {
                let found = store.setup().map(|s| s.found).unwrap_or_default();
                (!found.is_empty()).then(|| {
                    untracked(move || {
                        let mut list = view().gap_2().child(ui::section_label("Found here"));
                        for (index, model) in found.into_iter().enumerate() {
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
                                            .child(text(model.name).text_size(px(type_scale::BASE)))
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
                                drop(open_url(url.clone()));
                            },
                        ))
                        .child(ui::code(sign_in.url.clone()).number_of_lines(2));
                    if sign_in.paste {
                        card = card
                            .child(lead(
                                "When the browser shows it cannot reach the page it ends on, \
                                 copy that page's address and paste it here.",
                            ))
                            .child(field(
                                "http://localhost:1455/auth/callback?code=…",
                                redirect,
                            ))
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
            .child(ui::section_label("Or connect"))
            .child(dynamic(move || {
                let providers = store.setup().map(|s| s.providers).unwrap_or_default();
                Some(untracked(move || {
                    let mut list = view().gap_2();
                    for option in providers {
                        let preset = option.preset.clone();
                        let sign_in = !option.needs_key;
                        let action = move || {
                            if sign_in {
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
                                                .child(
                                                    text(option.name.clone())
                                                        .text_size(px(type_scale::BASE)),
                                                )
                                                .child(lead(billing_words(option.billing))),
                                        )
                                        .child(
                                            btn(
                                                if sign_in { "Sign in" } else { "Use a key" },
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
                                                drop(open_url(url.clone()));
                                            },
                                        ));
                                    }
                                    if needs_base_url {
                                        form = form.child(labeled(
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
                                    form.child(labeled(
                                        "API key",
                                        secret_field("Paste the key", key),
                                    ))
                                    .child(
                                        view()
                                            .flex_row()
                                            .items_center()
                                            .gap_3()
                                            .child(
                                                btn(
                                                    "Check key",
                                                    Variant::Secondary,
                                                    Size::Sm,
                                                    check,
                                                )
                                                .disabled(move || {
                                                    store.setup_busy()
                                                        || key.with(|k| k.trim().is_empty())
                                                }),
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
                                                .child(lead(format!(
                                                    "The key works. Its account offers {} models.",
                                                    found.models.len()
                                                )))
                                                .child(labeled(
                                                    "Model",
                                                    view().flex_row().child(ui::choice(
                                                        "Model",
                                                        "Choose a model",
                                                        options,
                                                        move || {
                                                            Some(model.get())
                                                                .filter(|m| !m.is_empty())
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
            .child(problem(store)),
    )
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
            .child(lead(move || {
                let model = store
                    .setup()
                    .and_then(|s| s.model)
                    .map(|m| format!("{m} is connected. "))
                    .unwrap_or_default();
                format!(
                    "{model}You can also talk to your agent in a chat app. This is optional: \
                     you can do it later in Settings."
                )
            }))
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
                            store.setup_step(SetupAction::Finish)
                        })),
                    Some(wait) => view()
                        .gap_3()
                        .child(lead(format!(
                            "Now send any message to {} from your phone. The first person to \
                             message it becomes its owner, so do it now. Its answer arrives there.",
                            wait.bot
                        )))
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
                                            drop(open_url(help.clone()));
                                        },
                                    ))
                                    .child(labeled(
                                        "Bot token",
                                        secret_field("Paste the bot's token", token),
                                    ));
                                if option.needs_app_token {
                                    form = form.child(labeled(
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
                btn("Skip for now", Variant::Ghost, Size::Sm, move || {
                    store.setup_step(SetupAction::SkipChannel)
                })
                .disabled(move || store.setup_busy()),
            ),
    )
}
