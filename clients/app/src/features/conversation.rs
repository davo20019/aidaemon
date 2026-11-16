//! A conversation: its transcript (Markdown replies, streamed as they
//! arrive), the approvals it waits on, and the composer. A new
//! conversation is the same page with no session yet.

use aidaemon_client_core::state::Line;
use aidaemon_client_protocol::{AttachmentRef, MidRun, ToolCallInfo};
use nsh_base::{MarkdownOptions, TextViewLinks, button, input, text_view};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use super::approvals;
use crate::store::{ActivityMode, DraftFileState, Felt, Store};
use crate::ui::{self, Glyph, Size, Variant, btn_icon, colors, radius, size, type_scale, working};

/// A row of the transcript.
#[derive(Debug, Clone, PartialEq)]
enum Row {
    /// A day break: "Today", "Yesterday", "Mar 3".
    Day(String),
    User(String, u64, Vec<AttachmentRef>),
    Assistant(String, u64, Vec<AttachmentRef>),
    Note(String),
    /// The reply being written and what the agent is doing.
    Live,
    Approval(String),
    /// The tools a reply's turn used, how many seconds it took, and what
    /// each was given and gave back as far as known.
    Activity(Vec<String>, u64, Vec<ToolCallInfo>),
    /// The last turn failed or was stopped: send its message again.
    Retry,
    /// Work handed to the background runs; its report comes here.
    Background,
}

impl Row {
    fn key(&self, index: usize) -> String {
        match self {
            Self::Live => "live".into(),
            Self::Retry => "retry".into(),
            Self::Background => "background".into(),
            Self::Approval(id) => format!("approval-{id}"),
            Self::Day(label) => format!("day-{label}-{index}"),
            _ => format!("line-{index}"),
        }
    }
}

fn rows(store: Store, session: &str) -> Vec<Row> {
    let now = crate::store::now_ms();
    let mode = store.activity_mode();
    let mut rows: Vec<Row> = Vec::new();
    let mut last_day = None;
    for line in store.lines(session) {
        let at = match &line {
            Line::User { at_ms, .. } | Line::Assistant { at_ms, .. } => *at_ms,
            Line::Note { .. } | Line::Activity { .. } => 0,
        };
        if at > 0 {
            let day = ui::time::day(at);
            if last_day != Some(day) {
                last_day = Some(day);
                rows.push(Row::Day(ui::time::day_label(at, now)));
            }
        }
        rows.push(match line {
            Line::User {
                text,
                at_ms,
                attachments,
                ..
            } => Row::User(text, at_ms, attachments),
            Line::Assistant {
                text,
                at_ms,
                attachments,
                ..
            } => Row::Assistant(text, at_ms, attachments),
            Line::Note { text } => Row::Note(text),
            Line::Activity { .. } if mode == ActivityMode::Hidden => continue,
            Line::Activity { tools, secs, calls } => Row::Activity(tools, secs, calls),
        });
    }
    if store.live(session).is_some() {
        rows.push(Row::Live);
    } else if store.in_background(session) {
        rows.push(Row::Background);
    } else if store.can_retry(session) {
        rows.push(Row::Retry);
    }
    rows.extend(
        store
            .approvals_for(session)
            .into_iter()
            .map(|a| Row::Approval(a.id)),
    );
    rows
}

/// The page for `session` (`None`: a new conversation).
pub fn page(store: Store, session: Option<String>) -> View {
    let body = match &session {
        Some(id) => transcript(store, id.clone()).into_element(),
        None => welcome(store).into_element(),
    };
    view()
        .test_id("conversation")
        .flex_1()
        .min_w(px(0.0))
        .child(header(store, session.clone()))
        .child(view().flex_1().child(body))
        .child(offers(store, session.clone()))
        .child(notices(store))
        .child(running_work(store, session.clone()))
        .child(match session.clone() {
            // A group hears what is sent from here: the owner says so first.
            Some(id) if untracked(|| store.is_group(&id)) => {
                let gate = id.clone();
                when(move || store.may_post(&gate), {
                    let session = session.clone();
                    move || composer(store, session.clone())
                })
                .otherwise(move || group_gate(store, id.clone()))
                .into_element()
            }
            _ => composer(store, session).into_element(),
        })
}

/// This conversation's work running in the background, over the composer
/// where the eye already is: what it is, how long it has run, Cancel.
fn running_work(store: Store, session: Option<String>) -> View {
    let Some(session) = session else {
        return view();
    };
    // While a turn runs, its own Stop stops everything it started (the
    // daemon lists the turn itself as a task too): one control at a time.
    // This is the work that outlives a turn.
    let tasks = computed(move || {
        if store.is_running(&session) {
            return Vec::new();
        }
        store
            .running()
            .into_iter()
            .filter(|t| t.session_id == session)
            .collect::<Vec<_>>()
    });
    view().items_center().px_4().child(each(
        tasks,
        |t| t.id.clone(),
        move |task, _| {
            let id = task.with_untracked(|t| t.id.clone());
            let started = task.with_untracked(|t| t.started_at_ms);
            view()
                .w_full()
                .max_w(px(size::COLUMN))
                .flex_row()
                .items_center()
                .gap_2()
                .pl_1()
                .child(ui::icon(Glyph::Work).text_color(colors::ACCENT))
                .child(
                    text(move || task.with(|t| t.description.clone()))
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .text_size(px(ui::fit::small()))
                        .text_color(colors::MUTED),
                )
                // How long it has run: a clock tick a second.
                .child(
                    text(move || {
                        store.tick();
                        ui::time::span(crate::store::now_ms().saturating_sub(started) / 1000)
                    })
                    .tabular_nums()
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                )
                .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                    store.cancel(&id)
                }))
        },
    ))
}

/// In a group conversation, instead of the composer until the owner
/// chooses: what is sent from here is posted for everyone in it.
fn group_gate(store: Store, session: String) -> View {
    let bot = untracked(|| {
        crate::store::channel_name(&store.channel(&session).unwrap_or_default(), &session)
    });
    view().items_center().px_4().pb_4().pt_2().child(
        ui::surface()
            .w_full()
            .max_w(px(size::COLUMN))
            .p_3()
            .gap_2()
            .child(
                text(format!(
                    "Other people are in this chat. A message sent from here is \
                     posted there by {bot}, for everyone in it to see."
                ))
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
            )
            .child(view().flex_row().gap_2().child(ui::btn(
                "Post in this group",
                Variant::Secondary,
                ui::Size::Sm,
                move || store.allow_posting(&session),
            ))),
    )
}

/// What the daemon offers in this conversation: features the agent asked
/// for (a card that turns one on), and prompts to press under its reply.
fn offers(store: Store, session: Option<String>) -> View {
    let Some(session) = session else {
        return view();
    };
    let for_features = session.clone();
    let for_prompts = session.clone();
    view()
        .items_center()
        .px_4()
        .child(dynamic(move || {
            let requests = store.feature_requests(&for_features);
            let for_features = for_features.clone();
            (!requests.is_empty()).then(|| {
                untracked(move || {
                    let mut list = view().w_full().items_center();
                    for (feature, reason) in requests {
                        let (what, how) = feature_words(feature);
                        let enable = if feature == aidaemon_client_protocol::Feature::Voice {
                            "Set it up"
                        } else {
                            "Turn it on"
                        };
                        list = list.child(
                            ui::banner(
                                format!("Your agent needs {what} {reason}. {how}"),
                                ui::Tone::Info,
                            )
                            .child(ui::btn(enable, Variant::Primary, ui::Size::Sm, {
                                let session = for_features.clone();
                                move || {
                                    if feature == aidaemon_client_protocol::Feature::Voice {
                                        store.ask(
                                            &session,
                                            "Set up voice-note transcription for me".into(),
                                        );
                                        store.dismiss_feature(feature);
                                    } else {
                                        store.enable_feature(feature);
                                        // macOS grants screen and input use in
                                        // System Settings.
                                        if feature
                                            == aidaemon_client_protocol::Feature::ComputerUse
                                            && cfg!(target_os = "macos")
                                        {
                                            drop(open_url(
                                                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
                                            ));
                                        }
                                    }
                                }
                            }))
                            .child(ui::btn(
                                "Not now",
                                Variant::Ghost,
                                ui::Size::Sm,
                                move || store.dismiss_feature(feature),
                            )),
                        );
                    }
                    list
                })
            })
        }))
        .child(dynamic(move || {
            if store.is_running(&for_prompts) {
                return None;
            }
            let prompts = store.suggestions(&for_prompts);
            (!prompts.is_empty()).then(|| {
                let session = for_prompts.clone();
                untracked(move || {
                    let mut row = view()
                        .w_full()
                        .max_w(px(size::COLUMN))
                        .flex_row()
                        .flex_wrap()
                        .gap_2()
                        .py_2();
                    for prompt in prompts {
                        let session = session.clone();
                        let text = prompt.clone();
                        row =
                            row.child(ui::btn(prompt, Variant::Outline, ui::Size::Sm, move || {
                                store.ask(&session, text.clone())
                            }));
                    }
                    row
                })
            })
        }))
}

/// A feature as the card names it, and what turning it on does.
fn feature_words(feature: aidaemon_client_protocol::Feature) -> (&'static str, &'static str) {
    use aidaemon_client_protocol::Feature;
    match feature {
        Feature::Browser => ("to browse websites", "Turning it on restarts the daemon."),
        Feature::ComputerUse => (
            "to see and use this computer's screen",
            "Turning it on restarts the daemon; your computer then asks to allow it.",
        ),
        Feature::Voice => (
            "to understand voice notes",
            "It needs a speech model on this computer; your agent can set it up.",
        ),
        Feature::Unknown => ("a feature", ""),
    }
}

/// What the agent's view moves say: a move offered while the owner types,
/// and a short note after one was taken.
fn notices(store: Store) -> View {
    view()
        .items_center()
        .px_4()
        .child(when(
            move || store.suggestion().is_some(),
            move || {
                ui::banner(
                    move || store.suggestion().unwrap_or_default(),
                    ui::Tone::Accent,
                )
                .child(ui::btn(
                    format!("Go  {}", key_label("mod+g")),
                    Variant::Primary,
                    ui::Size::Sm,
                    move || store.take_suggestion(),
                ))
                .child(ui::btn(
                    "Dismiss",
                    Variant::Ghost,
                    ui::Size::Sm,
                    move || store.dismiss_suggestion(),
                ))
            },
        ))
        // What the daemon refused or reported (a message it could not take).
        .child(when(
            move || store.notice().is_some(),
            move || {
                ui::banner(
                    move || store.notice().unwrap_or_default(),
                    ui::Tone::Neutral,
                )
                .child(ui::btn(
                    "Dismiss",
                    Variant::Ghost,
                    ui::Size::Sm,
                    move || store.dismiss_notice(),
                ))
            },
        ))
}

fn header(store: Store, session: Option<String>) -> View {
    let title = session.clone();
    let chat = session
        .clone()
        .filter(|id| store.source(id) != crate::store::Source::App);
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .border_b_1()
        .border_color(colors::LINE)
        // Narrow windows reach the panes from here; wide ones when folded.
        .child(when(
            move || store.is_compact() || !store.sidebar_open(),
            move || {
                // A phone's list is a drawer of conversations, not a pane
                // to unfold: it gets its own picture, unlike the details'.
                let glyph = if untracked(|| store.is_compact()) {
                    Glyph::Chat
                } else {
                    Glyph::Sidebar
                };
                // Something elsewhere waits on the owner (amber) or works
                // (accent): a dot on the button, so the list need not be
                // opened to know.
                let elsewhere = move |is: fn(&Store, &str) -> bool| {
                    let here = store.selected();
                    store
                        .session_ids()
                        .iter()
                        .any(|id| Some(id) != here.as_ref() && is(&store, id))
                };
                view()
                    .relative()
                    .child(ui::icon_btn(glyph, "Conversations", Size::Md, move || {
                        store.open_sidebar()
                    }))
                    .child(when(
                        move || elsewhere(Store::needs_you) || elsewhere(Store::is_working),
                        move || {
                            ui::corner_dot(move || {
                                if elsewhere(Store::needs_you) {
                                    ui::Tone::Warning
                                } else {
                                    ui::Tone::Accent
                                }
                            })
                        },
                    ))
            },
        ))
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .px_2()
                // A new conversation has no title yet: the bar says what
                // it is, a chat or work on code.
                .child(match title {
                    Some(id) => text(move || store.title(&id))
                        .role(AccessibilityRole::Header)
                        .number_of_lines(1)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD)
                        .into_element(),
                    None => view().items_center().child(new_kind(store)).into_element(),
                })
                // Where it lives (a chat's platform and bot) and the
                // project it works in, under its title.
                .child(when(
                    {
                        let chat = chat.clone();
                        // A new conversation's project is picked on the page.
                        let started = session.is_some();
                        move || started && (chat.is_some() || store.current_project().is_some())
                    },
                    move || {
                        let chat = chat.clone();
                        // A chat on Telegram, Discord or Slack: its mark, then
                        // the bot (and "group"), without the platform's name.
                        let glyph = chat
                            .as_deref()
                            .and_then(|id| untracked(|| store.channel(id)))
                            .and_then(|channel| ui::platform_glyph(&channel));
                        let mark = glyph.map(|glyph| {
                            ui::icon(glyph)
                                .text_color(colors::FAINT)
                                .accessibility_elements_hidden(true)
                        });
                        let line = text(move || {
                            let project = store
                                .current_project()
                                .and_then(|id| store.project(&id))
                                .map(|p| match p.path {
                                    Some(path) => format!(
                                        "{} · {}",
                                        p.name,
                                        super::sidebar::short_path(&path)
                                    ),
                                    None => p.name,
                                });
                            let place = chat.as_deref().map(|id| {
                                let place = store.place(id);
                                match glyph {
                                    Some(_) => place
                                        .split_once(" · ")
                                        .map_or(place.clone(), |(_, rest)| rest.to_owned()),
                                    None => place,
                                }
                            });
                            match (place, project) {
                                (Some(place), Some(project)) => format!("{place} · {project}"),
                                (Some(place), None) => place,
                                (None, project) => project.unwrap_or_default(),
                            }
                        })
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT);
                        let mut row = view().flex_row().items_center().gap_1();
                        if let Some(mark) = mark {
                            row = row.child(mark);
                        }
                        row.child(line)
                    },
                )),
        )
        .child(when(
            {
                let s = session.clone();
                move || s.is_some() && !store.is_compact()
            },
            {
                let pin = session.clone();
                move || {
                    let pin = pin.clone();
                    let label = pin.clone();
                    let id = pin.clone().unwrap_or_default();
                    view()
                        .flex_row()
                        .child(dynamic(move || {
                            let pinned = store.is_pinned(&id);
                            let pin = pin.clone();
                            Some(untracked(move || {
                                ui::icon_btn(
                                    if pinned { Glyph::Pinned } else { Glyph::Pin },
                                    if pinned { "Unpin" } else { "Pin to the top" },
                                    Size::Md,
                                    move || {
                                        if let Some(id) = &pin {
                                            store.toggle_pin(id)
                                        }
                                    },
                                )
                            }))
                        }))
                        .child(ui::icon_btn(
                            Glyph::Copy,
                            "Copy conversation as Markdown",
                            Size::Md,
                            move || {
                                let _ = &label;
                                if let Some(text) = store.transcript_markdown() {
                                    set_clipboard(text);
                                    store.say("Copied the conversation");
                                }
                            },
                        ))
                }
            },
        ))
        .child(when(
            move || store.is_compact(),
            move || {
                // A new conversation in the same project.
                ui::icon_btn(Glyph::Compose, "New conversation", Size::Md, move || {
                    store.new_in_project(store.current_project())
                })
            },
        ))
        .child(when(
            {
                // Only a coding conversation has details: a chat's steps
                // are under each reply.
                let started = session.is_some();
                move || {
                    started
                        && store.coding_on_screen()
                        && (store.is_compact() || !store.inspector_open())
                }
            },
            move || {
                view().child(dynamic(move || {
                    // A phone shows the changes themselves, the point of
                    // a coding conversation there: an icon like the bar's
                    // others, its count of files on its corner.
                    if store.is_compact() {
                        let files = store.changed_files().unwrap_or(0);
                        let label = match files {
                            0 => "Changes".to_owned(),
                            1 => "Changes, 1 file".to_owned(),
                            n => format!("Changes, {n} files"),
                        };
                        return Some(untracked(move || {
                            let button = view().relative().child(ui::icon_btn(
                                Glyph::Changes,
                                label,
                                Size::Md,
                                move || store.show_changes(),
                            ));
                            if files > 0 {
                                button.child(ui::corner_count(move || files))
                            } else {
                                button
                            }
                            .into_element()
                        }));
                    }
                    Some(untracked(move || {
                        ui::icon_btn(Glyph::Inspector, "Changes and steps", Size::Md, move || {
                            store.open_inspector()
                        })
                        .into_element()
                    }))
                }))
            },
        ))
}

/// What a new conversation is, in its header: a chat, or work on code in
/// a project's folder. Said before the first word, so the two never look
/// like the same thing.
fn new_kind(store: Store) -> View {
    ui::segmented(
        "What this conversation is",
        &["Chat", "Code"],
        move || usize::from(store.new_is_code()),
        move |at| {
            if (at == 1) != untracked(|| store.new_is_code()) {
                store.place_new(at == 1);
            }
        },
    )
}

/// A new conversation: what the header's switch chose. Both ways have
/// the same shape, a question over a room of one height (empty for a
/// chat; the project list for code), so switching changes the words and
/// nothing moves. By the composer on a phone, in the middle of a window.
/// The sidebar already lists past conversations.
fn welcome(store: Store) -> View {
    let code = computed(move || store.new_is_code());
    view()
        .flex_1()
        .items_center()
        .px_4()
        .py_4()
        .child(view().flex_1())
        .child(
            view()
                .w_full()
                .items_center()
                .gap_3()
                .child(
                    text(move || {
                        if code.get() {
                            "Which project is this for?"
                        } else {
                            "What can I help with?"
                        }
                    })
                    .role(AccessibilityRole::Header)
                    .text_center()
                    .text_size(px(ui::fit::title()))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(colors::TEXT),
                )
                .child(
                    view()
                        .w_full()
                        .items_center()
                        .min_h(px(if ui::touch() { 72.0 } else { 56.0 }))
                        .child(when(move || code.get(), move || code_start(store))),
                ),
        )
        // On a phone all of it sits by the composer, under the thumb; in a
        // window it is centred.
        .child(view().style_when(!ui::touch(), |s| s.flex_1()))
}

/// Opens the sidebar's form for a new project (the drawer on a phone).
fn new_project_form(store: Store) {
    if !store.sidebar_open() {
        store.open_sidebar();
    }
    // The form is at the end of the list of projects.
    store.show_sidebar_view(crate::store::SidebarView::Projects);
    store.show_project_form(true);
}

/// Code: the project list, with a "+" for a new one, and under it the
/// chosen project's folder (or what to do). With no project yet, New
/// project in the list's place.
fn code_start(store: Store) -> View {
    let any = computed(move || !store.projects().is_empty());
    view().w_full().items_center().child(dynamic(move || {
        let any = any.get();
        Some(untracked(move || {
            if !any {
                return view()
                    .items_center()
                    .gap_2()
                    .child(ui::btn(
                        "New project",
                        Variant::Outline,
                        ui::Size::Sm,
                        move || new_project_form(store),
                    ))
                    .child(
                        text("A project is a folder on the daemon's computer")
                            .number_of_lines(1)
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    );
            }
            // One card names the project and its folder; a press opens the
            // projects (recent first) with New project last.
            view().w_full().max_w(px(420.0)).child(ui::pick_list(
                ui::Picker {
                    title: "Choose a project",
                    items: move || {
                        super::sidebar::project_choices(store, store.current_project(), None)
                    },
                    pick: move |id| store.set_new_project(Some(id)),
                    action: Some(ui::PickAction {
                        label: "New project",
                        glyph: Glyph::Plus,
                        run: std::rc::Rc::new(move || new_project_form(store)),
                    }),
                },
                move || match store.current_project().and_then(|id| store.project(&id)) {
                    Some(p) => format!("Project: {}, change", p.name),
                    None => "Choose a project".to_owned(),
                },
                project_card(store),
                true,
            ))
        }))
    }))
}

/// What the Code page's card shows: the chosen project and its folder, or
/// that one is still to be chosen.
fn project_card(store: Store) -> View {
    let chosen = computed(move || store.current_project().and_then(|id| store.project(&id)));
    view()
        .w_full()
        .min_h(px(if ui::touch() { 60.0 } else { 52.0 }))
        .flex_row()
        .items_center()
        .gap_2()
        .px_3()
        .rounded(px(ui::radius::CONTROL))
        .border_1()
        .border_color(colors::EDGE)
        .bg(colors::SURFACE)
        .child(
            ui::icon(Glyph::Folder)
                .text_color(colors::MUTED)
                .accessibility_elements_hidden(true),
        )
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .child(
                    text(move || {
                        chosen.with(|p| {
                            p.as_ref()
                                .map_or_else(|| "Choose a project".to_owned(), |p| p.name.clone())
                        })
                    })
                    .number_of_lines(1)
                    .text_size(px(ui::fit::body()))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(move || {
                        if chosen.with(Option::is_some) {
                            colors::TEXT
                        } else {
                            colors::MUTED
                        }
                    }),
                )
                .child(
                    text(move || {
                        chosen.with(|p| match p {
                            Some(p) => match &p.path {
                                Some(path) => super::sidebar::short_path(path),
                                None => "No folder".to_owned(),
                            },
                            None => "The agent works in its folder".to_owned(),
                        })
                    })
                    .number_of_lines(1)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                ),
        )
        .child(
            ui::icon(Glyph::Down)
                .text_color(colors::FAINT)
                .accessibility_elements_hidden(true),
        )
}

/// The transcript, and a "Jump to latest" pill while the owner reads back.
fn transcript(store: Store, session: String) -> View {
    let id = session.clone();
    let all = computed(move || rows(store, &id));
    let last_seen = signal(usize::MAX);
    let list = element_ref();
    view()
        .flex_1()
        .child(
            transcript_list(store, session, all, last_seen)
                .element_ref(&list)
                .flex_1(),
        )
        .child(when(
            move || {
                let len = all.with(Vec::len);
                let seen = last_seen.get();
                seen != usize::MAX && len > 0 && seen + 2 < len
            },
            move || {
                view()
                    .absolute()
                    .bottom(px(12.0))
                    .left(px(0.0))
                    .right(px(0.0))
                    .items_center()
                    .child(
                        ui::btn(
                            "Jump to latest ↓",
                            Variant::Secondary,
                            Size::Sm,
                            move || list.scroll_to_end(true),
                        )
                        .shadow_md(),
                    )
            },
        ))
}

fn transcript_list(
    store: Store,
    session: String,
    all: Memo<Vec<Row>>,
    last_seen: Signal<usize>,
) -> VirtualList {
    let count = all.with_untracked(Vec::len);
    virtual_list(
        move || all.with(Vec::len),
        move |at| {
            let row = computed(move || all.with(|r| r.get(at.index()).cloned()));
            let kind = computed(move || {
                row.with(|r| {
                    r.as_ref().map(|r| match r {
                        Row::User(..) => 0,
                        Row::Assistant(..) => 1,
                        Row::Note(_) => 2,
                        Row::Live => 3,
                        Row::Approval(_) => 4,
                        Row::Day(_) => 5,
                        Row::Activity(..) => 6,
                        Row::Retry => 7,
                        Row::Background => 8,
                    })
                })
            });
            let session = session.clone();
            view().items_center().child(dynamic(move || {
                let kind = kind.get()?;
                let session = session.clone();
                Some(untracked(move || {
                    let column = view().w_full().max_w(px(size::COLUMN)).px_4().py_2();
                    let content = move || {
                        row.with(|r| match r {
                            Some(
                                Row::User(t, ..)
                                | Row::Assistant(t, ..)
                                | Row::Note(t)
                                | Row::Day(t),
                            ) => t.clone(),
                            _ => String::new(),
                        })
                    };
                    let at = move || {
                        row.with(|r| match r.as_ref() {
                            Some(Row::User(_, at, _) | Row::Assistant(_, at, _)) => *at,
                            _ => 0,
                        })
                    };
                    let attached = move || {
                        row.with(|r| match r {
                            Some(Row::User(_, _, files) | Row::Assistant(_, _, files)) => {
                                files.clone()
                            }
                            _ => Vec::new(),
                        })
                    };
                    match kind {
                        0 => column
                            .items_end()
                            .gap_1()
                            .child(files(store, attached, true))
                            .child(when(
                                move || !content().trim().is_empty(),
                                move || user_bubble(content),
                            ))
                            .child(stamp(at)),
                        1 => column
                            .gap_1()
                            .child(when(
                                move || !content().trim().is_empty(),
                                move || markdown(move || content().into(), || false),
                            ))
                            .child(files(store, attached, false))
                            .child(reply_footer(store, content, at)),
                        2 => column.child(note(content)),
                        5 => column.child(day_break(content)),
                        3 => column.child(live(store, session)),
                        6 => column.child(activity(
                            store,
                            move || {
                                row.with(|r| match r {
                                    Some(Row::Activity(tools, secs, calls)) => {
                                        (tools.clone(), *secs, calls.clone())
                                    }
                                    _ => (Vec::new(), 0, Vec::new()),
                                })
                            },
                            false,
                        )),
                        7 => column.child(retry(store, session)),
                        8 => column.child(ui::working(
                            "Working in the background · the result comes here",
                        )),
                        _ => column.child(dynamic(move || {
                            let id = row.with(|r| match r {
                                Some(Row::Approval(id)) => Some(id.clone()),
                                _ => None,
                            })?;
                            let approval = store.approvals().into_iter().find(|a| a.id == id)?;
                            Some(untracked(|| approvals::card(store, approval)))
                        })),
                    }
                }))
            }))
        },
    )
    .keys(move || {
        all.with(|r| {
            r.iter()
                .enumerate()
                .map(|(i, row)| row.key(i))
                .collect::<Vec<_>>()
        })
    })
    .on_viewable_items_change(move |items: &nsh_core::element::ViewableItems| {
        if let Some((_, last)) = items.viewable {
            last_seen.set(last);
        }
    })
    .estimated_item_size(px(96.0))
    .initial_scroll_index(count.saturating_sub(1))
    .follow_end(px(48.0))
    .content_padding_top(px(12.0))
    .content_padding_bottom(px(12.0))
}

/// What a reply's turn did: one quiet line ("Used 4 tools · 12s") that
/// opens to every tool in order, and each tool with details opens to what
/// it was given and gave back. The owner's activity mode sets whether the
/// list starts open.
fn activity(
    store: Store,
    data: impl Fn() -> (Vec<String>, u64, Vec<ToolCallInfo>) + Clone + 'static,
    live: bool,
) -> View {
    let open = signal(untracked(|| store.activity_mode()) == ActivityMode::Expanded);
    let summary = {
        let data = data.clone();
        move || {
            let (tools, secs, _) = data();
            activity_summary(&tools, secs, live)
        }
    };
    view()
        .gap_1()
        .child(ui::disclosure(summary, open).accessibility_label("Tools this reply used"))
        .child(when(
            move || open.get(),
            move || {
                let data = data.clone();
                view().pl(px(4.0)).gap(px(2.0)).child(dynamic(move || {
                    let (tools, _, calls) = data();
                    Some(untracked(move || {
                        let mut list = view().gap(px(2.0));
                        for (n, tool) in tools.iter().enumerate() {
                            list = list.child(tool_call(n + 1, tool, call_of(&calls, n, tool)));
                        }
                        list
                    }))
                }))
            },
        ))
}

/// The details recorded for the `n`th tool, when they are its and say
/// anything.
fn call_of(calls: &[ToolCallInfo], n: usize, tool: &str) -> Option<ToolCallInfo> {
    calls
        .get(n)
        .filter(|c| c.name == tool && !(c.input.is_empty() && c.output.is_empty()))
        .cloned()
}

/// One tool of the list: its row, and what it did once opened.
fn tool_call(number: usize, tool: &str, call: Option<ToolCallInfo>) -> View {
    let Some(call) = call else {
        return ui::tool_row(number, tool.to_owned(), None);
    };
    let open = signal(false);
    view()
        .child(ui::tool_row(number, tool.to_owned(), Some(open)))
        .child(when(
            move || open.get(),
            move || ui::call_detail(call.input.clone(), call.output.clone()),
        ))
}

/// "Used 4 tools: read_file ×2, web_search, terminal · 12s"; while the
/// turn runs, "2 tools so far: read_file, terminal".
fn activity_summary(tools: &[String], secs: u64, live: bool) -> String {
    let mut counted: Vec<(&str, usize)> = Vec::new();
    for tool in tools {
        match counted.iter_mut().find(|(name, _)| name == tool) {
            Some((_, n)) => *n += 1,
            None => counted.push((tool, 1)),
        }
    }
    let names = counted
        .iter()
        .take(3)
        .map(|(name, n)| {
            if *n > 1 {
                format!("{name} ×{n}")
            } else {
                (*name).to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let more = if counted.len() > 3 { ", …" } else { "" };
    let count = tools.len();
    let noun = if count == 1 { "tool" } else { "tools" };
    let took = match secs {
        0 => String::new(),
        1..60 => format!(" · {secs}s"),
        _ => format!(" · {}m {}s", secs / 60, secs % 60),
    };
    if live {
        format!("{count} {noun} so far: {names}{more}")
    } else {
        format!("Used {count} {noun}: {names}{more}{took}")
    }
}

/// Under a failed or stopped turn: send its message again.
fn retry(store: Store, session: String) -> View {
    view().flex_row().child(ui::btn_icon(
        Glyph::Retry,
        "Retry",
        Variant::Outline,
        Size::Sm,
        move || store.retry(&session),
    ))
}

/// When a message was said, quietly: "3:42 PM".
fn stamp(at: impl Fn() -> u64 + 'static) -> Text {
    text(move || {
        let at = at();
        if at == 0 {
            String::new()
        } else {
            ui::time::clock(at)
        }
    })
    .tabular_nums()
    .text_size(px(type_scale::XS))
    .text_color(colors::FAINT)
}

/// Under a reply: its time and Copy.
fn reply_footer(
    store: Store,
    content: impl Fn() -> String + Clone + 'static,
    at: impl Fn() -> u64 + 'static,
) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(stamp(at))
        // An icon, as chat apps keep it: the reply is what is read.
        .child(ui::icon_btn(Glyph::Copy, "Copy", Size::Sm, move || {
            set_clipboard(content());
            store.say("Copied the reply");
        }))
}

/// "Today", centred between hairlines.
fn day_break(label: impl Fn() -> String + 'static) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_3()
        .py_1()
        .child(view().flex_1().h(px(1.0)).bg(colors::LINE))
        .child(
            text(label)
                .role(AccessibilityRole::Header)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::FAINT),
        )
        .child(view().flex_1().h(px(1.0)).bg(colors::LINE))
}

/// The files sent with a message: pictures in place, other files as chips
/// that open them. `mine`: the owner's (shown at the right).
fn files(store: Store, attached: impl Fn() -> Vec<AttachmentRef> + 'static, mine: bool) -> View {
    view().child(dynamic(move || {
        let files = attached();
        if files.is_empty() {
            return None;
        }
        Some(untracked(move || {
            let mut list = view().flex_row().flex_wrap().gap_2();
            if mine {
                list = list.justify_end();
            }
            for file in files {
                let Some(url) = store.file_url(&file) else {
                    list = list.child(ui::file_chip(file.name.clone(), "not available"));
                    continue;
                };
                let open = url.clone();
                if file.is_image() && idiom() != Idiom::Terminal {
                    let pixels = file.width.zip(file.height);
                    let name = file.name.clone();
                    let id = file.id.clone();
                    list = list.child(ui::picture(
                        url,
                        format!("Picture: {}", file.name),
                        pixels,
                        move || {
                            store.view_picture(ui::Viewed {
                                id: id.clone(),
                                url: open.clone(),
                                name: name.clone(),
                                pixels,
                            });
                        },
                    ));
                } else {
                    list = list.child(
                        button()
                            .accessibility_label(format!("Open {}", file.name))
                            .on_press(move || {
                                open_file(open.clone());
                            })
                            .child(ui::file_chip(file.name.clone(), "")),
                    );
                }
            }
            list
        }))
    }))
}

/// The files attached to the message being written, each with a way to
/// take it off.
fn draft_files(store: Store) -> View {
    view().child(dynamic(move || {
        let files = store.draft_files();
        if files.is_empty() {
            return None;
        }
        Some(untracked(move || {
            let mut list = view()
                .flex_row()
                .flex_wrap()
                .gap_1()
                .px(px(4.0))
                .pt(px(2.0));
            for file in files {
                let number = file.number;
                let status = match &file.state {
                    DraftFileState::Uploading => "attaching…",
                    DraftFileState::Ready(_) => "",
                    DraftFileState::Failed(_) => "failed",
                };
                list = list.child(
                    view()
                        .flex_row()
                        .items_center()
                        .child(ui::file_chip(file.name.clone(), status))
                        .child(ui::icon_btn(
                            Glyph::Close,
                            "Remove the file",
                            Size::Sm,
                            move || store.remove_draft_file(number),
                        )),
                );
            }
            list
        }))
    }))
}

/// Open a message's file in the system's browser or viewer. Called right in
/// the press: a browser opens a new tab only for the tap itself, so the
/// request is not put off to a task (whether it opened is not awaited).
fn open_file(url: String) {
    drop(open_url(url));
}

fn user_bubble(content: impl Fn() -> String + 'static) -> View {
    view()
        .max_w(percentage(0.8))
        .rounded(px(radius::SURFACE))
        .bg(colors::BUBBLE)
        .px_3()
        .py_2()
        .child(
            text(content)
                .selectable(true)
                .text_size(px(ui::fit::read()))
                .line_height(px(ui::fit::read_lh())),
        )
}

fn note(content: impl Fn() -> String + 'static) -> View {
    view().flex_row().gap_2().child(
        text(content)
            .text_size(px(type_scale::SM))
            .text_color(colors::MUTED),
    )
}

/// A reply, as the agent's Markdown. HTML in it is shown as text: what the
/// model writes never becomes markup.
fn markdown(
    content: impl Fn() -> SharedString + 'static,
    streaming: impl Fn() -> bool + 'static,
) -> impl IntoElement {
    text_view()
        .markdown(move || Some(content()))
        .options(move || MarkdownOptions {
            streaming: streaming(),
            html: false,
            ..MarkdownOptions::default()
        })
        .links(TextViewLinks::Open)
        .selectable(true)
        .gap_3()
        .text_size(px(ui::fit::read()))
        .line_height(px(ui::fit::read_lh()))
        .text_color(colors::TEXT)
}

/// The turn running now: its streamed text and what it is doing.
fn live(store: Store, session: String) -> View {
    let (stop, stopping) = (session.clone(), session.clone());
    let streamed = {
        let session = session.clone();
        computed(move || store.live(&session).map(|l| l.streamed).unwrap_or_default())
    };
    let status = computed({
        let session = session.clone();
        move || {
            // Waiting on its owner is the news, not the last tool it ran.
            let waiting = store.approvals_for(&session).len();
            if waiting > 0 {
                return if waiting == 1 {
                    "Waiting for your approval".to_string()
                } else {
                    format!("Waiting for {waiting} approvals")
                };
            }
            store
                .live(&session)
                .and_then(|l| l.status)
                .unwrap_or_else(|| "Working…".into())
        }
    });
    // The tools so far, a line that opens to each in place (what the
    // reply keeps once done): the steps without leaving the chat.
    let so_far = {
        let session = session.clone();
        move || {
            store
                .live(&session)
                .map(|l| (l.tools, 0, l.calls))
                .unwrap_or_default()
        }
    };
    let has_tools = {
        let so_far = so_far.clone();
        computed(move || !so_far().0.is_empty())
    };
    view()
        .gap_2()
        .child(when(has_tools, move || {
            activity(store, so_far.clone(), true)
        }))
        .child(when(
            move || streamed.with(|s| !s.is_empty()),
            move || markdown(move || streamed.get().into(), || true),
        ))
        // What it is doing, and Stop beside it: where the eye already is.
        // Once asked, Stop says so until the turn ends.
        .child(
            // The line, then a quiet Stop at its end.
            view()
                .flex_row()
                .items_center()
                .gap_3()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(working(move || status.get())),
                )
                .child(
                    when(
                        move || store.is_stopping(&stopping),
                        || ui::status("Stopping…", ui::Tone::Warning),
                    )
                    .otherwise(move || {
                        let stop = stop.clone();
                        btn_icon(Glyph::Stop, "Stop", Variant::Ghost, Size::Sm, move || {
                            store.stop(&stop)
                        })
                    }),
                ),
        )
}

fn composer(store: Store, session: Option<String>) -> View {
    // At least one line: shorter, the browser shows a scroll bar for the
    // rest of it (a phone's line is 24 pt).
    let height = signal(ui::fit::read_lh());
    let online = move || store.is_online();
    let can_send = move || {
        online()
            && !store.needs_project()
            && (!store.draft().trim().is_empty()
                || store
                    .draft_files()
                    .iter()
                    .any(|f| matches!(f.state, DraftFileState::Ready(_))))
    };
    let running = {
        let session = session.clone();
        computed(move || session.as_deref().is_some_and(|id| store.is_running(id)))
    };
    let ongoing = session.is_some();
    let field = element_ref();
    let dragging = signal(false);
    // Enter's steer, held a moment so ⌘↩ can make it a queue instead.
    let steer_pending = stored(false);
    let owner = Owner::current();
    let touch = matches!(
        idiom(),
        nsh_core::platform::Idiom::Phone | nsh_core::platform::Idiom::Pad
    );
    let terminal = idiom() == nsh_core::platform::Idiom::Terminal;
    // While the conversation works, Enter adds the message to the work and
    // Mod+Enter queues it for after; Up on an empty composer brings back
    // the last message sent here.
    on_key_with(
        "mod+enter",
        KeyOptions {
            when: nsh_core::keymap::When::Always,
            ..KeyOptions::default()
        },
        move |_| {
            if running.get_untracked() {
                // The field's own submit (see `on_submit`) does not send it.
                steer_pending.replace(false);
                store.submit_placed(Some(other_placement(store)));
            } else {
                store.submit();
            }
        },
    );
    on_key_with(
        "up",
        KeyOptions {
            when: nsh_core::keymap::When::Always,
            pass_through: true,
            ..KeyOptions::default()
        },
        move |_| {
            store.recall_last();
        },
    );
    // Back to the composer when a conversation is picked or a pane
    // closes: typing goes here, not to a list that is gone (a phone keeps
    // its keyboard down until the composer is tapped).
    if !touch {
        effect(move || {
            if store.composer_focus() > 0 {
                untracked(|| field.focus());
            }
        });
    }
    let slash = computed(move || store.slash_matches());
    // A chat's messages from here also go to the chat itself.
    let chat_note = session
        .as_deref()
        .filter(|id| untracked(|| store.source(id)) != crate::store::Source::App)
        .map(|id| untracked(|| store.place(id)));
    // Telegram, Discord or Slack: its mark leads the note, which names the
    // bot without the platform.
    let chat_glyph = session
        .as_deref()
        .and_then(|id| untracked(|| store.channel(id)))
        .and_then(|channel| ui::platform_glyph(&channel))
        .filter(|_| chat_note.is_some());
    let chat_note = chat_note.map(|place| match chat_glyph {
        Some(_) => format!(
            "Also posted in {}",
            place
                .split_once(" · ")
                .map_or(place.as_str(), |(_, rest)| rest)
        ),
        None => format!("Also posted in {place}"),
    });
    view()
        .items_center()
        .px_4()
        .pb_4()
        .pt_2()
        .child(when(
            move || slash.with(|m| !m.is_empty()),
            move || slash_list(store, slash),
        ))
        .child(match session.clone() {
            Some(id) => queued_list(store, id),
            None => view(),
        })
        .child(
            ui::surface()
                .w_full()
                .max_w(px(size::COLUMN))
                .p(px(6.0))
                .gap_1()
                .focus_within(|s| s.border_color(colors::ACCENT))
                // Files dragged from the desktop attach where they are let
                // go; the border says so while they are over it.
                .border_color(move || {
                    if dragging.get() {
                        colors::ACCENT
                    } else {
                        colors::EDGE
                    }
                })
                .on_file_drag_enter(move |drag: &FileDragEvent| {
                    if store.takes_files() && !drag.uris.is_empty() {
                        dragging.set(true);
                    }
                })
                .on_file_drag_leave(move || dragging.set(false))
                .on_file_drop(move |drop: &FileDropEvent| {
                    dragging.set(false);
                    if store.takes_files() {
                        attach_dropped(store, drop.uris.clone(), drop.mime_types.clone());
                    }
                })
                .child(when(
                    move || dragging.get(),
                    || {
                        text("Drop to attach")
                            .px_1()
                            .text_size(px(type_scale::XS))
                            .text_color(colors::ACCENT)
                    },
                ))
                .child(draft_files(store))
                .child(
                    input()
                        .element_ref(&field)
                        // A pasted picture (a screenshot) is attached; its
                        // text, if any, still goes in.
                        .on_paste(move |paste: &PasteEvent| {
                            if !store.takes_files() {
                                return;
                            }
                            for file in &paste.files {
                                store.attach_file(file.name.clone(), file.mime.clone(), &file.data);
                            }
                        })
                        .value(move || store.draft())
                        .on_change(move |text: &str| store.set_draft(text.to_owned()))
                        .multiline(true)
                        // Phones keep Return for new lines; Send sends.
                        .submit_behavior(if touch {
                            SubmitBehavior::Newline
                        } else {
                            SubmitBehavior::Submit
                        })
                        .on_submit(move |_: &str| {
                            if running.get_untracked() {
                                // A browser's field submits on ⌘↩ too, before
                                // the Queue shortcut hears it: wait a moment
                                // for the shortcut to claim it.
                                steer_pending.replace(true);
                                owner.run(|| {
                                    spawn(async move {
                                        timer(std::time::Duration::from_millis(40)).await;
                                        if steer_pending.replace(false) {
                                            store.submit_placed(Some(enter_placement(store)));
                                        }
                                    })
                                });
                            } else {
                                store.submit();
                            }
                        })
                        // A phone would raise its keyboard over the page.
                        .auto_focus(!touch)
                        // A drop of a few seconds is not news: the field
                        // reads as usual, and what is sent goes on reconnect.
                        .placeholder(move || match store.felt() {
                            Felt::Online if store.needs_project() => {
                                "Choose a project first".to_owned()
                            }
                            Felt::Online => {
                                // The field says what to write here.
                                let files = store.draft_files();
                                let ask = if files.iter().any(|f| f.marked_from.is_some()) {
                                    "Say what to do with the marked part"
                                } else if !files.is_empty() {
                                    "Say what to do with it, or just send"
                                } else if running.get() {
                                    "Add to the work or queue what's next"
                                } else if ongoing {
                                    "Reply"
                                } else if store.new_is_code() {
                                    "Describe the change, fix or task"
                                } else {
                                    "Ask anything, or tell it what to do"
                                };
                                // A phone's field is narrow: the hint about
                                // commands is the list that `/` brings.
                                if touch {
                                    ask.to_owned()
                                } else {
                                    format!("{ask} (/ for commands)")
                                }
                            }
                            Felt::Connecting | Felt::NeedsPairing | Felt::NoDaemon => {
                                "Connecting to the daemon…".to_owned()
                            }
                            // Short enough for a phone's field: what is
                            // written now still sends on reconnect.
                            Felt::Reconnecting => "Reconnecting…".to_owned(),
                            Felt::Down { .. } => "Offline: sends once it's back".to_owned(),
                        })
                        .placeholder_color(|| color(colors::FAINT))
                        .caret_color(|| color(colors::ACCENT))
                        .selection_color(|| color(colors::ACCENT))
                        .on_content_size_change(move |size: &nsh_core::geometry::Size<Points>| {
                            height.set(size.height.0.clamp(ui::fit::read_lh(), 180.0));
                        })
                        .h(move || px(height.get() + 16.0))
                        .px(px(8.0))
                        .py(px(8.0))
                        .text_size(px(ui::fit::read()))
                        .line_height(px(ui::fit::read_lh()))
                        .text_color(colors::TEXT)
                        .label("Message"),
                )
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_1()
                        .child(when(
                            move || !terminal && store.takes_files(),
                            move || {
                                let mut ways = Vec::new();
                                if untracked(|| store.takes_pictures()) {
                                    ways.push(ui::MenuAction::new("Take a picture", move || {
                                        store.take_picture()
                                    }));
                                }
                                ways.push(ui::MenuAction::new("Pictures", move || {
                                    store.pick_pictures()
                                }));
                                ways.push(ui::MenuAction::new("Files", move || store.pick_files()));
                                ui::press_menu(Glyph::Attach, "Attach", ways)
                            },
                        ))
                        .child(dictation(store, terminal))
                        // Model and effort, in the same row; a phone's row
                        // has no room for them beside the working hint.
                        .child(when(move || !(touch && running.get()), {
                            let session = session.clone();
                            move || super::model_picker::model_picker(store, session.clone())
                        }))
                        .child(when(
                            move || chat_glyph.is_some() && !running.get() && !terminal,
                            move || {
                                ui::icon(chat_glyph.unwrap_or(Glyph::Chat))
                                    .text_color(colors::FAINT)
                                    .role(AccessibilityRole::Image)
                                    .label(chat_glyph.map_or("", |g| g.platform_name()))
                            },
                        ))
                        .child(
                            text(move || match (running.get(), touch) {
                                (true, true) if store.enter_queues() => {
                                    "Working · Send sends after · Now tells it now".to_owned()
                                }
                                (true, true) => {
                                    "Working · Send tells it now · Queue sends after".to_owned()
                                }
                                (true, false) => {
                                    let (enter, other) = if store.enter_queues() {
                                        ("sends after", "tells it now")
                                    } else {
                                        ("tells it now", "sends after")
                                    };
                                    let stops = if terminal {
                                        format!("Esc or {} stops", key_label("mod+c"))
                                    } else {
                                        "Esc stops".to_owned()
                                    };
                                    format!(
                                        "Working · Enter {enter} · {} {other} · {stops}",
                                        key_label("mod+enter")
                                    )
                                }
                                (false, true) => chat_note.clone().unwrap_or_default(),
                                // Terminals rarely send Shift+Enter; Alt+Enter
                                // works in all of them.
                                (false, false) if terminal => {
                                    format!(
                                        "Enter to send · {} for a new line",
                                        key_label("alt+enter")
                                    )
                                }
                                (false, false) => match &chat_note {
                                    Some(note) => format!("{note} · Enter to send"),
                                    None => "Enter to send · Shift+Enter for a new line".to_owned(),
                                },
                            })
                            .flex_1()
                            .px(px(8.0))
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                        )
                        .child(when(
                            move || running.get(),
                            move || {
                                // The other way: Queue, or Now when Enter queues.
                                view().child(dynamic(move || {
                                    let queues = store.enter_queues();
                                    Some(untracked(move || {
                                        let (label, name) = if queues {
                                            ("Now", "Now: the running work reads it")
                                        } else {
                                            ("Queue", "Queue: send after the running work")
                                        };
                                        ui::btn(label, Variant::Ghost, ui::Size::Sm, move || {
                                            store.submit_placed(Some(other_placement(store)))
                                        })
                                        .disabled(move || !can_send())
                                        .accessibility_label(name)
                                    }))
                                }))
                            },
                        ))
                        .child(
                            {
                                let send = move || {
                                    // A phone answers the press (nothing elsewhere).
                                    haptic(nsh_core::Haptic::Impact(nsh_core::HapticImpact::Light));
                                    if running.get_untracked() {
                                        store.submit_placed(Some(enter_placement(store)));
                                    } else {
                                        store.submit();
                                    }
                                };
                                // The arrow alone says Send; a terminal's
                                // glyph needs the word.
                                if terminal {
                                    ui::btn_icon(
                                        Glyph::Send,
                                        "Send",
                                        Variant::Primary,
                                        ui::Size::Sm,
                                        send,
                                    )
                                } else {
                                    ui::icon_btn_primary(Glyph::Send, "Send", send)
                                }
                            }
                            .disabled(move || !can_send())
                            .accessibility_label(move || {
                                Some(SharedString::from(
                                    if running.get() && store.enter_queues() {
                                        "Send after the running work"
                                    } else if running.get() {
                                        "Send now: the running work reads it"
                                    } else {
                                        "Send"
                                    },
                                ))
                            }),
                        ),
                ),
        )
}

/// Where Enter (and Send) puts a message while the agent works.
fn enter_placement(store: Store) -> MidRun {
    if store.enter_queues() {
        MidRun::Queue
    } else {
        MidRun::Steer
    }
}

/// Where ⌘↩ (and the second button) puts it: the other way.
fn other_placement(store: Store) -> MidRun {
    if store.enter_queues() {
        MidRun::Steer
    } else {
        MidRun::Queue
    }
}

/// Messages sent to wait for the running work, above the composer: each
/// can be edited in place or taken out until the agent takes it.
fn queued_list(store: Store, session: String) -> View {
    let items = {
        let session = session.clone();
        computed(move || store.queued(&session))
    };
    let editing = signal(None::<u64>);
    let draft = signal(String::new());
    view()
        .w_full()
        .max_w(px(size::COLUMN))
        .gap_1()
        .pb_1()
        .child(when(
            move || !items.with(Vec::is_empty),
            || ui::section_label("Waiting to send"),
        ))
        .child(each(
            items,
            |m| m.id,
            move |message, _| {
                let id = message.with_untracked(|m| m.id);
                let (save_session, remove_session) = (session.clone(), session.clone());
                view().child(
                    when(
                        move || editing.get() == Some(id),
                        move || {
                            let save_session = save_session.clone();
                            view()
                                .gap_1()
                                .child(ui::area("What to send", draft).label("Waiting message"))
                                .child(
                                    view()
                                        .flex_row()
                                        .gap_1()
                                        .child(ui::btn(
                                            "Save",
                                            Variant::Primary,
                                            ui::Size::Sm,
                                            move || {
                                                editing.set(None);
                                                store.edit_queued(
                                                    &save_session,
                                                    id,
                                                    Some(draft.get_untracked()),
                                                );
                                            },
                                        ))
                                        .child(ui::btn(
                                            "Cancel",
                                            Variant::Ghost,
                                            ui::Size::Sm,
                                            move || editing.set(None),
                                        )),
                                )
                        },
                    )
                    .otherwise(move || {
                        let remove_session = remove_session.clone();
                        ui::surface()
                            .flex_row()
                            .items_center()
                            .gap_2()
                            .px_3()
                            .py_1()
                            .child(ui::icon(Glyph::Clock).text_color(colors::FAINT))
                            // Read by the running work, or run after it.
                            .child(dynamic(move || {
                                let after = message.with(|m| m.after);
                                Some(untracked(move || {
                                    ui::with_hint(
                                        if after {
                                            "Runs as its own turn when the work ends"
                                        } else {
                                            "The running work reads it at its next step"
                                        },
                                        ui::badge(
                                            if after { "After" } else { "Now" },
                                            if after {
                                                ui::Tone::Neutral
                                            } else {
                                                ui::Tone::Accent
                                            },
                                        ),
                                    )
                                }))
                            }))
                            .child(
                                text(move || message.with(|m| m.text.clone()))
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .number_of_lines(2)
                                    .text_size(px(type_scale::SM))
                                    .text_color(colors::MUTED),
                            )
                            .child(ui::btn("Edit", Variant::Ghost, ui::Size::Sm, move || {
                                draft.set(message.with_untracked(|m| m.text.clone()));
                                editing.set(Some(id));
                            }))
                            .child(ui::icon_btn(
                                Glyph::Close,
                                "Remove from the queue",
                                ui::Size::Sm,
                                move || store.edit_queued(&remove_session, id, None),
                            ))
                    }),
                )
            },
        ))
}

/// The slash commands completing what is typed. Tab takes the first.
fn slash_list(store: Store, matches: Memo<Vec<aidaemon_client_protocol::SlashCommand>>) -> View {
    on_key_with(
        "tab",
        KeyOptions {
            when: nsh_core::keymap::When::Always,
            ..KeyOptions::default()
        },
        move |_| {
            if let Some(first) = matches.with_untracked(|m| m.first().cloned()) {
                store.complete_slash(&first.name);
            }
        },
    );
    ui::surface()
        .w_full()
        .max_w(px(size::COLUMN))
        .mb_1()
        .p_1()
        .child(each(
            move || matches.get(),
            |c: &aidaemon_client_protocol::SlashCommand| c.name.clone(),
            move |c, index| {
                let name = c.get_untracked().name.clone();
                let description = c.get_untracked().description.clone();
                let pick = name.clone();
                ui::row(|| false, move || store.complete_slash(&pick))
                    .accessibility_label(format!("/{name}: {description}"))
                    .child(ui::code(format!("/{name}")))
                    .child(
                        text(description)
                            .flex_1()
                            .number_of_lines(1)
                            .text_size(px(type_scale::SM))
                            .text_color(colors::MUTED),
                    )
                    .child(when(move || index.get() == 0, || ui::kbd("Tab")))
            },
        ))
}

/// Attaches files dropped from the desktop: read off the engine's thread,
/// each by its name, as the paperclip's are. Directories and files that
/// cannot be read say so.
#[cfg(not(target_arch = "wasm32"))]
fn attach_dropped(store: Store, uris: Vec<String>, mimes: Vec<String>) {
    spawn(async move {
        let read = background(async move {
            uris.iter()
                .zip(mimes.iter().chain(std::iter::repeat(&String::new())))
                .map(|(uri, mime)| {
                    let path = file_path(uri);
                    let name = std::path::Path::new(&path)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.clone());
                    let bytes = std::fs::metadata(&path)
                        .ok()
                        .filter(|m| m.is_file())
                        .and_then(|_| std::fs::read(&path).ok());
                    (name, mime.clone(), bytes)
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap_or_default();
        for (name, mime, bytes) in read {
            match bytes {
                Some(bytes) => {
                    let mime = if mime.is_empty() {
                        "application/octet-stream".to_owned()
                    } else {
                        mime
                    };
                    store.attach_file(name, mime, &bytes)
                }
                None => store.say(format!(
                    "{name} could not be attached (a folder, or unreadable)"
                )),
            }
        }
    });
}

/// The browser hands no paths: nothing to read.
#[cfg(target_arch = "wasm32")]
fn attach_dropped(_store: Store, _uris: Vec<String>, _mimes: Vec<String>) {}

/// The local path of a `file:` URL, its `%XX` escapes decoded.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn file_path(uri: &str) -> String {
    let rest = uri.strip_prefix("file://").unwrap_or(uri);
    // `file://localhost/…` and `file:///…` both name a local path.
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    // Windows: `/C:/…` is `C:/…`.
    let rest = match rest.as_bytes() {
        [b'/', drive, b':', ..] if drive.is_ascii_alphabetic() => &rest[1..],
        _ => rest,
    };
    let bytes = rest.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&rest[i + 1..i + 3], 16)
        {
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The microphone: speaking writes into the composer, after what is
/// already there, as the platform's recognizer hears it. Pressing it again
/// stops listening (the last words are kept). Shown only where the
/// platform recognizes speech.
fn dictation(store: Store, terminal: bool) -> View {
    use std::{cell::RefCell, rc::Rc};

    use nsh_core::element::SpeechRecognition;
    use nsh_core::speech::{SpeechEvent, SpeechRequest};

    let available = signal(false);
    if !terminal {
        spawn(async move { available.set(speech_available().await) });
    }
    let listening = signal(false);
    let session: Stored<Option<Rc<RefCell<SpeechRecognition>>>> = stored(None);
    // The session outlives the button pressed to start it (the button
    // becomes Stop): it runs under the composer.
    let owner = Owner::current();
    let press = move || {
        if listening.get_untracked() {
            // The recognizer finishes what it heard, then the session ends.
            session.with(|s| {
                if let Some(s) = s {
                    s.borrow().stop()
                }
            });
            return;
        }
        let before = store.draft();
        let lead = if before.trim().is_empty() {
            String::new()
        } else {
            format!("{} ", before.trim_end())
        };
        let running = Rc::new(RefCell::new(recognize_speech(SpeechRequest::default())));
        session.replace(Some(running.clone()));
        listening.set(true);
        owner.run(|| {
            spawn(async move {
                loop {
                    let event = std::future::poll_fn(|cx| running.borrow_mut().poll_next(cx)).await;
                    match event {
                        Some(SpeechEvent::Result { text, .. }) => {
                            store.set_draft(format!("{lead}{text}"))
                        }
                        Some(SpeechEvent::End { error }) => {
                            if let Some(note) = error.and_then(speech_note) {
                                store.say(note);
                            }
                            break;
                        }
                        None => break,
                        Some(_) => {}
                    }
                }
                session.replace(None);
                listening.set(false);
            })
        });
    };
    view().child(when(
        move || available.get(),
        move || {
            view().child(dynamic(move || {
                let on = listening.get();
                Some(untracked(move || {
                    if on {
                        ui::icon_btn(Glyph::Stop, "Stop listening", Size::Md, press)
                            .text_color(colors::ACCENT)
                    } else {
                        ui::icon_btn(Glyph::Mic, "Speak instead of typing", Size::Md, press)
                    }
                }))
            }))
        },
    ))
}

/// What to tell the owner when listening failed; nothing when it only
/// heard nothing.
fn speech_note(error: nsh_core::speech::SpeechError) -> Option<&'static str> {
    use nsh_core::speech::SpeechError;
    match error {
        SpeechError::NoSpeech => None,
        SpeechError::NotAllowed => Some(
            "The microphone or speech recognition is not allowed for AIdaemon in the system's settings",
        ),
        SpeechError::AudioCapture => Some("No microphone could be used"),
        SpeechError::LanguageNotSupported => Some("Speech in this language is not recognized here"),
        _ => Some("Listening stopped: speech recognition failed"),
    }
}

#[cfg(test)]
mod drop_tests {
    use super::file_path;

    #[test]
    fn file_urls_become_paths() {
        assert_eq!(file_path("file:///tmp/a%20b.png"), "/tmp/a b.png");
        assert_eq!(file_path("file://localhost/tmp/x"), "/tmp/x");
        assert_eq!(
            file_path("file:///C:/Users/synthetic/n%C3%A9.txt"),
            "C:/Users/synthetic/né.txt"
        );
        assert_eq!(file_path("file:///tmp/100%"), "/tmp/100%");
    }
}
