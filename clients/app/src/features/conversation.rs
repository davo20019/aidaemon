//! A conversation: its transcript (Markdown replies, streamed as they
//! arrive), the approvals it waits on, and the composer. A new
//! conversation is the same page with no session yet.

use aidaemon_client_core::state::Line;
use aidaemon_client_protocol::MidRun;
use nsh_base::{MarkdownOptions, TextViewLinks, input, text_view};
use nsh_core::prelude::*;

use super::approvals;
use crate::store::{ActivityMode, Connection, Store};
use crate::ui::{self, Glyph, Size, Variant, btn_icon, colors, radius, size, type_scale, working};

/// A row of the transcript.
#[derive(Debug, Clone, PartialEq)]
enum Row {
    /// A day break: "Today", "Yesterday", "Mar 3".
    Day(String),
    User(String, u64),
    Assistant(String, u64),
    Note(String),
    /// The reply being written and what the agent is doing.
    Live,
    Approval(String),
    /// The tools a reply's turn used, and how many seconds it took.
    Activity(Vec<String>, u64),
    /// The last turn failed or was stopped: send its message again.
    Retry,
}

impl Row {
    fn key(&self, index: usize) -> String {
        match self {
            Self::Live => "live".into(),
            Self::Retry => "retry".into(),
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
            Line::User { text, at_ms, .. } => Row::User(text, at_ms),
            Line::Assistant { text, at_ms } => Row::Assistant(text, at_ms),
            Line::Note { text } => Row::Note(text),
            Line::Activity { .. } if mode == ActivityMode::Hidden => continue,
            Line::Activity { tools, secs } => Row::Activity(tools, secs),
        });
    }
    if store.live(session).is_some() {
        rows.push(Row::Live);
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
        None => welcome().into_element(),
    };
    view()
        .flex_1()
        .min_w(px(0.0))
        .child(header(store, session.clone()))
        .child(view().flex_1().child(body))
        .child(notices(store))
        .child(composer(store, session))
}

/// What the agent's view moves say: a move offered while the owner types,
/// and a short note after one was taken.
fn notices(store: Store) -> View {
    // A note leaves by itself after a few seconds.
    effect(move || {
        if store.toast().is_some() {
            set_timeout(std::time::Duration::from_secs(6), move || {
                store.dismiss_toast()
            });
        }
    });
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
        .child(when(
            move || store.toast().is_some() && store.suggestion().is_none(),
            move || {
                ui::banner(move || store.toast().unwrap_or_default(), ui::Tone::Neutral).child(
                    ui::btn("Dismiss", Variant::Ghost, ui::Size::Sm, move || {
                        store.dismiss_toast()
                    }),
                )
            },
        ))
}

fn header(store: Store, session: Option<String>) -> View {
    let title = session.clone();
    let running = session.clone();
    let stop = session.clone();
    let waiting = computed(move || store.approvals().len());
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
                ui::icon_btn(Glyph::Sidebar, "Conversations", Size::Md, move || {
                    store.open_sidebar()
                })
            },
        ))
        .child(
            text(move || match &title {
                Some(id) => store.title(id),
                None => "New conversation".into(),
            })
            .flex_1()
            .px_2()
            .role(AccessibilityRole::Header)
            .number_of_lines(1)
            .text_size(px(ui::fit::body()))
            .font_weight(FontWeight::SEMIBOLD),
        )
        .child(when(
            move || running.as_deref().is_some_and(|id| store.is_running(id)),
            move || {
                let stopping = {
                    let stop = stop.clone();
                    move || stop.as_deref().is_some_and(|id| store.is_stopping(id))
                };
                let stop = stop.clone();
                let press = move || {
                    if let Some(id) = &stop {
                        store.stop(id);
                    }
                };
                // A phone's bar keeps room for the title: Stop is a glyph.
                // Once asked, it says so until the turn ends.
                when(stopping, || ui::status("Stopping…", ui::Tone::Warning)).otherwise(move || {
                    let press = press.clone();
                    if ui::touch() || store.is_compact() {
                        ui::icon_btn(Glyph::Stop, "Stop", Size::Md, press).into_element()
                    } else {
                        btn_icon(Glyph::Stop, "Stop", Variant::Outline, Size::Sm, press)
                            .into_element()
                    }
                })
            },
        ))
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
                ui::icon_btn(Glyph::Plus, "New conversation", Size::Md, move || {
                    store.select(None)
                })
            },
        ))
        .child(when(
            move || store.is_compact() || !store.inspector_open(),
            move || {
                ui::icon_btn_count(
                    Glyph::Approval,
                    "Activity",
                    move || waiting.get(),
                    move || store.open_inspector(),
                )
            },
        ))
}

fn welcome() -> View {
    ui::empty(
        "What should we work on?",
        "Ask anything, start a coding task, or hand off work to run in the background. Approvals and live progress show up here as the agent works.",
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
                                Row::User(t, _) | Row::Assistant(t, _) | Row::Note(t) | Row::Day(t),
                            ) => t.clone(),
                            _ => String::new(),
                        })
                    };
                    let at = move || {
                        row.with(|r| match r.as_ref() {
                            Some(Row::User(_, at) | Row::Assistant(_, at)) => *at,
                            _ => 0,
                        })
                    };
                    match kind {
                        0 => column
                            .items_end()
                            .gap_1()
                            .child(user_bubble(content))
                            .child(stamp(at)),
                        1 => column
                            .gap_1()
                            .child(markdown(move || content().into(), || false))
                            .child(reply_footer(store, content, at)),
                        2 => column.child(note(content)),
                        5 => column.child(day_break(content)),
                        3 => column.child(live(store, session)),
                        6 => column.child(activity(store, move || {
                            row.with(|r| match r {
                                Some(Row::Activity(tools, secs)) => (tools.clone(), *secs),
                                _ => (Vec::new(), 0),
                            })
                        })),
                        7 => column.child(retry(store, session)),
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
/// opens to every tool in order. The owner's activity mode sets whether it
/// starts open.
fn activity(store: Store, data: impl Fn() -> (Vec<String>, u64) + Clone + 'static) -> View {
    let open = signal(untracked(|| store.activity_mode()) == ActivityMode::Expanded);
    let summary = {
        let data = data.clone();
        move || {
            let (tools, secs) = data();
            activity_summary(&tools, secs)
        }
    };
    view()
        .gap_1()
        .child(ui::disclosure(summary, open).accessibility_label("Tools this reply used"))
        .child(when(
            move || open.get(),
            move || {
                let data = data.clone();
                view().pl(px(22.0)).gap(px(2.0)).child(dynamic(move || {
                    let (tools, _) = data();
                    Some(untracked(move || {
                        let mut list = view().gap(px(2.0));
                        for (n, tool) in tools.iter().enumerate() {
                            list = list.child(
                                view()
                                    .flex_row()
                                    .gap_2()
                                    .child(
                                        text(format!("{}.", n + 1))
                                            .text_size(px(type_scale::XS))
                                            .text_color(colors::FAINT),
                                    )
                                    .child(ui::code(tool.clone()).text_color(colors::MUTED)),
                            );
                        }
                        list
                    }))
                }))
            },
        ))
}

/// "Used 4 tools: read_file ×2, web_search, terminal · 12s".
fn activity_summary(tools: &[String], secs: u64) -> String {
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
    format!("Used {count} {noun}: {names}{more}{took}")
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
        .child(ui::btn_icon(
            Glyph::Copy,
            "Copy",
            Variant::Ghost,
            Size::Sm,
            move || {
                set_clipboard(content());
                store.say("Copied the reply");
            },
        ))
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
    let streamed = {
        let session = session.clone();
        computed(move || store.live(&session).map(|l| l.streamed).unwrap_or_default())
    };
    let status = computed(move || {
        let live = store.live(&session);
        let status = live
            .as_ref()
            .and_then(|l| l.status.clone())
            .unwrap_or_else(|| "Working…".into());
        match live.map(|l| l.tools.len()).unwrap_or(0) {
            0 => status,
            1 => format!("{status}  ·  1 tool so far"),
            n => format!("{status}  ·  {n} tools so far"),
        }
    });
    view()
        .gap_2()
        .child(when(
            move || streamed.with(|s| !s.is_empty()),
            move || markdown(move || streamed.get().into(), || true),
        ))
        .child(working(move || status.get()))
}

fn composer(store: Store, session: Option<String>) -> View {
    let height = signal(22.0_f32);
    let online = move || store.is_online();
    let can_send = move || online() && !store.draft().trim().is_empty();
    let running = {
        let session = session.clone();
        computed(move || session.as_deref().is_some_and(|id| store.is_running(id)))
    };
    let field = element_ref();
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
                store.submit_placed(Some(MidRun::Queue));
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
    let slash = computed(move || store.slash_matches());
    view()
        .items_center()
        .px_4()
        .pb_4()
        .pt_2()
        .child(when(
            move || slash.with(|m| !m.is_empty()),
            move || slash_list(store, slash),
        ))
        .child(
            ui::surface()
                .w_full()
                .max_w(px(size::COLUMN))
                .p(px(6.0))
                .gap_1()
                .focus_within(|s| s.border_color(colors::ACCENT))
                .child(
                    input()
                        .element_ref(&field)
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
                                store.submit_placed(Some(MidRun::Steer));
                            } else {
                                store.submit();
                            }
                        })
                        .auto_focus(true)
                        .placeholder(move || match store.connection() {
                            Connection::Online => "Message AIdaemon… (/ for commands)".to_owned(),
                            Connection::Connecting | Connection::NeedsPairing => {
                                "Connecting to the daemon…".to_owned()
                            }
                            Connection::Offline { .. } => {
                                "Offline. Messages send once the daemon is back.".to_owned()
                            }
                        })
                        .placeholder_color(|| color(colors::FAINT))
                        .caret_color(|| color(colors::ACCENT))
                        .selection_color(|| color(colors::ACCENT))
                        .on_content_size_change(move |size: &nsh_core::geometry::Size<Points>| {
                            height.set(size.height.0.clamp(22.0, 180.0));
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
                        .child(
                            text(move || match (running.get(), touch) {
                                (true, true) => {
                                    "Working · Add joins the work, Queue waits".to_owned()
                                }
                                (true, false) if terminal => format!(
                                    "Working · Enter adds to it · {} queues · Esc or {} stops",
                                    key_label("mod+enter"),
                                    key_label("mod+c")
                                ),
                                (true, false) => format!(
                                    "Working · Enter adds to it · {} queues · Esc stops",
                                    key_label("mod+enter")
                                ),
                                (false, true) => String::new(),
                                // Terminals rarely send Shift+Enter; Alt+Enter
                                // works in all of them.
                                (false, false) if terminal => {
                                    format!(
                                        "Enter to send · {} for a new line",
                                        key_label("alt+enter")
                                    )
                                }
                                (false, false) => {
                                    "Enter to send · Shift+Enter for a new line".to_owned()
                                }
                            })
                            .flex_1()
                            .px(px(8.0))
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                        )
                        .child(when(
                            move || running.get(),
                            move || {
                                ui::btn("Queue", Variant::Ghost, ui::Size::Sm, move || {
                                    store.submit_placed(Some(MidRun::Queue))
                                })
                                .disabled(move || !can_send())
                                .accessibility_label("Queue: send after the current work")
                            },
                        ))
                        .child(
                            ui::btn_icon(
                                Glyph::Send,
                                "Send",
                                Variant::Primary,
                                ui::Size::Sm,
                                move || {
                                    if running.get_untracked() {
                                        store.submit_placed(Some(MidRun::Steer));
                                    } else {
                                        store.submit();
                                    }
                                },
                            )
                            .disabled(move || !can_send())
                            .accessibility_label(move || {
                                Some(SharedString::from(if running.get() {
                                    "Add to the current work"
                                } else {
                                    "Send"
                                }))
                            }),
                        ),
                ),
        )
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
