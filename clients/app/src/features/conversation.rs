//! A conversation: its transcript (Markdown replies, streamed as they
//! arrive), the approvals it waits on, and the composer. A new
//! conversation is the same page with no session yet.

use aidaemon_client_core::state::Line;
use nsh_base::{MarkdownOptions, TextViewLinks, input, text_view};
use nsh_core::prelude::*;

use super::approvals;
use crate::store::{Connection, Store};
use crate::ui::{self, Glyph, Size, Variant, btn_icon, colors, radius, size, type_scale, working};

/// A row of the transcript.
#[derive(Debug, Clone, PartialEq)]
enum Row {
    User(String),
    Assistant(String),
    Note(String),
    /// The reply being written and what the agent is doing.
    Live,
    Approval(String),
}

impl Row {
    fn key(&self, index: usize) -> String {
        match self {
            Self::Live => "live".into(),
            Self::Approval(id) => format!("approval-{id}"),
            _ => format!("line-{index}"),
        }
    }
}

fn rows(store: Store, session: &str) -> Vec<Row> {
    let mut rows: Vec<Row> = store
        .lines(session)
        .into_iter()
        .map(|line| match line {
            Line::User { text, .. } => Row::User(text),
            Line::Assistant { text } => Row::Assistant(text),
            Line::Note { text } => Row::Note(text),
        })
        .collect();
    if store.live(session).is_some() {
        rows.push(Row::Live);
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
        .child(composer(store, session))
}

fn header(store: Store, session: Option<String>) -> View {
    let title = session.clone();
    let running = session.clone();
    let stop = session.clone();
    view()
        .h(px(44.0))
        .flex_row()
        .items_center()
        .gap_2()
        .px_4()
        .border_b_1()
        .border_color(colors::LINE)
        .child(
            text(move || match &title {
                Some(id) => store.title(id),
                None => "New conversation".into(),
            })
            .flex_1()
            .number_of_lines(1)
            .text_size(px(type_scale::BASE))
            .font_weight(FontWeight::SEMIBOLD),
        )
        .child(when(
            move || running.as_deref().is_some_and(|id| store.is_running(id)),
            move || {
                let stop = stop.clone();
                btn_icon(Glyph::Stop, "Stop", Variant::Outline, Size::Sm, move || {
                    if let Some(id) = &stop {
                        store.stop(id);
                    }
                })
            },
        ))
}

fn welcome() -> View {
    ui::empty(
        "What should we work on?",
        "Ask anything, start a coding task, or hand off work to run in the background. Approvals and live progress show up here as the agent works.",
    )
}

fn transcript(store: Store, session: String) -> VirtualList {
    let id = session.clone();
    let all = computed(move || rows(store, &id));
    let count = all.with_untracked(Vec::len);
    virtual_list(
        move || all.with(Vec::len),
        move |at| {
            let row = computed(move || all.with(|r| r.get(at.index()).cloned()));
            let kind = computed(move || {
                row.with(|r| {
                    r.as_ref().map(|r| match r {
                        Row::User(_) => 0,
                        Row::Assistant(_) => 1,
                        Row::Note(_) => 2,
                        Row::Live => 3,
                        Row::Approval(_) => 4,
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
                            Some(Row::User(t) | Row::Assistant(t) | Row::Note(t)) => t.clone(),
                            _ => String::new(),
                        })
                    };
                    match kind {
                        0 => column.items_end().child(user_bubble(content)),
                        1 => column.child(markdown(move || content().into(), || false)),
                        2 => column.child(note(content)),
                        3 => column.child(live(store, session)),
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
    .estimated_item_size(px(96.0))
    .initial_scroll_index(count.saturating_sub(1))
    .follow_end(px(48.0))
    .content_padding_top(px(12.0))
    .content_padding_bottom(px(12.0))
    .flex_1()
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
                .text_size(px(type_scale::READ))
                .line_height(px(type_scale::LH_READ)),
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
        .text_size(px(type_scale::READ))
        .line_height(px(type_scale::LH_READ))
        .text_color(colors::TEXT)
}

/// The turn running now: its streamed text and what it is doing.
fn live(store: Store, session: String) -> View {
    let streamed = {
        let session = session.clone();
        computed(move || store.live(&session).map(|l| l.streamed).unwrap_or_default())
    };
    let status = computed(move || {
        store
            .live(&session)
            .and_then(|l| l.status)
            .unwrap_or_else(|| "Working…".into())
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
    view().items_center().px_4().pb_4().pt_2().child(
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
                    .submit_behavior(SubmitBehavior::Submit)
                    .on_submit(move |_: &str| store.submit())
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
                    .text_size(px(type_scale::READ))
                    .line_height(px(type_scale::LH_READ))
                    .text_color(colors::TEXT)
                    .label("Message"),
            )
            .child(
                view()
                    .flex_row()
                    .items_center()
                    .gap_1()
                    .child(
                        text(move || {
                            if running.get() {
                                "Working · Esc stops".to_owned()
                            } else {
                                "Enter to send · Shift+Enter for a new line".to_owned()
                            }
                        })
                        .flex_1()
                        .px(px(8.0))
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT),
                    )
                    .child(
                        ui::btn_icon(
                            Glyph::Send,
                            "Send",
                            Variant::Primary,
                            ui::Size::Sm,
                            move || store.submit(),
                        )
                        .disabled(move || !can_send())
                        .accessibility_label("Send"),
                    ),
            ),
    )
}
