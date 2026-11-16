//! The app's look over `nsh_base`'s behaviour-only parts: switches,
//! selects, sections, tabs, tables, menus, confirmations, toasts and
//! resizable panes. `nsh_base` owns the keyboard, focus, roles and the
//! terminal's rendering; these add the tokens. Features call these and
//! never style a base part themselves.

use std::time::Duration;

use nsh_base::{
    Anchor, CommandEntry, CommandGroup, CommandItem, SelectEntry, SelectItem, SelectValue,
    ToastManager, ToastMotion, ToastOptions, ToastStackState, accordion, accordion_header,
    accordion_item, accordion_panel, accordion_trigger, alert_dialog, command, context_menu,
    dialog, h_resizable, menu, menu_item, popover, progress, resizable_panel, select, sheet,
    switch, tab, table, table_body, table_cell, table_head, table_header, table_row, tabs, toast,
    toast_stack,
};
use nsh_core::prelude::*;
use nsh_core::time::now;

use super::controls::{Glyph, icon, with_hint};
use super::motion;
use super::theme::{colors, fit, radius, text as type_scale};

/// An on/off setting that takes effect at once: a track and thumb, then
/// `label`. The owner keeps the value; `change` gets the asked-for one.
pub fn toggle(
    label: impl Into<SharedString>,
    on: impl Fn() -> bool + Copy + 'static,
    change: impl Fn(bool) + 'static,
) -> nsh_base::SwitchElement {
    let label = label.into();
    switch_only(label.clone(), on, change).child(
        text(label)
            .text_size(px(fit::small()))
            .text_color(colors::TEXT),
    )
}

/// [`toggle`] without its words, for a row that names the setting itself;
/// `label` is still what assistive technology reads.
pub fn switch_only(
    label: impl Into<SharedString>,
    on: impl Fn() -> bool + Copy + 'static,
    change: impl Fn(bool) + 'static,
) -> nsh_base::SwitchElement {
    let label = label.into();
    switch()
        .checked(on)
        .accessibility_label(label.clone())
        .hit_slop(EdgeInsets::all(px(6.0)))
        .flex_row()
        .items_center()
        .gap_2()
        .track_style(move |s| {
            s.w(px(30.0))
                .h(px(18.0))
                .rounded_full()
                .p(px(2.0))
                .bg(if on() { colors::ACCENT } else { colors::BORDER })
                .transition_colors()
                .duration(motion::FAST)
        })
        .thumb_style(move |s| {
            s.w(px(14.0))
                .h(px(14.0))
                .rounded_full()
                .bg(colors::SURFACE)
                .shadow_sm()
                .translate_x(px(if on() { 12.0 } else { 0.0 }))
                .transition_transform()
                .duration(motion::BASE)
        })
        .disabled_style(|s| s.opacity(0.45))
        .on_change(move |next: &bool| change(*next))
}

/// One value from a list: a trigger that shows the chosen title (or
/// `placeholder`, faint), opening a list (a sheet on phones). `items` are
/// (value, title). `searchable` adds a filter field above them.
pub fn choice(
    label: impl Into<SharedString>,
    placeholder: impl Into<SharedString>,
    items: Vec<(String, String)>,
    value: impl Fn() -> Option<String> + Copy + 'static,
    change: impl Fn(Option<String>) + 'static,
) -> nsh_base::SelectElement {
    choice_sized(label, placeholder, items, value, change, false)
}

/// [`choice`] as a small, borderless trigger, for a toolbar row (the
/// composer's model and effort).
pub fn choice_compact(
    label: impl Into<SharedString>,
    placeholder: impl Into<SharedString>,
    items: Vec<(String, String)>,
    value: impl Fn() -> Option<String> + Copy + 'static,
    change: impl Fn(Option<String>) + 'static,
) -> nsh_base::SelectElement {
    choice_sized(label, placeholder, items, value, change, true)
}

fn choice_sized(
    label: impl Into<SharedString>,
    placeholder: impl Into<SharedString>,
    items: Vec<(String, String)>,
    value: impl Fn() -> Option<String> + Copy + 'static,
    change: impl Fn(Option<String>) + 'static,
    compact: bool,
) -> nsh_base::SelectElement {
    let label = label.into();
    let placeholder = placeholder.into();
    let titles = items.clone();
    let shown = move || {
        value().and_then(|v| {
            titles
                .iter()
                .find(|(id, _)| *id == v)
                .map(|(_, title)| title.clone())
                .or(Some(v))
        })
    };
    let chosen = shown.clone();
    let searchable = items.len() > 8;
    select()
        .items(
            items
                .into_iter()
                .map(|(id, title)| SelectItem::new(id, title))
                .collect::<Vec<_>>(),
        )
        .value(move || SelectValue::Controlled(value().map(SharedString::from)))
        .placeholder(placeholder.clone())
        .searchable(searchable)
        .search_placeholder("Type to find one")
        .search_accessibility_label(format!("Find {label}"))
        .accessibility_label(label)
        .trigger(
            view()
                .h(px(match (super::touch(), compact) {
                    (true, _) => 44.0,
                    (false, true) => 26.0,
                    (false, false) => 30.0,
                }))
                .min_w(px(if compact { 0.0 } else { 180.0 }))
                .max_w(px(if compact { 220.0 } else { f32::MAX }))
                .flex_row()
                .items_center()
                .gap(px(if compact { 4.0 } else { 8.0 }))
                .px(px(if compact { 6.0 } else { 12.0 }))
                .rounded(px(radius::CONTROL))
                .border_1()
                .border_color(if compact {
                    colors::SURFACE
                } else {
                    colors::EDGE
                })
                .bg(colors::SURFACE)
                .transition_colors()
                .duration(motion::FAST)
                .hover(|s| s.border_color(colors::MUTED))
                .focus_visible(|s| s.border_color(colors::ACCENT))
                .focus(super::controls::focus_band)
                .child(
                    text(move || {
                        chosen()
                            .map(SharedString::from)
                            .unwrap_or_else(|| placeholder.clone())
                    })
                    .flex_1()
                    .number_of_lines(1)
                    .text_size(px(if compact {
                        type_scale::XS
                    } else {
                        fit::small()
                    }))
                    .text_color(move || {
                        if shown().is_some() && !compact {
                            colors::TEXT
                        } else {
                            colors::FAINT
                        }
                    }),
                )
                .child(
                    text("▾")
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT)
                        .accessibility_elements_hidden(true),
                ),
        )
        // A window's list hangs from its trigger; a phone's is a sheet
        // over the dimmed page, as the pickers' are.
        .popup_style(|s| {
            if super::touch() {
                s.bg(colors::SURFACE)
                    .rounded_t(px(radius::SURFACE))
                    .shadow_lg()
                    .p_2()
                    .pb_safe_area()
                    .max_h(percentage(0.72))
            } else {
                s.bg(colors::SURFACE)
                    .border_1()
                    .border_color(colors::EDGE)
                    .rounded(px(radius::CONTROL))
                    .shadow_lg()
                    .p_1()
                    .max_h(px(320.0))
            }
        })
        .backdrop_style(|s| s.bg(colors::SCRIM))
        .search_class("px-2 py-1.5 mb-1 text-sm border-b border-border")
        .item_class("rounded-md focus:bg-accent selected:bg-accent")
        .render_item(|e: &SelectEntry| {
            text(e.title.clone())
                .px_2()
                .py(px(if super::touch() { 14.0 } else { 5.0 }))
                .number_of_lines(1)
                .text_size(px(fit::small()))
                .text_color(colors::TEXT)
        })
        .empty(
            text("No match")
                .px_2()
                .py_1()
                .text_size(px(fit::small()))
                .text_color(colors::FAINT),
        )
        .on_change(move |next: &Option<SharedString>| change(next.as_ref().map(|v| v.to_string())))
}

/// A list of sections that open and close (several at once): each a
/// heading row with a turning chevron over its content. `open` holds the
/// open sections' ids.
pub fn sections(open: Signal<Vec<SharedString>>, items: Vec<(String, String, View)>) -> View {
    let mut list = accordion()
        .multiple(true)
        .value(open)
        .on_change(move |next: &Vec<SharedString>| open.set(next.clone()))
        .gap_1();
    for (id, title, content) in items {
        let key = SharedString::from(id.clone());
        let expanded = key.clone();
        list = list.child(
            accordion_item()
                .value(key)
                .gap_1()
                .child(
                    accordion_header().level(3).child(
                        accordion_trigger()
                            .flex_row()
                            .items_center()
                            .gap_1()
                            .min_h(px(fit::row() - 6.0))
                            .px_1()
                            .rounded(px(radius::SMALL))
                            .transition_colors()
                            .duration(motion::FAST)
                            .hover(|s| s.bg(colors::HOVER))
                            .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
                            .focus(super::controls::focus_band)
                            .child(
                                text(move || {
                                    if open.with(|o| o.contains(&expanded)) {
                                        "▾"
                                    } else {
                                        "▸"
                                    }
                                })
                                .w(px(fit::icon_slot()))
                                .text_center()
                                .text_size(px(type_scale::XS))
                                .text_color(colors::FAINT)
                                .accessibility_elements_hidden(true),
                            )
                            .child(
                                text(title.clone())
                                    .number_of_lines(1)
                                    .text_size(px(type_scale::SM))
                                    .text_color(colors::MUTED),
                            ),
                    ),
                )
                .child(
                    accordion_panel()
                        .reveal(true)
                        .accessibility_label(title)
                        .child(content),
                ),
        );
    }
    view().child(list)
}

/// A row of tabs over one page at a time: `labels` in order, `selected`
/// the open one's index.
pub fn tab_bar(selected: Signal<usize>, labels: &[&'static str]) -> View {
    let mut bar = tabs()
        .flex_row()
        .gap_1()
        .border_b_1()
        .border_color(colors::LINE)
        .indicator_style(|s| s.bottom(px(-1.0)).h(px(2.0)).bg(colors::ACCENT));
    for (index, label) in labels.iter().copied().enumerate() {
        bar = bar.child(
            tab()
                .selected(move || selected.get() == index)
                .accessibility_label(label)
                .px_3()
                .h(px(if super::touch() { 44.0 } else { 30.0 }))
                .justify_center()
                .rounded_t(px(radius::SMALL))
                .transition_colors()
                .duration(motion::FAST)
                .hover(|s| s.bg(colors::HOVER))
                .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
                .focus(super::controls::focus_band)
                .on_press(move || selected.set(index))
                .child(
                    text(label)
                        .text_size(px(type_scale::SM))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(move || {
                            if selected.get() == index {
                                colors::TEXT
                            } else {
                                colors::MUTED
                            }
                        }),
                ),
        );
    }
    view().child(bar)
}

/// A table with a table's semantics: a header row of `heads`, then `rows`
/// of cells (the last column is for actions, kept to its content).
pub fn data_table(
    label: impl Into<SharedString>,
    heads: &[&'static str],
    rows: Vec<Vec<View>>,
) -> View {
    let columns = heads.len();
    let cell_style = move |index: usize, cell: nsh_base::TableCellElement| {
        if index + 1 == columns {
            cell.flex_row().justify_end()
        } else {
            cell.flex_1().min_w(px(0.0))
        }
    };
    let mut head = table_row()
        .flex_row()
        .items_center()
        .gap_3()
        .px_3()
        .py_1p5()
        .border_b_1()
        .border_color(colors::LINE);
    for (index, title) in heads.iter().copied().enumerate() {
        let cell = table_head().child(
            text(title)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::FAINT),
        );
        head = head.child(if index + 1 == columns {
            cell.flex_row().justify_end().min_w(px(60.0))
        } else {
            cell.flex_1().min_w(px(0.0))
        });
    }
    let mut body = table_body();
    let count = rows.len();
    for (r, cells) in rows.into_iter().enumerate() {
        let mut row = table_row()
            .flex_row()
            .items_center()
            .gap_3()
            .px_3()
            .py_1p5();
        if r + 1 < count {
            row = row.border_b_1().border_color(colors::LINE);
        }
        for (index, content) in cells.into_iter().enumerate() {
            row = row.child(cell_style(index, table_cell().child(content)));
        }
        body = body.child(row);
    }
    view()
        .rounded(px(radius::SURFACE))
        .border_1()
        .border_color(colors::LINE)
        .bg(colors::SURFACE)
        .child(
            table()
                .accessibility_label(label.into())
                .child(table_header().child(head))
                .child(body),
        )
}

/// How full something is, 0 to 1, as a thin bar (`label` names it for
/// screen readers).
pub fn meter(label: impl Into<SharedString>, share: f32) -> View {
    view().child(
        progress()
            .value(share.clamp(0.0, 1.0) * 100.0)
            .accessibility_label(label.into())
            .h(px(4.0))
            .track_style(|s| {
                s.h_full()
                    .w_full()
                    .rounded(px(radius::SMALL))
                    .bg(colors::INSET)
            })
            .indicator_style(|s| s.h_full().rounded(px(radius::SMALL)).bg(colors::ACCENT)),
    )
}

/// Asks before something that cannot be undone, in the platform's own
/// alert: `title`, `detail`, then `action` (destructive) or Cancel.
/// `then` runs only on `action`.
pub fn confirm(
    title: impl Into<SharedString>,
    detail: impl Into<SharedString>,
    action: impl Into<SharedString>,
    then: impl FnOnce() + 'static,
) {
    let answer = alert_dialog()
        .title(title.into())
        .description(detail.into())
        .cancel("cancel", "Cancel")
        .destructive("go", action.into())
        .show();
    spawn(async move {
        if answer
            .await
            .is_ok_and(|r| r.action.as_deref() == Some("go"))
        {
            then();
        }
    });
}

/// What a row's menu offers: a label and what it does.
pub struct MenuAction {
    pub label: SharedString,
    pub run: Box<dyn Fn()>,
    /// Shown only while this holds (an action that only makes sense now).
    pub shown: Option<std::rc::Rc<dyn Fn() -> bool>>,
}

impl MenuAction {
    pub fn new(label: impl Into<SharedString>, run: impl Fn() + 'static) -> Self {
        Self {
            label: label.into(),
            run: Box::new(run),
            shown: None,
        }
    }

    /// Show it only while `shown` holds.
    pub fn shown_when(mut self, shown: impl Fn() -> bool + 'static) -> Self {
        self.shown = Some(std::rc::Rc::new(shown));
        self
    }
}

/// One row of a menu.
fn menu_row(action: MenuAction) -> impl IntoElement {
    let run = action.run;
    let shown = action.shown.clone();
    menu_item()
        .style_when(move || shown.as_ref().is_some_and(|f| !f()), |s| s.hidden())
        .accessibility_label(action.label.clone())
        .px_2()
        .py(px(if super::touch() { 12.0 } else { 5.0 }))
        .rounded(px(radius::SMALL))
        .focus(|s| s.bg(colors::HOVER))
        .hover(|s| s.bg(colors::HOVER))
        .on_select(run)
        .child(
            text(action.label)
                .text_size(px(fit::small()))
                .text_color(colors::TEXT),
        )
}

fn menu_surface(s: nsh_core::element::StyleRefinement) -> nsh_core::element::StyleRefinement {
    s.bg(colors::SURFACE)
        .border_1()
        .border_color(colors::EDGE)
        .rounded(px(radius::CONTROL))
        .shadow_lg()
        .p_1()
        .min_w(px(180.0))
        .transition_all()
        .duration(motion::BASE)
}

/// `trigger` with a menu on a secondary press (right click, a long press,
/// the context-menu key).
pub fn with_menu(trigger: impl IntoElement, actions: Vec<MenuAction>) -> View {
    let mut menu = context_menu()
        .trigger(trigger)
        .popup_style(menu_surface)
        // It grows from where it was asked for, and fades as it goes.
        .popup_starting_class("opacity-0 scale-95")
        .popup_ending_class("opacity-0 scale-95");
    for action in actions {
        menu = menu.child(menu_row(action));
    }
    view().child(menu)
}

/// A square glyph that opens a menu when pressed (the composer's Attach):
/// a popup at the glyph in a window, a sheet at the bottom of a phone.
/// `label` names it for screen readers.
pub fn press_menu(
    glyph: super::Glyph,
    label: impl Into<SharedString>,
    actions: Vec<MenuAction>,
) -> View {
    let side = if super::touch() { 44.0 } else { 32.0 };
    let mut menu = menu()
        .trigger(
            view()
                .role(AccessibilityRole::Button)
                .label(label.into())
                .h(px(side))
                .w(px(side))
                .items_center()
                .justify_center()
                .rounded(px(radius::CONTROL))
                .transition_colors()
                .duration(motion::FAST)
                .hover(|s| s.bg(colors::HOVER))
                .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
                .focus(super::controls::focus_band)
                .child(
                    super::icon(glyph)
                        .text_color(colors::MUTED)
                        .accessibility_elements_hidden(true),
                ),
        )
        .popup_style(menu_surface)
        .popup_starting_class("opacity-0 scale-95")
        .popup_ending_class("opacity-0 scale-95");
    for action in actions {
        menu = menu.child(menu_row(action));
    }
    view().child(menu)
}

/// A short note for the owner: what happened, and (`detail`) what it was
/// about. `done`: a finished piece of work, marked with a check. `opens`:
/// the conversation a press on it opens.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Note {
    pub title: String,
    pub detail: Option<String>,
    pub done: bool,
    pub opens: Option<String>,
}

impl From<String> for Note {
    fn from(title: String) -> Self {
        Self {
            title,
            ..Self::default()
        }
    }
}

/// The app's toasts: short notes that leave by themselves (paused while
/// hovered or touched; a terminal's status line shows the newest). A card
/// of the theme's surface, not an inverted slab. On a touch screen they
/// come in under the top bar and are swiped up, clear of the composer a
/// thumb is on; with a pointer, at the bottom over the composer, swiped
/// right. `current` is the note to show, `clear` is called once it is
/// taken, `open` opens a note's conversation.
pub fn toasts(
    current: impl Fn() -> Option<Note> + 'static,
    clear: impl Fn() + 'static,
    open: impl Fn(&str) + Clone + 'static,
) -> impl IntoElement {
    let manager = stored(ToastManager::<u64, Note>::new(ToastMotion::sonner()));
    let shown = signal(Vec::new());
    let state = ToastStackState::new();
    let next = stored(0_u64);
    let sync = move || {
        shown.set(manager.with(|m| {
            m.visible(3)
                .map(|(id, t, s)| (*id, t.clone(), s))
                .collect::<Vec<_>>()
        }))
    };
    effect(move || {
        if let Some(note) = current() {
            let id = next.replace(next.get() + 1);
            manager.with(|m| {
                m.push(
                    id,
                    note,
                    ToastOptions::timeout(Some(Duration::from_secs(6))),
                    now(),
                )
            });
            untracked(&clear);
            sync();
        }
    });
    set_interval(Duration::from_millis(50), move || {
        if manager.with(|m| m.advance(now(), state.is_expanded()).changed) {
            sync();
        }
    });
    let top = super::touch();
    // Over the page, not beside it: a layer that takes no presses itself,
    // so only the toasts do.
    let layer = view()
        .absolute()
        .left(px(0.0))
        .right(px(0.0))
        .items_center()
        .pointer_events(PointerEvents::BoxNone);
    let layer = if top {
        layer.top(px(fit::bar() + 8.0)).pt_safe_area()
    } else {
        layer.bottom(px(88.0))
    };
    layer.child(
        toast_stack()
            .placement(if top {
                Anchor::TopCenter
            } else {
                Anchor::BottomCenter
            })
            .state(state)
            .w(px(380.0))
            .max_w(percentage(0.92))
            .child(each(
                shown,
                |(id, _, _)| *id,
                move |item, _| {
                    let note = item.with_untracked(|(_, n, _)| n.clone());
                    let id = item.with_untracked(|(id, _, _)| *id);
                    let said = match &note.detail {
                        Some(detail) => format!("{}: {detail}", note.title),
                        None => note.title.clone(),
                    };
                    let open = open.clone();
                    toast()
                        .transition_status(move || item.get().2)
                        .accessibility_label(Some(SharedString::from(said)))
                        .swipe_direction(Some(if top {
                            nsh_base::SwipeDirection::Up
                        } else {
                            nsh_base::SwipeDirection::Right
                        }))
                        .on_dismiss_request(move || {
                            manager.with(|m| m.dismiss(&id, now()));
                        })
                        .rounded(px(radius::SURFACE))
                        .bg(colors::SURFACE)
                        .border_1()
                        .border_color(colors::BORDER)
                        .shadow_lg()
                        .transition_all()
                        .duration(motion::ENTER)
                        .starting_class(if top {
                            "opacity-0 -translate-y-4"
                        } else {
                            "opacity-0 translate-y-4"
                        })
                        .ending_class("opacity-0 scale-95")
                        .child(note_body(
                            note,
                            move |session: &str| {
                                manager.with(|m| m.dismiss(&id, now()));
                                open(session);
                            },
                            move || {
                                manager.with(|m| m.dismiss(&id, now()));
                            },
                        ))
                },
            )),
    )
}

/// A toast's content: the check of finished work, its title and what it
/// was about (two lines at most); a note that opens a conversation is a
/// button that does, with a chevron to say so. A close button ends it.
fn note_body(note: Note, open: impl Fn(&str) + 'static, close: impl Fn() + 'static) -> View {
    let mut words = view().flex_1().min_w(px(0.0)).gap(px(2.0)).child(
        text(note.title.clone())
            .number_of_lines(2)
            .text_size(px(fit::small()))
            .font_weight(if note.detail.is_some() {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            })
            .text_color(colors::TEXT),
    );
    if let Some(detail) = note.detail.clone() {
        words = words.child(
            text(detail)
                .number_of_lines(2)
                .text_size(px(fit::small()))
                .text_color(colors::MUTED),
        );
    }
    let mut row = view().flex_row().items_center().gap_3().px_4().py_3();
    if note.done {
        row = row.child(
            icon(Glyph::Check)
                .text_color(super::Tone::Success.color())
                .accessibility_elements_hidden(true),
        );
    }
    row = row.child(words);
    // Closed with a press too: a swipe is not something a card says.
    let close = view().pr_2().child(super::icon_btn(
        Glyph::Close,
        "Dismiss",
        super::Size::Sm,
        close,
    ));
    let body = match note.opens {
        Some(session) => view().flex_1().min_w(px(0.0)).child(
            nsh_base::button()
                .accessibility_label(SharedString::from(format!(
                    "Open {}",
                    note.detail.as_deref().unwrap_or(&note.title)
                )))
                .on_press(move || open(&session))
                .rounded(px(radius::SURFACE))
                .hover(|s| s.bg(colors::HOVER))
                .child(
                    row.child(
                        icon(Glyph::Chevron)
                            .text_color(colors::FAINT)
                            .accessibility_elements_hidden(true),
                    ),
                ),
        ),
        None => row.flex_1().min_w(px(0.0)),
    };
    view().flex_row().items_center().child(body).child(close)
}

/// A pane at an edge of a narrow window: it slides in from `side` over the
/// page, which dims behind it, and back out (in place when the owner asked
/// for less motion). A press on the dimmed page, Escape or the back button
/// close it (`close`). Its content's own heading names it.
pub fn drawer(
    side: Placement,
    open: impl Fn() -> bool + 'static,
    close: impl Fn() + 'static,
    content: impl IntoElement,
) -> nsh_base::SheetElement {
    sheet()
        .side(side)
        .open(move || Some(open()))
        .on_open_change(move |event: &nsh_base::SheetEvent| {
            if let nsh_base::SheetEvent::OpenChange(false, _) = event {
                close();
            }
        })
        .popup_style(|s| {
            s.h_full()
                .py_safe_area()
                .w(percentage(0.86))
                .max_w(px(380.0))
                .bg(colors::SIDEBAR)
                .shadow_lg()
        })
        .backdrop_style(|s| s.bg(colors::SCRIM))
        .child(content)
}

/// Side-by-side panes the owner resizes by dragging the line between them.
pub fn split() -> nsh_base::ResizablePanelGroupElement {
    h_resizable()
        .handle_style(|s| s.w(px(1.0)).bg(colors::LINE))
        .handle_accessibility_label("Resize")
}

/// The edge a [`pane`] lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Left,
    Right,
}

/// A pane of a [`split`] at `edge`: `size` wide at first, between `min` and
/// `max` as the owner drags it, shown while `open`. Opening, it slides in
/// from its edge: its width grows from none while what it holds keeps its
/// full width and moves in with it, so nothing rewraps on the way; closing,
/// it slides back out. With reduced motion it changes at once.
pub fn pane(
    edge: Edge,
    size: f32,
    min: f32,
    max: f32,
    open: impl Fn() -> bool + Copy + 'static,
    content: impl IntoElement,
) -> nsh_base::ResizablePanelElement {
    // The width the owner chose (a drag changes it), and the width shown
    // now: the same while still, between none and it while sliding.
    let chosen = signal(size);
    let shown = signal(if untracked(open) { size } else { 0.0 });
    let sliding = stored(None::<Spawned<()>>);
    effect(move || {
        let opening = open();
        let from = shown.get_untracked();
        let to = if opening { chosen.get_untracked() } else { 0.0 };
        if let Some(timer) = sliding.replace(None) {
            timer.cancel();
        }
        if (from - to).abs() < 0.5
            || reduce_motion()
            || idiom() == nsh_core::platform::Idiom::Terminal
        {
            shown.set(to);
            return;
        }
        let started = nsh_core::time::now();
        let span = if opening { motion::ENTER } else { motion::BASE };
        let timer = set_interval(std::time::Duration::from_millis(16), move || {
            let t = (started.elapsed().as_secs_f32() * 1000.0 / span).min(1.0);
            // Decelerating in, accelerating out.
            let eased = if opening {
                1.0 - (1.0 - t).powi(3)
            } else {
                t * t * t
            };
            shown.set(from + (to - from) * eased);
            if t >= 1.0
                && let Some(timer) = sliding.replace(None)
            {
                timer.cancel();
            }
        });
        sliding.replace(Some(timer));
    });
    resizable_panel()
        .visible(move || shown.get() > 0.5)
        .size(move || Some(px(shown.get())))
        .min_size(move || px(min.min(shown.get())))
        .max_size(Some(px(max)))
        // A side pane keeps its width as the window changes; the page takes
        // the difference.
        .fixed(true)
        .on_resize(move |width: &Points| {
            // Only the owner's drag, not the slide, changes the choice.
            if sliding.with(|s| s.is_none()) && open() {
                chosen.set(width.0);
                shown.set(width.0);
            }
        })
        .overflow_hidden()
        .child(
            view()
                .h_full()
                .w(move || px(chosen.get()))
                .translate_x(move || match edge {
                    Edge::Left => px(shown.get() - chosen.get()),
                    Edge::Right => px(0.0),
                })
                .child(content),
        )
}

/// The pane of a [`split`] that takes the rest of the room.
pub fn pane_rest(content: impl IntoElement) -> nsh_base::ResizablePanelElement {
    resizable_panel().child(content)
}

/// One choice of a [`pick_list`]: what it is, a line under it (a folder,
/// a provider; empty for none), the section it shows in, its glyph, and
/// whether it is the one in effect (one per section).
#[derive(Debug, Clone, PartialEq)]
pub struct PickItem {
    pub id: String,
    pub title: String,
    pub detail: String,
    pub group: Option<String>,
    pub glyph: Option<super::Glyph>,
    pub checked: bool,
}

impl PickItem {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            detail: String::new(),
            group: None,
            glyph: None,
            checked: false,
        }
    }

    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    pub fn group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    pub fn glyph(mut self, glyph: super::Glyph) -> Self {
        self.glyph = Some(glyph);
        self
    }

    pub fn checked(mut self, checked: bool) -> Self {
        self.checked = checked;
        self
    }
}

/// A row after the choices that does something else (New project).
pub struct PickAction {
    pub label: &'static str,
    pub glyph: super::Glyph,
    pub run: std::rc::Rc<dyn Fn()>,
}

/// What a [`pick_list`] or [`pick_dialog`] offers: its heading, its
/// choices (read when it opens), what a pick does, and an action row.
pub struct Picker<I, P> {
    pub title: &'static str,
    pub items: I,
    pub pick: P,
    pub action: Option<PickAction>,
}

/// One value from a list of named things with a line each (projects and
/// their folders, models and their providers): `trigger` shows what is
/// chosen (`name` says it to screen readers); a press opens the choices
/// under a heading, with a search past six of them, sections, a check on
/// what is in effect and the action last. A sheet from the bottom on touch
/// screens, a popover at the trigger in a window. `wide` stretches the
/// trigger across its row. Built on `nsh_base`'s Command, so its keys,
/// search and long lists are nsh's.
pub fn pick_list<I, P>(
    picker: Picker<I, P>,
    name: impl Fn() -> String + 'static,
    trigger: impl IntoElement,
    wide: bool,
) -> View
where
    I: Fn() -> Vec<PickItem> + Copy + 'static,
    P: Fn(String) + Clone + 'static,
{
    let open = signal(false);
    let trigger = view()
        .role(AccessibilityRole::Button)
        .label(move || SharedString::from(name()))
        .style_when(wide, |s| s.w_full())
        .rounded(px(radius::CONTROL))
        .transition_colors()
        .duration(motion::FAST)
        .hover(|s| s.bg(colors::HOVER))
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .focus(super::controls::focus_band)
        .child(trigger);
    if super::touch() {
        view()
            .style_when(wide, |s| s.w_full())
            .child(trigger.on_press(move || open.set(true)))
            .child(pick_sheet(picker, open))
    } else {
        let body = pick_content(picker, open, false);
        view().style_when(wide, |s| s.w_full()).child(
            popover()
                .open(move || Some(open.get()))
                .on_open_change(move |event: &nsh_base::PopoverEvent| {
                    if let nsh_base::PopoverEvent::OpenChange(now_open, _) = event {
                        open.set(*now_open);
                    }
                })
                .trigger(trigger)
                .popup_style(|s| {
                    s.w(px(380.0))
                        .bg(colors::SURFACE)
                        .border_1()
                        .border_color(colors::EDGE)
                        .rounded(px(radius::SURFACE))
                        .shadow_lg()
                        .transition_all()
                        .duration(motion::BASE)
                })
                .popup_starting_class("opacity-0 scale-95")
                .popup_ending_class("opacity-0 scale-95")
                .child(body.build()),
        )
    }
}

/// The same choices, opened by the app rather than a trigger (a menu's
/// "Move to project…"): a sheet on touch screens, a centred dialog in a
/// window. `open` holds whether it shows.
pub fn pick_dialog<I, P>(picker: Picker<I, P>, open: Signal<bool>) -> View
where
    I: Fn() -> Vec<PickItem> + Copy + 'static,
    P: Fn(String) + Clone + 'static,
{
    if super::touch() {
        return view().child(pick_sheet(picker, open));
    }
    let body = pick_content(picker, open, false);
    view().child(when(
        move || open.get(),
        move || {
            dialog()
                .on_cancel(move || open.set(false))
                .items_center()
                .pt(px(120.0))
                .child(
                    view()
                        .w(px(420.0))
                        .max_w(percentage(0.92))
                        .rounded(px(radius::SURFACE))
                        .border_1()
                        .border_color(colors::EDGE)
                        .bg(colors::SURFACE)
                        .shadow_lg()
                        .child(body.build()),
                )
        },
    ))
}

fn pick_sheet<I, P>(picker: Picker<I, P>, open: Signal<bool>) -> nsh_base::SheetElement
where
    I: Fn() -> Vec<PickItem> + Copy + 'static,
    P: Fn(String) + Clone + 'static,
{
    sheet()
        .side(Placement::Bottom)
        .open(move || Some(open.get()))
        .on_open_change(move |event: &nsh_base::SheetEvent| {
            if let nsh_base::SheetEvent::OpenChange(false, _) = event {
                open.set(false);
            }
        })
        .popup_style(|s| {
            s.w_full()
                .max_h(percentage(0.72))
                .rounded_t(px(radius::SURFACE))
                .bg(colors::SURFACE)
                .pb_safe_area()
                .shadow_lg()
        })
        .backdrop_style(|s| s.bg(colors::SCRIM))
        .child(pick_content(picker, open, true).build())
}

/// The picker's content, made again each time it opens (so the checks
/// and recent order are current).
#[derive(Clone)]
struct PickContent(std::rc::Rc<dyn Fn() -> View>);

impl PickContent {
    fn build(&self) -> View {
        (self.0)()
    }
}

fn pick_content<I, P>(picker: Picker<I, P>, open: Signal<bool>, touch: bool) -> PickContent
where
    I: Fn() -> Vec<PickItem> + Copy + 'static,
    P: Fn(String) + Clone + 'static,
{
    let Picker {
        title,
        items,
        pick,
        action,
    } = picker;
    let action = action.map(std::rc::Rc::new);
    PickContent(std::rc::Rc::new(move || {
        let pick = pick.clone();
        let action = action.clone();
        view().child(dynamic(move || {
            if !open.get() {
                return None;
            }
            let pick = pick.clone();
            let action = action.clone();
            Some(untracked(move || {
                pick_body(title, items(), pick, action, open, touch)
            }))
        }))
    }))
}

fn pick_body(
    title: &'static str,
    items: Vec<PickItem>,
    pick: impl Fn(String) + 'static,
    action: Option<std::rc::Rc<PickAction>>,
    open: Signal<bool>,
    touch: bool,
) -> View {
    let searchable = items.len() > 6;
    let glyphs = std::rc::Rc::new(
        items
            .iter()
            .filter_map(|i| i.glyph.map(|g| (i.id.clone(), g)))
            .collect::<std::collections::HashMap<_, _>>(),
    );
    let rows = items
        .into_iter()
        .map(|item| {
            let row = CommandItem::new(item.id, item.title)
                .keywords([item.detail])
                .checked(item.checked);
            match item.group {
                Some(group) => row.group(group),
                None => row,
            }
        })
        .collect::<Vec<_>>();
    let row_h = if touch { 52.0 } else { 40.0 };
    let mut list = command()
        .items(rows)
        .searchable(searchable)
        .placeholder("Find one")
        // A phone's keyboard would cover the list it opens over.
        .auto_focus(searchable && !touch)
        .accessibility_label(title)
        .search_accessibility_label(format!("Find in {title}"))
        .list_style(move |s| s.max_h(px(row_h * 6.5)))
        .item_class("rounded-md focus:bg-accent")
        .render_item(move |e: &CommandEntry| {
            let detail = e.keywords.first().cloned().unwrap_or_default();
            let glyph = glyphs.get(e.id.as_ref()).copied();
            view()
                .min_h(px(row_h))
                .flex_row()
                .items_center()
                .gap_2()
                .px_2()
                .child(match glyph {
                    Some(glyph) => super::icon(glyph)
                        .text_color(colors::MUTED)
                        .accessibility_elements_hidden(true),
                    None => view(),
                })
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(
                            text(e.label.clone())
                                .number_of_lines(1)
                                .text_size(px(fit::body()))
                                .text_color(colors::TEXT),
                        )
                        .child(if detail.is_empty() {
                            view().into_element()
                        } else {
                            text(detail)
                                .number_of_lines(1)
                                .text_size(px(type_scale::XS))
                                .text_color(colors::FAINT)
                                .into_element()
                        }),
                )
                .child(if e.checked {
                    super::icon(super::Glyph::Check)
                        .text_color(colors::ACCENT)
                        .accessibility_elements_hidden(true)
                } else {
                    view()
                })
        })
        .render_group(|g: &CommandGroup| {
            text(g.heading.clone())
                .px_2()
                .pt_2()
                .pb_1()
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::FAINT)
        })
        .on_cancel(move || open.set(false))
        .on_select(move |id: &SharedString| {
            open.set(false);
            pick(id.to_string());
        });
    if let Some(action) = action {
        let run = action.run.clone();
        list = list.footer(
            view()
                .mt_1()
                .pt_1()
                .border_t_1()
                .border_color(colors::LINE)
                .child(
                    super::row(
                        || false,
                        move || {
                            open.set(false);
                            run()
                        },
                    )
                    .min_h(px(row_h))
                    .accessibility_label(action.label)
                    .child(
                        super::icon(action.glyph)
                            .text_color(colors::MUTED)
                            .accessibility_elements_hidden(true),
                    )
                    .child(
                        text(action.label)
                            .text_size(px(fit::body()))
                            .text_color(colors::TEXT),
                    ),
                ),
        );
    }
    view()
        .p_2()
        .gap_1()
        .child(when(
            move || touch,
            || {
                view().items_center().pb_1().child(
                    view()
                        .w(px(36.0))
                        .h(px(4.0))
                        .rounded(px(2.0))
                        .bg(colors::BORDER)
                        .accessibility_elements_hidden(true),
                )
            },
        ))
        .child(
            text(title)
                .role(AccessibilityRole::Header)
                .px_2()
                .py_1()
                .text_size(px(fit::body()))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(colors::TEXT),
        )
        .child(list)
}

/// A picture sent in a conversation, opened over the page.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewed {
    /// Its attachment id.
    pub id: String,
    pub url: String,
    pub name: String,
    /// Its size in pixels, when the daemon knows it.
    pub pixels: Option<(u32, u32)>,
}

/// One button of the viewer: what it says and does.
pub struct ViewerAction {
    pub label: &'static str,
    pub glyph: Glyph,
    pub run: std::rc::Rc<dyn Fn(&Viewed)>,
}

/// Marks drawn over a picture: strokes of points, each from 0 to 1 across
/// the picture's width and height.
pub type Strokes = Vec<Vec<(f32, f32)>>;

/// A stroke's width, as a share of the picture's longer side (the drawing
/// and the picture made from it use the same).
pub const MARK_WIDTH: f32 = 0.012;
/// The marks' colour: a red that reads on most pictures.
pub const MARK_COLOR: &str = "#ff3b30";

/// Puts a marked picture on the message.
pub type MarkUp = std::rc::Rc<dyn Fn(&Viewed, Strokes)>;

/// What the viewer offers besides Close.
pub struct ViewerActions {
    /// Save a copy (the web) or open it outside the app (a window).
    pub save: ViewerAction,
    /// The system's share sheet, where there is one.
    pub share: Option<ViewerAction>,
    /// Draw over the picture and put the result on the message being
    /// written, where this returns true now; `mark_up` closes the viewer
    /// when done.
    pub can_mark_up: std::rc::Rc<dyn Fn() -> bool>,
    pub mark_up: MarkUp,
    /// It opened on a picture (the share sheet fetches it ahead).
    pub opened: std::rc::Rc<dyn Fn(&Viewed)>,
}

/// The picture viewer: the picture over a dark backdrop, as large as fits,
/// its name, Mark up, Share, the save action and Close above it. A press on
/// the picture or the backdrop, Escape or back closes it. Marking up draws
/// over the picture with a finger or the mouse; Add to message puts the
/// marked picture on the message. Mount once; `viewing` holds what it
/// shows.
pub fn picture_viewer(viewing: Signal<Option<Viewed>>, actions: ViewerActions) -> View {
    let actions = std::rc::Rc::new(actions);
    view().child(dynamic(move || {
        let shown = viewing.get()?;
        let actions = actions.clone();
        (actions.opened)(&shown);
        Some(untracked(move || viewer(shown, actions, viewing)))
    }))
}

fn viewer(
    shown: Viewed,
    actions: std::rc::Rc<ViewerActions>,
    viewing: Signal<Option<Viewed>>,
) -> View {
    let close = move || viewing.set(None);
    let marking = signal(false);
    let strokes: Signal<Strokes> = signal(Vec::new());
    // The room for the picture, and so the picture's box: as large as fits,
    // in its own shape (the marks are laid over exactly the picture).
    let room = signal((0.0f32, 0.0f32));
    let ratio = shown
        .pixels
        .filter(|(w, h)| *w > 0 && *h > 0)
        .map(|(w, h)| w as f32 / h as f32);
    let fitted = move || {
        let (w, h) = room.get();
        let ratio = ratio.unwrap_or(1.0);
        let width = w.min(h * ratio).max(0.0);
        (width, width / ratio)
    };

    let pressable = |glyph: Glyph, label: &'static str, run: std::rc::Rc<dyn Fn(&Viewed)>| {
        let shown = shown.clone();
        viewer_btn(glyph, label, move || run(&shown))
    };
    let mut looking = view().flex_row().items_center().gap_2();
    if ratio.is_some() {
        looking = looking.child(when(
            {
                let can = actions.can_mark_up.clone();
                move || can()
            },
            move || viewer_btn(Glyph::Pen, "Mark up", move || marking.set(true)),
        ));
    }
    if let Some(share) = &actions.share {
        looking = looking.child(pressable(share.glyph, share.label, share.run.clone()));
    }
    looking = looking
        .child(pressable(
            actions.save.glyph,
            actions.save.label,
            actions.save.run.clone(),
        ))
        .child(viewer_btn(Glyph::Close, "Close", close));

    let add = {
        let shown = shown.clone();
        let actions = actions.clone();
        move || {
            let marks = strokes.get_untracked();
            if marks.is_empty() {
                return;
            }
            // It closes the viewer once the picture is on the message (work
            // spawned here ends with the viewer).
            (actions.mark_up)(&shown, marks);
        }
    };
    let drawing = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(when(
            move || strokes.with(|s| !s.is_empty()),
            move || {
                viewer_btn(Glyph::Undo, "Undo", move || {
                    strokes.update(|s| {
                        s.pop();
                    })
                })
            },
        ))
        .child(viewer_text_btn("Cancel", false, move || {
            strokes.set(Vec::new());
            marking.set(false);
        }))
        .child(when(
            move || strokes.with(|s| !s.is_empty()),
            move || viewer_text_btn("Add to message", true, add.clone()),
        ));

    let bar = view()
        .flex_row()
        .items_center()
        .gap_2()
        .px_3()
        .py_2()
        .min_h(px(if super::touch() { 60.0 } else { 52.0 }))
        .child(
            text({
                let name = shown.name.clone();
                move || {
                    if marking.get() && super::touch() {
                        String::new()
                    } else if marking.get() {
                        "Draw over what matters".to_owned()
                    } else {
                        name.clone()
                    }
                }
            })
            .flex_1()
            .min_w(px(0.0))
            .number_of_lines(1)
            .text_size(px(type_scale::SM))
            .text_color(colors::ON_LIGHTBOX),
        )
        .child(when(move || !marking.get(), move || looking.clone()))
        .child(when(move || marking.get(), move || drawing.clone()));

    // The marks as line art over the picture.
    let marks = svg(move || {
        let (w, h) = fitted();
        nsh_core::element::ImageSrc::svg_markup(&marks_svg(&strokes.get(), w, h))
    })
    .absolute()
    .inset_0();
    // Where a drag draws: an empty layer over all, so a finger's place is
    // in its own frame.
    let start = signal((0.0f32, 0.0f32));
    let pad = view()
        .absolute()
        .inset_0()
        .claim_touches(TouchClaim::on_start().ask(move |request: &ClaimRequest| {
            if marking.get_untracked() {
                request.claim();
            }
        }))
        .keep_touches(true)
        .on_responder_grant(move |e: &ResponderEvent| {
            let (w, h) = fitted();
            if w <= 0.0 || h <= 0.0 {
                return;
            }
            let at = (e.location.x.0 / w, e.location.y.0 / h);
            start.set(at);
            strokes.update(|s| s.push(vec![at]));
        })
        .on_responder_move(move |e: &ResponderEvent| {
            let (w, h) = fitted();
            if w <= 0.0 || h <= 0.0 {
                return;
            }
            let (x0, y0) = start.get_untracked();
            let at = (
                (x0 + e.gesture.delta.x.0 / w).clamp(0.0, 1.0),
                (y0 + e.gesture.delta.y.0 / h).clamp(0.0, 1.0),
            );
            strokes.update(|s| {
                if let Some(last) = s.last_mut() {
                    last.push(at);
                }
            });
        });
    let picture = view()
        .relative()
        .w(move || px(fitted().0))
        .h(move || px(fitted().1))
        .child(
            image(shown.url.clone())
                .w_full()
                .h_full()
                .object_fit(ResizeMode::Contain),
        )
        .child(marks)
        .child(when(move || marking.get(), move || pad.clone()));

    // Zoom: pinch, Ctrl + wheel, or a double press (in, then back out).
    // Marking up starts from the whole picture, so it zooms back out.
    let zoomer = element_ref();
    let zoomed = signal(1.0f32);
    let last_press = signal(None::<nsh_core::time::Instant>);
    let commanded = signal(None::<nsh_core::time::Instant>);
    let reports = signal(false);
    let picture = picture.on_press(move || {
        let at = now();
        let double = last_press
            .get_untracked()
            .is_some_and(|before| at.duration_since(before) < Duration::from_millis(350));
        if double {
            last_press.set(None);
            let to = if zoomed.get_untracked() > 1.05 {
                1.0
            } else {
                2.5
            };
            // Set here: while the zoom animates, the scroll reports pass
            // through the old scales (ignored for a moment below).
            zoomed.set(to);
            commanded.set(Some(at));
            zoomer.zoom_to(to, None, true);
        } else {
            last_press.set(Some(at));
        }
    });
    let zoomable = scroll()
        .w_full()
        .h_full()
        .zoom_range(1.0, 5.0)
        .zoomable(move || !marking.get())
        .center_content(true)
        .scrollbar(false)
        .element_ref(&zoomer)
        .on_scroll(move |e: &ScrollEvent| {
            let settling = commanded
                .get_untracked()
                .is_some_and(|at| now().duration_since(at) < Duration::from_millis(700));
            // The web reports no scale (always 1): trust a report only
            // once one has shown another scale.
            if (e.zoom_scale - 1.0).abs() > f32::EPSILON {
                reports.set(true);
            }
            if !settling && reports.get_untracked() {
                zoomed.set(e.zoom_scale);
            }
        })
        // A frame the scroll view's size centers the picture (the web
        // does not centre a scroll view's content).
        .child(
            view()
                .w(move || px(room.get().0))
                .h(move || px(room.get().1))
                .items_center()
                .justify_center()
                .child(picture),
        );
    effect(move || {
        if marking.get() {
            zoomed.set(1.0);
            zoomer.zoom_to(1.0, None, false);
        }
    });

    view().child(
        dialog()
            .label(format!("Picture: {}", shown.name))
            .on_cancel(move || {
                if marking.get_untracked() {
                    strokes.set(Vec::new());
                    marking.set(false);
                } else {
                    close();
                }
            })
            .bg(colors::LIGHTBOX)
            .child(
                view()
                    .flex_1()
                    .w_full()
                    .h_full()
                    .flex_col()
                    .pt_safe_area()
                    .pb_safe_area()
                    .test_id("picture-viewer")
                    .child(bar)
                    .child(
                        // The backdrop: a press here closes, unless
                        // marking up or zoomed in (the picture itself
                        // takes presses: two zoom).
                        view()
                            .flex_1()
                            .min_h(px(0.0))
                            .w_full()
                            .items_center()
                            .justify_center()
                            .focusable(false)
                            .on_press(move || {
                                if !marking.get_untracked() && zoomed.get_untracked() <= 1.05 {
                                    close();
                                }
                            })
                            .child(
                                view()
                                    .w_full()
                                    .h_full()
                                    .p_4()
                                    .items_center()
                                    .justify_center()
                                    .on_layout(move |b: &Bounds<Points>| {
                                        room.set((
                                            (b.size.width.0 - 32.0).max(0.0),
                                            (b.size.height.0 - 32.0).max(0.0),
                                        ))
                                    })
                                    .child(zoomable),
                            ),
                    ),
            ),
    )
}

/// The marks as an SVG of `w` by `h` points.
fn marks_svg(strokes: &Strokes, w: f32, h: f32) -> String {
    let width = MARK_WIDTH * w.max(h);
    let mut paths = String::new();
    for stroke in strokes {
        let Some(((x, y), rest)) = stroke.split_first() else {
            continue;
        };
        paths.push_str(&format!("<path d=\"M{:.1} {:.1}", x * w, y * h));
        if rest.is_empty() {
            // A tap: a dot.
            paths.push_str(" l0.01 0");
        }
        for (x, y) in rest {
            paths.push_str(&format!(" L{:.1} {:.1}", x * w, y * h));
        }
        paths.push_str("\"/>");
    }
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{h:.0}" viewBox="0 0 {w:.1} {h:.1}" fill="none" stroke="{MARK_COLOR}" stroke-width="{width:.1}" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>"#
    )
}

/// A round glyph button over the viewer's dark backdrop.
fn viewer_btn(glyph: Glyph, label: &'static str, on_press: impl Fn() + 'static) -> View {
    let side = if super::touch() { 44.0 } else { 36.0 };
    with_hint(
        label,
        nsh_base::button()
            .w(px(side))
            .h(px(side))
            .rounded_full()
            .items_center()
            .justify_center()
            .bg(colors::LIGHTBOX_CONTROL)
            .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
            .accessibility_label(label)
            .on_press(on_press)
            .child(
                icon(glyph)
                    .text_color(colors::ON_LIGHTBOX)
                    .accessibility_elements_hidden(true),
            ),
    )
}

/// A pill with a word over the viewer's backdrop; `main`: filled.
fn viewer_text_btn(label: &'static str, main: bool, on_press: impl Fn() + 'static) -> View {
    let side = if super::touch() { 44.0 } else { 36.0 };
    view().child(
        nsh_base::button()
            .h(px(side))
            .px_4()
            .rounded_full()
            .items_center()
            .justify_center()
            .bg(if main {
                colors::ON_LIGHTBOX
            } else {
                colors::LIGHTBOX_CONTROL
            })
            .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
            .accessibility_label(label)
            .on_press(on_press)
            .child(
                text(label)
                    .text_size(px(type_scale::SM))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(if main {
                        colors::LIGHTBOX_INK
                    } else {
                        colors::ON_LIGHTBOX
                    }),
            ),
    )
}
