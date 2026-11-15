//! The app's look over `nsh_base`'s behaviour-only parts: switches,
//! selects, sections, tabs, tables, menus, confirmations, toasts and
//! resizable panes. `nsh_base` owns the keyboard, focus, roles and the
//! terminal's rendering; these add the tokens. Features call these and
//! never style a base part themselves.

use std::time::Duration;

use nsh_base::{
    Anchor, SelectEntry, SelectItem, SelectValue, ToastManager, ToastMotion, ToastOptions,
    ToastStackState, accordion, accordion_header, accordion_item, accordion_panel,
    accordion_trigger, alert_dialog, context_menu, h_resizable, menu_item, progress,
    resizable_panel, select, sheet, switch, tab, table, table_body, table_cell, table_head,
    table_header, table_row, tabs, toast, toast_stack,
};
use nsh_core::prelude::*;
use nsh_core::time::now;

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
                    colors::BORDER
                })
                .bg(colors::SURFACE)
                .transition_colors()
                .duration(motion::FAST)
                .hover(|s| s.border_color(colors::FAINT))
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
        .popup_style(|s| {
            s.bg(colors::SURFACE)
                .border_1()
                .border_color(colors::BORDER)
                .rounded(px(radius::CONTROL))
                .shadow_lg()
                .p_1()
                .max_h(px(320.0))
        })
        .search_class("px-2 py-1.5 mb-1 text-sm border-b border-border")
        .item_class("rounded-sm focus:bg-accent")
        .render_item(|e: &SelectEntry| {
            text(e.title.clone())
                .px_2()
                .py(px(5.0))
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

/// `trigger` with a menu on a secondary press (right click, a long press,
/// the context-menu key).
pub fn with_menu(trigger: impl IntoElement, actions: Vec<MenuAction>) -> View {
    let mut menu = context_menu()
        .trigger(trigger)
        .popup_style(|s| {
            s.bg(colors::SURFACE)
                .border_1()
                .border_color(colors::BORDER)
                .rounded(px(radius::CONTROL))
                .shadow_lg()
                .p_1()
                .min_w(px(180.0))
                .transition_all()
                .duration(motion::BASE)
        })
        // It grows from where it was asked for, and fades as it goes.
        .popup_starting_class("opacity-0 scale-95")
        .popup_ending_class("opacity-0 scale-95");
    for action in actions {
        let run = action.run;
        let shown = action.shown.clone();
        menu = menu.child(
            menu_item()
                .style_when(move || shown.as_ref().is_some_and(|f| !f()), |s| s.hidden())
                .accessibility_label(action.label.clone())
                .px_2()
                .py(px(5.0))
                .rounded(px(radius::SMALL))
                .focus(|s| s.bg(colors::HOVER))
                .hover(|s| s.bg(colors::HOVER))
                .on_select(run)
                .child(
                    text(action.label)
                        .text_size(px(fit::small()))
                        .text_color(colors::TEXT),
                ),
        );
    }
    view().child(menu)
}

/// The app's toasts: short notes at the bottom that leave by themselves
/// (paused while hovered; the terminal's status line shows the newest).
/// `current` is the note to show; `clear` is called once it is taken.
pub fn toasts(
    current: impl Fn() -> Option<String> + 'static,
    clear: impl Fn() + 'static,
) -> impl IntoElement {
    let manager = stored(ToastManager::<u64, SharedString>::new(ToastMotion::sonner()));
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
                    note.into(),
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
    // Over the page, not beside it: a layer at the bottom that takes no
    // presses itself, so only the toasts do.
    view()
        .absolute()
        .left(px(0.0))
        .right(px(0.0))
        .bottom(px(88.0))
        .items_center()
        .pointer_events(PointerEvents::BoxNone)
        .child(
            toast_stack()
                .placement(Anchor::BottomCenter)
                .state(state)
                .w(px(360.0))
                .max_w(percentage(0.92))
                .child(each(
                    shown,
                    |(id, _, _)| *id,
                    move |item, _| {
                        toast()
                            .transition_status(move || item.get().2)
                            .accessibility_label(move || Some(item.get().1))
                            .on_dismiss_request(move || {
                                manager.with(|m| m.dismiss(&item.get().0, now()));
                            })
                            .rounded(px(radius::SURFACE))
                            .px_4()
                            .py_3()
                            .bg(colors::PRIMARY)
                            .shadow_lg()
                            .transition_all()
                            .duration(motion::ENTER)
                            .starting_class("opacity-0 translate-y-4")
                            .ending_class("opacity-0 scale-95")
                            .child(
                                text(move || item.get().1)
                                    .text_size(px(fit::small()))
                                    .text_color(colors::ON_PRIMARY),
                            )
                    },
                )),
        )
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
