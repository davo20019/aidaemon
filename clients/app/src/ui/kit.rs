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
    // The colour and thumb are bound to `on`, so the switch follows its
    // value on its own: style closures run once, bound values rerun.
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
                .bg(move || if on() { colors::ACCENT } else { colors::BORDER })
                .transition_colors()
                .duration(motion::FAST)
        })
        .thumb_style(move |s| {
            s.w(px(14.0))
                .h(px(14.0))
                .rounded_full()
                .bg(colors::SURFACE)
                .shadow_sm()
                .translate_x(move || px(if on() { 12.0 } else { 0.0 }))
                .transition_transform()
                .duration(motion::BASE)
        })
        .disabled_style(|s| s.opacity(super::controls::DISABLED_OPACITY))
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
        .disabled_style(|s| s.opacity(super::controls::DISABLED_OPACITY))
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
                    super::icon(super::Glyph::Down)
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

/// What `open` shows under its trigger, unfolding: a region that clips it
/// grows from nothing to the content's height and back on a spring, what
/// is below moving with it; nothing fades or overlaps. It opens on a
/// softer spring than it closes ([`motion::unfold_open`]), and the content
/// stays until the region has closed. The mechanism is `nsh_base`'s
/// collapsible reveal (one layout animation of the commit that flips
/// `open`), whose single spring is too quick to open a long list. With
/// reduced motion, and in a terminal, it comes and goes at once.
pub fn unfold<E: IntoElement>(open: Signal<bool>, content: impl Fn() -> E + 'static) -> View {
    let region = element_ref();
    let unfolding = use_context::<Unfolding>();
    let initial = untracked(|| open.get());
    // The content is in the tree: while open, and while it closes.
    let mounted = signal(initial);
    let shown = stored(initial);
    // Each change of `open` ends what a closing reveal waits for.
    let generation = stored(0u64);
    effect(move || {
        let opening = open.get();
        untracked(|| {
            if shown.replace(opening) == opening {
                return;
            }
            let current = generation.get().wrapping_add(1);
            generation.replace(current);
            let animated = !reduce_motion() && idiom() != nsh_core::platform::Idiom::Terminal;
            let transition = if opening {
                motion::unfold_open()
            } else {
                motion::unfold_close()
            };
            if animated {
                region.animate_layout(transition);
                if let Some(unfolding) = unfolding {
                    unfolding.hold(transition.run_ms());
                }
            }
            if opening || !animated {
                mounted.set(opening);
                return;
            }
            // Out once the region has closed: the platform reports no end.
            set_timeout(
                Duration::from_millis(transition.run_ms() as u64),
                move || {
                    if generation.get() == current {
                        mounted.set(false);
                    }
                },
            );
        });
    });
    view()
        .element_ref(&region)
        .overflow_hidden()
        .style_with(move |s| if open.get() { s } else { s.h(px(0.0)) })
        .child(when(move || mounted.get(), content))
}

/// Whether an [`unfold`] is moving inside a list that follows its end
/// (the transcript). Such a list jumps straight to the new end when what is
/// at its end grows, which would yank the view while the region slides
/// open; while this holds, it keeps its place instead and the content
/// grows below the fold. The list provides it ([`Unfolding::provide`]) and
/// reads [`Unfolding::active`]; each unfold below it holds it while it
/// moves.
#[derive(Clone, Copy)]
pub struct Unfolding(Signal<u32>);

impl Unfolding {
    /// One for the views built after this call to find.
    pub fn provide() -> Self {
        let unfolding = Self(signal(0));
        provide_context(unfolding);
        unfolding
    }

    /// Some unfold is moving.
    pub fn active(self) -> bool {
        self.0.get() > 0
    }

    /// Held for `ms`, from now.
    fn hold(self, ms: f32) {
        self.0.update(|n| *n += 1);
        set_timeout(Duration::from_millis(ms as u64), move || {
            self.0.update(|n| *n = n.saturating_sub(1))
        });
    }
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

/// Asks yes or no in the platform's own alert: `then(true)` on `yes`,
/// `then(false)` on `no`; dismissed without an answer, neither runs.
pub fn yes_or_no(
    title: impl Into<SharedString>,
    detail: impl Into<SharedString>,
    yes: impl Into<SharedString>,
    no: impl Into<SharedString>,
    then: impl FnOnce(bool) + 'static,
) {
    let answer = alert_dialog()
        .title(title.into())
        .description(detail.into())
        .cancel("no", no.into())
        .action("yes", yes.into())
        .show();
    spawn(async move {
        match answer.await.ok().and_then(|r| r.action) {
            Some(id) if id.as_ref() == "yes" => then(true),
            Some(id) if id.as_ref() == "no" => then(false),
            _ => {}
        }
    });
}

/// Offers to do something now, in the platform's own alert: `title`,
/// `detail`, then `action` or `later`. `then` runs only on `action`.
pub fn now_or_later(
    title: impl Into<SharedString>,
    detail: impl Into<SharedString>,
    action: impl Into<SharedString>,
    later: impl Into<SharedString>,
    then: impl FnOnce() + 'static,
) {
    let answer = alert_dialog()
        .title(title.into())
        .description(detail.into())
        .cancel("later", later.into())
        .action("go", action.into())
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
    context_menu_of(trigger, actions, None)
}

/// [`with_menu`] for a row among rows like it, one of which may already
/// be marked as the current one: the menu says which row it is about.
/// `open` holds while the menu is up, for the row to mark itself with
/// [`menu_target`]; on a touch screen, where the sheet can cover the row,
/// `title` heads the sheet.
pub fn with_menu_about(
    title: impl Fn() -> String + 'static,
    open: Signal<bool>,
    trigger: impl IntoElement,
    actions: Vec<MenuAction>,
) -> View {
    context_menu_of(trigger, actions, Some((Box::new(title), open)))
}

/// The look of the row a menu is about, while the menu is up: filled and
/// ringed, so it reads over the scrim and apart from the current row.
pub fn menu_target(s: nsh_core::element::StyleRefinement) -> nsh_core::element::StyleRefinement {
    s.bg(colors::SELECTED)
        .outline(px(2.0))
        .outline_color(colors::ACCENT)
}

/// What a menu is about: its name, and the signal that holds while it is up.
type About = (Box<dyn Fn() -> String>, Signal<bool>);

fn context_menu_of(
    trigger: impl IntoElement,
    actions: Vec<MenuAction>,
    about: Option<About>,
) -> View {
    let mut menu = context_menu()
        .trigger(trigger)
        .popup_style(menu_surface)
        // A phone's sheet sits over a dimmed page, as the app's others do.
        .backdrop_style(|s| s.bg(colors::SCRIM))
        // It grows from where it was asked for, and fades as it goes.
        .popup_starting_class("opacity-0 scale-95")
        .popup_ending_class("opacity-0 scale-95");
    if let Some((title, open)) = about {
        menu = menu.on_open_change(move |event: &nsh_base::MenuEvent| {
            if let nsh_base::MenuEvent::OpenChange(now_open, _) = event {
                open.set(*now_open);
            }
        });
        if super::touch() {
            menu = menu.child(
                view()
                    .test_id("menu-title")
                    .px_2()
                    .pt_2()
                    .pb_2()
                    .mb_1()
                    .border_b_1()
                    .border_color(colors::EDGE)
                    .child(
                        text(title)
                            .number_of_lines(1)
                            .text_size(px(type_scale::XS))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(colors::MUTED),
                    ),
            );
        }
    }
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
        .backdrop_style(|s| s.bg(colors::SCRIM))
        .popup_starting_class("opacity-0 scale-95")
        .popup_ending_class("opacity-0 scale-95");
    for action in actions {
        menu = menu.child(menu_row(action));
    }
    view().child(menu)
}

/// A short note for the owner: what happened, and (`detail`) what it was
/// about. `done`: a finished piece of work, marked with a check. `opens`:
/// the conversation a press on it opens. `undo`: it says something the
/// owner just did that an Undo button on it takes back; the words are
/// handed to the toasts' `undo`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Note {
    pub title: String,
    pub detail: Option<String>,
    pub done: bool,
    pub opens: Option<String>,
    pub undo: Option<String>,
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
/// taken, `open` opens a note's conversation, `undo` takes back what a
/// note with Undo said.
pub fn toasts(
    current: impl Fn() -> Option<Note> + 'static,
    clear: impl Fn() + 'static,
    open: impl Fn(&str) + Clone + 'static,
    undo: impl Fn(&str) + Clone + 'static,
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
    // An overlay of its own, there only while a note shows: it goes over
    // whatever was already up (a phone's list of conversations, a dialog),
    // where a note about what was just done there has to be read.
    view().child(when(
        move || shown.with(|s: &Vec<_>| !s.is_empty()),
        move || {
            let (open, undo) = (open.clone(), undo.clone());
            let top = super::touch();
            // Over the page, not beside it: a layer that takes no presses itself,
            // so only the toasts do.
            let layer = overlay()
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
                            let undo = undo.clone();
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
                                    move |what: &str| {
                                        manager.with(|m| m.dismiss(&id, now()));
                                        undo(what);
                                    },
                                    move || {
                                        manager.with(|m| m.dismiss(&id, now()));
                                    },
                                ))
                        },
                    )),
            )
        },
    ))
}

/// A toast's content: the check of finished work, its title and what it
/// was about (two lines at most); a note that opens a conversation is a
/// button that does, with a chevron to say so; one that can be taken back
/// has Undo. A close button ends it.
fn note_body(
    note: Note,
    open: impl Fn(&str) + 'static,
    undo: impl Fn(&str) + 'static,
    close: impl Fn() + 'static,
) -> View {
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
    let mut card = view().flex_row().items_center().child(body);
    if let Some(what) = note.undo {
        card = card.child(super::btn(
            "Undo",
            super::Variant::Ghost,
            super::Size::Sm,
            move || undo(&what),
        ));
    }
    card.child(close)
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

/// A picture filling its place, fitted whole, that zooms where it is:
/// a press or click (in to 2.5x, then back out), pinch, or Ctrl + wheel.
/// `pixels` gives its shape (square until known).
pub fn zoomable_picture(url: String, label: String, pixels: Option<(u32, u32)>) -> View {
    let room = signal((0.0f32, 0.0f32));
    let ratio = pixels
        .filter(|(w, h)| *w > 0 && *h > 0)
        .map_or(1.0, |(w, h)| w as f32 / h as f32);
    let fitted = move || {
        let (w, h) = room.get();
        let width = w.min(h * ratio).max(0.0);
        (width, width / ratio)
    };
    let zoomer = element_ref();
    let zoomed = signal(1.0f32);
    let commanded = signal(None::<nsh_core::time::Instant>);
    let reports = signal(false);
    let picture = nsh_base::button()
        .w(move || px(fitted().0))
        .h(move || px(fitted().1))
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .accessibility_label(label)
        .on_press(move || {
            let to = if zoomed.get_untracked() > 1.05 {
                1.0
            } else {
                2.5
            };
            // Set here: while the zoom animates, the scroll reports pass
            // through the old scales (ignored for a moment below).
            zoomed.set(to);
            commanded.set(Some(now()));
            zoomer.zoom_to(to, None, true);
        })
        .child(image(url).w_full().h_full().object_fit(ResizeMode::Contain));
    let zoomable = scroll()
        .w_full()
        .h_full()
        .zoom_range(1.0, 5.0)
        .zoomable(|| true)
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
    view()
        .flex_1()
        .min_h(px(0.0))
        .w_full()
        .overflow_hidden()
        .test_id("zoomable-picture")
        .on_layout(move |b: &Bounds<Points>| room.set((b.size.width.0, b.size.height.0)))
        .child(zoomable)
}

/// The live screen's column beside the picture on a phone held sideways,
/// in points: the status, the typing row and the answers.
const LIVE_SIDE: f32 = 300.0;

/// A touch screen held sideways: the picture takes the full height and the
/// live screen's lines go in a column beside it, so the page is not
/// squeezed into a strip between them.
pub fn live_sideways() -> bool {
    let window = nsh_core::element::window();
    super::theme::touch() && window.width.0 > window.height.0
}

/// The room a touch screen has for the live picture while it controls the
/// page, in points: the screen less the status lines over the picture and
/// the typing row and answers under it (held sideways, less the column
/// beside it). The daemon lays the agent's page out for it, so the page
/// fills the phone instead of a wide strip. `None` on a pointer screen: a
/// window that size suits the page as it is.
pub fn live_room() -> Option<(u32, u32)> {
    /// The live screen's own lines, above and below the picture.
    const CHROME: f32 = 210.0;
    if !super::theme::touch() {
        return None;
    }
    let window = nsh_core::element::window();
    let safe = nsh_core::element::safe_area();
    let (w, h) = if live_sideways() {
        // The picture's box (its padding, the gap to the column) and the
        // column; the box's own height less its padding.
        (
            window.width.0 - safe.left.0 - safe.right.0 - LIVE_SIDE - 24.0,
            window.height.0 - safe.top.0 - safe.bottom.0 - 16.0,
        )
    } else {
        (
            window.width.0 - 20.0,
            window.height.0 - safe.top.0 - safe.bottom.0 - CHROME,
        )
    };
    // The daemon takes 160 at least.
    (w >= 200.0 && h >= 120.0).then(|| (w.round() as u32, h.round() as u32))
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

/// One picture of a live view: its `data:` URI and its shape.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveFrame {
    pub uri: String,
    pub w: u32,
    pub h: u32,
}

/// A key going down (`true`) or up: `code`, `key` and the modifiers.
pub type KeySender = dyn Fn(bool, &str, &str, Vec<&'static str>);

/// What the live view shows and does: the agent's browser tab, watched or
/// controlled. Every closure reads or acts through the store.
/// What typing does where the live view's focus is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveTyping {
    /// It takes none (a button, a link, nothing focused).
    None,
    /// It takes text.
    Text,
    /// It takes a secret: what is typed shows masked.
    Secret,
}

pub struct LiveScreen {
    pub open: std::rc::Rc<dyn Fn() -> bool>,
    pub title: std::rc::Rc<dyn Fn() -> String>,
    /// "Watching", "You're in control", or why it could not open.
    pub status: std::rc::Rc<dyn Fn() -> String>,
    pub frame: Signal<Option<LiveFrame>>,
    pub controlling: std::rc::Rc<dyn Fn() -> bool>,
    /// Where typing goes, in words; `Some(None)`: nothing takes typing;
    /// `None`: not known yet.
    pub focus: std::rc::Rc<dyn Fn() -> Option<Option<String>>>,
    /// The focused field's place, as frame fractions `[x, y, w, h]`.
    pub focus_rect: std::rc::Rc<dyn Fn() -> Option<[f32; 4]>>,
    /// What typing does where the focus is.
    pub typing: std::rc::Rc<dyn Fn() -> LiveTyping>,
    /// Characters the last Type sent.
    pub typed: std::rc::Rc<dyn Fn() -> Option<usize>>,
    pub can_control: std::rc::Rc<dyn Fn() -> bool>,
    /// The agent waits for the owner here (a handoff): Done and Not now.
    pub waiting: std::rc::Rc<dyn Fn() -> bool>,
    /// A press, move or release at a fraction of the picture.
    pub pointer: std::rc::Rc<dyn Fn(&'static str, f32, f32)>,
    /// The same with a button (`left`, `right`, `middle`): mouse mode.
    pub button_pointer: std::rc::Rc<dyn Fn(&'static str, f32, f32, &'static str)>,
    /// A key going down (`true`) or up, by `code`, `key` and modifiers.
    pub key_event: std::rc::Rc<KeySender>,
    /// A mouse and keyboard drive the view, as in a remote desktop window;
    /// otherwise touch (a tap clicks, a drag scrolls, a field types).
    pub mouse: bool,
    /// Point the agent at a fraction of the picture (watching, not
    /// controlling): it gets what is there as the owner's message.
    pub point: std::rc::Rc<dyn Fn(f32, f32)>,
    pub can_point: std::rc::Rc<dyn Fn() -> bool>,
    /// A scroll at a fraction of the picture, by page pixels.
    pub wheel: std::rc::Rc<dyn Fn(f32, f32, f32, f32)>,
    pub text: std::rc::Rc<dyn Fn(String)>,
    /// A key by its `code` and `key`.
    pub key: std::rc::Rc<dyn Fn(&'static str, &'static str)>,
    pub take: std::rc::Rc<dyn Fn()>,
    pub give_back: std::rc::Rc<dyn Fn()>,
    pub done: std::rc::Rc<dyn Fn()>,
    pub not_now: std::rc::Rc<dyn Fn()>,
    pub close: std::rc::Rc<dyn Fn()>,
    /// The room for the picture changed (a phone turned), as
    /// [`live_room`] gives it.
    pub room_changed: std::rc::Rc<dyn Fn((u32, u32))>,
}

/// The live view, over everything while it is open.
pub fn live_screen(screen: LiveScreen) -> View {
    let screen = std::rc::Rc::new(screen);
    view().child(dynamic(move || {
        if !(screen.open)() {
            return None;
        }
        let screen = screen.clone();
        Some(untracked(move || live_dialog(screen)))
    }))
}

fn live_dialog(screen: std::rc::Rc<LiveScreen>) -> View {
    let room = signal((0.0f32, 0.0f32));
    let frame = screen.frame;
    let fitted = move || {
        let (w, h) = room.get();
        let ratio = frame
            .with(|f| f.as_ref().map(|f| (f.w.max(1), f.h.max(1))))
            .map_or(16.0 / 10.0, |(w, h)| w as f32 / h as f32);
        let width = w.min(h * ratio).max(0.0);
        (width, width / ratio)
    };
    let controlling = {
        let c = screen.controlling.clone();
        move || c()
    };
    // A phone that turns between portrait and landscape: its new room goes
    // to the daemon, which lays the page out again for it.
    {
        let changed = screen.room_changed.clone();
        effect(move || {
            if let Some(room) = live_room() {
                untracked(|| changed(room));
            }
        });
    }

    // Pinch to zoom the picture, as a photo: `zoom` times the fitted size,
    // its top-left `pan` points from the box's (never past an edge).
    let zoom = signal(1.0f32);
    let pan = signal((0.0f32, 0.0f32));
    let shown = move || {
        let (w, h) = fitted();
        let z = zoom.get();
        (w * z, h * z)
    };
    let clamp_pan = move |(x, y): (f32, f32), z: f32| {
        let (w, h) = fitted();
        (x.clamp(w - w * z, 0.0), y.clamp(h - h * z, 0.0))
    };
    // A place in the box as a fraction of the page.
    let to_page = move |x: f32, y: f32| {
        let (w, h) = fitted();
        let z = zoom.get_untracked();
        let (px_, py_) = pan.get_untracked();
        if w <= 0.0 || h <= 0.0 {
            return (0.0, 0.0);
        }
        (
            ((x - px_) / (w * z)).clamp(0.0, 1.0),
            ((y - py_) / (h * z)).clamp(0.0, 1.0),
        )
    };
    // A two-finger touch: its spread, zoom and the page spot under its
    // middle when it began; once seen, the touch neither clicks nor scrolls.
    let pinch = signal(None::<(f32, f32, (f32, f32))>);
    let pinched = signal(false);

    // The phone's own keyboard: a tap focuses an unseen field laid over the
    // spot (then over the page's field once the page says which), so the
    // keyboard opens and what is typed shows in the page itself.
    let keys = element_ref();
    let keys_at = signal(None::<[f32; 4]>);
    let keys_open = signal(false);
    // What the unseen field holds: one space to start, so a Backspace on
    // the keyboard is a change too (it deletes in the page).
    const KEEP: &str = " ";
    let typed = signal(KEEP.to_owned());
    let sent_text = signal(KEEP.to_owned());

    // Point mode: the next tap tells the agent what is there.
    let pointing = signal(false);
    // Where the owner last tapped, marked until the page answers with a
    // new picture.
    let tapped = signal(None::<(f32, f32)>);
    {
        let seen = signal(0u64);
        effect(move || {
            let changed = frame.with(|f| f.as_ref().map(|f| f.uri.len() as u64));
            if let Some(len) = changed
                && len != seen.get_untracked()
            {
                seen.set(len);
                tapped.set(None);
            }
        });
    }
    // Where a finger or the pointer acts: an empty layer over the picture,
    // so its place is in the picture's own frame.
    let start = signal((0.0f32, 0.0f32));
    // As on a phone's own browser: a tap clicks where it lands, a drag
    // scrolls the page under the finger.
    let dragging = signal(false);
    let scrolled = signal((0.0f32, 0.0f32));
    // The page's size in its own pixels, to turn a drag into a scroll.
    let page_size = move || {
        frame
            .with(|f| f.as_ref().map(|f| (f.w as f32, f.h as f32)))
            .unwrap_or((1280.0, 720.0))
    };
    let pad = {
        let pointer = screen.pointer.clone();
        let wheel = screen.wheel.clone();
        let point = screen.point.clone();
        let controlling = controlling.clone();
        let c_move = controlling.clone();
        let c_up = controlling.clone();
        view()
            .absolute()
            .inset_0()
            .claim_touches(TouchClaim::on_start().ask(move |request: &ClaimRequest| {
                let _ = &controlling;
                request.claim();
            }))
            .keep_touches(true)
            .on_touch(move |e: &nsh_core::TouchEvent| {
                // Two fingers zoom about their middle and pan as they move.
                // The fingers down are the event's own list: it names every
                // finger on the screen, also one that came down off the
                // picture (whose lifting this view never hears), so nothing
                // is kept between events.
                let lifted = matches!(
                    e.phase,
                    nsh_core::TouchPhase::End | nsh_core::TouchPhase::Cancel
                );
                // Positions on the screen, turned into the picture's own
                // through the finger this view heard (its target is the
                // picture; another finger's is wherever it came down).
                let Some(about) = e.points.iter().find(|p| p.changed) else {
                    return;
                };
                let (ox, oy) = (
                    about.page.x.0 - about.location.x.0,
                    about.page.y.0 - about.location.y.0,
                );
                let down: Vec<(u32, f32, f32)> = e
                    .points
                    .iter()
                    .filter(|p| !(lifted && p.changed))
                    .map(|p| (p.id, p.page.x.0 - ox, p.page.y.0 - oy))
                    .collect();
                if down.len() < 2 {
                    pinch.set(None);
                    return;
                }
                let (a, b) = (down[0], down[1]);
                let spread = (a.1 - b.1).hypot(a.2 - b.2).max(1.0);
                let middle = ((a.1 + b.1) / 2.0, (a.2 + b.2) / 2.0);
                pinched.set(true);
                let Some((spread0, zoom0, spot)) = pinch.get_untracked() else {
                    let z = zoom.get_untracked();
                    let (px_, py_) = pan.get_untracked();
                    pinch.set(Some((
                        spread,
                        z,
                        ((middle.0 - px_) / z, (middle.1 - py_) / z),
                    )));
                    return;
                };
                let z = (zoom0 * spread / spread0).clamp(1.0, 4.0);
                zoom.set(z);
                pan.set(clamp_pan((middle.0 - spot.0 * z, middle.1 - spot.1 * z), z));
            })
            .on_responder_grant(move |e: &ResponderEvent| {
                let (w, h) = fitted();
                if w <= 0.0 || h <= 0.0 {
                    return;
                }
                start.set(to_page(e.location.x.0, e.location.y.0));
                dragging.set(false);
                pinched.set(false);
                scrolled.set((0.0, 0.0));
            })
            .on_responder_move(move |e: &ResponderEvent| {
                let (w, h) = fitted();
                if w <= 0.0 || h <= 0.0 || pinched.get_untracked() {
                    return;
                }
                let (dx, dy) = (e.gesture.delta.x.0, e.gesture.delta.y.0);
                if !dragging.get_untracked() && dx.hypot(dy) < 10.0 {
                    return;
                }
                dragging.set(true);
                let (sx, sy) = scrolled.get_untracked();
                let (step_x, step_y) = (dx - sx, dy - sy);
                scrolled.set((dx, dy));
                // Zoomed in, the finger first moves the picture, as a
                // phone's browser does; what is left scrolls the page.
                let z = zoom.get_untracked();
                let (px_, py_) = pan.get_untracked();
                let moved = clamp_pan((px_ + step_x, py_ + step_y), z);
                pan.set(moved);
                let (left_x, left_y) = (step_x - (moved.0 - px_), step_y - (moved.1 - py_));
                if !c_move() || left_x.hypot(left_y) < 0.5 {
                    return;
                }
                // Scroll by the rest, in the page's pixels; the page follows
                // the finger.
                let (pw, ph) = page_size();
                let (x0, y0) = start.get_untracked();
                wheel(x0, y0, -left_x / (w * z) * pw, -left_y / (h * z) * ph);
            })
            .on_responder_release(move |_: &ResponderEvent| {
                if dragging.get_untracked() || pinched.get_untracked() {
                    return;
                }
                let at = start.get_untracked();
                if c_up() || pointing.get_untracked() {
                    tapped.set(Some(at));
                }
                if c_up() {
                    pointer("down", at.0, at.1);
                    pointer("up", at.0, at.1);
                    // Open the keyboard now, inside the tap (a phone opens
                    // it for nothing later); the page's answer closes it
                    // when the spot takes no typing.
                    typed.set(KEEP.to_owned());
                    sent_text.set(KEEP.to_owned());
                    keys_at.set(Some([at.0 - 0.02, at.1 - 0.02, 0.04, 0.04]));
                    keys.focus();
                } else if pointing.get_untracked() {
                    pointing.set(false);
                    point(at.0, at.1);
                }
            })
    };
    // Mouse and keyboard, as in a remote desktop window: the pointer moves
    // and hovers, presses and drags are the mouse's own, the wheel scrolls,
    // a right click is a right click.
    let mouse = screen.mouse;
    let mouse_pad = {
        let frac = move |x: f32, y: f32| {
            let (w, h) = fitted();
            ((x / w).clamp(0.0, 1.0), (y / h).clamp(0.0, 1.0))
        };
        let (down, drag, up, hover, menu) = (
            screen.button_pointer.clone(),
            screen.button_pointer.clone(),
            screen.button_pointer.clone(),
            screen.button_pointer.clone(),
            screen.button_pointer.clone(),
        );
        let wheel = screen.wheel.clone();
        let (c_claim, c_hover, c_menu, c_wheel) = (
            controlling.clone(),
            controlling.clone(),
            controlling.clone(),
            controlling.clone(),
        );
        view()
            .absolute()
            .inset_0()
            .claim_touches(TouchClaim::on_start().ask(move |request: &ClaimRequest| {
                if c_claim() {
                    request.claim();
                }
            }))
            .keep_touches(true)
            .on_responder_grant(move |e: &ResponderEvent| {
                let at = frac(e.location.x.0, e.location.y.0);
                start.set(at);
                down("down", at.0, at.1, "left");
            })
            .on_responder_move(move |e: &ResponderEvent| {
                let (w, h) = fitted();
                if w <= 0.0 || h <= 0.0 {
                    return;
                }
                let (x0, y0) = start.get_untracked();
                drag(
                    "move",
                    (x0 + e.gesture.delta.x.0 / w).clamp(0.0, 1.0),
                    (y0 + e.gesture.delta.y.0 / h).clamp(0.0, 1.0),
                    "left",
                );
            })
            .on_responder_release(move |e: &ResponderEvent| {
                let (w, h) = fitted();
                let (x0, y0) = start.get_untracked();
                let at = if w > 0.0 && h > 0.0 {
                    (
                        (x0 + e.gesture.delta.x.0 / w).clamp(0.0, 1.0),
                        (y0 + e.gesture.delta.y.0 / h).clamp(0.0, 1.0),
                    )
                } else {
                    (x0, y0)
                };
                up("up", at.0, at.1, "left");
            })
            .on_mouse_move(move |e: &MouseEvent| {
                // A press drags through the responder; this is the hover.
                if !c_hover() || e.pressed_button.is_some() {
                    return;
                }
                let at = frac(e.position.x.0, e.position.y.0);
                hover("move", at.0, at.1, "left");
            })
            .on_context_menu(move |e: &ContextMenuEvent| {
                if !c_menu() {
                    return;
                }
                let at = frac(e.position.x.0, e.position.y.0);
                menu("down", at.0, at.1, "right");
                menu("up", at.0, at.1, "right");
            })
            .on_wheel(move |e: &WheelEvent| {
                if !c_wheel() {
                    return;
                }
                let at = frac(e.position.x.0, e.position.y.0);
                let (dx, dy) = match e.delta {
                    WheelDelta::Points(p) => (p.x.0, p.y.0),
                    WheelDelta::Lines(l) => (l.x * 40.0, l.y * 40.0),
                };
                wheel(at.0, at.1, dx, dy);
            })
    };
    // The desktop app's keys go straight to the page while in control
    // (the web app captures them in the page script instead).
    #[cfg(not(target_arch = "wasm32"))]
    if mouse {
        for (keystroke, code, key, mods) in desktop_keys() {
            let send = screen.key_event.clone();
            let controlling = controlling.clone();
            on_key(&keystroke, move |_| {
                if controlling() {
                    send(true, &code, &key, mods.clone());
                    send(false, &code, &key, mods.clone());
                }
            });
        }
    }

    // Typing goes straight into the page as you type: each change of the
    // unseen field sends what was added, and deletes what was removed.
    let send_text = screen.text.clone();
    let delete_key = screen.key.clone();
    let on_typed = move |now: &str| {
        let before = sent_text.get_untracked();
        let common = before
            .chars()
            .zip(now.chars())
            .take_while(|(a, b)| a == b)
            .count();
        for _ in 0..before.chars().count().saturating_sub(common) {
            delete_key("Backspace", "Backspace");
        }
        let added: String = now.chars().skip(common).collect();
        if !added.is_empty() {
            send_text(added);
        }
        // Emptied: start again from the kept space, so the next Backspace
        // is a change too.
        let now = if now.is_empty() { KEEP } else { now };
        sent_text.set(now.to_owned());
        typed.set(now.to_owned());
    };
    // The page's answer: the field it focused takes the unseen field's
    // place; nothing to type into closes the keyboard.
    {
        let focus = screen.focus.clone();
        let focus_rect = screen.focus_rect.clone();
        let typing = screen.typing.clone();
        effect(move || {
            let known = focus().is_some();
            match (typing(), focus_rect()) {
                (LiveTyping::Text | LiveTyping::Secret, Some(rect)) => keys_at.set(Some(rect)),
                // A button, a link or nothing: the keyboard has nothing to
                // type into.
                (LiveTyping::None, _) if known && keys_open.get_untracked() => {
                    keys.blur();
                    keys_at.set(None);
                }
                _ => {}
            }
        });
    }
    let password = {
        let typing = screen.typing.clone();
        move || typing() == LiveTyping::Secret
    };
    let unseen = {
        let key = screen.key.clone();
        nsh_base::input()
            .element_ref(&keys)
            .value(move || typed.get())
            .on_change(on_typed)
            .masked(password)
            .auto_correct(false)
            .auto_capitalize(AutoCapitalize::None)
            .auto_fill(nsh_core::element::AutoFill::Off)
            .on_submit(move |_: &str| key("Enter", "Enter"))
            .on_focus(move || keys_open.set(true))
            .on_blur(move || keys_open.set(false))
            .absolute()
            .left(move || px(keys_at.get().map_or(0.0, |r| r[0] * shown().0)))
            .top(move || px(keys_at.get().map_or(0.0, |r| r[1] * shown().1)))
            .w(move || px(keys_at.get().map_or(1.0, |r| (r[2] * shown().0).max(1.0))))
            .h(move || px(keys_at.get().map_or(1.0, |r| (r[3] * shown().1).max(1.0))))
            // 16: a phone's browser zooms into a smaller field.
            .text_size(px(16.0))
            .text_color(colors::TRANSPARENT)
            .caret_color(|| color(colors::TRANSPARENT))
            .selection_color(|| color(colors::TRANSPARENT))
            .bg(colors::TRANSPARENT)
            .border_color(colors::TRANSPARENT)
            .p_0()
    };
    let point_pad = pad.clone();
    let content = view()
        .child(unseen)
        .absolute()
        .left(move || px(pan.get().0))
        .top(move || px(pan.get().1))
        .w(move || px(shown().0))
        .h(move || px(shown().1))
        .child(when(
            move || frame.with(Option::is_some),
            move || {
                image(move || {
                    nsh_core::element::ImageSrc::from(
                        frame.with(|f| f.as_ref().map(|f| f.uri.clone()).unwrap_or_default()),
                    )
                })
                .w_full()
                .h_full()
                .object_fit(ResizeMode::Contain)
            },
        ))
        .child(when(
            move || frame.with(Option::is_none),
            move || {
                view()
                    .w_full()
                    .h_full()
                    .items_center()
                    .justify_center()
                    .child(
                        text("Connecting to the screen…")
                            .text_size(px(type_scale::SM))
                            .text_color(colors::ON_LIGHTBOX),
                    )
            },
        ))
        .child({
            // The field that has the keyboard, outlined by the app itself:
            // a page in the background may not draw its own focus.
            let focus_rect = screen.focus_rect.clone();
            dynamic(move || {
                let [x, y, w, h] = focus_rect()?;
                Some(untracked(move || {
                    let (fw, fh) = shown();
                    view()
                        .absolute()
                        .left(px(x * fw - 3.0))
                        .top(px(y * fh - 3.0))
                        .w(px(w * fw + 6.0))
                        .h(px(h * fh + 6.0))
                        .rounded(px(6.0))
                        .border_2()
                        .border_color(colors::TAP_RING)
                        .into_element()
                }))
            })
        })
        .child(dynamic(move || {
            let (x, y) = tapped.get()?;
            Some(untracked(move || {
                let (w, h) = shown();
                view()
                    .absolute()
                    .left(px(x * w - 14.0))
                    .top(px(y * h - 14.0))
                    .w(px(28.0))
                    .h(px(28.0))
                    .rounded_full()
                    .border_2()
                    .border_color(colors::TAP_RING)
                    .into_element()
            }))
        }));
    let in_control = controlling.clone();
    let picture = view()
        .relative()
        .overflow_hidden()
        // In control, the picture is framed, as a remote session's screen.
        .style_when(in_control, |s| s.border_2().border_color(colors::TAP_RING))
        .w(move || px(fitted().0))
        .h(move || px(fitted().1))
        .child(content)
        .child(when(
            {
                let c = controlling.clone();
                let _ = c;
                move || !mouse
            },
            move || pad.clone(),
        ))
        .child(when(
            {
                let c = controlling.clone();
                move || mouse && c()
            },
            move || mouse_pad.clone(),
        ))
        .child(when(
            {
                let c = controlling.clone();
                move || mouse && !c() && pointing.get()
            },
            move || point_pad.clone(),
        ));

    let status_line = {
        let status = screen.status.clone();
        text(move || status())
            .text_size(px(type_scale::XS))
            .text_color(colors::ON_LIGHTBOX)
    };
    let title = {
        let title = screen.title.clone();
        text(move || title())
            .flex_1()
            .min_w(px(0.0))
            .number_of_lines(1)
            .text_size(px(type_scale::SM))
            .font_weight(FontWeight::MEDIUM)
            .text_color(colors::ON_LIGHTBOX)
    };
    let close = screen.close.clone();
    let bar = view()
        .flex_row()
        .items_center()
        .gap_2()
        .px_3()
        .py_2()
        .min_h(px(if super::touch() { 60.0 } else { 52.0 }))
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .gap_1()
                .child(title)
                .child(status_line)
                .child(when(
                    {
                        let c = controlling.clone();
                        move || c() && mouse
                    },
                    || {
                        text("Your mouse and keyboard drive the page.")
                            .text_size(px(type_scale::XS))
                            .text_color(colors::ON_LIGHTBOX)
                    },
                ))
                .child(when(
                    {
                        let c = controlling.clone();
                        move || c() && !mouse
                    },
                    || {
                        text("Tap to click · drag to scroll · pinch to zoom")
                            .text_size(px(type_scale::XS))
                            .text_color(colors::ON_LIGHTBOX)
                    },
                )),
        )
        // Hides the view: a handoff keeps waiting (the card opens it again).
        .child(viewer_btn(Glyph::Close, "Hide the live view", move || {
            close()
        }));

    // Control and the handoff's answers: one bar at the bottom, always
    // there (never under the keyboard's room), whose main button says how
    // to go back to the agent.
    let take = screen.take.clone();
    let give = screen.give_back.clone();
    let done = screen.done.clone();
    let not_now = screen.not_now.clone();
    let can_control = screen.can_control.clone();
    let waiting = screen.waiting.clone();
    let can_point = screen.can_point.clone();
    let answers = view()
        .flex_row()
        .flex_wrap()
        .items_center()
        .justify_center()
        .gap_2()
        .px_3()
        .child(when(
            {
                let c = controlling.clone();
                move || !c() && can_point()
            },
            move || {
                view().child(dynamic(move || {
                    let on = pointing.get();
                    Some(untracked(move || {
                        viewer_text_btn(
                            if on { "Tap what you mean" } else { "Point" },
                            on,
                            move || pointing.set(!on),
                        )
                        .into_element()
                    }))
                }))
            },
        ))
        .child(when(
            {
                let (c, can) = (controlling.clone(), can_control.clone());
                move || !c() && can()
            },
            {
                let waiting = waiting.clone();
                move || {
                    let take = take.clone();
                    // While the agent waits, Done is the main answer.
                    viewer_text_btn("Take control", !waiting(), move || take())
                }
            },
        ))
        // The agent asked: Done hands back and lets it carry on.
        .child(when(
            {
                let w = waiting.clone();
                move || w()
            },
            move || {
                let (done, not_now) = (done.clone(), not_now.clone());
                view()
                    .flex_row()
                    .gap_2()
                    .child(viewer_text_btn(
                        "Done, back to the agent",
                        true,
                        move || done(),
                    ))
                    .child(viewer_text_btn("Not now", false, move || not_now()))
            },
        ))
        // The owner took over on their own: Give back ends it.
        .child(when(
            {
                let (c, w) = (controlling.clone(), waiting.clone());
                move || c() && !w()
            },
            move || {
                let give = give.clone();
                viewer_text_btn("Give back to the agent", true, move || give())
            },
        ));

    let key = screen.key.clone();
    let press = |label: &'static str, code: &'static str, name: &'static str| {
        let key = key.clone();
        viewer_text_btn(label, false, move || key(code, name))
    };
    let focus = screen.focus.clone();
    let sent = screen.typed.clone();
    let typing_kind = screen.typing.clone();
    let focus_line = text(move || {
        let takes_text = typing_kind() != LiveTyping::None;
        match (focus(), takes_text, sent(), keys_open.get()) {
            (Some(Some(words)), true, Some(_), true) => format!("Typing into {words}"),
            (Some(Some(words)), true, None, true) => format!("✓ Your keyboard types into {words}"),
            (Some(Some(words)), true, _, false) => {
                format!("{words} is selected: tap Keyboard to type")
            }
            (Some(Some(words)), false, ..) => format!("{words} is selected"),
            _ => "Tap a box on the page and your keyboard opens".to_owned(),
        }
    })
    .text_size(px(type_scale::SM))
    .font_weight(FontWeight::MEDIUM)
    .text_color(colors::ON_LIGHTBOX);
    let typing = view().gap_2().px_3().child(focus_line).child(
        view()
            .flex_row()
            .flex_wrap()
            .justify_center()
            .gap_2()
            .child(when(
                {
                    let typing = screen.typing.clone();
                    move || !keys_open.get() && typing() != LiveTyping::None
                },
                // A press is a gesture, so the keyboard may open again.
                move || viewer_text_btn("Keyboard", true, move || keys.focus()),
            ))
            .child(press("Tab", "Tab", "Tab"))
            .child(press("Delete", "Backspace", "Backspace")),
    );

    let close = screen.close.clone();
    // Held sideways the status goes in the column beside the picture.
    let sideways = live_sideways;
    let top_bar = bar.clone();
    view().child(
        dialog()
            .label("The agent's screen")
            .on_cancel(move || close())
            .bg(colors::LIVE_BACKDROP)
            .child(
                view()
                    .flex_1()
                    .w_full()
                    .h_full()
                    .flex_col()
                    .style_when(sideways, |s| s.flex_row().px_safe_area())
                    .gap_2()
                    .bg(colors::LIVE_BACKDROP)
                    .pt_safe_area()
                    .pb_safe_area()
                    .test_id("live-screen")
                    .child(when(move || !sideways(), move || top_bar.clone()))
                    .child(
                        view()
                            .flex_1()
                            .min_h(px(0.0))
                            .min_w(px(0.0))
                            .w_full()
                            .p_2()
                            .items_center()
                            .justify_center()
                            .on_layout(move |b: &Bounds<Points>| {
                                room.set((
                                    (b.size.width.0 - 16.0).max(0.0),
                                    (b.size.height.0 - 16.0).max(0.0),
                                ))
                            })
                            .child(picture),
                    )
                    .child(
                        view()
                            .gap_2()
                            .style_when(sideways, |s| s.w(px(LIVE_SIDE)).h_full().justify_center())
                            .child(when(sideways, move || bar.clone()))
                            .child(when(
                                {
                                    let c = controlling.clone();
                                    move || c() && !mouse
                                },
                                move || typing.clone(),
                            ))
                            .child(answers)
                            .child(view().h(px(8.0))),
                    ),
            ),
    )
}

/// The keys the desktop app sends straight to a page in mouse mode, each
/// as `(keystroke, code, key, modifiers)`: letters, digits and punctuation
/// with and without Shift (the US layout's), the named keys, and the
/// editing shortcuts. Shortcuts the platform keeps (Command+Q) stay its own.
#[cfg(not(target_arch = "wasm32"))]
fn desktop_keys() -> Vec<(String, String, String, Vec<&'static str>)> {
    let mut keys = Vec::new();
    for c in 'a'..='z' {
        let code = format!("Key{}", c.to_ascii_uppercase());
        keys.push((c.to_string(), code.clone(), c.to_string(), vec![]));
        keys.push((
            format!("shift+{c}"),
            code,
            c.to_ascii_uppercase().to_string(),
            vec!["shift"],
        ));
    }
    let shifted_digits = [')', '!', '@', '#', '$', '%', '^', '&', '*', '('];
    for d in 0..10u8 {
        let c = char::from(b'0' + d);
        let code = format!("Digit{c}");
        keys.push((c.to_string(), code.clone(), c.to_string(), vec![]));
        keys.push((
            format!("shift+{c}"),
            code,
            shifted_digits[usize::from(d)].to_string(),
            vec!["shift"],
        ));
    }
    for (plain, shifted, code) in [
        ('`', '~', "Backquote"),
        ('-', '_', "Minus"),
        ('=', '+', "Equal"),
        ('[', '{', "BracketLeft"),
        (']', '}', "BracketRight"),
        ('\\', '|', "Backslash"),
        (';', ':', "Semicolon"),
        ('\'', '"', "Quote"),
        (',', '<', "Comma"),
        ('.', '>', "Period"),
        ('/', '?', "Slash"),
    ] {
        keys.push((plain.to_string(), code.into(), plain.to_string(), vec![]));
        keys.push((
            format!("shift+{plain}"),
            code.into(),
            shifted.to_string(),
            vec!["shift"],
        ));
    }
    keys.push(("space".into(), "Space".into(), " ".into(), vec![]));
    keys.push((
        "shift+space".into(),
        "Space".into(),
        " ".into(),
        vec!["shift"],
    ));
    for (name, code, key) in [
        ("enter", "Enter", "Enter"),
        ("escape", "Escape", "Escape"),
        ("tab", "Tab", "Tab"),
        ("backspace", "Backspace", "Backspace"),
        ("delete", "Delete", "Delete"),
        ("up", "ArrowUp", "ArrowUp"),
        ("down", "ArrowDown", "ArrowDown"),
        ("left", "ArrowLeft", "ArrowLeft"),
        ("right", "ArrowRight", "ArrowRight"),
        ("home", "Home", "Home"),
        ("end", "End", "End"),
        ("pageup", "PageUp", "PageUp"),
        ("pagedown", "PageDown", "PageDown"),
    ] {
        keys.push((name.into(), code.into(), key.into(), vec![]));
        if !matches!(name, "enter" | "escape") {
            keys.push((
                format!("shift+{name}"),
                code.into(),
                key.into(),
                vec!["shift"],
            ));
        }
    }
    // Command on Apple platforms, Control elsewhere: what `mod` means.
    let modifier = if cfg!(target_os = "macos") {
        "meta"
    } else {
        "ctrl"
    };
    for c in ['a', 'c', 'v', 'x', 'z', 'y'] {
        let code = format!("Key{}", c.to_ascii_uppercase());
        keys.push((format!("mod+{c}"), code, c.to_string(), vec![modifier]));
    }
    keys.push((
        "shift+mod+z".into(),
        "KeyZ".into(),
        "z".into(),
        vec![modifier, "shift"],
    ));
    for (name, code) in [
        ("left", "ArrowLeft"),
        ("right", "ArrowRight"),
        ("backspace", "Backspace"),
    ] {
        keys.push((format!("alt+{name}"), code.into(), code.into(), vec!["alt"]));
        keys.push((
            format!("mod+{name}"),
            code.into(),
            code.into(),
            vec![modifier],
        ));
    }
    keys
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod desktop_key_tests {
    use nsh_core::keymap::Keystroke;
    use nsh_core::platform::Os;

    /// Every key the live view binds parses, and no platform keeps it.
    #[test]
    fn live_view_keys_parse_and_are_free_on_every_desktop() {
        let keys = super::desktop_keys();
        assert!(keys.len() > 100);
        for (keystroke, ..) in &keys {
            for (os, apple) in [(Os::MacOs, true), (Os::Windows, false), (Os::Linux, false)] {
                let parsed = Keystroke::parse_with(keystroke, apple)
                    .unwrap_or_else(|e| panic!("{keystroke:?}: {e:?}"));
                assert_eq!(parsed.reserved(os), None, "{keystroke:?} on {os:?}");
            }
        }
    }
}
