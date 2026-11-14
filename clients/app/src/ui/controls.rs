//! The app's styled controls over `nsh_base`. Features call these and never
//! style a base control themselves: one button means one hover, one press
//! and one focus ring everywhere.

use nsh_base::{button, input, tooltip};
use nsh_core::element::AutoFill;
use nsh_core::prelude::*;

use super::motion;
use super::theme::{MONO, Tone, colors, fit, radius, size, text as type_scale, touch};

/// What a button means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The one main action of a surface.
    Primary,
    /// An ordinary action on a fill.
    Secondary,
    /// A quiet action: no fill until hovered.
    Ghost,
    /// An action framed by a line.
    Outline,
    /// Destroys or denies.
    Danger,
    /// Approves: the safe answer of an approval.
    Success,
}

/// A control's size: height, padding and text together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    Sm,
    Md,
}

impl Size {
    /// On touch every size is a full 44 pt target or more.
    fn height(self) -> f32 {
        match (self, touch()) {
            (Self::Sm, false) => size::CONTROL_SM,
            (Self::Md, false) => size::CONTROL_MD,
            (Self::Sm, true) => 44.0,
            (Self::Md, true) => 48.0,
        }
    }

    fn pad_x(self) -> f32 {
        match (self, touch()) {
            (Self::Sm, false) => 8.0,
            (Self::Md, false) => 12.0,
            (_, true) => 16.0,
        }
    }

    fn text(self) -> f32 {
        match (self, touch()) {
            (Self::Sm, false) => type_scale::SM,
            (Self::Md, false) => type_scale::BASE,
            (_, true) => 15.0,
        }
    }
}

fn fill(variant: Variant) -> Option<AdaptiveColor> {
    match variant {
        Variant::Primary => Some(colors::PRIMARY),
        Variant::Secondary => Some(colors::HOVER),
        Variant::Ghost | Variant::Outline => None,
        Variant::Danger => Some(colors::DANGER),
        Variant::Success => Some(colors::SUCCESS),
    }
}

fn ink(variant: Variant) -> AdaptiveColor {
    match variant {
        Variant::Primary => colors::ON_PRIMARY,
        Variant::Danger | Variant::Success => colors::ON_STATUS,
        _ => colors::TEXT,
    }
}

/// The frame every button shares: size, radius, focus ring, hover wash,
/// press settle, disabled look.
fn frame(variant: Variant, size: Size, scale: f32) -> nsh_base::ButtonElement {
    let base = button()
        .h(px(size.height()))
        .flex_row()
        .items_center()
        .justify_center()
        .gap_1p5()
        .rounded(px(radius::CONTROL))
        .transition_all()
        .duration(motion::FAST)
        .ease_out()
        .focus_visible(|s| {
            s.outline(px(2.0))
                .outline_offset(px(2.0))
                .outline_color(colors::ACCENT)
        })
        .active(move |s| s.scale(scale))
        .disabled_style(|s| s.opacity(0.45));
    let base = match fill(variant) {
        Some(color) => base.bg(color).hover(|s| s.opacity(0.88)),
        None => base
            .hover(|s| s.bg(colors::HOVER))
            .active(|s| s.bg(colors::SELECTED)),
    };
    if variant == Variant::Outline {
        base.border_1().border_color(colors::BORDER)
    } else {
        base
    }
}

/// A text button.
pub fn btn(
    label: impl Into<SharedString>,
    variant: Variant,
    size: Size,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    let label = label.into();
    frame(variant, size, motion::PRESS_SCALE)
        .px(px(size.pad_x()))
        // Screen readers take the name from here, not from the text inside.
        .accessibility_label(label.clone())
        .on_press(on_press)
        .child(
            text(label)
                .text_size(px(size.text()))
                .font_weight(FontWeight::MEDIUM)
                .number_of_lines(1)
                .text_color(ink(variant)),
        )
}

/// A button with a leading glyph.
pub fn btn_icon(
    glyph: Glyph,
    label: impl Into<SharedString>,
    variant: Variant,
    size: Size,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    let label = label.into();
    frame(variant, size, motion::PRESS_SCALE)
        .px(px(size.pad_x()))
        .accessibility_label(label.clone())
        .on_press(on_press)
        .child(
            icon(glyph)
                .text_color(ink(variant))
                .accessibility_elements_hidden(true),
        )
        .child(
            text(label)
                .text_size(px(size.text()))
                .font_weight(FontWeight::MEDIUM)
                .number_of_lines(1)
                .text_color(ink(variant)),
        )
}

/// A square glyph button. `label` is what screen readers and the tooltip
/// say, and what a terminal shows.
pub fn icon_btn(
    glyph: Glyph,
    label: &'static str,
    size: Size,
    on_press: impl Fn() + 'static,
) -> View {
    with_hint(
        label,
        frame(Variant::Ghost, size, motion::PRESS_SCALE_ICON)
            .w(px(size.height()))
            .accessibility_label(label)
            .on_press(on_press)
            .child(
                icon(glyph)
                    .text_color(colors::MUTED)
                    .accessibility_elements_hidden(true),
            ),
    )
}

/// `control` with the app's tooltip.
pub fn with_hint(label: impl IntoReactive<SharedString>, control: impl IntoElement) -> View {
    view().tooltip(hint(label)).child(control)
}

/// A glyph button with a count over it (waiting approvals). `label` names
/// it for screen readers, with the count.
pub fn icon_btn_count(
    glyph: Glyph,
    label: &'static str,
    count: impl Fn() -> usize + Copy + 'static,
    on_press: impl Fn() + 'static,
) -> View {
    with_hint(
        label,
        frame(Variant::Ghost, Size::Md, motion::PRESS_SCALE_ICON)
            .w(px(Size::Md.height()))
            .accessibility_label(move || {
                Some(SharedString::from(match count() {
                    0 => label.to_owned(),
                    n => format!("{label}, {n} waiting"),
                }))
            })
            .on_press(on_press)
            .child(
                icon(glyph)
                    .text_color(colors::MUTED)
                    .accessibility_elements_hidden(true),
            )
            .child(when(
                move || count() > 0,
                move || {
                    text(move || count().to_string())
                        .absolute()
                        .top(px(4.0))
                        .right(px(2.0))
                        .min_w(px(18.0))
                        .h(px(18.0))
                        .px(px(4.0))
                        .rounded_full()
                        .text_center()
                        .bg(colors::WARNING)
                        .text_color(colors::ON_STATUS)
                        .text_size(px(type_scale::XS))
                        .line_height(px(18.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .accessibility_elements_hidden(true)
                },
            )),
    )
}

/// The tooltip every control uses.
pub fn hint(label: impl IntoReactive<SharedString>) -> nsh_base::TooltipElement {
    tooltip()
        .placement(Placement::Bottom)
        .rounded(px(radius::SMALL))
        .bg(colors::PRIMARY)
        .px_2()
        .py_1()
        .child(
            text(label)
                .text_size(px(type_scale::XS))
                .text_color(colors::ON_PRIMARY),
        )
}

/// The app's glyphs: one family of plain symbols, so they read the same in
/// a window and in a terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    Plus,
    Send,
    Stop,
    Close,
    Search,
    Chat,
    Work,
    Approval,
    Inspector,
    Sidebar,
    Settings,
    Copy,
    Check,
    Warning,
    Dot,
    Retry,
    Command,
    Tool,
    Chevron,
    Pin,
    Pinned,
    System,
    Inbox,
}

impl Glyph {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Plus => "+",
            Self::Send => "↑",
            Self::Stop => "■",
            Self::Close => "✕",
            Self::Search => "⌕",
            Self::Chat => "◇",
            Self::Work => "◎",
            Self::Approval => "◆",
            Self::Inspector => "▥",
            Self::Sidebar => "▤",
            Self::Settings => "⚙",
            Self::Copy => "⧉",
            Self::Check => "✓",
            Self::Warning => "!",
            Self::Dot => "●",
            Self::Retry => "↻",
            Self::Command => "⌘",
            Self::Tool => "›",
            Self::Chevron => "›",
            Self::Pin => "☆",
            Self::Pinned => "★",
            Self::System => "◫",
            Self::Inbox => "◆",
        }
    }
}

/// A glyph in a fixed square slot, so rows keep their lanes.
pub fn icon(glyph: Glyph) -> Text {
    text(glyph.symbol())
        .w(px(fit::icon_slot()))
        .text_center()
        .text_size(px(fit::icon_glyph()))
        .line_height(px(fit::icon_slot()))
}

/// A one-line text field.
pub fn field(placeholder: &'static str, value: Signal<String>) -> Input {
    input()
        .value(move || value.get())
        .on_change(move |text: &str| value.set(text.to_owned()))
        .placeholder(placeholder)
        .placeholder_color(|| color(colors::FAINT))
        .caret_color(|| color(colors::ACCENT))
        .selection_color(|| color(colors::ACCENT))
        .auto_fill(AutoFill::Off)
        .h(px(if touch() { 48.0 } else { size::CONTROL_MD }))
        .px_3()
        // 16 on touch: mobile browsers zoom into smaller fields.
        .text_size(px(if touch() { 16.0 } else { type_scale::BASE }))
        .text_color(colors::TEXT)
        .rounded(px(radius::CONTROL))
        .bg(colors::SURFACE)
        .border_1()
        .border_color(colors::BORDER)
        .transition_colors()
        .duration(motion::FAST)
        .focus(|s| s.border_color(colors::ACCENT))
}

/// A small label with a tone: a status, a count, a risk.
pub fn badge(label: impl IntoReactive<SharedString>, tone: Tone) -> View {
    view()
        .flex_row()
        .items_center()
        .h(px(18.0))
        .px(px(6.0))
        .rounded(px(radius::SMALL))
        .bg(tone.tint())
        .child(
            text(label)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .number_of_lines(1)
                .text_color(tone.color()),
        )
}

/// A status dot before a word: the word carries the meaning, the dot the
/// colour.
pub fn status(label: impl IntoReactive<SharedString>, tone: Tone) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_1p5()
        .child(
            text(Glyph::Dot.symbol())
                .text_size(px(8.0))
                .text_color(tone.color()),
        )
        .child(
            text(label)
                .flex_1()
                .min_w(px(0.0))
                .text_size(px(fit::small()))
                .number_of_lines(2)
                .text_color(colors::MUTED),
        )
}

/// A keyboard key.
pub fn kbd(keys: impl IntoReactive<SharedString>) -> View {
    view()
        .h(px(18.0))
        .px(px(5.0))
        .justify_center()
        .rounded(px(radius::SMALL))
        .border_1()
        .border_color(colors::BORDER)
        .bg(colors::SURFACE)
        .child(
            text(keys)
                .font_family(MONO)
                .text_size(px(type_scale::XS))
                .text_color(colors::MUTED),
        )
}

/// A hairline between regions.
pub fn separator() -> View {
    view().h(px(1.0)).bg(colors::LINE)
}

/// A vertical hairline.
pub fn separator_v() -> View {
    view().w(px(1.0)).bg(colors::LINE)
}

/// Small caps over a group.
pub fn section_label(label: impl IntoReactive<SharedString>) -> Text {
    text(label)
        .text_size(px(type_scale::XS))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(colors::FAINT)
        .px_2()
        .pt_3()
        .pb_1()
}

/// A selectable row in a list (the sidebar's sessions, a menu-like list).
/// The wash fills its whole hit area; selection is stronger than hover.
pub fn row(
    selected: impl Fn() -> bool + 'static,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    button()
        .min_h(px(fit::row()))
        .flex_row()
        .items_center()
        .gap_2()
        .px_2()
        .rounded(px(radius::CONTROL))
        .transition_colors()
        .duration(motion::FAST)
        .hover(|s| s.bg(colors::HOVER))
        .active(|s| s.bg(colors::SELECTED))
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .style_when(selected, |s| s.bg(colors::SELECTED))
        .on_press(on_press)
}

/// A framed surface: cards, dialogs' bodies, the composer.
pub fn surface() -> View {
    view()
        .rounded(px(radius::SURFACE))
        .border_1()
        .border_color(colors::BORDER)
        .bg(colors::SURFACE)
}

/// An empty state: what is missing and what to do about it.
pub fn empty(title: &'static str, detail: impl IntoReactive<SharedString>) -> View {
    view()
        .flex_1()
        .items_center()
        .justify_center()
        .gap_2()
        .p_8()
        .child(
            text(title)
                .role(AccessibilityRole::Header)
                .text_size(px(type_scale::TITLE))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(colors::TEXT),
        )
        .child(
            text(detail)
                .text_size(px(type_scale::BASE))
                .text_center()
                .max_w(px(420.0))
                .text_color(colors::MUTED),
        )
}

/// Work going on: three dots that pulse in turn.
pub fn working(label: impl IntoReactive<SharedString>) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(
            text("●")
                .text_size(px(8.0))
                .text_color(colors::ACCENT)
                .animate_pulse()
                .accessibility_elements_hidden(true),
        )
        .child(
            text(label)
                .text_size(px(type_scale::SM))
                .number_of_lines(1)
                .text_color(colors::MUTED),
        )
}

/// A short identifier in monospace.
pub fn code(value: impl IntoReactive<SharedString>) -> Text {
    text(value)
        .font_family(MONO)
        .text_size(px(type_scale::SM))
        .text_color(colors::TEXT)
}

/// A message strip in the content column: what happened or is offered,
/// then its actions (added by the caller as children).
pub fn banner(message: impl IntoReactive<SharedString>, tone: Tone) -> View {
    view()
        .w_full()
        .max_w(px(size::COLUMN))
        .flex_row()
        .items_center()
        .gap_2()
        .mt_2()
        .px_3()
        .py_2()
        .rounded(px(radius::CONTROL))
        .bg(tone.tint())
        .role(AccessibilityRole::Alert)
        .child(
            text(message)
                .flex_1()
                .text_size(px(type_scale::SM))
                .line_height(px(type_scale::LH_SM))
                .text_color(colors::TEXT),
        )
}

/// A thin bar showing a share (0–1) for scale; hidden from screen readers
/// (the row it sits in says the numbers).
pub fn meter(share: f32) -> View {
    view()
        .h(px(4.0))
        .rounded(px(radius::SMALL))
        .bg(colors::INSET)
        .accessibility_elements_hidden(true)
        .child(
            view()
                .h_full()
                .w(percentage(share.clamp(0.01, 1.0)))
                .rounded(px(radius::SMALL))
                .bg(colors::ACCENT),
        )
}

/// The terminal's key line: one quiet row of "Key what" pairs.
pub fn key_line(keys: impl Fn() -> String + 'static) -> View {
    view()
        .flex_row()
        .px_2()
        .border_t_1()
        .border_color(colors::LINE)
        .child(
            text(keys)
                .flex_1()
                .number_of_lines(1)
                .text_size(px(type_scale::XS))
                .text_color(colors::MUTED),
        )
}
