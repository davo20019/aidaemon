//! The app's styled controls over `nsh_base`. Features call these and never
//! style a base control themselves: one button means one hover, one press
//! and one focus ring everywhere.

use nsh_base::{button, input, tooltip};
use nsh_core::element::AutoFill;
use nsh_core::prelude::*;

use super::motion;
use super::theme::{MONO, Tone, colors, radius, size, text as type_scale};

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
    fn height(self) -> f32 {
        match self {
            Self::Sm => size::CONTROL_SM,
            Self::Md => size::CONTROL_MD,
        }
    }

    fn pad_x(self) -> f32 {
        match self {
            Self::Sm => 8.0,
            Self::Md => 12.0,
        }
    }

    fn text(self) -> f32 {
        match self {
            Self::Sm => type_scale::SM,
            Self::Md => type_scale::BASE,
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
        .focus_visible(|s| s.outline(px(2.0)).outline_offset(px(2.0)).outline_color(colors::ACCENT))
        .active(move |s| s.scale(scale))
        .disabled_style(|s| s.opacity(0.45));
    let base = match fill(variant) {
        Some(color) => base.bg(color).hover(|s| s.opacity(0.88)),
        None => base.hover(|s| s.bg(colors::HOVER)).active(|s| s.bg(colors::SELECTED)),
    };
    if variant == Variant::Outline {
        base.border_1().border_color(colors::BORDER)
    } else {
        base
    }
}

/// A text button.
pub fn btn(
    label: impl IntoReactive<SharedString>,
    variant: Variant,
    size: Size,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    frame(variant, size, motion::PRESS_SCALE)
        .px(px(size.pad_x()))
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
    label: impl IntoReactive<SharedString>,
    variant: Variant,
    size: Size,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    frame(variant, size, motion::PRESS_SCALE)
        .px(px(size.pad_x()))
        .on_press(on_press)
        .child(icon(glyph).text_color(ink(variant)))
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
            .child(icon(glyph).text_color(colors::MUTED)),
    )
}

/// `control` with the app's tooltip.
pub fn with_hint(label: impl IntoReactive<SharedString>, control: impl IntoElement) -> View {
    view().tooltip(hint(label)).child(control)
}

/// The tooltip every control uses.
pub fn hint(label: impl IntoReactive<SharedString>) -> nsh_base::TooltipElement {
    tooltip()
        .placement(Placement::Bottom)
        .rounded(px(radius::SMALL))
        .bg(colors::PRIMARY)
        .px_2()
        .py_1()
        .child(text(label).text_size(px(type_scale::XS)).text_color(colors::ON_PRIMARY))
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
        }
    }
}

/// A glyph in a fixed square slot, so rows keep their lanes.
pub fn icon(glyph: Glyph) -> Text {
    text(glyph.symbol())
        .w(px(size::ICON))
        .text_center()
        .text_size(px(type_scale::BASE))
        .line_height(px(size::ICON))
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
        .h(px(size::CONTROL_MD))
        .px_3()
        .text_size(px(type_scale::BASE))
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
        .child(text(Glyph::Dot.symbol()).text_size(px(8.0)).text_color(tone.color()))
        .child(
            text(label)
                .text_size(px(type_scale::SM))
                .number_of_lines(1)
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
pub fn row(selected: impl Fn() -> bool + 'static, on_press: impl Fn() + 'static) -> nsh_base::ButtonElement {
    button()
        .min_h(px(size::ROW))
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
                .animate_pulse(),
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
