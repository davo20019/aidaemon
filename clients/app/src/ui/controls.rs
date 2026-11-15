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
    Memory,
    /// Attach a file.
    Attach,
    /// A file (not a picture).
    File,
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
            Self::Memory => "◎",
            Self::Attach => "⊕",
            Self::File => "▭",
        }
    }
}

impl Glyph {
    /// The glyph as line art: Lucide's shapes (ISC licence,
    /// https://lucide.dev), on a 24-unit grid, drawn in the text colour.
    fn shapes(self) -> &'static str {
        match self {
            Self::Plus => r#"<path d="M5 12h14M12 5v14"/>"#,
            Self::Send => r#"<path d="m5 12 7-7 7 7M12 19V5"/>"#,
            Self::Stop => {
                r#"<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>"#
            }
            Self::Close => r#"<path d="M18 6 6 18M6 6l12 12"/>"#,
            Self::Search => r#"<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>"#,
            Self::Chat => {
                r#"<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>"#
            }
            Self::Work => r#"<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>"#,
            Self::Approval => {
                r#"<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>"#
            }
            Self::Inspector => {
                r#"<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18"/>"#
            }
            Self::Sidebar => {
                r#"<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/>"#
            }
            Self::Settings => {
                r#"<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/>"#
            }
            Self::Copy => {
                r#"<rect x="8" y="8" width="14" height="14" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>"#
            }
            Self::Check => r#"<path d="M20 6 9 17l-5-5"/>"#,
            Self::Warning => {
                r#"<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>"#
            }
            Self::Dot => r#"<circle cx="12" cy="12" r="4" fill="currentColor"/>"#,
            Self::Retry => {
                r#"<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>"#
            }
            Self::Command => {
                r#"<path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3"/>"#
            }
            Self::Tool => {
                r#"<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>"#
            }
            Self::Chevron => r#"<path d="m9 18 6-6-6-6"/>"#,
            Self::Pin => {
                r#"<path d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>"#
            }
            Self::Pinned => {
                r#"<path fill="currentColor" d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>"#
            }
            Self::System => {
                r#"<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>"#
            }
            Self::Inbox => {
                r#"<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>"#
            }
            Self::Memory => {
                r#"<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>"#
            }
            Self::Attach => {
                r#"<path d="m16 6-8.414 8.586a2 2 0 0 0 2.829 2.829l8.414-8.586a4 4 0 1 0-5.657-5.657l-8.379 8.551a6 6 0 1 0 8.485 8.485l8.379-8.551"/>"#
            }
            Self::File => {
                r#"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>"#
            }
        }
    }

    /// The glyph's SVG document, stroked in `currentColor`.
    fn markup(self) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{}</svg>"#,
            self.shapes()
        )
    }
}

/// A conversation's state at a glance, in an icon slot: a filled dot
/// (working or a new reply) or a ring (quiet), in `color`. A terminal shows
/// ● or ○.
pub fn state_dot(
    filled: impl Fn() -> bool + Clone + 'static,
    color: impl Fn() -> AdaptiveColor + Clone + 'static,
) -> View {
    let slot = view()
        .w(px(fit::icon_slot()))
        .h(px(fit::icon_slot()))
        .items_center()
        .justify_center()
        .accessibility_elements_hidden(true);
    if idiom() == nsh_core::platform::Idiom::Terminal {
        return slot.child(
            text(move || if filled() { "●" } else { "○" })
                .text_size(px(8.0))
                .text_color(color),
        );
    }
    let fill = color.clone();
    slot.child(
        view()
            .w(px(7.0))
            .h(px(7.0))
            .rounded_full()
            .border(px(1.5))
            .border_color(color)
            .bg(move || {
                if filled() {
                    fill()
                } else {
                    colors::TRANSPARENT
                }
            }),
    )
}

/// A glyph in a fixed square slot, so rows keep their lanes: line art in a
/// window and in a browser, the plain symbol in a terminal (which has only
/// cells). It takes the text colour it is given.
pub fn icon(glyph: Glyph) -> View {
    let slot = view()
        .w(px(fit::icon_slot()))
        .h(px(fit::icon_slot()))
        .items_center()
        .justify_center();
    if idiom() == nsh_core::platform::Idiom::Terminal {
        return slot.child(
            text(glyph.symbol())
                .text_center()
                .text_size(px(fit::icon_glyph()))
                .line_height(px(fit::icon_slot())),
        );
    }
    let side = px(fit::icon_glyph());
    slot.child(
        svg(nsh_core::element::ImageSrc::svg_markup(&glyph.markup()))
            .w(side)
            .h(side),
    )
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

/// A one-line field for a secret: masked, never filled in by the platform.
pub fn secret_field(placeholder: &'static str, value: Signal<String>) -> Input {
    field(placeholder, value).masked(true).auto_correct(false)
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
    // A touch screen has no keys to name: the hint would only take room.
    if touch() {
        return view();
    }
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

/// A picture sent with a message: fitted to 320 points wide (less on a
/// narrow screen), rounded like a card, pressable to open it full size.
pub fn picture(
    src: String,
    label: String,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    button()
        .max_w(px(320.0))
        .max_w(percentage(0.8))
        .rounded(px(radius::SURFACE))
        .overflow_hidden()
        .border_1()
        .border_color(colors::BORDER)
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .accessibility_label(label)
        .on_press(on_press)
        .child(image(src).w_full().max_h(px(360.0)))
}

/// A file sent with a message, or attached to one being written: its name
/// as a small chip. `status` is a word after it ("attaching…").
pub fn file_chip(name: impl Into<SharedString>, status: impl Into<SharedString>) -> View {
    let status: SharedString = status.into();
    let mut chip = view()
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .h(px(fit::row()))
        .max_w(px(260.0))
        .rounded(px(radius::CONTROL))
        .border_1()
        .border_color(colors::BORDER)
        .bg(colors::SURFACE)
        .child(icon(Glyph::File).text_color(colors::MUTED))
        .child(
            text(name.into())
                .number_of_lines(1)
                .flex_shrink(1.0)
                .text_size(px(type_scale::SM))
                .text_color(colors::TEXT),
        );
    if !status.is_empty() {
        chip = chip.child(
            text(status)
                .number_of_lines(1)
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        );
    }
    chip
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

/// A quiet pill that opens and closes what is under it: a chevron that
/// turns, then its label. The owner of `open` renders the content.
pub fn disclosure(
    label: impl IntoReactive<SharedString>,
    open: Signal<bool>,
) -> nsh_base::ButtonElement {
    button()
        .flex_row()
        .items_center()
        .gap_1()
        .self_start()
        .min_h(px(fit::row() - 4.0))
        .pl_1()
        .pr_2()
        .rounded(px(radius::CONTROL))
        .border_1()
        .border_color(colors::LINE)
        .transition_colors()
        .duration(motion::FAST)
        .hover(|s| s.bg(colors::HOVER).border_color(colors::BORDER))
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .child(
            icon(Glyph::Chevron)
                .text_color(colors::FAINT)
                .rotate(move || deg(if open.get() { 90.0 } else { 0.0 }))
                .transition_transform()
                .duration(motion::FAST)
                .accessibility_elements_hidden(true),
        )
        .child(
            text(label)
                .number_of_lines(1)
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
        )
        .on_press(move || {
            motion::next_unfold();
            open.update(|o| *o = !*o)
        })
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
