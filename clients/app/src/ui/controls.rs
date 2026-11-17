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
        // A terminal draws no outline: the focused control is a band.
        .focus(focus_band)
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
    label: impl Into<SharedString>,
    size: Size,
    on_press: impl Fn() + 'static,
) -> View {
    let label: SharedString = label.into();
    with_hint(
        label.clone(),
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

/// The one main action as a filled, round glyph button (the composer's
/// Send): smaller than a text button, a full touch target with its margin.
/// `label` is what screen readers and the tooltip say.
pub fn icon_btn_primary(
    glyph: Glyph,
    label: impl Into<SharedString>,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    let side = if touch() { 40.0 } else { size::CONTROL_SM };
    frame(Variant::Primary, Size::Sm, motion::PRESS_SCALE_ICON)
        .h(px(side))
        .w(px(side))
        .rounded(px(side / 2.0))
        .accessibility_label(label.into())
        .on_press(on_press)
        .child(
            icon(glyph)
                .text_color(ink(Variant::Primary))
                .accessibility_elements_hidden(true),
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
    /// A coding conversation's changes: a file with a diff's marks.
    Changes,
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
    /// Automations: what runs on its own.
    Clock,
    /// The previous one (a diff's change).
    Up,
    /// The next one.
    Down,
    /// Side by side.
    Columns,
    /// Speak instead of typing.
    Mic,
    /// A folder.
    Folder,
    /// Save a copy.
    Download,
    /// Open outside the app.
    External,
    /// Share with another app.
    Share,
    /// Draw over a picture.
    Pen,
    /// Take back the last one.
    Undo,
    /// Write a new one.
    Compose,
    /// Back to where this came from.
    Back,
    /// More to do with it.
    More,
    /// Drag to reorder.
    Grip,
    /// Chat platforms, by their marks (Simple Icons, CC0): where a
    /// conversation came from.
    Telegram,
    Discord,
    Slack,
    /// Several people: a group chat.
    Group,
    /// Model providers, by their marks (Simple Icons, CC0).
    OpenAi,
    Anthropic,
    Gemini,
    OpenRouter,
    DeepSeek,
    Kimi,
    MiniMax,
    Cloudflare,
    Ollama,
}

/// The mark of the chat platform a `channel` or platform name belongs to
/// ("telegram:coding_bot", "Telegram", "slack"): Telegram, Discord or
/// Slack; none for any other.
pub fn platform_glyph(channel: &str) -> Option<Glyph> {
    let platform = channel
        .split([':', ' '])
        .next()
        .unwrap_or("")
        .to_lowercase();
    match platform.as_str() {
        "telegram" => Some(Glyph::Telegram),
        "discord" => Some(Glyph::Discord),
        "slack" => Some(Glyph::Slack),
        _ => None,
    }
}

/// The mark of a model provider, from its catalog preset (`openai_chatgpt`,
/// `google_ai_studio`...): `None` for one with no mark of its own.
pub fn provider_glyph(preset: &str) -> Option<Glyph> {
    match preset {
        "openai" | "openai_chatgpt" => Some(Glyph::OpenAi),
        "anthropic_native" | "anthropic" | "anthropic_openrouter" => Some(Glyph::Anthropic),
        "google_ai_studio" => Some(Glyph::Gemini),
        "openrouter" => Some(Glyph::OpenRouter),
        "deepseek" => Some(Glyph::DeepSeek),
        "moonshot" => Some(Glyph::Kimi),
        "minimax" => Some(Glyph::MiniMax),
        "cloudflare_gateway" => Some(Glyph::Cloudflare),
        "ollama" => Some(Glyph::Ollama),
        _ => None,
    }
}

impl Glyph {
    /// A platform mark's name, for screen readers; empty for other glyphs.
    pub const fn platform_name(self) -> &'static str {
        match self {
            Self::Telegram => "Telegram",
            Self::Discord => "Discord",
            Self::Slack => "Slack",
            _ => "",
        }
    }
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
            Self::Changes => "±",
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
            Self::Clock => "◷",
            Self::Up => "↑",
            Self::Down => "↓",
            Self::Columns => "◫",
            Self::Mic => "◉",
            Self::Folder => "▸",
            Self::Download => "⤓",
            Self::External => "↗",
            Self::Share => "⇪",
            Self::Pen => "✎",
            Self::Undo => "↶",
            Self::Compose => "✎",
            Self::Back => "‹",
            Self::More => "⋮",
            Self::Grip => "⠿",
            Self::Telegram => "✈",
            Self::Discord => "◈",
            Self::Slack => "#",
            Self::Group => "⚇",
            Self::OpenAi => "◍",
            Self::Anthropic => "A",
            Self::Gemini => "✦",
            Self::OpenRouter => "⇄",
            Self::DeepSeek => "D",
            Self::Kimi => "K",
            Self::MiniMax => "M",
            Self::Cloudflare => "☁",
            Self::Ollama => "O",
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
            Self::Share => {
                r#"<path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8M16 6l-4-4-4 4M12 2v13"/>"#
            }
            Self::Pen => r#"<path d="M12 20h9M16.4 3.6a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4Z"/>"#,
            Self::Undo => r#"<path d="M9 14 4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>"#,
            Self::Compose => {
                r#"<path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7M18.4 2.6a2.1 2.1 0 1 1 3 3L12 15l-4 1 1-4Z"/>"#
            }
            Self::Back => r#"<path d="m15 18-6-6 6-6"/>"#,
            Self::Telegram => {
                r#"<path fill="currentColor" stroke="none" d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/>"#
            }
            Self::Discord => {
                r#"<path fill="currentColor" stroke="none" d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z"/>"#
            }
            Self::Slack => {
                r#"<path fill="currentColor" stroke="none" d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z"/>"#
            }
            Self::OpenAi => {
                r#"<path fill="currentColor" stroke="none" d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z"/>"#
            }
            Self::Anthropic => {
                r#"<path fill="currentColor" stroke="none" d="M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z"/>"#
            }
            Self::Gemini => {
                r#"<path fill="currentColor" stroke="none" d="M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81"/>"#
            }
            Self::OpenRouter => {
                r#"<path fill="currentColor" stroke="none" d="M16.778 1.844v1.919q-.569-.026-1.138-.032-.708-.008-1.415.037c-1.93.126-4.023.728-6.149 2.237-2.911 2.066-2.731 1.95-4.14 2.75-.396.223-1.342.574-2.185.798-.841.225-1.753.333-1.751.333v4.229s.768.108 1.61.333c.842.224 1.789.575 2.185.799 1.41.798 1.228.683 4.14 2.75 2.126 1.509 4.22 2.11 6.148 2.236.88.058 1.716.041 2.555.005v1.918l7.222-4.168-7.222-4.17v2.176c-.86.038-1.611.065-2.278.021-1.364-.09-2.417-.357-3.979-1.465-2.244-1.593-2.866-2.027-3.68-2.508.889-.518 1.449-.906 3.822-2.59 1.56-1.109 2.614-1.377 3.978-1.466.667-.044 1.418-.017 2.278.02v2.176L24 6.014Z"/>"#
            }
            Self::DeepSeek => {
                r#"<path fill="currentColor" stroke="none" d="M23.748 4.651c-.254-.124-.364.113-.512.233-.051.04-.094.09-.137.137-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.155-.708-.311-.955-.65-.172-.24-.219-.509-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.094.172.187.129.323-.082.28-.18.553-.266.833-.055.179-.137.218-.328.14a5.5 5.5 0 0 1-1.737-1.179c-.857-.828-1.631-1.743-2.597-2.46a12 12 0 0 0-.689-.47c-.985-.957.13-1.743.387-1.836.27-.098.094-.433-.778-.428-.872.003-1.67.295-2.687.685a3 3 0 0 1-.465.136 9.6 9.6 0 0 0-2.883-.101c-1.885.21-3.39 1.1-4.497 2.622C.082 8.776-.231 10.854.152 13.02c.403 2.284 1.568 4.175 3.36 5.653 1.857 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.132-.284 4.994-1.86.47.234.962.328 1.78.398.629.058 1.235-.031 1.705-.129.735-.155.684-.836.418-.961-2.155-1.004-1.682-.595-2.112-.926 1.095-1.295 2.768-3.598 3.284-6.733.05-.346.115-.834.108-1.114-.004-.171.035-.238.23-.257a4.2 4.2 0 0 0 1.545-.475c1.397-.763 1.96-2.016 2.093-3.517.02-.23-.004-.467-.247-.588M11.58 18.168c-2.088-1.642-3.101-2.183-3.52-2.16-.39.024-.32.472-.234.763.09.288.207.487.371.74.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.168-1.361-.801-2.5-1.86-3.301-3.306-.775-1.393-1.225-2.888-1.299-4.482-.02-.385.094-.522.477-.592a4.7 4.7 0 0 1 1.53-.038c2.131.311 3.946 1.264 5.467 2.774.868.86 1.525 1.887 2.202 2.89.72 1.066 1.494 2.082 2.48 2.915.348.291.626.513.892.677-.802.09-2.14.109-3.055-.615zm1.001-6.44a.306.306 0 0 1 .415-.287.3.3 0 0 1 .113.074.3.3 0 0 1 .086.214c0 .17-.136.307-.308.307a.303.303 0 0 1-.306-.307m3.11 1.596c-.2.081-.4.151-.591.16a1.25 1.25 0 0 1-.798-.254c-.274-.23-.47-.358-.551-.758a1.7 1.7 0 0 1 .015-.588c.07-.327-.007-.537-.238-.727-.188-.156-.426-.199-.689-.199a.6.6 0 0 1-.254-.078.253.253 0 0 1-.114-.358 1 1 0 0 1 .192-.21c.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.392.451.462.576.685.915.176.264.336.536.446.848.066.194-.02.353-.25.45"/>"#
            }
            Self::Kimi => {
                r#"<path fill="currentColor" stroke="none" d="M21.765.351C22.998.351 24 1.353 24 2.586S22.998 4.82 21.765 4.82h-1.974c-.15 0-.26-.12-.26-.26V2.586A2.237 2.237 0 0 1 21.765.35M9.41 13.388l8.447-8.377c.16-.16.07-.471-.14-.471h-4.55s-.1.02-.14.06l-9.099 9.029c-.14.14-.35.02-.35-.21V4.81c0-.15-.1-.27-.221-.27H.22c-.12 0-.22.12-.22.27v18.57c0 .15.1.27.22.27h3.137c.12 0 .22-.12.22-.27v-3.79c0-.08.03-.16.08-.21l2.826-2.796c.07-.07.16-.08.241-.03l7.546 5.551a8.9 8.9 0 0 0 4.018 1.493c.12.01.23-.11.23-.27V19.76c0-.14-.08-.25-.19-.26a5.8 5.8 0 0 1-2.355-.942l-6.533-4.73c-.14-.09-.15-.32-.03-.441"/>"#
            }
            Self::MiniMax => {
                r#"<path fill="currentColor" stroke="none" d="M11.43 3.92a.86.86 0 1 0-1.718 0v14.236a1.999 1.999 0 0 1-3.997 0V9.022a.86.86 0 1 0-1.718 0v3.87a1.999 1.999 0 0 1-3.997 0V11.49a.57.57 0 0 1 1.139 0v1.404a.86.86 0 0 0 1.719 0V9.022a1.999 1.999 0 0 1 3.997 0v9.134a.86.86 0 0 0 1.719 0V3.92a1.998 1.998 0 1 1 3.996 0v11.788a.57.57 0 1 1-1.139 0zm10.572 3.105a2 2 0 0 0-1.999 1.997v7.63a.86.86 0 0 1-1.718 0V3.923a1.999 1.999 0 0 0-3.997 0v16.16a.86.86 0 0 1-1.719 0V18.08a.57.57 0 1 0-1.138 0v2a1.998 1.998 0 0 0 3.996 0V3.92a.86.86 0 0 1 1.719 0v12.73a1.999 1.999 0 0 0 3.996 0V9.023a.86.86 0 1 1 1.72 0v6.686a.57.57 0 0 0 1.138 0V9.022a2 2 0 0 0-1.998-1.997"/>"#
            }
            Self::Cloudflare => {
                r#"<path fill="currentColor" stroke="none" d="M16.5088 16.8447c.1475-.5068.0908-.9707-.1553-1.3154-.2246-.3164-.6045-.499-1.0615-.5205l-8.6592-.1123a.1559.1559 0 0 1-.1333-.0713c-.0283-.042-.0351-.0986-.021-.1553.0278-.084.1123-.1484.2036-.1562l8.7359-.1123c1.0351-.0489 2.1601-.8868 2.5537-1.9136l.499-1.3013c.0215-.0561.0293-.1128.0147-.168-.5625-2.5463-2.835-4.4453-5.5499-4.4453-2.5039 0-4.6284 1.6177-5.3876 3.8614-.4927-.3658-1.1187-.5625-1.794-.499-1.2026.119-2.1665 1.083-2.2861 2.2856-.0283.31-.0069.6128.0635.894C1.5683 13.171 0 14.7754 0 16.752c0 .1748.0142.3515.0352.5273.0141.083.0844.1475.1689.1475h15.9814c.0909 0 .1758-.0645.2032-.1553l.12-.4268zm2.7568-5.5634c-.0771 0-.1611 0-.2383.0112-.0566 0-.1054.0415-.127.0976l-.3378 1.1744c-.1475.5068-.0918.9707.1543 1.3164.2256.3164.6055.498 1.0625.5195l1.8437.1133c.0557 0 .1055.0263.1329.0703.0283.043.0351.1074.0214.1562-.0283.084-.1132.1485-.204.1553l-1.921.1123c-1.041.0488-2.1582.8867-2.5527 1.914l-.1406.3585c-.0283.0713.0215.1416.0986.1416h6.5977c.0771 0 .1474-.0489.169-.126.1122-.4082.1757-.837.1757-1.2803 0-2.6025-2.125-4.727-4.7344-4.727"/>"#
            }
            Self::Ollama => {
                r#"<path fill="currentColor" stroke="none" d="M16.361 10.26a.894.894 0 0 0-.558.47l-.072.148.001.207c0 .193.004.217.059.353.076.193.152.312.291.448.24.238.51.3.872.205a.86.86 0 0 0 .517-.436.752.752 0 0 0 .08-.498c-.064-.453-.33-.782-.724-.897a1.06 1.06 0 0 0-.466 0zm-9.203.005c-.305.096-.533.32-.65.639a1.187 1.187 0 0 0-.06.52c.057.309.31.59.598.667.362.095.632.033.872-.205.14-.136.215-.255.291-.448.055-.136.059-.16.059-.353l.001-.207-.072-.148a.894.894 0 0 0-.565-.472 1.02 1.02 0 0 0-.474.007Zm4.184 2c-.131.071-.223.25-.195.383.031.143.157.288.353.407.105.063.112.072.117.136.004.038-.01.146-.029.243-.02.094-.036.194-.036.222.002.074.07.195.143.253.064.052.076.054.255.059.164.005.198.001.264-.03.169-.082.212-.234.15-.525-.052-.243-.042-.28.087-.355.137-.08.281-.219.324-.314a.365.365 0 0 0-.175-.48.394.394 0 0 0-.181-.033c-.126 0-.207.03-.355.124l-.085.053-.053-.032c-.219-.13-.259-.145-.391-.143a.396.396 0 0 0-.193.032zm.39-2.195c-.373.036-.475.05-.654.086-.291.06-.68.195-.951.328-.94.46-1.589 1.226-1.787 2.114-.04.176-.045.234-.045.53 0 .294.005.357.043.524.264 1.16 1.332 2.017 2.714 2.173.3.033 1.596.033 1.896 0 1.11-.125 2.064-.727 2.493-1.571.114-.226.169-.372.22-.602.039-.167.044-.23.044-.523 0-.297-.005-.355-.045-.531-.288-1.29-1.539-2.304-3.072-2.497a6.873 6.873 0 0 0-.855-.031zm.645.937a3.283 3.283 0 0 1 1.44.514c.223.148.537.458.671.662.166.251.26.508.303.82.02.143.01.251-.043.482-.08.345-.332.705-.672.957a3.115 3.115 0 0 1-.689.348c-.382.122-.632.144-1.525.138-.582-.006-.686-.01-.853-.042-.57-.107-1.022-.334-1.35-.68-.264-.28-.385-.535-.45-.946-.03-.192.025-.509.137-.776.136-.326.488-.73.836-.963.403-.269.934-.46 1.422-.512.187-.02.586-.02.773-.002zm-5.503-11a1.653 1.653 0 0 0-.683.298C5.617.74 5.173 1.666 4.985 2.819c-.07.436-.119 1.04-.119 1.503 0 .544.064 1.24.155 1.721.02.107.031.202.023.208a8.12 8.12 0 0 1-.187.152 5.324 5.324 0 0 0-.949 1.02 5.49 5.49 0 0 0-.94 2.339 6.625 6.625 0 0 0-.023 1.357c.091.78.325 1.438.727 2.04l.13.195-.037.064c-.269.452-.498 1.105-.605 1.732-.084.496-.095.629-.095 1.294 0 .67.009.803.088 1.266.095.555.288 1.143.503 1.534.071.128.243.393.264.407.007.003-.014.067-.046.141a7.405 7.405 0 0 0-.548 1.873c-.062.417-.071.552-.071.991 0 .56.031.832.148 1.279L3.42 24h1.478l-.05-.091c-.297-.552-.325-1.575-.068-2.597.117-.472.25-.819.498-1.296l.148-.29v-.177c0-.165-.003-.184-.057-.293a.915.915 0 0 0-.194-.25 1.74 1.74 0 0 1-.385-.543c-.424-.92-.506-2.286-.208-3.451.124-.486.329-.918.544-1.154a.787.787 0 0 0 .223-.531c0-.195-.07-.355-.224-.522a3.136 3.136 0 0 1-.817-1.729c-.14-.96.114-2.005.69-2.834.563-.814 1.353-1.336 2.237-1.475.199-.033.57-.028.776.01.226.04.367.028.512-.041.179-.085.268-.19.374-.431.093-.215.165-.333.36-.576.234-.29.46-.489.822-.729.413-.27.884-.467 1.352-.561.17-.035.25-.04.569-.04.319 0 .398.005.569.04a4.07 4.07 0 0 1 1.914.997c.117.109.398.457.488.602.034.057.095.177.132.267.105.241.195.346.374.43.14.068.286.082.503.045.343-.058.607-.053.943.016 1.144.23 2.14 1.173 2.581 2.437.385 1.108.276 2.267-.296 3.153-.097.15-.193.27-.333.419-.301.322-.301.722-.001 1.053.493.539.801 1.866.708 3.036-.062.772-.26 1.463-.533 1.854a2.096 2.096 0 0 1-.224.258.916.916 0 0 0-.194.25c-.054.109-.057.128-.057.293v.178l.148.29c.248.476.38.823.498 1.295.253 1.008.231 2.01-.059 2.581a.845.845 0 0 0-.044.098c0 .006.329.009.732.009h.73l.02-.074.036-.134c.019-.076.057-.3.088-.516.029-.217.029-1.016 0-1.258-.11-.875-.295-1.57-.597-2.226-.032-.074-.053-.138-.046-.141.008-.005.057-.074.108-.152.376-.569.607-1.284.724-2.228.031-.26.031-1.378 0-1.628-.083-.645-.182-1.082-.348-1.525a6.083 6.083 0 0 0-.329-.7l-.038-.064.131-.194c.402-.604.636-1.262.727-2.04a6.625 6.625 0 0 0-.024-1.358 5.512 5.512 0 0 0-.939-2.339 5.325 5.325 0 0 0-.95-1.02 8.097 8.097 0 0 1-.186-.152.692.692 0 0 1 .023-.208c.208-1.087.201-2.443-.017-3.503-.19-.924-.535-1.658-.98-2.082-.354-.338-.716-.482-1.15-.455-.996.059-1.8 1.205-2.116 3.01a6.805 6.805 0 0 0-.097.726c0 .036-.007.066-.015.066a.96.96 0 0 1-.149-.078A4.857 4.857 0 0 0 12 3.03c-.832 0-1.687.243-2.456.698a.958.958 0 0 1-.148.078c-.008 0-.015-.03-.015-.066a6.71 6.71 0 0 0-.097-.725C8.997 1.392 8.337.319 7.46.048a2.096 2.096 0 0 0-.585-.041Zm.293 1.402c.248.197.523.759.682 1.388.03.113.06.244.069.292.007.047.026.152.041.233.067.365.098.76.102 1.24l.002.475-.12.175-.118.178h-.278c-.324 0-.646.041-.954.124l-.238.06c-.033.007-.038-.003-.057-.144a8.438 8.438 0 0 1 .016-2.323c.124-.788.413-1.501.696-1.711.067-.05.079-.049.157.013zm9.825-.012c.17.126.358.46.498.888.28.854.36 2.028.212 3.145-.019.14-.024.151-.057.144l-.238-.06a3.693 3.693 0 0 0-.954-.124h-.278l-.119-.178-.119-.175.002-.474c.004-.669.066-1.19.214-1.772.157-.623.434-1.185.68-1.382.078-.062.09-.063.159-.012z"/>"#
            }
            Self::Group => {
                r#"<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/><circle cx="9" cy="7" r="4"/>"#
            }
            Self::Grip => {
                r#"<circle cx="9" cy="12" r="1"/><circle cx="9" cy="5" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="15" cy="19" r="1"/>"#
            }
            Self::More => {
                r#"<circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>"#
            }
            Self::External => {
                r#"<path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>"#
            }
            Self::Download => {
                r#"<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>"#
            }
            Self::Clock => r#"<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>"#,
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
            Self::Changes => {
                r#"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M9 10h6M12 7v6M9 17h6"/>"#
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
            Self::Up => r#"<path d="m18 15-6-6-6 6"/>"#,
            Self::Down => r#"<path d="m6 9 6 6 6-6"/>"#,
            Self::Columns => {
                r#"<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/>"#
            }
            Self::Mic => {
                r#"<path d="M12 19v3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><rect x="9" y="2" width="6" height="13" rx="3"/>"#
            }
            Self::Folder => {
                r#"<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>"#
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
    // Narrow on a phone, where the name after it needs the room.
    let slot = view()
        .w(px(if touch() { 8.0 } else { fit::icon_slot() }))
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
        .border_color(colors::EDGE)
        .transition_colors()
        .duration(motion::FAST)
        .focus(|s| focus_band(s.border_color(colors::ACCENT)))
}

/// A field for a few lines of text (instructions, a note).
pub fn area(placeholder: &'static str, value: Signal<String>) -> Input {
    field(placeholder, value)
        .multiline(true)
        .h(px(size::CONTROL_MD * 3.0))
        .py_2()
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
                // Counts keep their width as they change.
                .tabular_nums()
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

/// One of a few ways, all in view (a new conversation's Chat or In a
/// project): a track, with the chosen one raised on it. `label` names the
/// group for assistive technology.
pub fn segmented(
    label: &'static str,
    options: &[&'static str],
    chosen: impl Fn() -> usize + Copy + 'static,
    choose: impl Fn(usize) + Clone + 'static,
) -> View {
    use nsh_base::{radio, radio_group};
    let terminal = idiom() == nsh_core::platform::Idiom::Terminal;
    // One stop of Tab whose arrow keys choose (the radio pattern: a choice
    // of one, as a mode switch is). Focus is a band or ring; the chosen
    // option is a raised segment, and in a terminal, which has no shadows,
    // the radio's own (●) mark.
    let mut group = radio_group()
        .flex_row()
        .p(px(3.0))
        .gap(px(2.0))
        .rounded(px(radius::CONTROL + 3.0))
        .bg(colors::HOVER);
    for (at, option) in options.iter().copied().enumerate() {
        let choose = choose.clone();
        group = group.child(
            radio()
                .checked(move || chosen() == at)
                .on_change(move |_: &bool| {
                    haptic(nsh_core::Haptic::Selection);
                    choose(at)
                })
                .h(px(if touch() { 38.0 } else { 26.0 }))
                .px(px(if touch() { 18.0 } else { 12.0 }))
                .items_center()
                .justify_center()
                .rounded(px(radius::CONTROL))
                .transition_colors()
                .duration(motion::FAST)
                .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
                .focus(focus_band)
                .style_when(
                    move || chosen() == at && !terminal,
                    |s| s.bg(colors::SURFACE).shadow_sm(),
                )
                .accessibility_label(option)
                .child(
                    text(option)
                        .text_size(px(fit::small()))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(move || {
                            if chosen() == at {
                                colors::TEXT
                            } else {
                                colors::MUTED
                            }
                        }),
                ),
        );
    }
    // What screen readers call the group.
    view().label(label).child(group)
}

/// One of a few filters side by side: outlined, and filled with the accent
/// when chosen.
pub fn chip(
    label: impl Into<SharedString>,
    selected: impl Fn() -> bool + Copy + 'static,
    on_press: impl Fn() + 'static,
) -> nsh_base::ButtonElement {
    let label = label.into();
    button()
        .h(px(if touch() { 32.0 } else { 24.0 }))
        .px_2()
        .items_center()
        .justify_center()
        .rounded(px(radius::CONTROL))
        .border_1()
        .border_color(colors::BORDER)
        .transition_colors()
        .duration(motion::FAST)
        .hover(|s| s.bg(colors::HOVER))
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .focus(focus_band)
        .style_when(selected, |s| {
            s.bg(Tone::Accent.tint()).border_color(colors::ACCENT)
        })
        .accessibility_label(label.clone())
        .on_press(on_press)
        .child(
            text(label)
                .text_size(px(type_scale::XS))
                .text_color(move || {
                    if selected() {
                        colors::ACCENT
                    } else {
                        colors::MUTED
                    }
                }),
        )
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
        .focus(focus_band)
        .style_when(selected, |s| s.bg(colors::SELECTED))
        .on_press(on_press)
}

/// A picture sent with a message, pressable to open it full size. A
/// picture from a link has no size until it loads, so it is laid out from
/// `pixels` (its width and height): at most 320 points wide and 360 tall,
/// and never wider than 80% of the column. Without them it gets a 4:3 box
/// and fits inside it.
pub fn picture(
    src: String,
    label: String,
    pixels: Option<(u32, u32)>,
    on_press: impl Fn() + 'static,
) -> View {
    const MAX_W: f32 = 320.0;
    const MAX_H: f32 = 360.0;
    let (width, ratio, fit) = match pixels {
        Some((w, h)) if w > 0 && h > 0 => {
            let (w, h) = (w as f32, h as f32);
            let ratio = w / h;
            (w.min(MAX_W).min(MAX_H * ratio), ratio, ResizeMode::Cover)
        }
        _ => (240.0, 4.0 / 3.0, ResizeMode::Contain),
    };
    view()
        .w(px(width))
        .max_w(percentage(0.8))
        .aspect_ratio(ratio)
        .test_id("picture")
        .child(
            button()
                .w_full()
                .h_full()
                .rounded(px(radius::SURFACE))
                .overflow_hidden()
                .border_1()
                .border_color(colors::BORDER)
                .bg(colors::SURFACE)
                .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
                .focus(focus_band)
                .accessibility_label(label)
                .on_press(on_press)
                .child(image(src).w_full().h_full().object_fit(fit)),
        )
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

/// A recessed block inside a surface: what a card last did, a quote.
pub fn inset() -> View {
    view()
        .rounded(px(radius::CONTROL))
        .bg(colors::INSET)
        .px_3()
        .py_2()
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
                .text_size(px(fit::title()))
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

/// Work going on, in a row's end: a small ring that turns (still, with
/// reduced motion). `label` is what screen readers hear.
pub fn spinner(label: &'static str) -> View {
    ring(label, true)
}

/// Work that was going on when the connection dropped: the same ring,
/// faint and still, since nobody can say yet whether it has finished.
pub fn spinner_stale(label: &'static str) -> View {
    ring(label, false)
}

fn ring(label: &'static str, live: bool) -> View {
    let tint = if live { colors::ACCENT } else { colors::FAINT };
    let ring = svg(nsh_core::element::ImageSrc::svg_markup(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke-width="2" stroke-linecap="round"><circle cx="8" cy="8" r="6" stroke="currentColor" stroke-opacity="0.25"/><path d="M8 2a6 6 0 0 1 6 6" stroke="currentColor"/></svg>"#,
    ))
    .w(px(14.0))
    .h(px(14.0))
    .text_color(tint);
    let ring = if reduce_motion() || !live {
        ring
    } else {
        ring.animate_spin()
    };
    view()
        .w(px(fit::icon_slot()))
        .h(px(fit::icon_slot()))
        .items_center()
        .justify_center()
        .role(AccessibilityRole::Progressbar)
        .label(label)
        .child(if idiom() == nsh_core::platform::Idiom::Terminal {
            view().child(text("◌").text_color(tint))
        } else {
            view().child(ring)
        })
}

/// A dot on a button's corner: something behind it wants a look
/// (`tone`: amber waits on the owner, accent works).
pub fn corner_dot(tone: impl Fn() -> Tone + 'static) -> View {
    view()
        .absolute()
        .top(px(6.0))
        .right(px(6.0))
        .w(px(8.0))
        .h(px(8.0))
        .rounded_full()
        .border(px(1.5))
        .border_color(colors::BACKGROUND)
        .bg(move || tone().color())
        .accessibility_elements_hidden(true)
}

/// A count on an icon button's corner (a coding conversation's changed
/// files): small, in the accent, ringed off what it sits on.
pub fn corner_count(count: impl Fn() -> usize + 'static) -> View {
    view()
        .absolute()
        .top(px(2.0))
        .right(px(0.0))
        .min_w(px(16.0))
        .h(px(16.0))
        .px(px(4.0))
        .rounded_full()
        .items_center()
        .justify_center()
        .border(px(1.5))
        .border_color(colors::BACKGROUND)
        .bg(colors::ACCENT)
        .accessibility_elements_hidden(true)
        .child(
            text(move || count().to_string())
                .tabular_nums()
                .text_size(px(10.0))
                .line_height(px(12.0))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(colors::ON_STATUS),
        )
}

/// The app's name at the top as a button home, as other chat apps do:
/// `inset` is the padding either side, so a caller lines its text up.
pub fn name_btn(label: &'static str, inset: f32, on_press: impl Fn() + 'static) -> View {
    with_hint(
        "Home",
        frame(Variant::Ghost, Size::Sm, motion::PRESS_SCALE)
            .px(px(inset))
            .accessibility_label(SharedString::from(format!("{label}, home")))
            .on_press(on_press)
            .child(
                text(label)
                    .text_size(px(fit::body()))
                    .font_weight(FontWeight::SEMIBOLD)
                    .accessibility_elements_hidden(true),
            ),
    )
}

/// A state as one small dot beside a name (the connection beside the
/// app's): `label` is what screen readers and the tooltip say, and a
/// press says it too, for a phone with no hover.
pub fn status_dot(
    tone: impl Fn() -> Tone + 'static,
    label: impl Fn() -> String + Clone + 'static,
    on_press: impl Fn() + 'static,
) -> View {
    let hint = label.clone();
    with_hint(
        move || SharedString::from(hint()),
        // The whole target to press, the dot at its start: it sits by the
        // name it belongs to.
        frame(Variant::Ghost, Size::Sm, motion::PRESS_SCALE_ICON)
            .w(px(Size::Sm.height()))
            .justify_start()
            .pl(px(2.0))
            .accessibility_label(move || SharedString::from(label()))
            .on_press(on_press)
            // The "●" glyph, a size up from a [`status`] line's: alone by a
            // name it carries the state by itself.
            .child(
                text(Glyph::Dot.symbol())
                    .text_size(px(12.0))
                    .text_color(move || tone().color())
                    .accessibility_elements_hidden(true),
            ),
    )
}

/// A small coloured dot and a word, at a row's end: "Needs you".
pub fn badge_dot(label: &'static str, tone: Tone) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_1()
        .child(
            view()
                .w(px(7.0))
                .h(px(7.0))
                .rounded_full()
                .bg(tone.color())
                .accessibility_elements_hidden(true),
        )
        .child(
            text(label)
                .text_size(px(if touch() { 12.0 } else { type_scale::XS }))
                .font_weight(FontWeight::MEDIUM)
                .text_color(tone.color()),
        )
}

/// Work going on: three dots that pulse in turn.
pub fn working(label: impl IntoReactive<SharedString>) -> View {
    // Three dots that brighten in turn, a wave that says it is alive
    // without moving the line.
    let mut dots = view()
        .flex_row()
        .flex_shrink(0.0)
        .items_center()
        .gap(px(3.0))
        .accessibility_elements_hidden(true);
    for n in 0..3 {
        dots = dots.child(
            view()
                .w(px(5.0))
                .h(px(5.0))
                .rounded_full()
                .bg(colors::ACCENT)
                .animation(
                    KeyframeAnimation::new(Keyframes::Pulse)
                        .duration(1200.0)
                        .delay(n as f32 * 200.0),
                ),
        );
    }
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .min_w(px(0.0))
        .child(dots)
        .child(
            text(label)
                // A long status ends in "…" instead of running under
                // what follows it on the line.
                .flex_1()
                .min_w(px(0.0))
                .text_size(px(type_scale::SM))
                // "· 3 tools so far", ticking: the line does not shift.
                .tabular_nums()
                .number_of_lines(1)
                .text_color(colors::MUTED),
        )
}

/// A quiet count at a row's end that opens and closes what is under it
/// ("12 tools ›", the chevron turning down when open): no border, so it
/// reads as part of the line it ends.
pub fn count_toggle(
    label: impl IntoReactive<SharedString>,
    open: Signal<bool>,
) -> nsh_base::ButtonElement {
    button()
        .flex_row()
        .items_center()
        .gap_1()
        .min_h(px(fit::row() - 8.0))
        .px_2()
        .rounded(px(radius::CONTROL))
        .transition_colors()
        .duration(motion::FAST)
        .hover(|s| s.bg(colors::HOVER))
        .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
        .focus(focus_band)
        .child(
            text(label)
                .number_of_lines(1)
                .tabular_nums()
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
        )
        .child(
            icon(Glyph::Chevron)
                .text_color(colors::FAINT)
                .rotate(move || deg(if open.get() { 90.0 } else { 0.0 }))
                .transition_transform()
                .duration(motion::FAST)
                .accessibility_elements_hidden(true),
        )
        .on_press(move || open.update(|o| *o = !*o))
}

/// A quiet pill that opens and closes what is under it: a chevron that
/// turns, then its label. The owner of `open` renders the content, in an
/// [`super::unfold`].
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
        .focus(focus_band)
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
                .flex_shrink(1.0)
                .min_w(px(0.0))
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
        )
        // Never wider than the column: a long label ends in "…".
        .max_w(percentage(1.0))
        .on_press(move || open.update(|o| *o = !*o))
}

/// One tool of a reply's activity list: its number and name. With details
/// to show it is a button whose chevron turns as `open` flips; the caller
/// renders the details under it ([`call_detail`]).
pub fn tool_row(number: usize, name: String, open: Option<Signal<bool>>) -> View {
    let label = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(
            text(format!("{number}."))
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        )
        .child(code(name.clone()).text_color(colors::MUTED));
    let Some(open) = open else {
        return view().pl(px(18.0)).child(label);
    };
    view().child(
        button()
            .flex_row()
            .items_center()
            .gap_1()
            .self_start()
            .min_h(px(fit::row() - 6.0))
            .pr_2()
            .rounded(px(radius::CONTROL))
            .transition_colors()
            .duration(motion::FAST)
            .hover(|s| s.bg(colors::HOVER))
            .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
            .focus(focus_band)
            .accessibility_label(format!("{name}: show what it was given and returned"))
            .child(
                icon(Glyph::Chevron)
                    .text_color(colors::FAINT)
                    .rotate(move || deg(if open.get() { 90.0 } else { 0.0 }))
                    .transition_transform()
                    .duration(motion::FAST)
                    .accessibility_elements_hidden(true),
            )
            .child(label)
            .on_press(move || open.update(|o| *o = !*o)),
    )
}

/// What one tool call was given and gave back, under its [`tool_row`]:
/// each a quiet label over selectable monospace text on an inset well.
pub fn call_detail(input: String, output: String) -> View {
    let part = |label: &'static str, body: String| {
        view()
            .gap_1()
            .child(
                text(label)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            )
            .child(
                view()
                    .w_full()
                    .px_2()
                    .py_1()
                    .rounded(px(radius::SMALL))
                    .bg(colors::INSET)
                    .child(
                        text(body)
                            .selectable(true)
                            .font_family(MONO)
                            .text_size(px(type_scale::XS))
                            .text_color(colors::TEXT),
                    ),
            )
    };
    let mut detail = view().pl(px(22.0)).pb_1().gap_2();
    if !input.is_empty() {
        detail = detail.child(part("Input", input));
    }
    detail = detail.child(if output.is_empty() {
        view().child(
            text("No result recorded")
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        )
    } else {
        part("Output", output)
    });
    detail
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

/// How a focused control shows in a terminal, which draws no outline or
/// border: a band of the selection wash behind it. Elsewhere nothing more
/// (the focus ring does it).
pub fn focus_band(s: nsh_core::element::StyleRefinement) -> nsh_core::element::StyleRefinement {
    if idiom() == nsh_core::platform::Idiom::Terminal {
        s.bg(colors::SELECTED)
    } else {
        s
    }
}

/// `control` under a visible caption: a form's field says what it is in
/// every idiom (a placeholder goes once something is typed, and an
/// accessibility label is not drawn).
pub fn labeled(caption: &'static str, control: impl IntoElement) -> View {
    view()
        .gap(px(4.0))
        .child(
            text(caption)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::MUTED),
        )
        .child(control)
}
