//! The app's look as tokens: a quiet graphite workbench (paper in light
//! appearance), one teal accent for the agent's own activity, and status
//! roles that always travel with a word or a glyph, never colour alone.
//!
//! Colours are [`AdaptiveColor`]s, so light and dark switch natively. In a
//! terminal they are read as meanings (a neutral is the terminal's
//! foreground, a saturated one its accent of that hue), so the same tokens
//! work there.

use nsh_core::element::{AdaptiveColor, Rgba, rgb, rgba};

/// `#rrggbb` at `alpha`.
const fn rgb_a(hex: u32, alpha: f32) -> Rgba {
    rgba(hex << 8 | (alpha * 255.0 + 0.5) as u32)
}

const fn tone(light: Rgba, dark: Rgba) -> AdaptiveColor {
    AdaptiveColor {
        light: light.to_hsla(),
        dark: dark.to_hsla(),
    }
}

/// Surfaces, text and lines.
pub mod colors {
    use super::{AdaptiveColor, rgb, tone};

    /// The window behind everything.
    pub const BACKGROUND: AdaptiveColor = tone(rgb(0xf7f7f5), rgb(0x0f1012));
    /// The sidebar and the inspector: a step away from the page.
    pub const SIDEBAR: AdaptiveColor = tone(rgb(0xefefec), rgb(0x141518));
    /// A card, a popup, the composer.
    pub const SURFACE: AdaptiveColor = tone(rgb(0xffffff), rgb(0x1a1b1f));
    /// A surface inside a surface: code, quoted tool output.
    pub const INSET: AdaptiveColor = tone(rgb(0xf1f1ee), rgb(0x0c0d0f));
    /// Hover wash on rows and quiet buttons.
    pub const HOVER: AdaptiveColor = tone(rgb(0xe8e8e4), rgb(0x22242a));
    /// Pressed and selected rows.
    pub const SELECTED: AdaptiveColor = tone(rgb(0xdfdfda), rgb(0x2a2c33));
    /// Borders of surfaces and controls.
    pub const BORDER: AdaptiveColor = tone(rgb(0xdcdcd6), rgb(0x2a2c31));
    /// Hairlines between rows and regions.
    pub const LINE: AdaptiveColor = tone(rgb(0xe6e6e1), rgb(0x202227));
    /// Body text.
    pub const TEXT: AdaptiveColor = tone(rgb(0x16171a), rgb(0xeceef1));
    /// Secondary text.
    pub const MUTED: AdaptiveColor = tone(rgb(0x5d6168), rgb(0x9196a0));
    /// Quiet text: timestamps, counts, hints.
    pub const FAINT: AdaptiveColor = tone(rgb(0x92969d), rgb(0x60646d));
    /// The agent's colour: running work, links, the focus ring, the caret.
    pub const ACCENT: AdaptiveColor = tone(rgb(0x0b7a6e), rgb(0x45d2bf));
    /// The main action's fill.
    pub const PRIMARY: AdaptiveColor = tone(rgb(0x16171a), rgb(0xeceef1));
    /// Text on [`PRIMARY`].
    pub const ON_PRIMARY: AdaptiveColor = tone(rgb(0xffffff), rgb(0x111214));
    /// Text on a status fill.
    pub const ON_STATUS: AdaptiveColor = tone(rgb(0xffffff), rgb(0x0f1012));
    /// The user's own messages.
    pub const BUBBLE: AdaptiveColor = tone(rgb(0xe9e9e5), rgb(0x23252b));
    /// Done, approved.
    pub const SUCCESS: AdaptiveColor = tone(rgb(0x1d7f45), rgb(0x55d68a));
    /// Needs a look.
    pub const WARNING: AdaptiveColor = tone(rgb(0xa45d00), rgb(0xf5b84a));
    /// Failed, denied, destructive.
    pub const DANGER: AdaptiveColor = tone(rgb(0xc2332f), rgb(0xff6b66));
    /// Information.
    pub const INFO: AdaptiveColor = tone(rgb(0x2f5fd0), rgb(0x83a9ff));
}

/// Soft fills behind status text (badges, banners, tool rows).
pub mod tints {
    use super::{AdaptiveColor, rgb_a, tone};

    pub const ACCENT: AdaptiveColor = tone(rgb_a(0x0b7a6e, 0.10), rgb_a(0x45d2bf, 0.13));
    pub const SUCCESS: AdaptiveColor = tone(rgb_a(0x1d7f45, 0.10), rgb_a(0x55d68a, 0.13));
    pub const WARNING: AdaptiveColor = tone(rgb_a(0xa45d00, 0.11), rgb_a(0xf5b84a, 0.14));
    pub const DANGER: AdaptiveColor = tone(rgb_a(0xc2332f, 0.10), rgb_a(0xff6b66, 0.14));
    pub const INFO: AdaptiveColor = tone(rgb_a(0x2f5fd0, 0.10), rgb_a(0x83a9ff, 0.14));
    pub const NEUTRAL: AdaptiveColor = tone(rgb_a(0x16171a, 0.06), rgb_a(0xeceef1, 0.08));
}

/// What a status means, whatever it is attached to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger,
    Info,
}

impl Tone {
    pub const fn color(self) -> AdaptiveColor {
        match self {
            Self::Neutral => colors::MUTED,
            Self::Accent => colors::ACCENT,
            Self::Success => colors::SUCCESS,
            Self::Warning => colors::WARNING,
            Self::Danger => colors::DANGER,
            Self::Info => colors::INFO,
        }
    }

    pub const fn tint(self) -> AdaptiveColor {
        match self {
            Self::Neutral => tints::NEUTRAL,
            Self::Accent => tints::ACCENT,
            Self::Success => tints::SUCCESS,
            Self::Warning => tints::WARNING,
            Self::Danger => tints::DANGER,
            Self::Info => tints::INFO,
        }
    }
}

/// Corner radii, largest surfaces softest; nested shapes concentric.
pub mod radius {
    /// Dialogs, the composer, cards.
    pub const SURFACE: f32 = 12.0;
    /// Controls: buttons, inputs, rows.
    pub const CONTROL: f32 = 7.0;
    /// Small parts: badges, keys.
    pub const SMALL: f32 = 4.0;
}

/// The type scale, in points.
pub mod text {
    /// Labels, metadata.
    pub const XS: f32 = 11.0;
    /// Secondary and dense rows.
    pub const SM: f32 = 12.5;
    /// Body and controls.
    pub const BASE: f32 = 13.5;
    /// Reading text (messages).
    pub const READ: f32 = 14.5;
    /// Page titles.
    pub const TITLE: f32 = 17.0;
    /// Line heights for each.
    pub const LH_SM: f32 = 17.0;
    pub const LH_BASE: f32 = 19.0;
    pub const LH_READ: f32 = 22.0;
}

/// Sizes of the desktop and web family.
pub mod size {
    /// Control heights.
    pub const CONTROL_SM: f32 = 26.0;
    pub const CONTROL_MD: f32 = 32.0;
    /// A sidebar row.
    pub const ROW: f32 = 30.0;
    /// The sidebar's width.
    pub const SIDEBAR: f32 = 248.0;
    /// The inspector's width.
    pub const INSPECTOR: f32 = 320.0;
    /// The conversation's readable column.
    pub const COLUMN: f32 = 760.0;
    /// Icon glyph box.
    pub const ICON: f32 = 16.0;
}

/// Monospace for code, identifiers and keys.
pub const MONO: &str = "monospace";
