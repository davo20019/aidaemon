//! Motion tokens. Things move only to explain a change: a hover wash fades
//! in fast and out without lag, a press settles, arriving content rises a
//! little, popups grow from their trigger. Reduced motion is honoured by
//! the platform (transitions take their end state at once).

/// Hover washes and colour changes.
pub const FAST: f32 = 120.0;
/// Presses, toggles, small reveals.
pub const BASE: f32 = 180.0;
/// Arriving rows, panels opening.
pub const ENTER: f32 = 260.0;

/// A narrow window's drawer sliding out: settling, so it eases to the
/// edge rather than leaving at full speed.
pub const DRAWER_OUT: f32 = 220.0;

/// The scale a pressed button settles to.
pub const PRESS_SCALE: f32 = 0.97;
/// The scale a pressed icon button settles to.
pub const PRESS_SCALE_ICON: f32 = 0.94;

/// Fades the next state change: a pane opening or closing, a page
/// replacing another. What comes fades in, what goes fades out; nothing
/// grows (a view shown from hidden would grow out of a corner).
pub fn next_layout() {
    animate(nsh_core::element::LayoutAnimation {
        create: Some(fade()),
        update: None,
        delete: Some(fade()),
    });
}

/// Animates a section unfolding: its content fades in and what is below it
/// moves down to make room (and back up when it folds).
pub fn next_unfold() {
    animate(nsh_core::element::LayoutAnimation::all(fade()));
}

/// An [`super::unfold`] opening: a soft spring, so a long list arrives
/// rather than snaps.
pub fn unfold_open() -> nsh_core::element::LayoutTransition {
    nsh_core::element::LayoutTransition::spring(nsh_core::element::Spring::new(ENTER + 60.0))
}

/// An [`super::unfold`] closing: quicker, out of the way.
pub fn unfold_close() -> nsh_core::element::LayoutTransition {
    nsh_core::element::LayoutTransition::spring(nsh_core::element::Spring::new(BASE))
}

fn fade() -> nsh_core::element::LayoutTransition {
    nsh_core::element::LayoutTransition::new(nsh_core::element::LayoutCurve::EaseInEaseOut, BASE)
}

/// A terminal has only cells: it changes at once.
fn animate(animation: nsh_core::element::LayoutAnimation) {
    if nsh_core::element::idiom() == nsh_core::platform::Idiom::Terminal {
        return;
    }
    nsh_core::prelude::animate_layout(animation);
}
