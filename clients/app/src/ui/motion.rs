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

/// The scale a pressed button settles to.
pub const PRESS_SCALE: f32 = 0.97;
/// The scale a pressed icon button settles to.
pub const PRESS_SCALE_ICON: f32 = 0.94;
