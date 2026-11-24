//! Pull to refresh: a list pulled down past its top reads everything again.
//! The pull itself drives the indicator on the UI thread (no engine work per
//! frame): a round chip travels in with the finger, grows and fades in, its
//! ring winds up as the pull nears the point where letting go refreshes,
//! and pops a little when it gets there. While the refresh runs the ring
//! turns and its centre breathes; when it ends the list settles back and
//! the chip leaves with it.
//!
//! A long conversation is read from its bottom, where a pull down cannot
//! reach the top: [`refreshable_at_end`] also refreshes when it is pulled
//! up past its end, with the same chip rising from the bottom.
//!
//! [`mark`] is the one part with a character: a mascot can take its place
//! later without touching the chip's motion.

use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

use nsh_core::prelude::*;

use super::theme::colors;

/// How tall the indicator is: how far a pull goes before letting go
/// refreshes, and the room kept above the list while it does.
pub const PULL_HEIGHT: f32 = 64.0;

/// The least time the indicator spins, so a quick answer still reads as
/// a refresh rather than a flicker.
const LEAST_SPIN: Duration = Duration::from_millis(650);

/// Where the chip arms (letting go now refreshes): a moment before the
/// full height, so the pop lands as the finger reaches it.
const ARMED: f32 = PULL_HEIGHT * 0.88;

/// `list` (a scroll view or list) refreshes when pulled down past its top:
/// `refresh` starts it, `refreshing` says the reads are still out. A
/// terminal has no pull: the list is returned as it is.
pub fn refreshable<S>(
    list: S,
    refreshing: impl Fn() -> bool + Copy + 'static,
    refresh: impl Fn() + Copy + 'static,
) -> View
where
    S: Scrollable + IntoElement + 'static,
{
    if idiom() == nsh_core::platform::Idiom::Terminal {
        return view().flex_1().child(list);
    }
    let pull = animated_value(0.0);
    let hold = signal(false);
    let shown = computed(move || hold.get() || refreshing());
    let list = list
        .refreshing(shown)
        .pull_value(&pull)
        .refresh_indicator(Some(px(PULL_HEIGHT)))
        .on_refresh(move |_: &()| {
            hold.set(true);
            set_timeout(LEAST_SPIN, move || hold.set(false));
            refresh();
        });
    view().flex_1().child(list).child(indicator(pull, shown))
}

/// As [`refreshable`], and pulled up past its end it refreshes too, the
/// chip rising from the bottom. `test_id` marks the list: in a browser the
/// page watches that pull itself (its scroll view has no bounce to read);
/// on a phone or tablet the list's own bounce past its end is the pull.
pub fn refreshable_at_end<S>(
    list: S,
    test_id: &'static str,
    refreshing: impl Fn() -> bool + Copy + 'static,
    refresh: impl Fn() + Copy + 'static,
) -> View
where
    S: Scrollable + Interactive + IntoElement + 'static,
{
    let idiom = idiom();
    if idiom == nsh_core::platform::Idiom::Terminal {
        return refreshable(list, refreshing, refresh);
    }
    let up = animated_value(0.0);
    // A refresh started from the end is out (and held for the least spin).
    let spinning = signal(false);
    let least = signal(false);
    let settle = move |to: f32| {
        let _ = Animation::timing(&up, to, Timing::new(250.0, Easing::EaseOut)).start();
    };
    // A pull reported at `pull`, `released` when the finger lifted.
    let pulled = move |pull: f32, released: bool| {
        if untracked(|| spinning.get()) {
            return;
        }
        if !released {
            up.set(pull.max(0.0));
        } else if pull >= ARMED {
            spinning.set(true);
            least.set(true);
            set_timeout(LEAST_SPIN, move || least.set(false));
            settle(PULL_HEIGHT);
            refresh();
        } else {
            settle(0.0);
        }
    };
    effect(move || {
        if spinning.get() && !least.get() && !refreshing() {
            spinning.set(false);
            settle(0.0);
        }
    });
    #[cfg(target_arch = "wasm32")]
    {
        use futures::StreamExt;
        let mut pulls = crate::link::web::end_pulls(test_id);
        spawn(async move {
            while let Some((pull, released)) = pulls.next().await {
                pulled(pull, released);
            }
        });
    }
    // Native scroll views bounce past their end while dragged: that
    // overshoot is the pull. Only a drag counts, not a fling's bounce.
    let touch = matches!(
        idiom,
        nsh_core::platform::Idiom::Phone | nsh_core::platform::Idiom::Pad
    ) && !cfg!(target_arch = "wasm32");
    let dragging = Rc::new(Cell::new(false));
    let over = Rc::new(Cell::new(0.0_f32));
    let list = list.test_id(test_id);
    let list = if touch {
        let (d1, d2, d3) = (dragging.clone(), dragging.clone(), dragging);
        let (o1, o2) = (over.clone(), over);
        list.on_scroll(move |e: &ScrollEvent| {
            let end = e.content_size.height.max(e.size.height);
            let past = (e.offset.y + e.size.height - end).0.max(0.0) * 0.5;
            o1.set(past);
            if d1.get() {
                pulled(past, false);
            }
        })
        .on_scroll_drag_start(move |_: &Point<Points>| d2.set(true))
        .on_scroll_drag_end(move |_: &Point<Points>| {
            d3.set(false);
            pulled(o2.get(), true);
        })
    } else {
        list
    };
    let shown = computed(move || spinning.get());
    view()
        .flex_1()
        .child(refreshable(list, refreshing, refresh))
        .child(indicator_at(up, shown, Edge::Bottom))
}

#[derive(Clone, Copy, PartialEq)]
enum Edge {
    Top,
    Bottom,
}

/// The chip over the list's top, following `pull`.
fn indicator(pull: AnimatedValue, refreshing: Memo<bool>) -> View {
    indicator_at(pull, refreshing, Edge::Top)
}

/// The chip over the list's `edge`, following `pull` in from it.
fn indicator_at(pull: AnimatedValue, refreshing: Memo<bool>, edge: Edge) -> View {
    let h = PULL_HEIGHT;
    let away = if edge == Edge::Top { -30.0 } else { 30.0 };
    let place = view().pointer_events(PointerEvents::None).absolute();
    let place = if edge == Edge::Top {
        place.top_0()
    } else {
        place.bottom_0()
    };
    place
        .left_0()
        .right_0()
        .h(px(h))
        .items_center()
        .justify_center()
        .role(AccessibilityRole::Progressbar)
        .label("Refreshing")
        .accessibility_elements_hidden(move || !refreshing.get())
        .child(
            view()
                .size(px(38.0))
                .rounded_full()
                .border_1()
                .bg(colors::SURFACE)
                .shadow_sm()
                .items_center()
                .justify_center()
                .bind_border_color(
                    &pull.interpolate_colors(
                        &[ARMED - 8.0, ARMED],
                        &[colors::BORDER, colors::ACCENT],
                    ),
                )
                .bind_opacity(&pull.interpolate(&[0.0, 20.0], &[0.0, 1.0]))
                .bind_translate_y(&pull.interpolate(&[0.0, h], &[away, 0.0]))
                // Grows with the pull, pops as it arms, settles.
                .bind_scale(
                    &pull.interpolate(&[0.0, ARMED - 8.0, ARMED, h * 1.4], &[0.5, 0.9, 1.1, 1.0]),
                )
                .child(mark(pull, refreshing)),
        )
}

/// What the chip holds: a ring that winds up with the pull and turns while
/// refreshing, around a dot that fills as the pull arms and breathes while
/// it works. Reduced motion keeps it still while refreshing.
pub fn mark(pull: AnimatedValue, refreshing: Memo<bool>) -> View {
    let h = PULL_HEIGHT;
    let turn = animated_value(0.0);
    let breath = animated_value(1.0);
    let spin = Animation::timing(&turn, 360.0, Timing::new(850.0, Easing::Linear)).looped(None);
    let breathe = Animation::sequence([
        Animation::timing(&breath, 0.55, Timing::new(420.0, Easing::EaseInOut)),
        Animation::timing(&breath, 1.0, Timing::new(420.0, Easing::EaseInOut)),
    ])
    .looped(None);
    let (spinning, breathing) = (spin.clone(), breathe.clone());
    effect(move || {
        if refreshing.get() && !reduce_motion() {
            turn.set(0.0);
            breath.set(1.0);
            let _ = spinning.start();
            let _ = breathing.start();
        } else {
            spinning.stop();
            breathing.stop();
            breath.set(1.0);
        }
    });
    cleanup(move || {
        spin.stop();
        breathe.stop();
    });
    let ring = svg(nsh_core::element::ImageSrc::svg_markup(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22" fill="none" stroke-width="2.25" stroke-linecap="round"><circle cx="11" cy="11" r="8.5" stroke="currentColor" stroke-opacity="0.18"/><path d="M11 2.5a8.5 8.5 0 1 1-8.5 8.5" stroke="currentColor"/></svg>"#,
    ))
    .w(px(22.0))
    .h(px(22.0))
    .text_color(colors::ACCENT);
    view()
        .size(px(22.0))
        .items_center()
        .justify_center()
        // Winds up with the pull…
        .bind_rotate(&pull.interpolate(&[0.0, h], &[-120.0, 200.0]))
        .child(view().absolute().bind_rotate(&turn).child(ring))
        // …around a dot that fills as it arms.
        .child(
            view()
                .absolute()
                .bind_scale(&pull.interpolate(&[h * 0.4, ARMED], &[0.0, 1.0]))
                .child(
                    view()
                        .size(px(7.0))
                        .rounded_full()
                        .bg(colors::ACCENT)
                        .bind_scale(&breath),
                ),
        )
}
