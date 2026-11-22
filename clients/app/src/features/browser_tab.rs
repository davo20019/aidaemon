//! The details' Browser tab: the agent's browser in this conversation,
//! watched in the pane. Showing the tab opens the live view of the
//! conversation's browser (drawn here, not over the window); leaving it
//! closes a view this tab opened. Taking control, a handoff's Done and the
//! rest are the live view's own.

use std::rc::Rc;

use nsh_core::prelude::*;

use crate::store::Store;
use crate::ui::{self, Size, Variant, btn};

pub fn browser_tab(store: Store) -> View {
    view().flex_1().min_h(px(0.0)).child(dynamic(move || {
        let session = store.selected()?;
        Some(untracked(move || browser_view(store, session)))
    }))
}

fn browser_view(store: Store, session: String) -> View {
    // The surface this tab opened a view of, closed when it goes.
    let opened = Rc::new(std::cell::RefCell::new(None::<String>));
    {
        let (session, opened) = (session.clone(), opened.clone());
        effect(move || {
            let Some(first) = store.surfaces_in(&session).into_iter().next() else {
                return;
            };
            // One of this conversation's own already shows: keep it.
            let showing = store.live_surface();
            if store.live_open()
                && showing.is_some_and(|s| s.session_id.as_deref() == Some(session.as_str()))
            {
                return;
            }
            if opened.borrow().as_deref() == Some(first.id.as_str()) {
                // Opened once already (and closed since, or refused): the
                // owner opens it again.
                return;
            }
            opened.replace(Some(first.id.clone()));
            untracked(|| store.watch_surface(&first.id, first.handoff_pending));
        });
    }
    {
        let opened = opened.clone();
        cleanup(move || {
            let mine = opened.borrow().clone();
            if mine.is_some()
                && untracked(|| store.live_surface()).map(|s| s.id) == mine
                && !untracked(|| store.live_surface()).is_some_and(|s| s.handoff_pending)
            {
                store.close_live();
            }
        });
    }
    let has_browser = {
        let session = session.clone();
        move || !store.surfaces_in(&session).is_empty()
    };
    let watch = {
        let session = session.clone();
        move || {
            if let Some(first) = store.surfaces_in(&session).into_iter().next() {
                store.watch_surface(&first.id, first.handoff_pending);
            }
        }
    };
    view()
        .flex_1()
        .min_h(px(0.0))
        .child(when(
            move || store.live_open() && store.live_docked(),
            move || {
                view()
                    .flex_1()
                    .min_h(px(0.0))
                    .child(ui::live_screen(crate::shell::live_screen(store, true)))
            },
        ))
        .child(when(
            move || !store.live_open() && has_browser(),
            move || {
                let watch = watch.clone();
                view()
                    .p_4()
                    .gap_2()
                    .items_start()
                    .child(ui::empty(
                        "The agent's browser",
                        "Watch it here; take control from the view when you need to.",
                    ))
                    .child(btn("Watch", Variant::Primary, Size::Md, watch))
            },
        ))
        .child(when(
            {
                let session = session.clone();
                move || store.surfaces_in(&session).is_empty()
            },
            || {
                view().p_4().child(ui::empty(
                    "No browser yet",
                    "When the agent opens a page in this conversation, it shows here.",
                ))
            },
        ))
}
