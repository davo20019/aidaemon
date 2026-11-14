//! Telling the owner an approval is waiting while they look elsewhere: a
//! system notification per new approval (closed once it is answered
//! anywhere), and the waiting count on the app's icon. Pressing a
//! notification brings that approval into view.

use std::collections::BTreeSet;

use nsh_core::notifications::{Notification, NotificationPermission};
use nsh_core::platform::Lifecycle;
use nsh_core::prelude::*;

use crate::store::Store;

const PREFIX: &str = "approval:";

/// Start watching; lives as long as its caller (the shell).
pub fn approvals(store: Store) {
    let seen = stored(BTreeSet::<String>::new());
    let asked = stored(false);
    effect(move || {
        let waiting = store.approvals();
        untracked(|| {
            set_badge(u32::try_from(waiting.len()).unwrap_or(u32::MAX));
            let ids: BTreeSet<String> = waiting.iter().map(|a| a.id.clone()).collect();
            // Answered here or elsewhere: its notification goes.
            for gone in seen.with(|s| s.difference(&ids).cloned().collect::<Vec<_>>()) {
                close_notification(format!("{PREFIX}{gone}"));
            }
            let new: Vec<_> = waiting
                .into_iter()
                .filter(|a| !seen.with(|s| s.contains(&a.id)))
                .collect();
            seen.with(|s| *s = ids);
            if new.is_empty() || lifecycle() == Lifecycle::Active {
                return;
            }
            let first_ask = !asked.replace(true);
            spawn(async move {
                let permission = if first_ask {
                    request_notification_permission().await
                } else {
                    notification_permission().await
                };
                if permission != NotificationPermission::Granted {
                    return;
                }
                for approval in new {
                    let body = format!("In {} · tap to review", store.title(&approval.session_id));
                    let _ = show_notification(Notification::new(
                        format!("{PREFIX}{}", approval.id),
                        format!("Approval needed: {}", approval.summary),
                        body,
                    ))
                    .await;
                }
            });
        });
    });
    // A press brings that approval into view.
    spawn(async move {
        loop {
            let id = next_notification_press().await;
            // No notifications here: nothing will ever be pressed.
            if id.is_empty() {
                break;
            }
            if let Some(approval) = id.strip_prefix(PREFIX) {
                store.show_approval(approval);
            }
        }
    });
}
