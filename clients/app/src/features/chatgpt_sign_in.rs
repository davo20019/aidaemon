//! The words and the paste button of a ChatGPT sign-in that may finish on
//! another device, shared by Settings and first-run setup.

use nsh_core::prelude::*;

use crate::store::Store;

/// Above the "Open the sign-in page" button, while an address may be
/// pasted.
pub fn intro() -> &'static str {
    if crate::link::MOBILE {
        "1. Sign in with your ChatGPT account on the page that opened."
    } else {
        "1. Sign in with your ChatGPT account on the page that opened. In a browser on \
         the daemon's computer, this card then finishes by itself."
    }
}

/// The paste step.
pub fn paste_step() -> &'static str {
    if crate::link::MOBILE {
        "2. The browser then shows a page that cannot load (\"This site can't be \
         reached\", its address starting with http://127.0.0.1). That is expected: copy \
         the whole address from the address bar, come back here and press Paste and finish."
    } else {
        "2. Signed in on a phone or another computer? Its browser then shows a page that \
         cannot load (\"This site can't be reached\", its address starting with \
         http://127.0.0.1). That is expected: copy the whole address from the address \
         bar, come back here and press Paste and finish."
    }
}

/// Read the clipboard into `redirect` and, when it holds anything, hand it
/// to `finish` (the daemon says if it is not the sign-in's address). An
/// empty clipboard, or one the browser would not let the page read, is
/// said on `store`.
pub fn paste_and_finish(
    store: Store,
    redirect: Signal<String>,
    finish: impl FnOnce(String) + 'static,
) {
    spawn(async move {
        let pasted = clipboard_string().await;
        let pasted = pasted.trim();
        if pasted.is_empty() {
            store.say("Couldn't read the clipboard: paste the address into the field");
            return;
        }
        redirect.set(pasted.to_owned());
        finish(pasted.to_owned());
    });
}
