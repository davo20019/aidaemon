//! Links into the app. A channel's handoff card links to the live view as
//! `…/client/v1/next/#surface=<id>`: opened in a browser, the app waits until
//! it is connected, opens that view with control (as the card's own "Open the
//! live view" does), and takes the part out of its address so a reload does
//! not open it again. The same link given to the open page (typed, or the
//! browser's back and forward) does the same.

use nsh_core::prelude::*;

use crate::store::Store;

/// The fragment key naming a live view.
const SURFACE: &str = "surface=";

/// Start listening; lives as long as its caller (the shell).
pub fn surface_links(store: Store) {
    let wanted = signal(None::<String>);
    spawn(async move {
        let mut url = app_url().await;
        loop {
            if let Some(at) = url.as_deref()
                && let Some(id) = surface_in(at)
            {
                let _ = set_app_url(without_surface(at));
                wanted.set(Some(id));
            }
            url = Some(next_app_url_change().await.url);
        }
    });
    effect(move || {
        let Some(id) = wanted.get() else {
            return;
        };
        if !store.is_online() {
            return;
        }
        untracked(|| {
            wanted.set(None);
            store.watch_surface(&id, true);
        });
    });
}

/// The live view an address names, if it names one.
fn surface_in(url: &str) -> Option<String> {
    let (_, fragment) = url.split_once('#')?;
    let id = fragment
        .split('&')
        .find_map(|part| part.strip_prefix(SURFACE))?;
    let valid = !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    valid.then(|| id.to_owned())
}

/// The address with the live view's part taken out (the rest kept).
fn without_surface(url: &str) -> String {
    let Some((base, fragment)) = url.split_once('#') else {
        return url.to_owned();
    };
    let rest: Vec<&str> = fragment
        .split('&')
        .filter(|part| !part.starts_with(SURFACE) && !part.is_empty())
        .collect();
    if rest.is_empty() {
        base.to_owned()
    } else {
        format!("{base}#{}", rest.join("&"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handoff_link_names_its_live_view_and_leaves_the_rest() {
        let link = "https://daemon.example.invalid/client/v1/next/#surface=srf_12ab";
        assert_eq!(surface_in(link).as_deref(), Some("srf_12ab"));
        assert_eq!(
            without_surface(link),
            "https://daemon.example.invalid/client/v1/next/"
        );
        let mixed = "https://d.example.invalid/#tab=chat&surface=srf_1";
        assert_eq!(surface_in(mixed).as_deref(), Some("srf_1"));
        assert_eq!(
            without_surface(mixed),
            "https://d.example.invalid/#tab=chat"
        );
    }

    #[test]
    fn other_addresses_name_no_live_view() {
        for url in [
            "https://d.example.invalid/client/v1/next/",
            "https://d.example.invalid/#surface=",
            "https://d.example.invalid/#surface=srf_1%3Cscript",
            "https://d.example.invalid/?surface=srf_1",
        ] {
            assert_eq!(surface_in(url), None, "{url}");
        }
    }
}
