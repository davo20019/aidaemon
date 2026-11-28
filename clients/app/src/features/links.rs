//! Links into the app. An e-ink reader's QR link (`…/pair-eink#n=…`) offers
//! to pair it (`aidaemon_client_core::eink`), read once and taken out of the
//! address at once. A channel's handoff card links to the live view as
//! `…/client/v1/next/#surface=<id>`: opened in a browser, the app waits until
//! it is connected, opens that view with control (as the card's own "Open the
//! live view" does), and only then takes the part out of its address, so a
//! reload before it connects still opens it and one after does not. The
//! same link given to the open page (typed, or the browser's back and
//! forward) does the same. The link opens the view to watch: taking control
//! of it is the owner's choice, asked for, since a link (from a page or a
//! message) is not the owner's word.
//!
//! Links out of the app ([`open_outside`]): what a reply, a page the agent
//! read or a daemon's answer links to opens only as a web or mail link, and
//! only `https:` without a question.

use nsh_core::prelude::*;

use crate::store::Store;

/// The fragment key naming a live view.
const SURFACE: &str = "surface=";

/// Start listening; lives as long as its caller (the shell).
pub fn surface_links(store: Store) {
    let wanted = signal(None::<(String, String)>);
    spawn(async move {
        let mut url = app_url().await;
        loop {
            if let Some(at) = url.as_deref()
                && let Some(id) = surface_in(at)
            {
                wanted.set(Some((id, at.to_owned())));
            }
            // An e-ink reader's QR link: its part leaves the address at
            // once, so a reload or a later visit never offers it again.
            if let Some(at) = url.as_deref()
                && let Some(link) = aidaemon_client_core::eink::parse_pair_link(at)
            {
                let _ = set_app_url(without_eink_link(at));
                store.open_eink_link(link);
            }
            url = Some(next_app_url_change().await.url);
        }
    });
    // Online (the page connects after it reads its address), Settings →
    // Connections offers the e-ink link; it takes it within a minute.
    effect(move || {
        if store.eink_link_waiting() && store.is_online() {
            untracked(|| store.open_settings_tab(crate::store::SettingsTab::Connections));
        }
    });
    effect(move || {
        let Some((id, at)) = wanted.get() else {
            return;
        };
        if !store.is_online() {
            return;
        }
        untracked(|| {
            wanted.set(None);
            store.watch_surface(&id, false);
            let _ = set_app_url(without_surface(&at));
            let id = id.clone();
            crate::ui::confirm(
                "Take control of this live view?",
                "A link opened this live view to watch. Take control only if you opened the link yourself.",
                "Take control",
                move || store.watch_surface(&id, true),
            );
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

/// The address an e-ink reader's QR link opened, without the link: the
/// app's own page (`…/pair-eink#n=…` to `…/`).
fn without_eink_link(url: &str) -> String {
    let base = url.split_once('#').map_or(url, |(base, _)| base);
    let base = base.trim_end_matches('/');
    match base.strip_suffix(aidaemon_client_core::eink::LINK_PATH) {
        Some(site) => format!("{site}/"),
        None => format!("{base}/"),
    }
}

/// What a press on a link out of the app may do, by the URL's scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outside {
    /// `https:`: opens.
    Open,
    /// `http:` or `mailto:`: opens once the owner has seen where it goes.
    Ask,
    /// Anything else (another app's scheme, `file:`, a pairing link, a
    /// relative path): never opened from content.
    Refuse,
}

/// The scheme of `url` (machine format: the RFC 3986 scheme before `:`).
fn scheme(url: &str) -> Option<String> {
    let (scheme, _) = url.trim().split_once(':')?;
    let valid = scheme.len() > 1
        && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then(|| scheme.to_ascii_lowercase())
}

pub fn outside(url: &str) -> Outside {
    match scheme(url).as_deref() {
        Some("https") => Outside::Open,
        Some("http" | "mailto") => Outside::Ask,
        _ => Outside::Refuse,
    }
}

/// Open `url` outside the app as [`outside`] allows; a refused one is
/// logged. Every link from content or from the daemon goes through here.
pub fn open_outside(url: String) {
    match outside(&url) {
        Outside::Open => drop(open_url(url)),
        Outside::Ask => crate::ui::confirm(
            "Open this link?",
            format!("It is not a secure web link: {url}"),
            "Open",
            move || drop(open_url(url)),
        ),
        Outside::Refuse => log::warn!("refused to open a link outside the app: {url}"),
    }
}

/// Open a URL the daemon sent (a sign-in page, where to get a key, a bot's
/// help page) outside the app: `https:` only. A daemon is not the owner, and
/// one this device was steered to can name any app's scheme.
pub fn open_https(url: String) {
    if outside(&url) == Outside::Open {
        drop(open_url(url));
    } else {
        log::warn!("refused to open a daemon's link that is not https: {url}");
    }
}

/// A press on a link in a document: anchors stay in it, the rest goes out
/// through [`open_outside`].
pub fn open_document_link(url: &str) {
    if !url.starts_with('#') {
        open_outside(url.to_owned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_eink_link_leaves_the_address_once_read() {
        assert_eq!(
            without_eink_link("https://app.aidaemon.ai/pair-eink#n=K7F2&s=9QXM4HTAPZ6CR&k=00"),
            "https://app.aidaemon.ai/"
        );
        assert_eq!(
            without_eink_link("https://daemon.example.test/client/v1/next/pair-eink/#n=K7F2"),
            "https://daemon.example.test/client/v1/next/"
        );
    }

    #[test]
    fn only_web_and_mail_links_leave_the_app() {
        assert_eq!(outside("https://docs.example.invalid/a"), Outside::Open);
        assert_eq!(outside(" HTTPS://docs.example.invalid"), Outside::Open);
        assert_eq!(outside("http://docs.example.invalid"), Outside::Ask);
        assert_eq!(outside("mailto:alice@example.invalid"), Outside::Ask);
        for url in [
            "aidaemon-pair:?gateway=https://evil.example.invalid",
            "vscode://file/etc/passwd",
            "ssh://host.example.invalid",
            "smb://host.example.invalid/share",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "tel:5550100",
            "C:\\Windows\\x.exe",
            "/client/v1/files/x",
            "no scheme here",
        ] {
            assert_eq!(outside(url), Outside::Refuse, "{url}");
        }
    }

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
