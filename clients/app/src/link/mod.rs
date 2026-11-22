//! The connection to the daemon's Client Gateway. On the desktops and in a
//! terminal it is `aidaemon-client-core`'s own link (the one the old TUI
//! uses), run on NSH's background threads; its updates come back to the engine thread and into the [`Store`].

#[cfg(target_arch = "wasm32")]
use nsh_core::prelude::*;

use crate::store::Store;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod random;
#[cfg(not(target_arch = "wasm32"))]
mod voice;
#[cfg(target_arch = "wasm32")]
pub mod web;

/// This app's version as the store and the daemon know it: "1.2.0 (14)".
pub fn app_version() -> String {
    let version = env!("AIDAEMON_APP_VERSION");
    match env!("AIDAEMON_APP_BUILD") {
        "" => version.to_owned(),
        build => format!("{version} ({build})"),
    }
}

/// The device woke (the app back in front, a computer back from sleep):
/// the link checks its socket, or tries again now. The browser's link
/// watches the page itself.
pub fn woke() {
    #[cfg(not(target_arch = "wasm32"))]
    native::woke();
}

/// Start the link for this platform.
pub fn start(store: Store) {
    #[cfg(not(target_arch = "wasm32"))]
    native::start(store);
    #[cfg(target_arch = "wasm32")]
    web::start(store);
}

/// The pairing offer the link that opened the app carried: offer id, code
/// and the daemon's fingerprint.
pub fn offered_pairing() -> Option<(String, String, Option<String>)> {
    #[cfg(target_arch = "wasm32")]
    return web::offered_pairing();
    #[cfg(not(target_arch = "wasm32"))]
    None
}

/// Pair this device with the offer, then connect.
pub async fn pair(
    store: Store,
    offer: String,
    code: String,
    name: String,
    fingerprint: Option<String>,
) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    return web::pair(store, offer, code, name, fingerprint).await;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (store, offer, code, name, fingerprint);
        Err("On the daemon's own machine this app signs in with `aidaemon client-token`.".into())
    }
}

/// Forget this device's pairing and show the pairing screen (the daemon
/// refused it).
pub fn pair_again(store: Store) {
    #[cfg(target_arch = "wasm32")]
    spawn(async move {
        web::unpair().await;
        store.needs_pairing();
    });
    #[cfg(not(target_arch = "wasm32"))]
    native::pair_again(store);
}

/// Whether this app signs in with a key it paired (a phone, or a computer
/// paired with a daemon elsewhere), so a refused key can be paired again.
/// The browser re-pairs on its own when its link carries an offer.
pub fn signs_in_paired() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    return native::is_paired();
    #[cfg(target_arch = "wasm32")]
    false
}

/// The owner's side of pairing a phone from this computer (a QR code of the
/// offer, approved here). Only the daemon's own computer offers it.
pub mod owner_pairing {
    use aidaemon_client_protocol::{Enrollment, PairingOffer};

    pub fn available() -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return super::native::can_offer_pairing();
        #[cfg(target_arch = "wasm32")]
        false
    }

    pub async fn offer() -> Result<PairingOffer, String> {
        #[cfg(not(target_arch = "wasm32"))]
        return super::native::offer_pairing().await;
        #[cfg(target_arch = "wasm32")]
        Err("Pair phones from the computer AIDaemon runs on.".into())
    }

    pub async fn status(offer_id: String) -> Result<Enrollment, String> {
        #[cfg(not(target_arch = "wasm32"))]
        return super::native::pairing_status(offer_id).await;
        #[cfg(target_arch = "wasm32")]
        {
            let _ = offer_id;
            Err("Pair phones from the computer AIDaemon runs on.".into())
        }
    }

    pub async fn decide(offer_id: String, approve: bool) -> Result<Enrollment, String> {
        #[cfg(not(target_arch = "wasm32"))]
        return super::native::decide_pairing(offer_id, approve).await;
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (offer_id, approve);
            Err("Pair phones from the computer AIDaemon runs on.".into())
        }
    }

    /// Whether a phone can reach the gateway the offer names.
    pub fn reachable_elsewhere(offer: &PairingOffer) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return offer
            .public_url
            .as_deref()
            .is_none_or(aidaemon_client_core::local::owner_pairing::reachable_elsewhere);
        #[cfg(target_arch = "wasm32")]
        {
            let _ = offer;
            true
        }
    }
}

/// Whether this platform pairs from the app (the browser does).
pub const PAIRS_IN_APP: bool = cfg!(target_arch = "wasm32");

/// Run the daemon on this computer, then connect (desktops only).
pub fn install_here(store: Store) {
    #[cfg(not(target_arch = "wasm32"))]
    native::install_here(store);
    #[cfg(target_arch = "wasm32")]
    let _ = store;
}

/// Pair with a daemon on another machine from its pairing link, then
/// connect (desktops; the browser pairs with [`pair`]).
pub async fn pair_remote(store: Store, link: String, name: String) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    return native::pair_remote(store, link, name).await;
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (store, link, name);
        Err("Open the pairing link in this browser instead.".into())
    }
}

/// Open a live view's frame socket for opening `epoch` (protocol 1.30).
pub fn open_live(store: Store, epoch: u64, url: String, ticket: String) {
    #[cfg(not(target_arch = "wasm32"))]
    native::open_live(store, epoch, url, ticket);
    #[cfg(target_arch = "wasm32")]
    web::open_live(store, epoch, url, ticket);
}

/// Open a conversation shell's socket for opening `epoch` (protocol 1.42).
pub fn open_shell(store: Store, epoch: u64, url: String, ticket: String) {
    #[cfg(not(target_arch = "wasm32"))]
    native::open_shell(store, epoch, url, ticket);
    #[cfg(target_arch = "wasm32")]
    web::open_shell(store, epoch, url, ticket);
}

/// Whether speech can be recognized here, to await in a task: the
/// platform's recognizer (macOS, iOS) or, in the browser, the page's Web
/// Speech API (NSH's web build wires no recognizer of its own).
pub async fn can_listen() -> bool {
    #[cfg(target_arch = "wasm32")]
    return web::dictation_available();
    #[cfg(not(target_arch = "wasm32"))]
    nsh_core::prelude::speech_available().await
}

/// A dictation session: the transcript as it grows, then its end. Dropping
/// it ends it at once.
pub struct Listening {
    #[cfg(not(target_arch = "wasm32"))]
    inner: nsh_core::element::SpeechRecognition,
    #[cfg(target_arch = "wasm32")]
    inner: web::Dictation,
}

impl Listening {
    /// Start listening as `request` asks.
    pub fn start(request: nsh_core::speech::SpeechRequest) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let inner = nsh_core::prelude::recognize_speech(request);
        #[cfg(target_arch = "wasm32")]
        let inner = web::Dictation::start(request);
        Self { inner }
    }

    /// The next event; `None` after the end was taken.
    pub fn poll_next(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<nsh_core::speech::SpeechEvent>> {
        self.inner.poll_next(cx)
    }

    /// Stop listening and finish what was heard: a last result, then the end.
    pub fn stop(&self) {
        self.inner.stop()
    }
}

/// Whether replies can be read aloud here: the page's voice in the
/// browser, the system's on a desktop ([`voice`]). Not in the terminal.
pub fn can_speak() -> bool {
    #[cfg(target_arch = "wasm32")]
    return web::speech_available();
    #[cfg(not(target_arch = "wasm32"))]
    {
        nsh_core::prelude::idiom() != nsh_core::platform::Idiom::Terminal && voice::available()
    }
}

/// Text being read aloud. `ended` runs once, on the thread that drives the
/// UI, when the reading is done or stopped. Dropping it stops it.
pub struct Speaking {
    #[cfg(target_arch = "wasm32")]
    inner: web::Speaker,
    #[cfg(not(target_arch = "wasm32"))]
    inner: voice::Reading,
}

impl Speaking {
    /// Read `text` aloud; `None` where nothing can speak.
    pub fn start(text: &str, ended: impl FnOnce() + 'static) -> Option<Self> {
        #[cfg(target_arch = "wasm32")]
        let inner = Some(web::Speaker::start(text, ended));
        #[cfg(not(target_arch = "wasm32"))]
        let inner = voice::Reading::start(text, ended);
        inner.map(|inner| Self { inner })
    }

    /// Stop reading now.
    pub fn stop(&self) {
        self.inner.stop();
    }
}

/// Whether this build can run the daemon on its own computer.
pub const RUNS_DAEMON: bool = cfg!(not(any(
    target_arch = "wasm32",
    target_os = "ios",
    target_os = "android"
)));

/// A phone or tablet app (iOS, Android): it never runs the daemon, only
/// pairs with one elsewhere.
pub const MOBILE: bool = cfg!(any(target_os = "ios", target_os = "android"));
