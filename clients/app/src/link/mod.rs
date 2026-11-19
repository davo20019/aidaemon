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
#[cfg(target_arch = "wasm32")]
pub mod web;

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
/// refused it). Only the browser pairs from the app.
pub fn pair_again(store: Store) {
    #[cfg(target_arch = "wasm32")]
    spawn(async move {
        web::unpair().await;
        store.needs_pairing();
    });
    #[cfg(not(target_arch = "wasm32"))]
    let _ = store;
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
/// browser, the system's on macOS. Not yet on Windows, Linux or in the
/// terminal (nsh has no speech output of its own).
pub fn can_speak() -> bool {
    #[cfg(target_arch = "wasm32")]
    return web::speech_available();
    #[cfg(target_os = "macos")]
    return nsh_core::prelude::idiom() != nsh_core::platform::Idiom::Terminal;
    #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
    false
}

/// Text being read aloud. `ended` runs once, on the thread that drives the
/// UI, when the reading is done or stopped. Dropping it stops it.
pub struct Speaking {
    #[cfg(target_arch = "wasm32")]
    _inner: web::Speaker,
    #[cfg(target_os = "macos")]
    child: std::sync::Arc<std::sync::Mutex<Option<std::process::Child>>>,
}

impl Speaking {
    /// Read `text` aloud; `None` where nothing can speak.
    #[allow(unused_variables)]
    pub fn start(text: &str, ended: impl FnOnce() + 'static) -> Option<Self> {
        #[cfg(target_arch = "wasm32")]
        return Some(Self {
            _inner: web::Speaker::start(text, ended),
        });
        #[cfg(target_os = "macos")]
        return macos_say(text, ended);
        #[cfg(not(any(target_arch = "wasm32", target_os = "macos")))]
        None
    }

    /// Stop reading now.
    pub fn stop(&self) {
        #[cfg(target_arch = "wasm32")]
        self._inner.stop();
        #[cfg(target_os = "macos")]
        if let Ok(mut child) = self.child.lock()
            && let Some(child) = child.as_mut()
        {
            let _ = child.kill();
        }
    }
}

impl Drop for Speaking {
    fn drop(&mut self) {
        self.stop();
    }
}

/// macOS's own voice: `say`, reading the text from its input.
#[cfg(target_os = "macos")]
fn macos_say(text: &str, ended: impl FnOnce() + 'static) -> Option<Speaking> {
    use std::io::Write as _;
    use std::sync::{Arc, Mutex};

    let mut child = std::process::Command::new("/usr/bin/say")
        .arg("-f")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut input) = child.stdin.take() {
        let _ = input.write_all(text.as_bytes());
    }
    let child = Arc::new(Mutex::new(Some(child)));
    let watched = child.clone();
    // Waited for off the engine thread; the end is reported back on it, so
    // `ended` (which touches signals) runs there.
    nsh_core::prelude::spawn(async move {
        let _ = nsh_core::prelude::blocking(move || {
            loop {
                let running = watched
                    .lock()
                    .ok()
                    .and_then(|mut child| child.as_mut().map(|c| matches!(c.try_wait(), Ok(None))))
                    .unwrap_or(false);
                if !running {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
        })
        .await;
        ended();
    });
    Some(Speaking { child })
}

/// Whether this build can run the daemon on its own computer.
pub const RUNS_DAEMON: bool = cfg!(not(target_arch = "wasm32"));
