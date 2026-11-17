//! The agent's browser tabs as live surfaces: the owner watches a tab over
//! the page's own screencast and, holding control, sends it pointer, key
//! and text input.
//!
//! A tab is one surface. Its [`SurfaceId`] is minted here and never shown to
//! the model; the model keeps naming the tab by its tab id (the surface's
//! agent reference). While the owner controls a tab, the session's action
//! lock is held, so the agent's next browser action in that session waits
//! (and its wait counts as time spent on the owner, not as work).
//!
//! Nothing the owner types is logged or stored. The tab is marked as one the
//! owner typed into, so the browser tool can warn before a script reads it.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_surfaces::{
    Codec, ExclusiveGuard, FrameSink, HandoffPrep, InputEvent, KeyAction, Modifier, PointedAt,
    PointerAction, PointerButton, Size, StreamHandle, StreamParams, SurfaceBackend, SurfaceCaps,
    SurfaceDescriptor, SurfaceError, SurfaceId, SurfaceKind, SurfaceNotifier, SurfaceObservation,
    clean_title,
};
use async_trait::async_trait;
use futures::StreamExt;
use tracing::debug;

use super::backend::{
    BrowserBackend, KeyInput, MouseButtonKind, MouseInput, MouseKind, PageHandle, SignInStore,
};
use super::redact_origin;
use super::session::BrowserSessionRegistry;

/// What is at a point of the page: its kind, its visible words (never a
/// field's value) and a selector the browser tool can use. `__X__` and
/// `__Y__` are the point's fractions, numbers only.
const POINT_SCRIPT: &str = r#"(() => {
  const x = __X__ * window.innerWidth, y = __Y__ * window.innerHeight;
  let el = document.elementFromPoint(x, y);
  if (!el) return null;
  el = el.closest('a,button,input,select,textarea,label,[role=button],[role=link],[onclick]') || el;
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute('type') || '').toLowerCase();
  const role = el.getAttribute('role') || '';
  const what = tag === 'a' || role === 'link' ? 'a link'
    : tag === 'button' || role === 'button' || type === 'submit' || type === 'button' ? 'a button'
    : tag === 'input' && type === 'password' ? 'a password field'
    : tag === 'input' || tag === 'textarea' ? 'a text field'
    : tag === 'select' ? 'a menu'
    : tag === 'img' ? 'a picture'
    : 'the ' + tag + ' element';
  const typed = tag === 'input' || tag === 'textarea' || tag === 'select';
  const words = (el.getAttribute('aria-label') || el.getAttribute('placeholder')
    || el.getAttribute('alt') || (typed ? '' : el.innerText) || el.getAttribute('title') || '')
    .trim().replace(/\s+/g, ' ').slice(0, 80);
  let selector;
  if (el.id) selector = '#' + CSS.escape(el.id);
  else if (el.getAttribute('name')) selector = tag + '[name="' + CSS.escape(el.getAttribute('name')) + '"]';
  else {
    const parts = [];
    let n = el;
    while (n && n.nodeType === 1 && parts.length < 6) {
      if (n.id) { parts.unshift('#' + CSS.escape(n.id)); break; }
      let part = n.tagName.toLowerCase();
      const same = n.parentElement ? [...n.parentElement.children].filter((c) => c.tagName === n.tagName) : [];
      if (same.length > 1) part += ':nth-of-type(' + (same.indexOf(n) + 1) + ')';
      parts.unshift(part);
      n = n.parentElement;
    }
    selector = parts.join(' > ');
  }
  return { what, text: words || null, selector };
})()"#;

/// How long one page read for the surface list may take.
const READ_WITHIN: Duration = Duration::from_millis(800);

/// State the browser tool and its live surfaces share.
#[derive(Default)]
pub struct LiveState {
    /// Sessions whose tab the owner controls now.
    owner_held: Mutex<HashSet<String>>,
    /// Tabs (target ids) the owner typed into.
    owner_typed: Mutex<HashSet<String>>,
    notifier: Mutex<Option<SurfaceNotifier>>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl LiveState {
    /// The owner holds this session's tab: the agent's next action waits.
    pub fn owner_holds(&self, session_id: &str) -> bool {
        lock(&self.owner_held).contains(session_id)
    }

    /// The owner typed into this tab.
    pub fn owner_typed(&self, target_id: &str) -> bool {
        lock(&self.owner_typed).contains(target_id)
    }

    /// Tell the registry the tabs may have changed.
    pub fn notify(&self) {
        if let Some(notifier) = lock(&self.notifier).as_ref() {
            notifier.notify();
        }
    }
}

/// Keeps a session's action lock while the owner controls one of its tabs.
struct OwnerHold {
    _guard: tokio::sync::OwnedMutexGuard<()>,
    live: Arc<LiveState>,
    session_id: String,
}

impl Drop for OwnerHold {
    fn drop(&mut self) {
        lock(&self.live.owner_held).remove(&self.session_id);
    }
}

/// The agent's browser tabs as surfaces.
pub struct BrowserSurface {
    backend: Arc<dyn BrowserBackend>,
    sessions: Arc<BrowserSessionRegistry>,
    live: Arc<LiveState>,
    /// `(session, target)` to the surface id minted for it.
    ids: Mutex<HashMap<(String, String), SurfaceId>>,
    /// Each tab's viewport in CSS pixels, as its frames last said.
    viewports: Arc<Mutex<HashMap<String, (f64, f64)>>>,
    /// A button held down on a tab (for drags).
    pressed: Mutex<HashMap<String, MouseButtonKind>>,
    size: Size,
}

impl BrowserSurface {
    pub(crate) fn new(
        backend: Arc<dyn BrowserBackend>,
        sessions: Arc<BrowserSessionRegistry>,
        live: Arc<LiveState>,
        size: Size,
    ) -> Self {
        Self {
            backend,
            sessions,
            live,
            ids: Mutex::new(HashMap::new()),
            viewports: Arc::new(Mutex::new(HashMap::new())),
            pressed: Mutex::new(HashMap::new()),
            size,
        }
    }

    /// Report tab changes to the registry from now on.
    pub fn set_notifier(&self, notifier: SurfaceNotifier) {
        *lock(&self.live.notifier) = Some(notifier);
    }

    fn id_for(&self, session_id: &str, target_id: &str) -> SurfaceId {
        lock(&self.ids)
            .entry((session_id.to_string(), target_id.to_string()))
            .or_insert_with(SurfaceId::random)
            .clone()
    }

    fn locate(&self, id: &SurfaceId) -> Result<(String, String), SurfaceError> {
        lock(&self.ids)
            .iter()
            .find(|(_, known)| *known == id)
            .map(|(key, _)| key.clone())
            .ok_or(SurfaceError::NotFound)
    }

    async fn page(
        &self,
        id: &SurfaceId,
    ) -> Result<(String, String, Arc<dyn PageHandle>), SurfaceError> {
        let (session, target) = self.locate(id)?;
        let tab = self
            .sessions
            .tab(&session, &target)
            .await
            .ok_or(SurfaceError::NotFound)?;
        Ok((session, target, tab.page))
    }

    async fn viewport(&self, target: &str, page: &Arc<dyn PageHandle>) -> (f64, f64) {
        if let Some(known) = lock(&self.viewports).get(target).copied() {
            return known;
        }
        match page.viewport().await {
            Ok(size) if size.0 > 0.0 && size.1 > 0.0 => {
                lock(&self.viewports).insert(target.to_string(), size);
                size
            }
            _ => (f64::from(self.size.w), f64::from(self.size.h)),
        }
    }
}

/// The host of a URL, nothing else.
fn site_of(url: &str) -> Option<String> {
    let origin = redact_origin(url);
    let host = origin.split_once("://").map_or(origin.as_str(), |(_, h)| h);
    (!host.is_empty() && url.contains("://")).then(|| host.to_string())
}

fn modifier_bits(mods: &[Modifier]) -> i64 {
    mods.iter()
        .map(|m| match m {
            Modifier::Alt => 1,
            Modifier::Ctrl => 2,
            Modifier::Meta => 4,
            Modifier::Shift => 8,
        })
        .fold(0, |a, b| a | b)
}

/// The Windows virtual key code CDP needs for an editing key to act.
fn virtual_key(code: &str) -> Option<i64> {
    let fixed = match code {
        "Backspace" => 8,
        "Tab" => 9,
        "Enter" | "NumpadEnter" => 13,
        "ShiftLeft" | "ShiftRight" => 16,
        "ControlLeft" | "ControlRight" => 17,
        "AltLeft" | "AltRight" => 18,
        "Escape" => 27,
        "Space" => 32,
        "PageUp" => 33,
        "PageDown" => 34,
        "End" => 35,
        "Home" => 36,
        "ArrowLeft" => 37,
        "ArrowUp" => 38,
        "ArrowRight" => 39,
        "ArrowDown" => 40,
        "Delete" => 46,
        _ => 0,
    };
    if fixed != 0 {
        return Some(fixed);
    }
    let letter = code.strip_prefix("Key").filter(|l| l.len() == 1);
    let digit = code.strip_prefix("Digit").filter(|d| d.len() == 1);
    letter
        .or(digit)
        .and_then(|c| c.chars().next())
        .map(|c| i64::from(u32::from(c.to_ascii_uppercase())))
}

/// What a key-down types, if anything.
fn typed_text(key: &str, code: &str, mods: &[Modifier]) -> Option<String> {
    if mods
        .iter()
        .any(|m| matches!(m, Modifier::Ctrl | Modifier::Meta))
    {
        return None;
    }
    if code == "Enter" || code == "NumpadEnter" {
        return Some("\r".to_string());
    }
    (key.chars().count() == 1).then(|| key.to_string())
}

fn backend_error(error: String) -> SurfaceError {
    SurfaceError::Backend(error)
}

#[async_trait]
impl SurfaceBackend for BrowserSurface {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::BrowserTab
    }

    async fn list(&self) -> Vec<SurfaceDescriptor> {
        let tabs = self.sessions.all_tabs().await;
        let mut out = Vec::with_capacity(tabs.len());
        let mut seen = HashSet::new();
        for (session_id, tab, active) in tabs {
            let id = self.id_for(&session_id, &tab.target_id);
            seen.insert((session_id.clone(), tab.target_id.clone()));
            let url = match tab.last_url.clone() {
                Some(url) => Some(url),
                None => tokio::time::timeout(READ_WITHIN, tab.page.url())
                    .await
                    .ok()
                    .flatten(),
            };
            let title = match tab.title.clone() {
                Some(title) => Some(title),
                None => tokio::time::timeout(READ_WITHIN, tab.page.title())
                    .await
                    .ok()
                    .flatten(),
            };
            let site = url.as_deref().and_then(site_of);
            out.push(SurfaceDescriptor {
                id,
                kind: SurfaceKind::BrowserTab,
                session_id: Some(session_id),
                agent_ref: Some(tab.tab_id.clone()),
                active,
                title: clean_title(
                    title
                        .as_deref()
                        .filter(|t| !t.trim().is_empty())
                        .or(site.as_deref())
                        .unwrap_or("New tab"),
                ),
                site,
                size: self.size,
                caps: SurfaceCaps::interactive(),
                opener: None,
            });
        }
        lock(&self.ids).retain(|key, _| seen.contains(key));
        out
    }

    async fn start_stream(
        &self,
        id: &SurfaceId,
        params: StreamParams,
        sink: FrameSink,
    ) -> Result<StreamHandle, SurfaceError> {
        let (_, target, page) = self.page(id).await?;
        let mut frames = page
            .start_screencast(params.max_w, params.max_h, params.quality)
            .await
            .map_err(backend_error)?;
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
        let min_gap = Duration::from_millis(1000 / u64::from(params.max_fps.max(1)));
        let viewports = self.viewports.clone();
        tokio::spawn(async move {
            loop {
                let frame = tokio::select! {
                    _ = &mut stop_rx => break,
                    frame = frames.next() => frame,
                };
                let Some(frame) = frame else { break };
                lock(&viewports).insert(target.clone(), (frame.css_w, frame.css_h));
                let w = frame.css_w.round().max(1.0) as u32;
                let h = frame.css_h.round().max(1.0) as u32;
                if !sink.push(Codec::Jpeg, w, h, frame.data) {
                    break;
                }
                // Pace the page: the next frame comes after the ack.
                tokio::time::sleep(min_gap).await;
                if page.ack_screencast(frame.session).await.is_err() {
                    break;
                }
            }
            let _ = page.stop_screencast().await;
            debug!(target = %target, "live view stopped");
        });
        Ok(StreamHandle::new(move || {
            let _ = stop_tx.send(());
        }))
    }

    async fn input(&self, id: &SurfaceId, event: &InputEvent) -> Result<(), SurfaceError> {
        let (session, target, page) = self.page(id).await?;
        self.sessions.touch(&session).await;
        match event {
            InputEvent::Pointer {
                action,
                x,
                y,
                button,
            } => {
                let (w, h) = self.viewport(&target, &page).await;
                let button = match button {
                    PointerButton::Left => MouseButtonKind::Left,
                    PointerButton::Right => MouseButtonKind::Right,
                    PointerButton::Middle => MouseButtonKind::Middle,
                };
                let (kind, button) = match action {
                    PointerAction::Down => {
                        lock(&self.pressed).insert(target.clone(), button);
                        (MouseKind::Pressed, button)
                    }
                    PointerAction::Up | PointerAction::Cancel => {
                        let held = lock(&self.pressed).remove(&target).unwrap_or(button);
                        (MouseKind::Released, held)
                    }
                    PointerAction::Move => (
                        MouseKind::Moved,
                        lock(&self.pressed)
                            .get(&target)
                            .copied()
                            .unwrap_or(MouseButtonKind::None),
                    ),
                };
                page.dispatch_mouse(&MouseInput {
                    kind,
                    x: x * w,
                    y: y * h,
                    button,
                    click_count: 1,
                    delta_x: 0.0,
                    delta_y: 0.0,
                    modifiers: 0,
                })
                .await
                .map_err(backend_error)
            }
            InputEvent::Wheel { x, y, dx, dy } => {
                let (w, h) = self.viewport(&target, &page).await;
                page.dispatch_mouse(&MouseInput {
                    kind: MouseKind::Wheel,
                    x: x * w,
                    y: y * h,
                    button: MouseButtonKind::None,
                    click_count: 0,
                    delta_x: *dx,
                    delta_y: *dy,
                    modifiers: 0,
                })
                .await
                .map_err(backend_error)
            }
            InputEvent::Key {
                action,
                code,
                key,
                mods,
            } => {
                lock(&self.live.owner_typed).insert(target);
                let down = *action == KeyAction::Down;
                page.dispatch_key(&KeyInput {
                    down,
                    code: code.clone(),
                    key: key.clone(),
                    text: if down {
                        typed_text(key, code, mods)
                    } else {
                        None
                    },
                    modifiers: modifier_bits(mods),
                    windows_virtual_key_code: virtual_key(code),
                })
                .await
                .map_err(backend_error)
            }
            InputEvent::Text { text } => {
                lock(&self.live.owner_typed).insert(target);
                page.insert_text(text).await.map_err(backend_error)
            }
        }
    }

    async fn observe(&self, id: &SurfaceId) -> Result<SurfaceObservation, SurfaceError> {
        let (_, _, page) = self.page(id).await?;
        let (session, target) = self.locate(id)?;
        let url = page.url().await;
        let password_field_present = page
            .evaluate("!!document.querySelector('input[type=password]')")
            .await
            .ok()
            .flatten()
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let agent_ref = self
            .sessions
            .tab(&session, &target)
            .await
            .map(|tab| tab.tab_id);
        Ok(SurfaceObservation {
            site: url.as_deref().and_then(site_of),
            title: clean_title(&page.title().await.unwrap_or_default()),
            password_field_present,
            agent_ref,
        })
    }

    async fn describe_point(
        &self,
        id: &SurfaceId,
        x: f64,
        y: f64,
    ) -> Result<PointedAt, SurfaceError> {
        let (session, target, page) = self.page(id).await?;
        let script = POINT_SCRIPT
            .replace("__X__", &format!("{:.6}", x.clamp(0.0, 1.0)))
            .replace("__Y__", &format!("{:.6}", y.clamp(0.0, 1.0)));
        let found = page
            .evaluate(&script)
            .await
            .map_err(backend_error)?
            .filter(serde_json::Value::is_object)
            .ok_or_else(|| SurfaceError::Refused("Nothing is there.".into()))?;
        let field = |key: &str| {
            found
                .get(key)
                .and_then(|v| v.as_str())
                .map(clean_title)
                .filter(|v| !v.is_empty())
        };
        let agent_ref = self
            .sessions
            .tab(&session, &target)
            .await
            .map(|tab| tab.tab_id);
        Ok(PointedAt {
            what: field("what").unwrap_or_else(|| "something".into()),
            text: field("text"),
            selector: found
                .get("selector")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty() && v.len() <= 300)
                .map(str::to_string),
            agent_ref,
        })
    }

    async fn exclusive(&self, id: &SurfaceId) -> Result<ExclusiveGuard, SurfaceError> {
        let (session, _) = self.locate(id)?;
        let action_lock = self
            .sessions
            .action_lock(&session)
            .await
            .ok_or(SurfaceError::NotFound)?;
        let guard = action_lock.lock_owned().await;
        lock(&self.live.owner_held).insert(session.clone());
        Ok(ExclusiveGuard::new(OwnerHold {
            _guard: guard,
            live: self.live.clone(),
            session_id: session,
        }))
    }

    async fn prepare_handoff(&self, id: &SurfaceId) -> Result<HandoffPrep, SurfaceError> {
        match self.backend.sign_in_store().await {
            SignInStore::Kept => {
                return Ok(HandoffPrep {
                    id: id.clone(),
                    keeps_sign_in: Some(true),
                });
            }
            SignInStore::NotKept => {
                return Ok(HandoffPrep {
                    id: id.clone(),
                    keeps_sign_in: Some(false),
                });
            }
            SignInStore::CanKeep => {}
        }
        let (session, _, page) = self.page(id).await?;
        // No agent action runs in the session while its browser moves.
        let action_lock = self
            .sessions
            .action_lock(&session)
            .await
            .ok_or(SurfaceError::NotFound)?;
        let _quiet = action_lock.lock_owned().await;
        let url = page.url().await;
        if !self
            .backend
            .keep_sign_ins_now()
            .await
            .map_err(backend_error)?
        {
            return Ok(HandoffPrep {
                id: id.clone(),
                keeps_sign_in: Some(true),
            });
        }
        // Every tab went with the old browser: reopen this one where it was.
        self.sessions.invalidate_all_pages().await;
        lock(&self.viewports).clear();
        self.backend.ensure_ready().await.map_err(backend_error)?;
        let (page, _) = self
            .sessions
            .get_or_create_page(&session, &*self.backend)
            .await
            .map_err(backend_error)?;
        if let Some(url) = url.filter(|u| {
            (u.starts_with("https://") || u.starts_with("http://"))
                && super::policy::validate_network_url(u).is_ok()
        }) {
            page.goto(&url).await.map_err(backend_error)?;
            self.sessions.update_active_tab_url(&session, url).await;
        }
        let target = self
            .sessions
            .active_target_id(&session)
            .await
            .ok_or(SurfaceError::NotFound)?;
        self.live.notify();
        Ok(HandoffPrep {
            id: self.id_for(&session, &target),
            keeps_sign_in: Some(true),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::BrowserTool;
    use crate::tool::backend::{MockBackend, MockCall};
    use aidaemon_surfaces::{Controller, LiveMode, SurfaceRegistry, SurfacesConfig};

    async fn setup(
        backend: MockBackend,
    ) -> (BrowserTool, Arc<MockBackend>, SurfaceRegistry, SurfaceId) {
        let backend = Arc::new(backend);
        let (media_tx, _rx) = tokio::sync::mpsc::channel(4);
        let tool = BrowserTool::with_backend(backend.clone(), media_tx);
        tool.sessions
            .get_or_create_page("app:s1", &*backend)
            .await
            .unwrap();
        let tab = tool.sessions.active_target_id("app:s1").await.unwrap();
        tool.sessions
            .update_active_tab_url("app:s1", "https://login.acme.example/signin?next=/x".into())
            .await;
        let surfaces = tool.surface_backend();
        let registry = SurfaceRegistry::new(SurfacesConfig::default());
        surfaces.set_notifier(registry.notifier());
        registry.add_backend(surfaces);
        let found = registry.find_for_agent("app:s1", Some(&tab)).await.unwrap();
        (tool, backend, registry, found.id)
    }

    async fn calls(backend: &MockBackend) -> Vec<MockCall> {
        backend.calls().lock().await.clone()
    }

    #[tokio::test]
    async fn a_tab_is_a_surface_named_by_its_site_only() {
        let (_tool, _backend, registry, id) = setup(MockBackend::new()).await;
        let info = registry.get(&id).unwrap();
        assert_eq!(info.site.as_deref(), Some("login.acme.example"));
        assert_eq!(info.session_id.as_deref(), Some("app:s1"));
        assert!(info.active);
        assert!(
            !info
                .id
                .as_str()
                .contains(info.agent_ref.as_deref().unwrap())
        );
    }

    #[tokio::test]
    async fn the_owner_watches_and_under_control_taps_and_types() {
        let (tool, backend, registry, id) = setup(MockBackend::new()).await;
        let mut live = registry
            .open_live(&id, "phone", LiveMode::View, StreamParams::default())
            .await
            .unwrap();
        let frame = loop {
            if let Some(frame) = live.frames.borrow_and_update().clone() {
                break frame;
            }
            live.frames.changed().await.unwrap();
        };
        assert_eq!(&*frame.data, b"\xff\xd8mock-frame");
        assert_eq!((frame.w, frame.h), (1280, 720));

        registry.take_control(&id, "phone").await.unwrap();
        assert!(tool.live.owner_holds("app:s1"));
        let tap = InputEvent::Pointer {
            action: PointerAction::Down,
            x: 0.5,
            y: 0.5,
            button: PointerButton::Left,
        };
        registry.input(&live.live_id, &tap).await.unwrap();
        registry
            .input(
                &live.live_id,
                &InputEvent::Text {
                    text: "synthetic-password".into(),
                },
            )
            .await
            .unwrap();
        registry
            .input(
                &live.live_id,
                &InputEvent::Key {
                    action: KeyAction::Down,
                    code: "Enter".into(),
                    key: "Enter".into(),
                    mods: vec![],
                },
            )
            .await
            .unwrap();
        let recorded = calls(&backend).await;
        assert!(recorded.contains(&MockCall::StartScreencast));
        assert!(recorded.contains(&MockCall::Mouse("Pressed".into(), 640, 360)));
        assert!(recorded.contains(&MockCall::InsertText(18)));
        assert!(recorded.contains(&MockCall::Key(true, "Enter".into(), true)));
        assert!(
            !format!("{recorded:?}").contains("synthetic-password"),
            "typed text is never recorded"
        );
        let tab = tool.sessions.active_target_id("app:s1").await.unwrap();
        assert!(tool.live.owner_typed(&tab));

        // While the owner holds the tab the agent's next action waits.
        let waiting =
            tokio::time::timeout(Duration::from_millis(200), tool.page_for("app:s1")).await;
        assert!(waiting.is_err(), "the agent waits for the owner");
        registry.release_control(&id, "phone").unwrap();
        assert_eq!(registry.get(&id).unwrap().controller, Controller::Agent);
        assert!(!tool.live.owner_holds("app:s1"));
        tokio::time::timeout(Duration::from_secs(1), tool.page_for("app:s1"))
            .await
            .expect("the agent goes on after release")
            .unwrap();
        registry.close_live(&live.live_id);
    }

    #[tokio::test]
    async fn a_handoff_moves_an_ephemeral_browser_onto_a_kept_profile() {
        let (_tool, backend, registry, id) =
            setup(MockBackend::new().with_sign_in_store(SignInStore::CanKeep)).await;
        let prep = registry.prepare_handoff(&id).await.unwrap();
        assert_eq!(prep.keeps_sign_in, Some(true));
        assert_ne!(prep.id, id, "the tab reopened under a new surface");
        let recorded = calls(&backend).await;
        assert!(recorded.contains(&MockCall::KeepSignIns));
        // The page reopens where it really was (the page's own URL).
        assert!(recorded.contains(&MockCall::Goto("https://mock.example/".into())));
        assert!(registry.get(&prep.id).is_some());
        // Kept now: the next handoff changes nothing.
        let again = registry.prepare_handoff(&prep.id).await.unwrap();
        assert_eq!(again.id, prep.id);
    }

    #[tokio::test]
    async fn an_ephemeral_browser_that_may_not_keep_says_so() {
        let (_tool, _backend, registry, id) = setup(MockBackend::new()).await;
        let prep = registry.prepare_handoff(&id).await.unwrap();
        assert_eq!(
            prep,
            HandoffPrep {
                id,
                keeps_sign_in: Some(false)
            }
        );
    }

    #[test]
    fn a_site_is_the_host_only() {
        assert_eq!(
            site_of("https://user:pw@login.acme.example/path?token=x").as_deref(),
            Some("login.acme.example")
        );
        assert_eq!(site_of("about:blank"), None);
    }

    #[test]
    fn editing_keys_carry_their_codes_and_printable_keys_type() {
        assert_eq!(virtual_key("Enter"), Some(13));
        assert_eq!(virtual_key("KeyA"), Some(65));
        assert_eq!(virtual_key("Digit7"), Some(55));
        assert_eq!(virtual_key("F13"), None);
        assert_eq!(typed_text("a", "KeyA", &[]).as_deref(), Some("a"));
        assert_eq!(typed_text("Enter", "Enter", &[]).as_deref(), Some("\r"));
        assert_eq!(typed_text("a", "KeyA", &[Modifier::Ctrl]), None);
        assert_eq!(typed_text("Backspace", "Backspace", &[]), None);
    }
}
