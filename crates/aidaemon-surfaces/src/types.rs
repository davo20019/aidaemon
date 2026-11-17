//! The domain types: what a surface is, the frames it shows and the input it
//! takes. No wire format lives here; the gateway maps these to its protocol.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use tokio::sync::watch;

/// An opaque, unguessable surface name. Never a backend's own key (a CDP
/// target id, a display number): backends mint one per surface and keep the
/// mapping to themselves.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SurfaceId(String);

impl SurfaceId {
    /// A fresh random id.
    pub fn random() -> Self {
        Self(format!("srf_{}", uuid::Uuid::new_v4().simple()))
    }

    /// An id read back from a client or a model. Only ids a backend minted
    /// ever match a surface, so any string is safe to hold.
    pub fn from_string(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SurfaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What kind of thing a surface is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SurfaceKind {
    /// One tab of the agent's browser.
    BrowserTab,
    /// A whole display of the computer the daemon runs on.
    Desktop,
    /// One application window.
    Window,
    /// A screen on another machine, reached through a Node.
    Node,
}

impl SurfaceKind {
    /// The kind's name in config and on the wire.
    pub fn name(self) -> &'static str {
        match self {
            Self::BrowserTab => "browser_tab",
            Self::Desktop => "desktop",
            Self::Window => "window",
            Self::Node => "node",
        }
    }
}

/// Pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Size {
    pub w: u32,
    pub h: u32,
}

/// What a backend can do with a surface.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceCaps {
    pub view: bool,
    pub pointer: bool,
    pub keys: bool,
    pub text: bool,
    pub clipboard: bool,
    pub files: bool,
    pub audio: bool,
}

impl SurfaceCaps {
    /// Viewing, pointer, keys and text: what every screen backend offers.
    pub fn interactive() -> Self {
        Self {
            view: true,
            pointer: true,
            keys: true,
            text: true,
            ..Self::default()
        }
    }

    /// Whether `event` is something this surface accepts.
    pub fn accepts(&self, event: &InputEvent) -> bool {
        match event {
            InputEvent::Pointer { .. } | InputEvent::Wheel { .. } => self.pointer,
            InputEvent::Key { .. } => self.keys,
            InputEvent::Text { .. } => self.text,
        }
    }
}

/// Who may act on a surface right now. Exactly one at a time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Controller {
    /// The agent acts when it has work there; nobody else holds it.
    #[default]
    Agent,
    /// The owner, from one paired device.
    Owner { device_id: String },
}

impl Controller {
    pub fn owner_device(&self) -> Option<&str> {
        match self {
            Self::Owner { device_id } => Some(device_id),
            Self::Agent => None,
        }
    }
}

/// A surface as its backend describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceDescriptor {
    pub id: SurfaceId,
    pub kind: SurfaceKind,
    /// The conversation it belongs to (a browser tab's session).
    pub session_id: Option<String>,
    /// The name the agent knows it by inside its session (a browser tab
    /// id), so a tool call can name it without a second id.
    pub agent_ref: Option<String>,
    /// The session's surface the agent is on now.
    pub active: bool,
    /// Already cleaned and capped by the backend (see [`clean_title`]).
    pub title: String,
    /// The host only: never a path or a query.
    pub site: Option<String>,
    pub size: Size,
    pub caps: SurfaceCaps,
    /// A popup's parent.
    pub opener: Option<SurfaceId>,
}

/// A surface as the registry shows it: the backend's description plus who
/// controls it and whether the owner was asked to do something there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceInfo {
    pub id: SurfaceId,
    pub kind: SurfaceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_ref: Option<String>,
    pub active: bool,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    pub size: Size,
    pub caps: SurfaceCaps,
    pub controller: Controller,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opener: Option<SurfaceId>,
    /// The agent asked the owner to do something here and waits.
    pub handoff_pending: bool,
}

impl SurfaceInfo {
    pub(crate) fn from_descriptor(
        descriptor: SurfaceDescriptor,
        controller: Controller,
        handoff_pending: bool,
    ) -> Self {
        Self {
            id: descriptor.id,
            kind: descriptor.kind,
            session_id: descriptor.session_id,
            agent_ref: descriptor.agent_ref,
            active: descriptor.active,
            title: descriptor.title,
            site: descriptor.site,
            size: descriptor.size,
            caps: descriptor.caps,
            controller,
            opener: descriptor.opener,
            handoff_pending,
        }
    }
}

/// How a frame's bytes are encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Codec {
    Jpeg,
    Png,
}

impl Codec {
    pub fn name(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Png => "png",
        }
    }
}

/// One picture of a surface. Lives only in a stream's one-frame slot: never
/// stored, logged or shown to a model.
#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    /// Per stream, rising.
    pub seq: u64,
    pub at_ms: u64,
    pub codec: Codec,
    pub w: u32,
    pub h: u32,
    /// Stands alone (every JPEG or PNG frame does).
    pub key: bool,
    /// Which page the picture shows: rises each time the screen shows a new
    /// document ([`FrameSink::new_page`]), so what was focused or pointed at
    /// on the old one is gone.
    pub page: u64,
    pub data: Arc<[u8]>,
}

impl fmt::Debug for Frame {
    // Never print a frame's pixels.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("seq", &self.seq)
            .field("codec", &self.codec)
            .field("w", &self.w)
            .field("h", &self.h)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// What a viewer wants from a stream. The backend may give less.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamParams {
    pub max_w: u32,
    pub max_h: u32,
    pub max_fps: u32,
    /// 1 to 100.
    pub quality: u8,
    /// The room the viewer has for the picture: while its device holds
    /// control, a surface that can lay itself out takes this size.
    pub fit: Option<Size>,
}

impl Default for StreamParams {
    fn default() -> Self {
        Self {
            max_w: 1280,
            max_h: 1280,
            max_fps: 10,
            quality: 70,
            fit: None,
        }
    }
}

/// Where a backend puts frames. It holds the newest frame only: a frame not
/// yet taken is replaced by the next one, so a slow viewer never makes the
/// daemon queue frames.
#[derive(Clone)]
pub struct FrameSink {
    tx: Arc<watch::Sender<Option<Arc<Frame>>>>,
    seq: Arc<AtomicU64>,
    page: Arc<AtomicU64>,
}

impl FrameSink {
    pub(crate) fn new() -> (Self, watch::Receiver<Option<Arc<Frame>>>) {
        let (tx, rx) = watch::channel(None);
        (
            Self {
                tx: Arc::new(tx),
                seq: Arc::new(AtomicU64::new(0)),
                page: Arc::new(AtomicU64::new(0)),
            },
            rx,
        )
    }

    /// A sink with its own receiver, for a backend's tests.
    #[cfg(any(test, feature = "testing"))]
    pub fn detached() -> (Self, watch::Receiver<Option<Arc<Frame>>>) {
        Self::new()
    }

    /// Publish a frame. Returns false once nobody can see it any more.
    pub fn push(&self, codec: Codec, w: u32, h: u32, data: Vec<u8>) -> bool {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let frame = Frame {
            seq,
            at_ms: now_ms(),
            codec,
            w,
            h,
            key: true,
            page: self.page.load(Ordering::Relaxed),
            data: data.into(),
        };
        self.tx.send_replace(Some(Arc::new(frame)));
        self.is_watched()
    }

    /// The screen moved to a new document (a link, a form sent, a
    /// redirect): frames from now on carry the next page number.
    pub fn new_page(&self) {
        self.page.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether anyone still watches.
    pub fn is_watched(&self) -> bool {
        self.tx.receiver_count() > 0
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<Option<Arc<Frame>>> {
        self.tx.subscribe()
    }
}

/// Keeps a backend's stream running; dropping it stops the stream.
pub struct StreamHandle {
    on_drop: Option<Box<dyn FnOnce() + Send + Sync>>,
}

impl StreamHandle {
    pub fn new(stop: impl FnOnce() + Send + Sync + 'static) -> Self {
        Self {
            on_drop: Some(Box::new(stop)),
        }
    }

    pub fn noop() -> Self {
        Self { on_drop: None }
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        if let Some(stop) = self.on_drop.take() {
            stop();
        }
    }
}

/// Held while the owner controls a surface: the backend keeps the agent's
/// own actions there waiting until it is dropped.
pub struct ExclusiveGuard(#[allow(dead_code)] Box<dyn Send + Sync>);

impl ExclusiveGuard {
    pub fn new(inner: impl Send + Sync + 'static) -> Self {
        Self(Box::new(inner))
    }
}

/// What a backend did to get a surface ready for the owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffPrep {
    /// The surface to hand over (a backend may have reopened it).
    pub id: SurfaceId,
    /// For a browser: whether a sign-in made there is kept for later tasks.
    /// `None` when that means nothing for the surface.
    pub keeps_sign_in: Option<bool>,
}

/// Where a surface is now, as the agent may be told after a handoff. Typed
/// facts only: never page text, never a path or a query.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceObservation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    pub title: String,
    /// A password field is on the page now (a sign-in is likely still
    /// waiting).
    pub password_field_present: bool,
    /// The name the agent knows the surface by now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_ref: Option<String>,
}

/// What the owner pointed at on a surface: enough for the agent to act on
/// it with its own tools. Never a field's value.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PointedAt {
    /// "a button", "a link", "a text field"...
    pub what: String,
    /// Its visible words (or label, placeholder), capped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A CSS selector the agent's browser tool can use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selector: Option<String>,
    /// The name the agent knows the surface by (a tab id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_ref: Option<String>,
    /// Where it is, as fractions of the frame: `[x, y, w, h]`, so a viewer
    /// can outline it whatever the surface itself draws.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<[f64; 4]>,
    /// What typing does there; `None` when it takes none (a button).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typing: Option<Typing>,
}

/// What typing does in an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Typing {
    /// Text that can be shown.
    Text,
    /// A secret (a password): never shown.
    Secret,
}

/// A pointer button.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerButton {
    #[default]
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerAction {
    Down,
    Move,
    Up,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyAction {
    Down,
    Up,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modifier {
    Shift,
    Ctrl,
    Alt,
    Meta,
}

/// The longest text one `text` event may carry.
pub const MAX_TEXT_CHARS: usize = 4096;
/// The longest `code` or `key` name.
const MAX_KEY_NAME_CHARS: usize = 32;

/// Input from the owner, the same for every kind of surface. Coordinates
/// are fractions of the frame (0 to 1), so a client may zoom and rotate
/// freely; the backend scales them to the surface.
///
/// Never logged, stored or shown to a model: `Debug` hides the text.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputEvent {
    Pointer {
        action: PointerAction,
        x: f64,
        y: f64,
        #[serde(default)]
        button: PointerButton,
    },
    Wheel {
        x: f64,
        y: f64,
        #[serde(default)]
        dx: f64,
        #[serde(default)]
        dy: f64,
    },
    Key {
        action: KeyAction,
        /// The physical key (`KeyA`, `Enter`, `ArrowLeft`), as the web's
        /// `KeyboardEvent.code`.
        code: String,
        /// What it produces (`a`, `Enter`), as `KeyboardEvent.key`.
        #[serde(default)]
        key: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        mods: Vec<Modifier>,
    },
    /// Committed text: what a phone keyboard, an IME, a paste or a password
    /// manager produces.
    Text { text: String },
}

impl fmt::Debug for InputEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pointer { action, .. } => write!(f, "Pointer({action:?})"),
            Self::Wheel { .. } => f.write_str("Wheel"),
            Self::Key { action, .. } => write!(f, "Key({action:?})"),
            Self::Text { text } => write!(f, "Text({} chars)", text.chars().count()),
        }
    }
}

impl InputEvent {
    /// Check the event is well formed: coordinates inside the frame, short
    /// names, bounded text.
    pub fn validate(&self) -> Result<(), SurfaceError> {
        let inside = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        match self {
            Self::Pointer { x, y, .. } if !(inside(*x) && inside(*y)) => {
                Err(SurfaceError::BadInput("pointer outside the frame"))
            }
            Self::Wheel { x, y, dx, dy }
                if !(inside(*x) && inside(*y) && dx.is_finite() && dy.is_finite())
                    || dx.abs() > 10_000.0
                    || dy.abs() > 10_000.0 =>
            {
                Err(SurfaceError::BadInput("wheel out of range"))
            }
            Self::Key { code, key, .. }
                if code.is_empty()
                    || code.chars().count() > MAX_KEY_NAME_CHARS
                    || key.chars().count() > MAX_KEY_NAME_CHARS =>
            {
                Err(SurfaceError::BadInput("bad key name"))
            }
            Self::Text { text } if text.is_empty() || text.chars().count() > MAX_TEXT_CHARS => {
                Err(SurfaceError::BadInput("text empty or too long"))
            }
            _ => Ok(()),
        }
    }

    /// Whether the event types something (text or a key press).
    pub fn types(&self) -> bool {
        matches!(self, Self::Text { .. } | Self::Key { .. })
    }
}

/// Why a surface operation failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    #[error("no such surface")]
    NotFound,
    #[error("no such live session")]
    NoLiveSession,
    #[error("live surfaces are turned off")]
    Disabled,
    #[error("{0}")]
    NotSupported(&'static str),
    #[error("{0}")]
    BadInput(&'static str),
    /// The caller may not do this now; the text says why, for the owner.
    #[error("{0}")]
    Refused(String),
    #[error("too many live sessions")]
    TooMany,
    /// The backend failed (the browser went away).
    #[error("{0}")]
    Backend(String),
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// The longest title a surface shows.
pub const MAX_TITLE_CHARS: usize = 120;

/// A page-made title made safe to show: control and bidi characters out,
/// whitespace collapsed, capped. Pages choose their titles, so a card shows
/// it as data, never as its own wording.
pub fn clean_title(raw: &str) -> String {
    let mut out = String::new();
    let mut space = false;
    for ch in raw.chars() {
        let bidi = matches!(ch, '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{200E}' | '\u{200F}');
        if ch.is_control() || bidi || ch == '\u{FEFF}' {
            space = !out.is_empty();
            continue;
        }
        if ch.is_whitespace() {
            space = !out.is_empty();
            continue;
        }
        if space {
            out.push(' ');
            space = false;
        }
        out.push(ch);
        if out.chars().count() >= MAX_TITLE_CHARS {
            out.push('…');
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_lose_control_and_bidi_characters() {
        assert_eq!(clean_title("  Sign\u{202E}in\n\tnow  "), "Sign in now");
        let long = "x".repeat(500);
        assert_eq!(clean_title(&long).chars().count(), MAX_TITLE_CHARS + 1);
    }

    #[test]
    fn input_is_checked() {
        let pointer = |x, y| InputEvent::Pointer {
            action: PointerAction::Down,
            x,
            y,
            button: PointerButton::Left,
        };
        assert!(pointer(0.5, 0.5).validate().is_ok());
        assert!(pointer(1.2, 0.5).validate().is_err());
        assert!(pointer(f64::NAN, 0.5).validate().is_err());
        let text = InputEvent::Text {
            text: "a".repeat(MAX_TEXT_CHARS + 1),
        };
        assert!(text.validate().is_err());
        assert!(
            InputEvent::Text {
                text: String::new()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn debug_never_shows_typed_text() {
        let event = InputEvent::Text {
            text: "hunter2".into(),
        };
        assert!(!format!("{event:?}").contains("hunter2"));
    }

    #[test]
    fn input_reads_from_the_wire_shape() {
        let event: InputEvent = serde_json::from_str(
            r#"{"type":"pointer","action":"down","x":0.25,"y":0.5,"button":"left"}"#,
        )
        .unwrap();
        assert!(matches!(event, InputEvent::Pointer { .. }));
        let event: InputEvent =
            serde_json::from_str(r#"{"type":"key","action":"down","code":"Enter","key":"Enter"}"#)
                .unwrap();
        assert!(event.types());
    }
}
