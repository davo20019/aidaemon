//! The desktop as a live surface (macOS): each display is a surface the
//! owner can watch, and with `[surfaces] desktop` on and control taken,
//! drive with the pointer and keys from a paired app.
//!
//! Frames are polled captures, scaled, sent only when the picture changed.
//! While the owner controls a display the agent's own `computer_use` calls
//! wait (their wait counts as time on the owner), and the computer shows a
//! notification so whoever sits at it knows it is being driven remotely.
//!
//! The lock screen and login window are out of reach: a normal process
//! cannot see or type into them.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_surfaces::{
    Codec, ExclusiveGuard, FrameSink, InputEvent, KeyAction, Modifier, PointerAction,
    PointerButton, Size, StreamHandle, StreamParams, SurfaceBackend, SurfaceCaps,
    SurfaceDescriptor, SurfaceError, SurfaceId, SurfaceKind, SurfaceObservation, clean_title,
};
use async_trait::async_trait;
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use xcap::Monitor;
use xcap::image::imageops::FilterType;

/// The owner controls a display now: the agent's desktop actions wait.
static OWNER_HOLDS: AtomicBool = AtomicBool::new(false);

/// Whether the owner controls the desktop now.
pub fn owner_holds() -> bool {
    OWNER_HOLDS.load(Ordering::SeqCst)
}

/// Wait while the owner controls the desktop; the time is the owner's.
pub async fn wait_for_owner() {
    if !owner_holds() {
        return;
    }
    let _waiting = aidaemon_channels::owner_wait::OwnerWaitClock::begin_waiting();
    while owner_holds() {
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

struct Hold;

impl Drop for Hold {
    fn drop(&mut self) {
        OWNER_HOLDS.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
struct Display {
    id: u32,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

/// The computer's displays as surfaces.
#[derive(Default)]
pub struct DesktopSurface {
    ids: Mutex<HashMap<u32, SurfaceId>>,
    displays: Mutex<HashMap<SurfaceId, Display>>,
    pressed: Mutex<Option<Button>>,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn backend_error(error: impl std::fmt::Display) -> SurfaceError {
    SurfaceError::Backend(error.to_string())
}

impl DesktopSurface {
    pub fn new() -> Self {
        Self::default()
    }

    fn display(&self, id: &SurfaceId) -> Result<Display, SurfaceError> {
        lock(&self.displays)
            .get(id)
            .copied()
            .ok_or(SurfaceError::NotFound)
    }
}

fn monitor(id: u32) -> Result<Monitor, SurfaceError> {
    Monitor::all()
        .map_err(backend_error)?
        .into_iter()
        .find(|m| m.id().ok() == Some(id))
        .ok_or(SurfaceError::NotFound)
}

/// One picture of a display, scaled to fit, as JPEG (PNG where this build
/// has no JPEG encoder).
fn capture(
    id: u32,
    max_w: u32,
    max_h: u32,
    quality: u8,
) -> Result<(Codec, u32, u32, Vec<u8>), String> {
    let image = monitor(id)
        .map_err(|e| e.to_string())?
        .capture_image()
        .map_err(|e| e.to_string())?;
    let (w, h) = (image.width().max(1), image.height().max(1));
    let scale = (max_w as f32 / w as f32)
        .min(max_h as f32 / h as f32)
        .min(1.0);
    let (nw, nh) = (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    );
    let image = if (nw, nh) == (w, h) {
        image
    } else {
        xcap::image::imageops::resize(&image, nw, nh, FilterType::Triangle)
    };
    let rgb = xcap::image::DynamicImage::ImageRgba8(image).to_rgb8();
    let mut jpeg = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, quality.clamp(10, 100));
    if rgb.write_with_encoder(encoder).is_ok() {
        return Ok((Codec::Jpeg, nw, nh, jpeg));
    }
    let mut png = Vec::new();
    rgb.write_to(
        &mut std::io::Cursor::new(&mut png),
        xcap::image::ImageFormat::Png,
    )
    .map_err(|e| e.to_string())?;
    Ok((Codec::Png, nw, nh, png))
}

fn key_of(code: &str, key: &str) -> Option<Key> {
    Some(match code {
        "Enter" | "NumpadEnter" => Key::Return,
        "Backspace" => Key::Backspace,
        "Tab" => Key::Tab,
        "Escape" => Key::Escape,
        "Space" => Key::Space,
        "Delete" => Key::Delete,
        "ArrowLeft" => Key::LeftArrow,
        "ArrowRight" => Key::RightArrow,
        "ArrowUp" => Key::UpArrow,
        "ArrowDown" => Key::DownArrow,
        "Home" => Key::Home,
        "End" => Key::End,
        "PageUp" => Key::PageUp,
        "PageDown" => Key::PageDown,
        _ => {
            let mut chars = key.chars();
            let first = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Key::Unicode(first)
        }
    })
}

fn modifier_key(modifier: Modifier) -> Key {
    match modifier {
        Modifier::Shift => Key::Shift,
        Modifier::Ctrl => Key::Control,
        Modifier::Alt => Key::Alt,
        Modifier::Meta => Key::Meta,
    }
}

/// Tell whoever sits at the computer it is driven from elsewhere.
fn notify_at_the_computer(text: &str) {
    let _ = std::process::Command::new("osascript")
        .args([
            "-e",
            &format!("display notification \"{text}\" with title \"AIdaemon\""),
        ])
        .spawn();
}

#[async_trait]
impl SurfaceBackend for DesktopSurface {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::Desktop
    }

    async fn list(&self) -> Vec<SurfaceDescriptor> {
        let monitors = tokio::task::spawn_blocking(|| {
            Monitor::all()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|m| {
                    Some((
                        Display {
                            id: m.id().ok()?,
                            x: m.x().ok()?,
                            y: m.y().ok()?,
                            w: m.width().ok()?,
                            h: m.height().ok()?,
                        },
                        m.name().unwrap_or_default(),
                        m.is_primary().unwrap_or(false),
                    ))
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap_or_default();
        let mut out = Vec::new();
        let mut displays = HashMap::new();
        for (display, name, primary) in monitors {
            let id = lock(&self.ids)
                .entry(display.id)
                .or_insert_with(SurfaceId::random)
                .clone();
            displays.insert(id.clone(), display);
            out.push(SurfaceDescriptor {
                id,
                kind: SurfaceKind::Desktop,
                session_id: None,
                agent_ref: None,
                active: primary,
                title: clean_title(if name.is_empty() { "Display" } else { &name }),
                site: None,
                size: Size {
                    w: display.w,
                    h: display.h,
                },
                caps: SurfaceCaps::interactive(),
                opener: None,
            });
        }
        *lock(&self.displays) = displays;
        out
    }

    async fn start_stream(
        &self,
        id: &SurfaceId,
        params: StreamParams,
        sink: FrameSink,
    ) -> Result<StreamHandle, SurfaceError> {
        let display = self.display(id)?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let gap = Duration::from_millis(1000 / u64::from(params.max_fps.max(1)));
        tokio::task::spawn_blocking(move || {
            let mut last = None;
            while !stopped.load(Ordering::SeqCst) && sink.is_watched() {
                let started = std::time::Instant::now();
                if let Ok((codec, w, h, bytes)) =
                    capture(display.id, params.max_w, params.max_h, params.quality)
                {
                    // Only a changed picture is sent.
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    let digest = hasher.finish();
                    if last != Some(digest) {
                        last = Some(digest);
                        if !sink.push(codec, w, h, bytes) {
                            break;
                        }
                    }
                }
                std::thread::sleep(gap.saturating_sub(started.elapsed()));
            }
        });
        Ok(StreamHandle::new(move || {
            stop.store(true, Ordering::SeqCst)
        }))
    }

    async fn input(&self, id: &SurfaceId, event: &InputEvent) -> Result<(), SurfaceError> {
        let display = self.display(id)?;
        let event = event.clone();
        let pressed = {
            let mut pressed = lock(&self.pressed);
            match &event {
                InputEvent::Pointer {
                    action: PointerAction::Down,
                    button,
                    ..
                } => {
                    let button = match button {
                        PointerButton::Left => Button::Left,
                        PointerButton::Right => Button::Right,
                        PointerButton::Middle => Button::Middle,
                    };
                    *pressed = Some(button);
                    Some(button)
                }
                InputEvent::Pointer {
                    action: PointerAction::Up | PointerAction::Cancel,
                    ..
                } => pressed.take(),
                _ => *pressed,
            }
        };
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
            let at = |x: f64, y: f64| {
                (
                    display.x + (x * f64::from(display.w)).round() as i32,
                    display.y + (y * f64::from(display.h)).round() as i32,
                )
            };
            match event {
                InputEvent::Pointer { action, x, y, .. } => {
                    let (px, py) = at(x, y);
                    enigo
                        .move_mouse(px, py, Coordinate::Abs)
                        .map_err(|e| e.to_string())?;
                    let direction = match action {
                        PointerAction::Down => Some(Direction::Press),
                        PointerAction::Up | PointerAction::Cancel => Some(Direction::Release),
                        PointerAction::Move => None,
                    };
                    if let (Some(direction), Some(button)) = (direction, pressed) {
                        enigo.button(button, direction).map_err(|e| e.to_string())?;
                    }
                }
                InputEvent::Wheel { x, y, dx, dy } => {
                    let (px, py) = at(x, y);
                    enigo
                        .move_mouse(px, py, Coordinate::Abs)
                        .map_err(|e| e.to_string())?;
                    let lines = |d: f64| (d / 40.0).round() as i32;
                    if lines(dy) != 0 {
                        enigo
                            .scroll(lines(dy), Axis::Vertical)
                            .map_err(|e| e.to_string())?;
                    }
                    if lines(dx) != 0 {
                        enigo
                            .scroll(lines(dx), Axis::Horizontal)
                            .map_err(|e| e.to_string())?;
                    }
                }
                InputEvent::Key {
                    action,
                    code,
                    key,
                    mods,
                } => {
                    let Some(main) = key_of(&code, &key) else {
                        return Ok(());
                    };
                    if action == KeyAction::Down {
                        for m in &mods {
                            enigo
                                .key(modifier_key(*m), Direction::Press)
                                .map_err(|e| e.to_string())?;
                        }
                        let result = enigo.key(main, Direction::Click);
                        for m in mods.iter().rev() {
                            let _ = enigo.key(modifier_key(*m), Direction::Release);
                        }
                        result.map_err(|e| e.to_string())?;
                    }
                }
                InputEvent::Text { text } => {
                    enigo.text(&text).map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        })
        .await
        .map_err(backend_error)?
        .map_err(SurfaceError::Backend)
    }

    async fn observe(&self, id: &SurfaceId) -> Result<SurfaceObservation, SurfaceError> {
        self.display(id)?;
        Ok(SurfaceObservation {
            site: None,
            title: "Desktop".into(),
            password_field_present: false,
            agent_ref: None,
        })
    }

    async fn exclusive(&self, id: &SurfaceId) -> Result<ExclusiveGuard, SurfaceError> {
        self.display(id)?;
        OWNER_HOLDS.store(true, Ordering::SeqCst);
        notify_at_the_computer("This computer is being controlled from your paired device.");
        Ok(ExclusiveGuard::new(Hold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_map_to_the_desktop_and_printable_keys_type_themselves() {
        assert!(matches!(key_of("Enter", "Enter"), Some(Key::Return)));
        assert!(matches!(key_of("KeyA", "a"), Some(Key::Unicode('a'))));
        assert!(key_of("F13", "F13").is_none());
    }

    #[test]
    fn a_hold_ends_when_dropped() {
        {
            OWNER_HOLDS.store(true, Ordering::SeqCst);
            let _hold = Hold;
            assert!(owner_holds());
        }
        assert!(!owner_holds());
    }
}
