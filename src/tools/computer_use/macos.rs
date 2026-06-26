//! macOS Accessibility + screen capture backend for `computer_use`.
#![allow(dead_code, clippy::too_many_arguments)]

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axuielement::ax_action::AX_PRESS_ACTION;
use axuielement::ax_attribute::attributes::{
    AX_DESCRIPTION_ATTRIBUTE, AX_ENABLED_ATTRIBUTE, AX_FOCUSED_ATTRIBUTE, AX_MAIN_ATTRIBUTE,
    AX_POSITION_ATTRIBUTE, AX_ROLE_ATTRIBUTE, AX_SIZE_ATTRIBUTE, AX_SUBROLE_ATTRIBUTE,
    AX_TITLE_ATTRIBUTE, AX_VALUE_ATTRIBUTE, AX_WINDOWS_ATTRIBUTE,
};
use axuielement::AXUIElement;
use axuielement::{is_process_trusted, is_process_trusted_with_prompt};
use enigo::{Button, Coordinate, Direction, Enigo, Key, Keyboard, Mouse, Settings};
use xcap::image::imageops::FilterType;
use xcap::image::RgbaImage;
use xcap::Window;

use crate::config::ComputerUseConfig;

use super::cache::{SnapshotCache, SnapshotKey};
use super::harness::{ComputerHarness, HarnessRequestContext};
use super::policy::{is_prohibited_bundle, is_text_input_element};
use super::types::{AppInfo, AppSnapshot, AxWalkLimits, ElementBounds, IndexedElement};

const ACCESSIBILITY_HELP: &str = "Accessibility permission is required. \
Grant it in System Settings → Privacy & Security → Accessibility for aidaemon, then retry.";
const SCREEN_RECORDING_HELP: &str = "Screen Recording permission is required. \
Grant it in System Settings → Privacy & Security → Screen Recording for aidaemon. \
This permission only takes effect after the daemon is restarted.";

const LOCKED_SCREEN_HELP: &str = "The Mac screen is locked — macOS routes all keyboard \
and mouse input to the lock screen, so computer_use cannot type or click in any app until \
it is unlocked. (Reading state and screenshots still work, which is why a screenshot may \
look correct.) Stop and ask the user to unlock the Mac, then retry.";

// CoreGraphics screen-capture authorization (macOS 10.15+). Preflight is a
// non-prompting probe; Request triggers the system prompt and registers the
// app in the Screen Recording pane so the user has something to toggle.
// CGSessionCopyCurrentDictionary exposes the login-session state (incl. screen
// lock); it returns NULL when there is no owning GUI session.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGRequestScreenCaptureAccess() -> bool;
    fn CGSessionCopyCurrentDictionary() -> *const std::ffi::c_void;
}

// Minimal CoreFoundation surface for reading one boolean out of the session
// dictionary. The framework is already linked transitively; declaring just
// these four entry points avoids taking a direct core-foundation crate dep
// (two major versions are already in the tree).
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringCreateWithCString(
        alloc: *const std::ffi::c_void,
        c_str: *const std::ffi::c_char,
        encoding: u32,
    ) -> *const std::ffi::c_void;
    fn CFDictionaryGetValue(
        dict: *const std::ffi::c_void,
        key: *const std::ffi::c_void,
    ) -> *const std::ffi::c_void;
    fn CFBooleanGetValue(boolean: *const std::ffi::c_void) -> u8;
    fn CFRelease(cf: *const std::ffi::c_void);
}

pub struct MacOsHarness {
    config: ComputerUseConfig,
}

impl MacOsHarness {
    pub fn new(config: ComputerUseConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ComputerHarness for MacOsHarness {
    fn check_permissions(&self) -> Result<(), String> {
        // Accessibility — needed for the AX tree walk and AX/enigo input.
        if !is_process_trusted() {
            // On the first failed check, ask macOS to show the grant dialog —
            // this also registers aidaemon in the Accessibility pane so the user
            // can toggle it on instead of hunting for it with the "+" button.
            static AX_PROMPTED: AtomicBool = AtomicBool::new(false);
            if !AX_PROMPTED.swap(true, Ordering::SeqCst) {
                let _ = is_process_trusted_with_prompt();
            }
            return Err(ACCESSIBILITY_HELP.to_string());
        }
        // Screen Recording — needed for window capture. Check it explicitly so
        // the model gets an accurate, actionable error instead of a downstream
        // "no capturable window" from the capture backend.
        // SAFETY: both are simple C predicates with no arguments.
        if !unsafe { CGPreflightScreenCaptureAccess() } {
            static SCREEN_PROMPTED: AtomicBool = AtomicBool::new(false);
            if !SCREEN_PROMPTED.swap(true, Ordering::SeqCst) {
                // Triggers the system prompt and registers aidaemon in the
                // Screen Recording pane. The grant only applies after restart.
                unsafe { CGRequestScreenCaptureAccess() };
            }
            return Err(SCREEN_RECORDING_HELP.to_string());
        }
        Ok(())
    }

    async fn list_apps(&self) -> Result<Vec<AppInfo>, String> {
        self.check_permissions()?;
        list_running_apps()
    }

    async fn launch_app(&self, app: &str) -> Result<AppInfo, String> {
        self.check_permissions()?;
        // Idempotent: if it is already running, just hand back the live instance.
        if let Ok(found) = resolve_app(app) {
            return Ok(found);
        }
        launch_app_process(app)?;
        // `open` returns as soon as the launch is initiated; the process can take
        // a moment to register with System Events, so poll until it appears.
        let timeout = Duration::from_secs(self.config.action_timeout_secs.max(5));
        let resolved = wait_for_running_app(app, timeout)?;
        if is_prohibited_bundle(&resolved.bundle_id) {
            return Err(format!(
                "App '{}' ({}) is blocked by computer_use policy",
                resolved.name, resolved.bundle_id
            ));
        }
        Ok(resolved)
    }

    async fn get_app_state(
        &self,
        app: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        self.check_permissions()?;
        let resolved = resolve_app(app)?;
        if is_prohibited_bundle(&resolved.bundle_id) {
            return Err(format!(
                "App '{}' ({}) is blocked by computer_use policy",
                resolved.name, resolved.bundle_id
            ));
        }
        capture_app(resolved, &self.config, cache, ctx)
    }

    async fn capture_screenshot(&self, app: &str) -> Result<AppSnapshot, String> {
        self.check_permissions()?;
        let resolved = resolve_app(app)?;
        if is_prohibited_bundle(&resolved.bundle_id) {
            return Err(format!(
                "App '{}' ({}) is blocked by computer_use policy",
                resolved.name, resolved.bundle_id
            ));
        }
        // Image only: no AX walk, no cache.store — so the element generation is
        // left untouched and a mutation issued after this screenshot is not stale.
        let png = capture_window_png(resolved.pid, &resolved.name, &self.config)?;
        Ok(AppSnapshot {
            generation: 0,
            bundle_id: resolved.bundle_id,
            app_name: resolved.name,
            pid: resolved.pid,
            window_id: 1,
            window_title: String::new(),
            elements: Vec::new(),
            truncated: false,
            png,
        })
    }

    async fn activate_app(
        &self,
        app: &str,
        generation: Option<u64>,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let resolved = resolve_app(app)?;
        let key = snapshot_key(resolved.bundle_id.clone(), ctx);
        if let Some(generation) = generation {
            cache.validate_generation(&key, generation)?;
        }
        activate_app(&resolved)?;
        capture_app(resolved, &self.config, cache, ctx)
    }

    async fn click(
        &self,
        app: &str,
        generation: u64,
        element_index: Option<u32>,
        x: Option<f64>,
        y: Option<f64>,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, Option<u32>, &'static str), String> {
        let resolved = resolve_app(app)?;
        let key = snapshot_key(resolved.bundle_id.clone(), ctx);
        cache.validate_generation(&key, generation)?;

        if let Some(index) = element_index {
            let element = cache.element_by_index(&key, generation, index)?.clone();
            // Prefer an AX press: it activates the control directly without
            // moving the real cursor or requiring the app to be frontmost, so a
            // GUI task works even while the user is looking at another window.
            // Only fall back to a synthetic cursor click (which needs the app in
            // the foreground) if the control can't be AX-pressed.
            if press_element_via_ax(resolved.pid, index, &self.config)? {
                let refreshed = capture_app(resolved, &self.config, cache, ctx)?;
                return Ok((refreshed, Some(index), "ax_press"));
            }
            ensure_foreground(&resolved)?;
            click_element(&element)?;
            let refreshed = capture_app(resolved, &self.config, cache, ctx)?;
            return Ok((refreshed, Some(index), "cursor"));
        }

        // Raw coordinate clicks can only go through the synthetic cursor, which
        // requires the target app to be frontmost.
        let (px, py) = match (x, y) {
            (Some(x), Some(y)) => (x, y),
            _ => return Err("click requires element_index or both x and y coordinates".to_string()),
        };
        ensure_foreground(&resolved)?;
        click_global_point(px, py)?;
        Ok((
            capture_app(resolved, &self.config, cache, ctx)?,
            None,
            "coordinate",
        ))
    }

    async fn type_text(
        &self,
        app: &str,
        generation: u64,
        element_index: Option<u32>,
        text: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let resolved = resolve_app(app)?;
        let key = snapshot_key(resolved.bundle_id.clone(), ctx);
        cache.validate_generation(&key, generation)?;
        // If a target field was named, focus it first so the keystrokes land
        // there instead of in whatever control currently holds focus. Prefer an
        // AX press (no foreground needed); fall back to a synthetic cursor click.
        if let Some(index) = element_index {
            let element = cache.element_by_index(&key, generation, index)?.clone();
            if !press_element_via_ax(resolved.pid, index, &self.config)? {
                ensure_foreground(&resolved)?;
                click_element(&element)?;
            }
            // Focusing a text field places the cursor without selecting, so
            // typing would insert into any leftover content (e.g. a stale URL in
            // the address bar). Select-all first so the new text replaces it.
            if is_text_input_element(&element) {
                ensure_foreground(&resolved)?;
                press_key_combo("Command+a")?;
            }
        }
        ensure_foreground(&resolved)?;
        let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
        enigo
            .text(text)
            .map_err(|e| format!("Failed to type text: {e}"))?;
        capture_app(resolved, &self.config, cache, ctx)
    }

    async fn press_key(
        &self,
        app: &str,
        generation: u64,
        key: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String> {
        let resolved = resolve_app(app)?;
        let key_id = snapshot_key(resolved.bundle_id.clone(), ctx);
        cache.validate_generation(&key_id, generation)?;
        ensure_foreground(&resolved)?;
        press_key_combo(key)?;
        capture_app(resolved, &self.config, cache, ctx)
    }

    async fn scroll(
        &self,
        app: &str,
        generation: u64,
        element_index: u32,
        direction: &str,
        pages: f64,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, u32), String> {
        let resolved = resolve_app(app)?;
        let key = snapshot_key(resolved.bundle_id.clone(), ctx);
        let _element = cache.element_by_index(&key, generation, element_index)?;
        ensure_foreground(&resolved)?;
        scroll_direction(direction, pages)?;
        Ok((
            capture_app(resolved, &self.config, cache, ctx)?,
            element_index,
        ))
    }

    async fn set_value(
        &self,
        app: &str,
        generation: u64,
        element_index: u32,
        value: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, u32), String> {
        let resolved = resolve_app(app)?;
        let key = snapshot_key(resolved.bundle_id.clone(), ctx);
        let element = cache
            .element_by_index(&key, generation, element_index)?
            .clone();
        ensure_foreground(&resolved)?;
        click_element(&element)?;
        let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
        enigo.text(value).map_err(|e| e.to_string())?;
        Ok((
            capture_app(resolved, &self.config, cache, ctx)?,
            element_index,
        ))
    }
}

fn snapshot_key(bundle_id: String, ctx: &HarnessRequestContext) -> SnapshotKey {
    SnapshotKey {
        task_id: ctx.task_id.clone(),
        session_id: ctx.session_id.clone(),
        bundle_id,
    }
}

fn capture_app(
    app: AppInfo,
    config: &ComputerUseConfig,
    cache: &mut SnapshotCache,
    ctx: &HarnessRequestContext,
) -> Result<AppSnapshot, String> {
    let app_element = AXUIElement::from_pid(app.pid)
        .ok_or_else(|| format!("No AX element for pid {}", app.pid))?;
    let window = focused_or_main_window(&app_element)?;
    let window_title =
        optional_string_attr(&window, AX_TITLE_ATTRIBUTE).unwrap_or_else(|| app.name.clone());
    let limits = AxWalkLimits::from_config(config);
    let (elements, truncated) = walk_tree(&window, limits);
    let png = capture_window_png(app.pid, &app.name, config)?;
    let snapshot = AppSnapshot {
        generation: 0,
        bundle_id: app.bundle_id,
        app_name: app.name,
        pid: app.pid,
        window_id: 1,
        window_title,
        elements,
        truncated,
        png,
    };
    let key = snapshot_key(snapshot.bundle_id.clone(), ctx);
    Ok(cache.store(key, snapshot))
}

fn list_running_apps() -> Result<Vec<AppInfo>, String> {
    // System Events' process list can mutate while AppleScript iterates it (an
    // app launches or quits mid-loop), which throws a transient
    // "Invalid index" error. Snapshot the process list into a local variable
    // first, wrap each process's property reads in `try` so one stale entry
    // can't abort the whole enumeration, and retry the whole call a few times.
    const SCRIPT: &str = r#"
set output to ""
tell application "System Events"
  set procList to (every application process whose background only is false)
  repeat with proc in procList
    try
      set procName to name of proc
      set procPid to unix id of proc
      set procBundle to ""
      try
        set procBundle to bundle identifier of proc
      end try
      set output to output & procName & tab & procPid & tab & procBundle & linefeed
    end try
  end repeat
end tell
return output
"#;
    let mut last_err = String::new();
    let mut stdout = String::new();
    for attempt in 0..3 {
        let out = Command::new("/usr/bin/osascript")
            .arg("-e")
            .arg(SCRIPT)
            .output()
            .map_err(|e| format!("Failed to list apps: {e}"))?;
        if out.status.success() {
            stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            last_err.clear();
            break;
        }
        last_err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        // Brief backoff lets the process table settle before retrying.
        if attempt < 2 {
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
    }
    if !last_err.is_empty() {
        return Err(format!("Failed to list apps: {last_err}"));
    }
    let mut apps = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            continue;
        }
        let name = parts[0].trim();
        let pid: i32 = parts[1].trim().parse().unwrap_or(-1);
        if pid <= 0 || name.is_empty() {
            continue;
        }
        let bundle_id = parts.get(2).copied().unwrap_or("").trim();
        let bundle_id = if bundle_id.is_empty() {
            bundle_id_for_pid(pid).unwrap_or_default()
        } else {
            bundle_id.to_string()
        };
        apps.push(AppInfo {
            name: name.to_string(),
            bundle_id,
            pid,
        });
    }
    apps.sort_by(|a, b| {
        a.name
            .to_ascii_lowercase()
            .cmp(&b.name.to_ascii_lowercase())
    });
    apps.dedup_by(|a, b| a.pid == b.pid);
    Ok(apps)
}

fn resolve_app(app: &str) -> Result<AppInfo, String> {
    let apps = list_running_apps()?;
    let needle = app.trim();
    if let Some(found) = apps
        .iter()
        .find(|a| a.bundle_id.eq_ignore_ascii_case(needle) || a.name.eq_ignore_ascii_case(needle))
    {
        return Ok(found.clone());
    }
    if let Some(found) = apps.iter().find(|a| {
        a.name
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase())
    }) {
        return Ok(found.clone());
    }
    Err(super::policy::no_running_app_message(app))
}

/// Start an installed app via `/usr/bin/open`. Arguments are passed without a
/// shell, so the app string can never be interpreted as code. Tries the value
/// as an application name first, then as a bundle identifier.
fn launch_app_process(app: &str) -> Result<(), String> {
    let by_name = Command::new("/usr/bin/open")
        .arg("-a")
        .arg(app)
        .output()
        .map_err(|e| format!("Failed to launch '{app}': {e}"))?;
    if by_name.status.success() {
        return Ok(());
    }
    let by_bundle = Command::new("/usr/bin/open")
        .arg("-b")
        .arg(app)
        .output()
        .map_err(|e| format!("Failed to launch '{app}': {e}"))?;
    if by_bundle.status.success() {
        return Ok(());
    }
    Err(format!(
        "Could not launch '{app}': not found as an application name or bundle id. \
         Check the exact name, or call list_apps to see what is already running. ({})",
        String::from_utf8_lossy(&by_name.stderr).trim()
    ))
}

/// Poll the running-app list until `app` appears or the timeout elapses.
fn wait_for_running_app(app: &str, timeout: Duration) -> Result<AppInfo, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(found) = resolve_app(app) {
            return Ok(found);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "Launched '{app}' but it did not register as running within {}s. It may still be \
                 starting — call list_apps or get_app_state again shortly.",
                timeout.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn bundle_id_for_pid(pid: i32) -> Option<String> {
    let script = format!(
        r#"tell application "System Events" to get bundle identifier of first application process whose unix id is {pid}"#
    );
    let out = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn focused_or_main_window(app: &AXUIElement) -> Result<AXUIElement, String> {
    let windows = windows_for_app(app)?;
    for window in &windows {
        if optional_bool_attr(window, AX_FOCUSED_ATTRIBUTE).unwrap_or(false)
            || optional_bool_attr(window, AX_MAIN_ATTRIBUTE).unwrap_or(false)
        {
            return Ok(window.clone());
        }
    }
    windows
        .into_iter()
        .next()
        .ok_or_else(|| "No accessible window found for app".to_string())
}

fn windows_for_app(app: &AXUIElement) -> Result<Vec<AXUIElement>, String> {
    let value = app
        .attribute(AX_WINDOWS_ATTRIBUTE)
        .map_err(|e| e.to_string())?;
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    Ok(items.iter().filter_map(|item| item.as_element()).collect())
}

struct WalkState {
    next_index: u32,
    nodes: u32,
    truncated: bool,
    started: Instant,
    limits: AxWalkLimits,
    elements: Vec<IndexedElement>,
    max_depth_seen: u32,
}

fn walk_tree(root: &AXUIElement, limits: AxWalkLimits) -> (Vec<IndexedElement>, bool) {
    let mut state = WalkState {
        // Incremented before assignment, so the first interactive element is 1.
        next_index: 0,
        nodes: 0,
        truncated: false,
        started: Instant::now(),
        limits,
        elements: Vec::new(),
        max_depth_seen: 0,
    };
    walk_node(root, 0, &mut state);
    if state.truncated && state.elements.is_empty() {
        // No interactive controls before the limit. The root role distinguishes
        // the real failure modes: a degraded/no-access tree shows the app
        // recursing into itself ("AXWindow" never appears), whereas a genuinely
        // deep app needs a higher depth/node budget.
        let root_role = optional_string_attr(root, AX_ROLE_ATTRIBUTE).unwrap_or_default();
        tracing::warn!(
            nodes = state.nodes,
            max_depth_seen = state.max_depth_seen,
            root_role,
            "computer_use AX walk truncated with no elements (check Accessibility permission / app depth)"
        );
    }
    (state.elements, state.truncated)
}

fn walk_node(element: &AXUIElement, depth: u32, state: &mut WalkState) {
    state.max_depth_seen = state.max_depth_seen.max(depth);
    if state.started.elapsed() > state.limits.max_duration
        || state.nodes >= state.limits.max_nodes
        || depth > state.limits.max_depth
    {
        state.truncated = true;
        return;
    }

    let role = optional_string_attr(element, AX_ROLE_ATTRIBUTE).unwrap_or_default();
    let title = pick_label(element);
    let enabled = optional_bool_attr(element, AX_ENABLED_ATTRIBUTE).unwrap_or(true);
    let subrole = optional_string_attr(element, AX_SUBROLE_ATTRIBUTE);
    let bounds = element_bounds(element);

    let interactive = is_interactive_role(&role);
    let context = !title.is_empty() && is_context_role(&role);
    if interactive || context {
        // Only interactive elements get a stable number the model can target;
        // context labels/value displays carry index 0 so they never shift the
        // button numbering when they appear/disappear (e.g. a result display).
        let index = if interactive {
            state.next_index = state.next_index.saturating_add(1);
            state.next_index
        } else {
            0
        };
        state.elements.push(IndexedElement {
            index,
            role,
            title,
            enabled,
            bounds,
            subrole,
            interactive,
        });
        state.nodes = state.nodes.saturating_add(1);
    }

    let children = match element.children() {
        Ok(children) => collapse_passthrough(element, children),
        Err(_) => return,
    };
    for child in children {
        walk_node(&child, depth.saturating_add(1), state);
        if state.truncated {
            return;
        }
    }
}

fn collapse_passthrough(parent: &AXUIElement, children: Vec<AXUIElement>) -> Vec<AXUIElement> {
    if children.len() != 1 {
        return children;
    }
    let role = optional_string_attr(parent, AX_ROLE_ATTRIBUTE).unwrap_or_default();
    // Use the *structural* label (title/description only, NOT AXValue): a wrapper
    // group that merely carries a value is still structurally empty and should be
    // collapsed, otherwise value-bearing groups exhaust the depth budget before
    // the real controls are reached.
    let title = structural_label(parent);
    if title.is_empty() && role.contains("Group") {
        if let Ok(grandchildren) = children[0].children() {
            return collapse_passthrough(&children[0], grandchildren);
        }
    }
    children
}

fn is_interactive_role(role: &str) -> bool {
    matches!(
        role,
        "AXButton"
            | "AXCheckBox"
            | "AXRadioButton"
            | "AXPopUpButton"
            | "AXMenuItem"
            | "AXTextField"
            | "AXTextArea"
            | "AXSlider"
            | "AXIncrementor"
            | "AXLink"
            | "AXComboBox"
            | "AXCell"
    ) || role.ends_with("Button")
}

fn is_context_role(role: &str) -> bool {
    role.contains("StaticText") || role.contains("Heading") || role.contains("Label")
}

fn pick_label(element: &AXUIElement) -> String {
    // Fall back to AXValue last so value-bearing displays (e.g. a calculator's
    // result field, a read-only text label) surface their content — these often
    // have an empty title/description but a meaningful value the model needs to
    // read back the outcome of an action.
    if let Some(label) = structural_label_opt(element) {
        return label;
    }
    optional_string_attr(element, AX_VALUE_ATTRIBUTE).unwrap_or_default()
}

/// Title/description label only (no AXValue). Used where a value must not count
/// as a meaningful label — notably passthrough-group collapsing.
fn structural_label(element: &AXUIElement) -> String {
    structural_label_opt(element).unwrap_or_default()
}

fn structural_label_opt(element: &AXUIElement) -> Option<String> {
    for attr in [AX_TITLE_ATTRIBUTE, AX_DESCRIPTION_ATTRIBUTE] {
        if let Some(value) = optional_string_attr(element, attr) {
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn optional_string_attr(element: &AXUIElement, name: &str) -> Option<String> {
    element.attribute(name).ok()?.and_then(|v| v.as_string())
}

fn optional_bool_attr(element: &AXUIElement, name: &str) -> Option<bool> {
    element.attribute(name).ok()?.and_then(|v| v.as_bool())
}

fn element_bounds(element: &AXUIElement) -> Option<ElementBounds> {
    let pos = element
        .attribute(AX_POSITION_ATTRIBUTE)
        .ok()?
        .and_then(|v| v.as_point())?;
    let size = element
        .attribute(AX_SIZE_ATTRIBUTE)
        .ok()?
        .and_then(|v| v.as_size())?;
    Some(ElementBounds {
        x: pos.x,
        y: pos.y,
        width: size.width,
        height: size.height,
    })
}

fn capture_window_png(
    pid: i32,
    app_name: &str,
    config: &ComputerUseConfig,
) -> Result<Vec<u8>, String> {
    let windows = Window::all().map_err(|e| e.to_string())?;
    // Match by pid or app name (case-insensitive); an app can run more than one
    // process, and the on-screen window may be owned by a sibling pid. Among the
    // matches, prefer a non-minimized window and the largest one — that is the
    // real document window rather than a menubar/popover/utility sliver.
    let mut candidates: Vec<Window> = windows
        .into_iter()
        .filter(|w| {
            w.pid().ok() == Some(pid as u32)
                || w.app_name()
                    .ok()
                    .is_some_and(|n| n.eq_ignore_ascii_case(app_name))
        })
        .collect();
    candidates.sort_by_key(|w| {
        let minimized = w.is_minimized().unwrap_or(false);
        let area = (w.width().unwrap_or(0) as u64) * (w.height().unwrap_or(0) as u64);
        // Non-minimized first, then largest area first.
        (minimized, std::cmp::Reverse(area))
    });
    let window = candidates
        .into_iter()
        .next()
        .ok_or_else(|| no_capturable_window_message(pid, app_name))?;
    let image = window.capture_image().map_err(|e| e.to_string())?;
    encode_png(resize_image(image, config)?, config)
}

/// Build an actionable error for when no on-screen window can be captured for an
/// app. `xcap` only sees windows on the *currently displayed* Space, so a window
/// living on another Mission Control Space or display is invisible here even
/// though the app is running and the accessibility tree can see it. Distinguish
/// "has windows, just not on this desktop" from "no window open at all", and
/// steer the model away from flailing into unrelated tools.
fn no_capturable_window_message(pid: i32, app_name: &str) -> String {
    let ax_window_count = AXUIElement::from_pid(pid)
        .and_then(|el| windows_for_app(&el).ok())
        .map(|w| w.len())
        .unwrap_or(0);
    if ax_window_count > 0 {
        format!(
            "Cannot capture {app_name}: it is running with {ax_window_count} window(s), but none \
             are on the desktop that is currently on screen — the window is most likely on another \
             Mission Control Space or display (or hidden/minimized). Ask the user to bring a \
             {app_name} window to the front on the active desktop, then retry. Do NOT switch to \
             terminal or other tools — computer_use cannot capture a window that is off the \
             current Space."
        )
    } else {
        format!(
            "Cannot capture {app_name}: it is running but has no open window. Ask the user to open \
             a {app_name} window (or open one yourself with a New Window shortcut after activating \
             it), then retry. Do NOT switch to terminal or other tools."
        )
    }
}

fn resize_image(image: RgbaImage, config: &ComputerUseConfig) -> Result<RgbaImage, String> {
    let (w, h) = (image.width(), image.height());
    let max_w = config.screenshot_max_width.max(1);
    let max_h = config.screenshot_max_height.max(1);
    if w <= max_w && h <= max_h {
        return Ok(image);
    }
    let scale = (max_w as f32 / w as f32).min(max_h as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    Ok(xcap::image::imageops::resize(
        &image,
        nw,
        nh,
        FilterType::Triangle,
    ))
}

fn encode_png(image: RgbaImage, config: &ComputerUseConfig) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    image
        .write_to(&mut cursor, xcap::image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    if buf.len() as u64 > config.screenshot_max_bytes {
        return Err(format!(
            "Screenshot exceeds screenshot_max_bytes ({})",
            config.screenshot_max_bytes
        ));
    }
    Ok(buf)
}

/// Ensure the target app is frontmost before sending synthetic keyboard/cursor
/// input. The OS routes synthetic events to whichever app currently has focus,
/// so if the target isn't frontmost we activate it and re-check rather than
/// dead-ending. Models rarely call `activate_app` on their own — they retry the
/// same blocked keystroke — so doing the activation here is what makes typing
/// into apps like Slack actually land instead of bouncing on "Activate first".
/// Returns true if the macOS login session reports the screen as locked. While
/// locked, loginwindow owns the foreground and synthetic input is undeliverable,
/// so this is a hard blocker we want to report distinctly rather than letting it
/// masquerade as a foreground mismatch.
fn screen_is_locked() -> bool {
    // SAFETY: standard CoreFoundation/CoreGraphics C calls. We own the copied
    // session dictionary and the created CFString, and release both; the value
    // fetched via CFDictionaryGetValue is not owned and is not released.
    unsafe {
        let dict = CGSessionCopyCurrentDictionary();
        if dict.is_null() {
            return false;
        }
        // kCFStringEncodingUTF8 = 0x0800_0100. Trailing NUL makes this a C string.
        let key_bytes = b"CGSSessionScreenIsLocked\0";
        let key = CFStringCreateWithCString(
            std::ptr::null(),
            key_bytes.as_ptr() as *const std::ffi::c_char,
            0x0800_0100,
        );
        let mut locked = false;
        if !key.is_null() {
            let value = CFDictionaryGetValue(dict, key);
            if !value.is_null() {
                locked = CFBooleanGetValue(value) != 0;
            }
            CFRelease(key);
        }
        CFRelease(dict);
        locked
    }
}

fn ensure_foreground(app: &AppInfo) -> Result<(), String> {
    // A locked screen blocks all synthetic input no matter how often we activate
    // — surface that plainly instead of looping on "foreground mismatch".
    if screen_is_locked() {
        return Err(LOCKED_SCREEN_HELP.to_string());
    }
    if verify_foreground(app).is_ok() {
        return Ok(());
    }
    activate_app(app)?;
    // Activation goes through the window server asynchronously; poll briefly for
    // focus to actually transfer before giving up.
    for _ in 0..12 {
        std::thread::sleep(Duration::from_millis(60));
        if verify_foreground(app).is_ok() {
            return Ok(());
        }
    }
    verify_foreground(app)
}

fn verify_foreground(app: &AppInfo) -> Result<(), String> {
    let system = axuielement::system_wide()
        .ok_or_else(|| "AX system-wide element unavailable".to_string())?;
    let focused_app = system
        .focused_application()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "No focused application".to_string())?;
    let focused_pid = focused_app.pid().map_err(|e| e.to_string())?;
    if focused_pid != app.pid {
        return Err(format!(
            "Foreground target mismatch — expected {} (pid {}). Activate the app first.",
            app.name, app.pid
        ));
    }
    Ok(())
}

fn activate_app(app: &AppInfo) -> Result<(), String> {
    // Activate by PID, not by name: the name could legitimately contain quotes
    // or other AppleScript metacharacters, so interpolating it into a script is
    // an injection risk. A PID is an integer with no escaping concerns.
    let script = format!(
        r#"tell application "System Events" to set frontmost of (first process whose unix id is {}) to true"#,
        app.pid
    );
    let out = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "Failed to activate app '{}': {}",
            app.name,
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// Try to activate the `target_index`-th interactive control via an AX press
/// (no cursor movement, no foreground requirement). Returns `Ok(true)` if the
/// control was pressed, `Ok(false)` if it can't be found or doesn't support a
/// press (caller falls back to a synthetic cursor click).
fn press_element_via_ax(
    pid: i32,
    target_index: u32,
    config: &ComputerUseConfig,
) -> Result<bool, String> {
    let app_element =
        AXUIElement::from_pid(pid).ok_or_else(|| format!("No AX element for pid {pid}"))?;
    let window = focused_or_main_window(&app_element)?;
    let limits = AxWalkLimits::from_config(config);
    let started = Instant::now();
    let mut counter = 0u32;
    let Some(live) =
        find_live_interactive(&window, 0, &limits, &started, &mut counter, target_index)
    else {
        return Ok(false);
    };
    Ok(live.perform_action(AX_PRESS_ACTION).is_ok())
}

/// Re-resolve the live `AXUIElement` for the `target`-th interactive control,
/// numbered identically to the snapshot walk (only interactive roles counted,
/// same depth-first order and passthrough collapsing).
fn find_live_interactive(
    element: &AXUIElement,
    depth: u32,
    limits: &AxWalkLimits,
    started: &Instant,
    counter: &mut u32,
    target: u32,
) -> Option<AXUIElement> {
    if started.elapsed() > limits.max_duration || depth > limits.max_depth {
        return None;
    }
    let role = optional_string_attr(element, AX_ROLE_ATTRIBUTE).unwrap_or_default();
    if is_interactive_role(&role) {
        *counter = counter.saturating_add(1);
        if *counter == target {
            return Some(element.clone());
        }
    }
    let children = match element.children() {
        Ok(children) => collapse_passthrough(element, children),
        Err(_) => return None,
    };
    for child in children {
        if let Some(found) = find_live_interactive(
            &child,
            depth.saturating_add(1),
            limits,
            started,
            counter,
            target,
        ) {
            return Some(found);
        }
    }
    None
}

fn click_element(element: &IndexedElement) -> Result<(), String> {
    let bounds = element
        .bounds
        .ok_or_else(|| format!("Element {} has no bounds for click", element.index))?;
    let cx = bounds.x + bounds.width / 2.0;
    let cy = bounds.y + bounds.height / 2.0;
    click_global_point(cx, cy)
}

fn click_global_point(x: f64, y: f64) -> Result<(), String> {
    let target_x = x.round() as i32;
    let target_y = y.round() as i32;
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
    enigo
        .move_mouse(target_x, target_y, Coordinate::Abs)
        .map_err(|e| e.to_string())?;
    if let Ok((cursor_x, cursor_y)) = read_mouse_position() {
        const CURSOR_TOLERANCE: i32 = 8;
        if (cursor_x - target_x).abs() > CURSOR_TOLERANCE
            || (cursor_y - target_y).abs() > CURSOR_TOLERANCE
        {
            return Err(
                "Cursor moved after positioning — aborting click to avoid hitting an unintended target"
                    .to_string(),
            );
        }
    }
    enigo
        .button(Button::Left, Direction::Click)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn read_mouse_position() -> Result<(i32, i32), String> {
    let output = Command::new("swift")
        .args([
            "-e",
            r#"import Cocoa
let p = NSEvent.mouseLocation
let screen = NSScreen.screens?.first?.frame.height ?? 0
print("\(Int(p.x)),\(Int(screen - p.y))")"#,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut parts = text.trim().split(',');
    let x: i32 = parts
        .next()
        .ok_or_else(|| "missing cursor x".to_string())?
        .trim()
        .parse()
        .map_err(|e| format!("invalid cursor x: {e}"))?;
    let y: i32 = parts
        .next()
        .ok_or_else(|| "missing cursor y".to_string())?
        .trim()
        .parse()
        .map_err(|e| format!("invalid cursor y: {e}"))?;
    Ok((x, y))
}

fn parse_key_token(part: &str) -> Result<Key, String> {
    match part.to_ascii_lowercase().as_str() {
        "return" | "enter" => Ok(Key::Return),
        "tab" => Ok(Key::Tab),
        "escape" | "esc" => Ok(Key::Escape),
        "space" | "spacebar" => Ok(Key::Space),
        "backspace" => Ok(Key::Backspace),
        "delete" | "del" => Ok(Key::Delete),
        // Navigation keys — models reach for these to scroll/move (e.g. PageDown
        // to scroll a feed), so accept the common spellings instead of erroring.
        "pagedown" | "page_down" | "pgdn" => Ok(Key::PageDown),
        "pageup" | "page_up" | "pgup" => Ok(Key::PageUp),
        "home" => Ok(Key::Home),
        "end" => Ok(Key::End),
        "up" | "uparrow" | "arrowup" => Ok(Key::UpArrow),
        "down" | "downarrow" | "arrowdown" => Ok(Key::DownArrow),
        "left" | "leftarrow" | "arrowleft" => Ok(Key::LeftArrow),
        "right" | "rightarrow" | "arrowright" => Ok(Key::RightArrow),
        "command" | "cmd" => Ok(Key::Meta),
        "shift" => Ok(Key::Shift),
        "control" | "ctrl" => Ok(Key::Control),
        "option" | "alt" => Ok(Key::Alt),
        other if other.chars().count() == 1 => Ok(Key::Unicode(other.chars().next().unwrap())),
        other => Err(format!("Unsupported key token '{other}'")),
    }
}

fn is_modifier_key(key: &Key) -> bool {
    matches!(key, Key::Meta | Key::Shift | Key::Control | Key::Alt)
}

fn press_key_combo(raw: &str) -> Result<(), String> {
    let keys = raw
        .split('+')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(parse_key_token)
        .collect::<Result<Vec<_>, _>>()?;
    let Some((main, modifiers)) = keys.split_last() else {
        return Err("empty key combo".to_string());
    };
    // Modifiers must stay held while the main key is pressed — clicking each
    // token independently (press+release) turns "Command+a" into a literal "a".
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
    for m in modifiers {
        enigo.key(*m, Direction::Press).map_err(|e| e.to_string())?;
    }
    let main_result = enigo.key(*main, Direction::Click);
    // Always release held modifiers, even if the main keypress failed, so we
    // never strand a modifier in the down state.
    for m in modifiers.iter().rev() {
        let _ = enigo.key(*m, Direction::Release);
    }
    main_result.map_err(|e| e.to_string())?;
    Ok(())
}

fn scroll_direction(direction: &str, pages: f64) -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default()).map_err(|e| e.to_string())?;
    let amount = (pages.abs() * 120.0).round() as i32;
    let delta = match direction.to_ascii_lowercase().as_str() {
        "up" => amount,
        "down" => -amount,
        "left" => amount,
        "right" => -amount,
        other => return Err(format!("Invalid scroll direction '{other}'")),
    };
    enigo
        .scroll(delta, enigo::Axis::Vertical)
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_combo(raw: &str) -> Vec<Key> {
        raw.split('+')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|p| parse_key_token(p).unwrap())
            .collect()
    }

    #[test]
    fn modifier_combo_separates_held_modifier_from_main_key() {
        // "Command+a" must hold Meta while clicking 'a' — not click each
        // independently (which would type a literal "a").
        let keys = parse_combo("Command+a");
        let (main, modifiers) = keys.split_last().unwrap();
        assert_eq!(modifiers.len(), 1);
        assert!(
            is_modifier_key(&modifiers[0]),
            "Command should be a modifier"
        );
        assert!(
            !is_modifier_key(main),
            "'a' should be the main key, not a modifier"
        );
    }

    #[test]
    fn single_key_has_no_modifiers() {
        let keys = parse_combo("Return");
        let (main, modifiers) = keys.split_last().unwrap();
        assert!(modifiers.is_empty());
        assert!(!is_modifier_key(main));
    }

    #[test]
    fn navigation_keys_are_supported() {
        for token in [
            "page_down",
            "PageDown",
            "pgdn",
            "down",
            "home",
            "space",
            "backspace",
        ] {
            assert!(
                parse_key_token(token).is_ok(),
                "expected '{token}' to be a supported key token"
            );
        }
        assert!(parse_key_token("definitely_not_a_key").is_err());
    }
}
