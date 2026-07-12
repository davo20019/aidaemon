use crate::config::ComputerUseConfig;

/// Running application entry returned by `list_apps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppInfo {
    pub name: String,
    pub bundle_id: String,
    pub pid: i32,
}

/// Bounds in global display points (AX coordinate space).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementBounds {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// Upper bound of the normalized coordinate space the model points in. The
/// model emits click coordinates as `(x, y)` each in `[0, NORMALIZED_COORD_MAX]`
/// over the screenshot it was shown — it never has to reason about global
/// desktop points or Retina pixels, which it cannot derive from a window-cropped
/// image. The harness translates via `normalized_to_global_point`.
pub const NORMALIZED_COORD_MAX: f64 = 1000.0;

/// A window's on-screen frame in global display **points** (top-left origin) —
/// the same coordinate space as `ElementBounds` and the synthetic-cursor click.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowFrame {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub height: f64,
}

/// Map a normalized image coordinate (each axis in `[0, NORMALIZED_COORD_MAX]`)
/// to a global display point using the window's point-space frame.
///
/// Resolution/Retina-independent by construction: normalization erases the
/// pixel scale, so the mapping needs only the window's origin and size in
/// points — never the screenshot's pixel dimensions or a scale factor. This is
/// what makes vision-driven clicking work: the model points at the image it can
/// see, and the harness does the geometry it cannot. Out-of-range inputs clamp
/// to the window so a click can never land off the target window.
pub fn normalized_to_global_point(nx: f64, ny: f64, frame: WindowFrame) -> (f64, f64) {
    let fx = (nx / NORMALIZED_COORD_MAX).clamp(0.0, 1.0);
    let fy = (ny / NORMALIZED_COORD_MAX).clamp(0.0, 1.0);
    (
        frame.origin_x + fx * frame.width,
        frame.origin_y + fy * frame.height,
    )
}

/// One accessibility element in a snapshot.
///
/// Only *interactive* elements receive a stable numeric `index` (the model
/// targets these by `element_index`). Non-interactive context elements (labels,
/// value displays) carry `index == 0` and `interactive == false`: they are shown
/// to the model for context but never consume a number, so adding/removing a
/// label (e.g. a calculator's result display) can't shift the buttons' indices.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedElement {
    pub index: u32,
    pub role: String,
    pub title: String,
    pub enabled: bool,
    pub bounds: Option<ElementBounds>,
    pub subrole: Option<String>,
    pub interactive: bool,
}

/// Window + tree metadata for a captured app state.
#[derive(Debug, Clone, PartialEq)]
pub struct AppSnapshot {
    pub generation: u64,
    pub bundle_id: String,
    pub app_name: String,
    pub pid: i32,
    pub window_id: u64,
    pub window_title: String,
    pub elements: Vec<IndexedElement>,
    pub truncated: bool,
    pub png: Vec<u8>,
}

/// Limits applied while walking the accessibility tree.
#[derive(Debug, Clone, Copy)]
pub struct AxWalkLimits {
    pub max_depth: u32,
    pub max_nodes: u32,
    pub max_duration: std::time::Duration,
}

impl AxWalkLimits {
    pub fn from_config(config: &ComputerUseConfig) -> Self {
        Self {
            max_depth: config.ax_max_depth.max(1),
            max_nodes: config.ax_max_nodes.max(1),
            max_duration: config.action_timeout(),
        }
    }

    /// Walk limits for a specific app. Browsers render web content as a very deep
    /// accessibility tree — a feed's Like/Comment buttons can sit 30+ levels down,
    /// far past the native-app default depth — so give browser windows a much
    /// deeper and wider budget. The duration cap still bounds the walk.
    pub fn for_app(config: &ComputerUseConfig, bundle_id: &str) -> Self {
        let mut limits = Self::from_config(config);
        if is_browser_bundle(bundle_id) {
            limits.max_depth = limits.max_depth.max(40);
            limits.max_nodes = limits.max_nodes.max(1200);
        }
        limits
    }
}

/// Whether a bundle id is a web browser (whose content nests far deeper than a
/// native app's controls).
pub fn is_browser_bundle(bundle_id: &str) -> bool {
    let b = bundle_id.to_ascii_lowercase();
    [
        "chrome", "chromium", "safari", "edgemac", "firefox", "brave", "arc", "opera", "vivaldi",
    ]
    .iter()
    .any(|name| b.contains(name))
}

/// Serialize a full accessibility tree for `get_app_state`.
pub fn format_full_tree(snapshot: &AppSnapshot) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "app={} bundle_id={} pid={} window_id={} window_title={}\n",
        snapshot.app_name,
        snapshot.bundle_id,
        snapshot.pid,
        snapshot.window_id,
        snapshot.window_title
    ));
    out.push_str(&format!("snapshot_generation={}\n", snapshot.generation));
    if snapshot.truncated {
        out.push_str("[TRUNCATED: AX tree walk hit depth/node/time limit]\n");
    }
    for el in &snapshot.elements {
        out.push_str(&format_element_line(el));
    }
    // Web content (browsers) often exposes no addressable AX elements. Point the
    // model at the coordinate-click fallback instead of leaving it with an empty
    // list and no path — the historical dead end that drove defection to shell.
    if !snapshot.elements.iter().any(|el| el.interactive) {
        out.push_str(
            "[no addressable accessibility elements — this is normal for web pages and \
             custom-drawn UIs. To interact, look at the screenshot and click by COORDINATE: \
             call click with normalized x and y (0-1000 over the screenshot; no \
             snapshot_generation needed). Do NOT switch to terminal/AppleScript.]\n",
        );
    }
    tracing::info!(
        kind = "full",
        app = %snapshot.app_name,
        elements = snapshot.elements.len(),
        ax_chars = out.len(),
        ax_tokens_est = out.len() / 4,
        "computer_use observation size"
    );
    out
}

/// Rough token cost of a rendered computer-use observation, for telemetry into
/// where a computer-use prompt's bytes go: the AX-tree text (chars/4, matching
/// `memory::context_window::estimate_tokens`) plus a fixed surrogate for an
/// attached screenshot. The AX tree is the variable, sometimes-huge part (a
/// browser full tree can be ~1200 nodes); the screenshot is small and capped.
pub fn observation_token_estimate(ax_text: &str, has_screenshot: bool) -> usize {
    let ax_tokens = ax_text.len() / 4;
    let screenshot_tokens = if has_screenshot { 1200 } else { 0 };
    ax_tokens + screenshot_tokens
}

/// Condensed refresh returned after mutating actions.
pub fn format_condensed_refresh(snapshot: &AppSnapshot, focus_index: Option<u32>) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "app={} bundle_id={} pid={} window_id={} window_title={}\n",
        snapshot.app_name,
        snapshot.bundle_id,
        snapshot.pid,
        snapshot.window_id,
        snapshot.window_title
    ));
    out.push_str(&format!("snapshot_generation={}\n", snapshot.generation));
    out.push_str("[condensed refresh after mutation]\n");
    if let Some(idx) = focus_index {
        if let Some(el) = snapshot.elements.iter().find(|e| e.index == idx) {
            out.push_str("focus_element:\n");
            out.push_str(&format_element_line(el));
        }
    }
    // List ALL interactive (clickable) controls, not just the first few. After a
    // click the layout's other buttons (e.g. a calculator's "9", "+", "=") must
    // still be visible by index, otherwise the model re-runs the full, expensive
    // get_app_state every step just to find the next target — which is what
    // burns the task token budget. Cap keeps dense apps bounded.
    const MAX_INTERACTIVE: usize = 60;
    let total_interactive = snapshot.elements.iter().filter(|e| e.interactive).count();
    let interactive: Vec<_> = snapshot
        .elements
        .iter()
        .filter(|e| e.interactive)
        .take(MAX_INTERACTIVE)
        .map(format_element_line)
        .collect();
    if !interactive.is_empty() {
        out.push_str("interactive_elements:\n");
        for line in interactive {
            out.push_str(&line);
        }
        if total_interactive > MAX_INTERACTIVE {
            out.push_str(
                "[more interactive elements omitted — call get_app_state for the full tree]\n",
            );
        }
    }
    if snapshot.truncated {
        out.push_str("[TRUNCATED]\n");
    }
    tracing::info!(
        kind = "condensed",
        app = %snapshot.app_name,
        elements = snapshot.elements.len(),
        ax_chars = out.len(),
        ax_tokens_est = out.len() / 4,
        "computer_use observation size"
    );
    out
}

fn format_element_line(el: &IndexedElement) -> String {
    let label = if el.title.is_empty() {
        el.role.clone()
    } else {
        format!("{} \"{}\"", el.role, el.title)
    };
    if el.interactive {
        let state = if el.enabled { "enabled" } else { "disabled" };
        // Numbered, clickable: "{index} {role} "{title}" {state}".
        format!("{} {} {}\n", el.index, label, state)
    } else {
        // Context only (label / value display): no number, can't be targeted.
        format!("- {label} (context)\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_center_maps_to_window_center() {
        // Window at global origin (100, 200), 800×600 points. Center of the
        // normalized space (500, 500) → window center in global points.
        let frame = WindowFrame {
            origin_x: 100.0,
            origin_y: 200.0,
            width: 800.0,
            height: 600.0,
        };
        let (gx, gy) = normalized_to_global_point(500.0, 500.0, frame);
        assert!((gx - 500.0).abs() < 1e-9, "gx={gx}");
        assert!((gy - 500.0).abs() < 1e-9, "gy={gy}");
    }

    #[test]
    fn normalized_maps_are_retina_scale_independent() {
        // Same window frame in POINTS produces the same global point regardless
        // of the screenshot's pixel resolution — because normalization erases
        // the pixel scale. A button 25% across / 75% down a Retina (2×) window
        // and the same button on a 1× window both map to the same global point.
        let frame = WindowFrame {
            origin_x: 0.0,
            origin_y: 0.0,
            width: 1000.0,
            height: 1000.0,
        };
        let (gx, gy) = normalized_to_global_point(250.0, 750.0, frame);
        assert!((gx - 250.0).abs() < 1e-9);
        assert!((gy - 750.0).abs() < 1e-9);
    }

    #[test]
    fn normalized_corners_and_out_of_range_clamp_to_window() {
        let frame = WindowFrame {
            origin_x: 10.0,
            origin_y: 20.0,
            width: 100.0,
            height: 200.0,
        };
        assert_eq!(normalized_to_global_point(0.0, 0.0, frame), (10.0, 20.0));
        assert_eq!(
            normalized_to_global_point(1000.0, 1000.0, frame),
            (110.0, 220.0)
        );
        // Out-of-range clamps to the window edges — never clicks off-window.
        assert_eq!(
            normalized_to_global_point(-50.0, 5000.0, frame),
            (10.0, 220.0)
        );
    }

    #[test]
    fn browsers_get_a_deeper_walk_budget() {
        let config = ComputerUseConfig {
            ax_max_depth: 12,
            ax_max_nodes: 500,
            ..Default::default()
        };
        let chrome = AxWalkLimits::for_app(&config, "com.google.Chrome");
        assert!(chrome.max_depth >= 40, "browser depth should be raised");
        assert!(
            chrome.max_nodes >= 1200,
            "browser node budget should be raised"
        );

        let native = AxWalkLimits::for_app(&config, "com.apple.calculator");
        assert_eq!(
            native.max_depth, 12,
            "native apps keep the configured depth"
        );
        assert_eq!(native.max_nodes, 500);

        assert!(is_browser_bundle("com.apple.Safari"));
        assert!(is_browser_bundle("org.mozilla.firefox"));
        assert!(!is_browser_bundle("com.tinyspeck.slackmacgap"));
    }

    fn sample_snapshot() -> AppSnapshot {
        AppSnapshot {
            generation: 3,
            bundle_id: "com.apple.calculator".to_string(),
            app_name: "Calculator".to_string(),
            pid: 123,
            window_id: 1,
            window_title: "Calculator".to_string(),
            elements: vec![
                IndexedElement {
                    index: 1,
                    role: "AXButton".to_string(),
                    title: "7".to_string(),
                    enabled: true,
                    bounds: None,
                    subrole: None,
                    interactive: true,
                },
                IndexedElement {
                    index: 2,
                    role: "AXButton".to_string(),
                    title: "8".to_string(),
                    enabled: true,
                    bounds: None,
                    subrole: None,
                    interactive: true,
                },
            ],
            truncated: false,
            png: vec![0x89, 0x50, 0x4E, 0x47],
        }
    }

    #[test]
    fn full_tree_includes_generation_and_elements() {
        let text = format_full_tree(&sample_snapshot());
        assert!(text.contains("snapshot_generation=3"));
        assert!(text.contains("1 AXButton \"7\" enabled"));
        // A tree with addressable elements does NOT nag about coordinates.
        assert!(!text.contains("click by COORDINATE"));
    }

    #[test]
    fn empty_tree_steers_to_coordinate_click() {
        // Web content case: no addressable elements. The model must be pointed
        // at coordinate clicking, not left with an empty list (which drove
        // shell/AppleScript defection on 2026-07-12).
        let mut snap = sample_snapshot();
        snap.elements.clear();
        let text = format_full_tree(&snap);
        assert!(text.contains("click by COORDINATE"), "got: {text}");
        assert!(text.contains("normalized x and y"));
        assert!(text.contains("Do NOT switch to terminal"));
    }

    #[test]
    fn non_interactive_only_tree_still_steers_to_coordinate_click() {
        // Context labels (interactive=false) don't count as addressable.
        let mut snap = sample_snapshot();
        for el in &mut snap.elements {
            el.interactive = false;
            el.index = 0;
        }
        assert!(format_full_tree(&snap).contains("click by COORDINATE"));
    }

    #[test]
    fn observation_estimate_counts_ax_text_and_screenshot() {
        let text = "x".repeat(400); // 400 chars -> 100 tokens
        assert_eq!(observation_token_estimate(&text, false), 100);
        // The screenshot surrogate is added only when an image is attached.
        assert_eq!(observation_token_estimate(&text, true), 1300);
        // A real full-tree render is non-trivially sized.
        let full = format_full_tree(&sample_snapshot());
        assert!(observation_token_estimate(&full, true) > 1200);
    }

    #[test]
    fn condensed_refresh_is_shorter_or_equal_for_small_trees() {
        let snap = sample_snapshot();
        let full = format_full_tree(&snap);
        let condensed = format_condensed_refresh(&snap, Some(1));
        assert!(condensed.contains("condensed refresh"));
        assert!(full.contains("snapshot_generation=3"));
    }
}
