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
