//! Live surfaces in the daemon: one registry for the process. The browser
//! tool adds its tabs, the Client Gateway serves them to the owner's apps,
//! and `hand_over` asks the owner to act on one.

use std::sync::{Arc, OnceLock};

use aidaemon_surfaces::SurfaceRegistry;

use crate::config::AppConfig;

static REGISTRY: OnceLock<SurfaceRegistry> = OnceLock::new();

#[cfg(feature = "browser")]
static BROWSER: OnceLock<Arc<crate::tools::BrowserTool>> = OnceLock::new();

/// The process's surface registry, made (and its upkeep started) on first
/// use. Needs a tokio runtime.
pub(crate) fn registry(config: &AppConfig) -> SurfaceRegistry {
    REGISTRY
        .get_or_init(|| {
            let registry = SurfaceRegistry::new(config.surfaces.registry_config());
            registry.start();
            registry
        })
        .clone()
}

/// The registry, when something made it.
pub(crate) fn running() -> Option<SurfaceRegistry> {
    REGISTRY.get().cloned()
}

/// The agent's browser shows its tabs as surfaces.
#[cfg(feature = "browser")]
pub(crate) fn add_browser(config: &AppConfig, browser: Arc<crate::tools::BrowserTool>) {
    let registry = registry(config);
    let backend = browser.surface_backend();
    backend.set_notifier(registry.notifier());
    registry.add_backend(backend);
    let _ = BROWSER.set(browser);
}

/// This computer's displays are surfaces too (`[surfaces] desktop`).
#[cfg(all(target_os = "macos", feature = "computer_use-macos"))]
pub(crate) fn add_desktop(config: &AppConfig) {
    registry(config).add_backend(Arc::new(
        aidaemon_runtime::tools::computer_use::live_desktop::DesktopSurface::new(),
    ));
}

/// Delete the sign-ins the agent's own browser kept.
pub(crate) async fn forget_sign_ins() -> Result<String, String> {
    #[cfg(feature = "browser")]
    if let Some(browser) = BROWSER.get() {
        return browser.forget_sign_ins().await;
    }
    Err("The agent's browser is not running.".to_string())
}

/// Where the owner's app opens a surface's live view: the gateway's address
/// a phone can reach (a tunnel or `public_url`), never a loopback one.
pub(crate) fn live_view_link() -> aidaemon_runtime::tools::hand_over::LiveViewLink {
    Arc::new(|id| {
        let gateway = crate::startup::client_gateway::running()?;
        let base = gateway.public_base();
        let reachable = base.starts_with("https://")
            && !base.contains("://127.0.0.1")
            && !base.contains("://localhost");
        reachable.then(|| format!("{base}/client/v1/next/#surface={id}"))
    })
}
