use async_trait::async_trait;

use super::cache::SnapshotCache;
use super::types::{AppInfo, AppSnapshot};

#[derive(Debug, Clone)]
pub struct HarnessRequestContext {
    pub task_id: String,
    pub session_id: String,
}

#[async_trait]
pub trait ComputerHarness: Send + Sync {
    fn check_permissions(&self) -> Result<(), String>;

    async fn list_apps(&self) -> Result<Vec<AppInfo>, String>;

    /// Launch an installed app that is not currently running and wait until it
    /// registers as a running process. Returns the app's identity (no screenshot
    /// is taken — capturing is gated behind per-app approval, which the caller
    /// runs before the first `get_app_state`). If the app is already running this
    /// is idempotent and returns the running instance.
    async fn launch_app(&self, app: &str) -> Result<AppInfo, String>;

    async fn get_app_state(
        &self,
        app: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String>;

    /// Capture only the app window image — no accessibility-tree walk and, by
    /// design, NO new cached snapshot. A screenshot is a pure observation: it
    /// must not advance the element generation, or every mutation issued after a
    /// screenshot would be rejected as stale (the model can't see the bumped
    /// number). The returned snapshot carries the PNG for delivery; its element
    /// list is empty and its generation field is unset (the dispatch reports the
    /// existing cached generation instead).
    async fn capture_screenshot(&self, app: &str) -> Result<AppSnapshot, String>;

    /// `generation` is optional: activation has no element target, so a stale
    /// snapshot can't misdirect it, and activating is often the first action on
    /// an app (before any get_app_state). When present it is still validated.
    async fn activate_app(
        &self,
        app: &str,
        generation: Option<u64>,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String>;

    async fn click(
        &self,
        app: &str,
        generation: u64,
        element_index: Option<u32>,
        x: Option<f64>,
        y: Option<f64>,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, Option<u32>, &'static str), String>;

    async fn type_text(
        &self,
        app: &str,
        generation: u64,
        text: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String>;

    async fn press_key(
        &self,
        app: &str,
        generation: u64,
        key: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<AppSnapshot, String>;

    async fn scroll(
        &self,
        app: &str,
        generation: u64,
        element_index: u32,
        direction: &str,
        pages: f64,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, u32), String>;

    async fn set_value(
        &self,
        app: &str,
        generation: u64,
        element_index: u32,
        value: &str,
        ctx: &HarnessRequestContext,
        cache: &mut SnapshotCache,
    ) -> Result<(AppSnapshot, u32), String>;
}
